"""Extract a PDF or build an English draft for bilingual review."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from regqa.ingest.artifacts import publish, sha256
from regqa.ingest.parse_pdf import parse_pdf
from regqa.ingest.structure import Corpus
from regqa.translate.checks import load_glossary
from regqa.translate.review_log import PRIORITY
from regqa.translate.translate_sections import TranslationSettings, translate_sections


def build_english(
    corpus: Corpus,
    model: str,
    config_path: Path,
    glossary_path: Path,
    prompt_log: Path,
    checkpoint_path: Path | None = None,
) -> Corpus:
    from google import genai
    from google.genai import errors, types

    settings = TranslationSettings.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    glossary = load_glossary(glossary_path)

    load_dotenv(".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in the environment or local .env before translation")
    translation_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(type=types.Type.STRING),
            "text": types.Schema(type=types.Type.STRING),
        },
        required=["title", "text"],
    )
    config = types.GenerateContentConfig(
        temperature=settings.temperature,
        response_mime_type="application/json",
        response_schema=translation_schema,
    )
    cached_responses: dict[str, str] = {}
    if checkpoint_path and checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cached_responses[str(entry["prompt_hash"])] = str(entry["response"])

    with (
        prompt_log.open("x", encoding="utf-8") as log,
        genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=settings.timeout_ms)
        ) as client,
    ):

        def generate(prompt: str) -> str:
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            log.write(json.dumps({"model": model, "prompt": prompt}, ensure_ascii=False) + "\n")
            log.flush()
            if prompt_hash in cached_responses:
                return cached_responses[prompt_hash]
            last_err: str | None = None
            for attempt in range(1, 6):
                try:
                    response = client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    break
                except errors.APIError as error:
                    detail = getattr(error, "message", "") or ""
                    last_err = f"Gemini API returned HTTP {error.code}"
                    if detail:
                        last_err = f"{last_err}: {detail}"
                    if error.code in {429, 500, 502, 503, 504} and attempt < 5:
                        time.sleep(5 * (2 ** (attempt - 1)))
                        continue
                    raise RuntimeError(last_err) from None
                except Exception as error:
                    last_err = f"Gemini request failed ({type(error).__name__})"
                    if attempt < 5:
                        time.sleep(5 * (2 ** (attempt - 1)))
                        continue
                    raise RuntimeError(last_err) from None
            else:
                raise RuntimeError(last_err or "Gemini request failed after retries")
            if (
                not response.candidates
                or response.candidates[0].finish_reason != types.FinishReason.STOP
                or not response.text
            ):
                raise ValueError("Gemini did not return a complete translation")
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8") as ckpt:
                    ckpt.write(
                        json.dumps(
                            {"prompt_hash": prompt_hash, "response": response.text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                cached_responses[prompt_hash] = response.text
            return response.text

        return translate_sections(corpus, glossary, settings, generate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Gemini model ID for an English draft")
    parser.add_argument("--priority", action="store_true", help="Draft the 16 review sections")
    parser.add_argument("--prompt-log", type=Path, help="New JSONL file for verbatim prompts")
    parser.add_argument("--translation-config", type=Path, default=Path("data/translation.yaml"))
    parser.add_argument("--glossary", type=Path, default=Path("data/glossary.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional JSONL checkpoint file to resume interrupted translation runs",
    )
    args = parser.parse_args()
    if bool(args.model) != bool(args.prompt_log):
        parser.error("--model and --prompt-log are required together")
    if args.priority and not args.model:
        parser.error("--priority requires --model")
    if args.model and args.output is None:
        parser.error("--model requires an explicit draft --output path")
    if args.pdf is None:
        pdfs = sorted(Path("data/source_id").glob("*.pdf"))
        if len(pdfs) != 1:
            parser.error("Provide a PDF path or place one PDF in data/source_id/")
        args.pdf = pdfs[0]
    default_output = args.output is None
    if default_output:
        args.output = Path(".tmp/corpus.id.json")
    if args.prompt_log and args.prompt_log.resolve() == args.output.resolve():
        parser.error("--output and --prompt-log must be different paths")
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        corpus = parse_pdf(args.pdf)
        if args.priority:
            sections = [s for s in corpus["sections"] if s["section_id"] in PRIORITY]
            if len(sections) != len(PRIORITY):
                raise ValueError("PDF is missing one or more of the 16 priority sections")
            corpus["sections"] = sections
        if args.model:
            corpus = build_english(
                corpus,
                args.model,
                args.translation_config,
                args.glossary,
                args.prompt_log,
                checkpoint_path=args.checkpoint,
            )
        payload = (json.dumps(corpus, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if default_output:
            args.output.parent.mkdir(exist_ok=True)
        publish(args.output.parent, {args.output.name: payload})
    except (OSError, ValueError, ImportError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")
    digest = sha256(payload)
    print(f"{len(corpus['sections'])} sections ({corpus['language']}) -> {args.output}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
