"""Experiment A1: Tokenizer analysis (ID vs EN tokens per passage, CPU-only).

Owner: Member A (see docs/project_spec.md §0.2, §10 A1).
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokenizerLanguageStats:
    """Tokenization statistics for one tokenizer on one language corpus."""

    tokenizer_id: str
    language: str
    n_sections: int
    n_nonempty_sections: int
    total_chars: int
    total_tokens: int
    mean_tokens_per_section: float
    median_tokens_per_section: float
    p95_tokens_per_section: float
    max_tokens_per_section: int
    chars_per_token: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerComparison:
    """ID vs. EN tokenization comparison for a single tokenizer."""

    tokenizer_id: str
    id_stats: TokenizerLanguageStats | None
    en_stats: TokenizerLanguageStats | None
    token_ratio_id_over_en: float | None
    matched_sections: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "id_stats": self.id_stats.to_dict() if self.id_stats else None,
            "en_stats": self.en_stats.to_dict() if self.en_stats else None,
            "token_ratio_id_over_en": self.token_ratio_id_over_en,
            "matched_sections": self.matched_sections,
        }


def _encode_length(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        return len(ids)
    if callable(tokenizer):
        return len(tokenizer(text))
    raise TypeError(f"Unsupported tokenizer object: {type(tokenizer).__name__}")


def count_corpus_tokens(
    sections: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    tokenizer_id: str,
    language: str,
) -> tuple[TokenizerLanguageStats, dict[str, int]]:
    """Count tokens per section for a structured corpus."""
    per_section: dict[str, int] = {}
    counts: list[int] = []
    total_chars = 0

    for sec in sections:
        sec_id = str(sec.get("section_id", ""))
        title = str(sec.get("title", "")).strip()
        text = str(sec.get("text", "")).strip()
        combined = f"{title}\n{text}".strip() if title else text
        total_chars += len(combined)
        n_tok = _encode_length(tokenizer, combined)
        per_section[sec_id] = n_tok
        if combined:
            counts.append(n_tok)

    total_tokens = int(sum(counts))
    if counts:
        mean_tok = round(float(np.mean(counts)), 2)
        median_tok = round(float(np.median(counts)), 2)
        p95_tok = round(float(np.percentile(counts, 95)), 2)
        max_tok = int(max(counts))
    else:
        mean_tok = median_tok = p95_tok = 0.0
        max_tok = 0

    chars_per_token = round(total_chars / total_tokens, 3) if total_tokens > 0 else 0.0

    stats = TokenizerLanguageStats(
        tokenizer_id=tokenizer_id,
        language=language,
        n_sections=len(sections),
        n_nonempty_sections=len(counts),
        total_chars=total_chars,
        total_tokens=total_tokens,
        mean_tokens_per_section=mean_tok,
        median_tokens_per_section=median_tok,
        p95_tokens_per_section=p95_tok,
        max_tokens_per_section=max_tok,
        chars_per_token=chars_per_token,
    )
    return stats, per_section


def compare_language_tokenization(
    *,
    tokenizer_id: str,
    tokenizer: Any,
    id_sections: Sequence[Mapping[str, Any]] | None = None,
    en_sections: Sequence[Mapping[str, Any]] | None = None,
) -> TokenizerComparison:
    """Compute ID, EN, and paired ID/EN token expansion ratio for one tokenizer."""
    if id_sections is None and en_sections is None:
        raise ValueError("At least one of id_sections or en_sections must be provided")

    id_stats: TokenizerLanguageStats | None = None
    en_stats: TokenizerLanguageStats | None = None
    id_map: dict[str, int] = {}
    en_map: dict[str, int] = {}

    if id_sections is not None:
        id_stats, id_map = count_corpus_tokens(
            id_sections, tokenizer, tokenizer_id=tokenizer_id, language="id"
        )
    if en_sections is not None:
        en_stats, en_map = count_corpus_tokens(
            en_sections, tokenizer, tokenizer_id=tokenizer_id, language="en"
        )

    ratio: float | None = None
    matched = 0
    if id_sections is not None and en_sections is not None:
        common = [sid for sid in id_map if sid in en_map and id_map[sid] > 0 and en_map[sid] > 0]
        matched = len(common)
        if common:
            id_sum = sum(id_map[sid] for sid in common)
            en_sum = sum(en_map[sid] for sid in common)
            ratio = round(id_sum / en_sum, 4) if en_sum > 0 else None

    return TokenizerComparison(
        tokenizer_id=tokenizer_id,
        id_stats=id_stats,
        en_stats=en_stats,
        token_ratio_id_over_en=ratio,
        matched_sections=matched,
    )


def _load_corpus_sections(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sections = data.get("sections")
    if not isinstance(sections, list):
        raise ValueError(f"{path}: expected a JSON object with a 'sections' list")
    return sections


def analyze_tokenizers(
    tokenizer_ids: Sequence[str],
    *,
    id_corpus_path: str | Path | None = None,
    en_corpus_path: str | Path | None = None,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> list[TokenizerComparison]:
    """Run Experiment A1 across one or more Hugging Face tokenizers (CPU-only)."""
    id_sections = _load_corpus_sections(id_corpus_path) if id_corpus_path else None
    en_sections = _load_corpus_sections(en_corpus_path) if en_corpus_path else None

    if tokenizer_loader is None:
        transformers_mod = importlib.import_module("transformers")
        tokenizer_loader = lambda tid: transformers_mod.AutoTokenizer.from_pretrained(tid)  # noqa: E731

    results: list[TokenizerComparison] = []
    for tid in tokenizer_ids:
        tok = tokenizer_loader(tid)
        results.append(
            compare_language_tokenization(
                tokenizer_id=tid,
                tokenizer=tok,
                id_sections=id_sections,
                en_sections=en_sections,
            )
        )
    return results


def format_tokenizer_markdown(comparisons: Sequence[TokenizerComparison]) -> str:
    """Format Experiment A1 results as a Markdown summary table."""
    header = (
        "| Tokenizer | ID Total Tok | ID Mean/Sec | ID p95 | ID Chars/Tok | "
        "EN Total Tok | EN Mean/Sec | ID/EN Ratio |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for c in comparisons:
        id_total = str(c.id_stats.total_tokens) if c.id_stats else "—"
        id_mean = f"{c.id_stats.mean_tokens_per_section:.1f}" if c.id_stats else "—"
        id_p95 = f"{c.id_stats.p95_tokens_per_section:.1f}" if c.id_stats else "—"
        id_cpt = f"{c.id_stats.chars_per_token:.2f}" if c.id_stats else "—"
        en_total = str(c.en_stats.total_tokens) if c.en_stats else "—"
        en_mean = f"{c.en_stats.mean_tokens_per_section:.1f}" if c.en_stats else "—"
        ratio = f"{c.token_ratio_id_over_en:.3f}" if c.token_ratio_id_over_en is not None else "—"
        rows.append(
            f"| {c.tokenizer_id} | {id_total} | {id_mean} | {id_p95} | {id_cpt} | "
            f"{en_total} | {en_mean} | {ratio} |"
        )
    return "\n".join(rows)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Experiment A1: ID vs EN tokenizer analysis.")
    parser.add_argument("--id-corpus", type=Path, help="Path to Indonesian corpus JSON")
    parser.add_argument("--en-corpus", type=Path, help="Path to English corpus JSON")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        required=True,
        help="One or more Hugging Face tokenizer IDs (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if args.id_corpus is None and args.en_corpus is None:
        parser.error("Provide at least one of --id-corpus or --en-corpus")

    comparisons = analyze_tokenizers(
        args.tokenizers,
        id_corpus_path=args.id_corpus,
        en_corpus_path=args.en_corpus,
    )
    print(format_tokenizer_markdown(comparisons))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([c.to_dict() for c in comparisons], indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    _cli()
