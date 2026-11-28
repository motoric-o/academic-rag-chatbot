"""Summarize human review and freeze a complete, reviewed corpus."""

import argparse
import csv
from datetime import date
from math import ceil
from pathlib import Path

import yaml

from regqa.evaluation.validate_eval import DEV_COUNT, validate_eval
from regqa.ingest.artifacts import publish, read_corpus, sha256
from regqa.translate.checks import load_glossary, normalized, translation_issues
from regqa.translate.translate_sections import TranslationSettings

FIELDS = (
    "section_id",
    "reviewer",
    "review_date",
    "status",
    "error_type",
    "source_excerpt",
    "draft_en_excerpt",
    "corrected_en_excerpt",
    "notes",
)
ERROR_TYPES = {
    "numeric_or_date",
    "modal_verb",
    "glossary_mismatch",
    "omission",
    "hallucination",
    "table_layout",
    "mistranslation",
}
# The 2024 handbook batch selected by the bilingual lead in docs/handoffs/a_to_b.md.
PRIORITY = frozenset(
    "4.2 6.1.9 6.1.10 6.2.1 6.2.2 8.1 8.2 8.3 8.4 8.5 8.6 8.7 9.1 9.2 9.3 9.4".split()
)


def read_review_log(path: Path, section_ids: set[str]) -> dict[str, dict[str, str]]:
    reviewed = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        if tuple(rows.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: unexpected review log header")
        for row in rows:
            line = f"{path}:{rows.line_num}"
            if None in row or None in row.values():
                raise ValueError(f"{line}: wrong number of columns")
            row = {field: value.strip() for field, value in row.items()}
            section_id = row["section_id"]
            if section_id not in section_ids or section_id in reviewed or not row["reviewer"]:
                raise ValueError(f"{line}: unknown or duplicate section, or missing reviewer")
            try:
                if date.fromisoformat(row["review_date"]).isoformat() != row["review_date"]:
                    raise ValueError
            except ValueError:
                raise ValueError(f"{line}: review_date must use YYYY-MM-DD") from None
            status, error = row["status"], row["error_type"]
            if not (
                (status == "approved" and error == "none")
                or (status in {"corrected", "flagged"} and error in ERROR_TYPES)
            ):
                raise ValueError(f"{line}: inconsistent review status and error type")
            excerpts = ("source_excerpt", "draft_en_excerpt", "corrected_en_excerpt")
            if status == "corrected" and not all(row[field] for field in excerpts):
                raise ValueError(f"{line}: corrected reviews require all three excerpts")
            reviewed[section_id] = row
    return reviewed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".tmp/corpus.id.json"))
    parser.add_argument("--log", type=Path, default=Path("data/review_log.csv"))
    parser.add_argument("--eval", type=Path, help="Human-authored eval JSONL")
    parser.add_argument("--reviewed", type=Path, help="Corrected English draft")
    parser.add_argument("--freeze", type=Path, help="New frozen corpus.json and SHA-256 sidecar")
    parser.add_argument("--glossary", type=Path, default=Path("data/glossary.yaml"))
    parser.add_argument("--translation-config", type=Path, default=Path("data/translation.yaml"))
    args = parser.parse_args()
    if args.freeze and (not args.reviewed or not args.eval):
        parser.error("--freeze requires --reviewed and --eval")
    if args.freeze and args.freeze.suffix != ".json":
        parser.error("--freeze expects a .json output path")
    try:
        source = read_corpus(args.source.read_bytes(), language="id")
        original = {s["section_id"]: s for s in source["sections"]}
        rows = read_review_log(args.log, set(original))
        required = PRIORITY.intersection(original)
        questions = validate_eval(args.eval, section_ids=set(original)) if args.eval else []
        required |= {span.section_id for q in questions for span in q.gold_spans}
        approved = {s for s, row in rows.items() if row["status"] != "flagged"}
        missing = sorted(required - approved)
        errors = sum(row["status"] != "approved" for row in rows.values())
        rate = f"{errors / len(rows):.1%}" if rows else "n/a"
        print(f"Reviewed: {len(rows)}/{len(original)} sections. Review error rate: {rate}.")
        if missing:
            print("Pending priority or eval sections: " + ", ".join(missing))
        if not args.freeze:
            return
        if (
            missing
            or rows.keys() - approved
            or len(approved) < ceil(0.15 * len(original))
            or sum(q.split == "dev" for q in questions) != DEV_COUNT
        ):
            raise ValueError(f"Freeze requires completed reviews and {DEV_COUNT} dev questions")
        payload = args.reviewed.read_bytes()
        english = read_corpus(payload, language="en")
        targets = {s["section_id"]: s for s in english["sections"]}
        if list(targets) != list(original) or english["source_sha256"] != source["source_sha256"]:
            raise ValueError("English corpus must cover the same sections and source PDF")
        glossary = load_glossary(args.glossary)
        settings = TranslationSettings.model_validate(
            yaml.safe_load(args.translation_config.read_text(encoding="utf-8"))
        )
        for section_id, target in targets.items():
            section = original[section_id]
            if any(section[field] != target[field] for field in ("page_start", "page_end")):
                raise ValueError(f"{section_id}: source page range changed")
            for field in ("title", "text"):
                issues = translation_issues(
                    section[field],
                    target[field],
                    glossary,
                    min_ratio=settings.min_length_ratio,
                    max_ratio=settings.max_length_ratio,
                )
                if issues:
                    raise ValueError(f"{section_id} {field}: " + ", ".join(issues))
            row = rows.get(section_id)
            if row and row["status"] == "corrected":
                final_text = normalized(f"{target['title']} {target['text']}")
                if normalized(row["corrected_en_excerpt"]) not in final_text:
                    raise ValueError(f"{section_id}: logged correction is missing from the draft")
        digest = sha256(payload)
        publish(
            args.freeze.parent,
            {
                args.freeze.name: payload,
                args.freeze.with_suffix(".sha256").name: (digest + "\n").encode("ascii"),
            },
        )
        print(f"Frozen corpus SHA-256 {digest}")
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
