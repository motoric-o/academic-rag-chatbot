"""Validate Member B's evaluation JSONL before calibration."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Fact = Annotated[list[Text], Field(min_length=1)]
FIXTURE = "SYNTHETIC_FIXTURE"


class GoldSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    section_id: Text
    paragraph: Text | None = None


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Text
    question: Text
    type: Literal[
        "factual",
        "numeric",
        "procedural",
        "multi_section",
        "conditional",
        "unanswerable",
        "ambiguous",
    ]
    answerable: bool
    gold_answer: str
    key_facts: list[Fact]
    gold_spans: list[GoldSpan]
    split: Literal["dev", "test"]
    annotator: Text
    notes: str = ""


def validate_eval(
    path: Path, *, section_ids: set[str] | None = None, synthetic: bool = False
) -> list[EvalQuestion]:
    questions: dict[str, EvalQuestion] = {}
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                question = EvalQuestion.model_validate_json(line)
                if question.id in questions:
                    raise ValueError(f"duplicate question ID {question.id}")
                if synthetic != (FIXTURE in question.notes):
                    raise ValueError(f"{FIXTURE} rows require --synthetic and vice versa")
                if question.answerable:
                    if question.type == "unanswerable":
                        raise ValueError("an unanswerable question cannot have answerable=true")
                    if question.gold_answer.strip() in {"", "NOT_FOUND"}:
                        raise ValueError("answerable question requires a gold answer")
                    if not question.key_facts or not question.gold_spans:
                        raise ValueError("answerable question requires key facts and gold spans")
                elif question.key_facts or question.gold_spans:
                    raise ValueError("unanswerable question requires empty facts and spans")
                elif question.gold_answer.strip() not in {"", "NOT_FOUND"}:
                    raise ValueError("unanswerable question requires NOT_FOUND or an empty answer")
                for span in question.gold_spans:
                    if section_ids is not None and span.section_id not in section_ids:
                        raise ValueError(f"unknown section ID {span.section_id}")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from None
            questions[question.id] = question
    if not questions:
        raise ValueError(f"{path}: no questions")
    return list(questions.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_file", type=Path)
    parser.add_argument("--corpus", type=Path, help="Check gold section IDs against a corpus JSON")
    parser.add_argument("--synthetic", action="store_true", help="Require synthetic fixture rows")
    parser.add_argument("--expected-dev", type=int, help="Required dev count before calibration")
    args = parser.parse_args()
    try:
        section_ids = None
        if args.corpus:
            corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
            if args.synthetic != (FIXTURE in corpus.get("notes", "")):
                raise ValueError(f"{FIXTURE} corpus requires --synthetic and vice versa")
            section_ids = {section["section_id"] for section in corpus["sections"]}
        questions = validate_eval(args.eval_file, section_ids=section_ids, synthetic=args.synthetic)
        counts = Counter(question.split for question in questions)
        if args.expected_dev is not None and counts["dev"] != args.expected_dev:
            raise ValueError(f"expected {args.expected_dev} dev questions, found {counts['dev']}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Valid evaluation set: {counts['dev']} dev, {counts['test']} test")


if __name__ == "__main__":
    main()
