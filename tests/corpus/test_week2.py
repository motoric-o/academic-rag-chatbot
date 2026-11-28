"""Regression checks for Week 2 translation review and CPU retrieval artifacts."""

import csv
import json
from pathlib import Path

import pytest

from regqa.evaluation.retrieval_metrics import found, retrieval_metrics
from regqa.evaluation.validate_eval import validate_eval
from regqa.ingest.structure import Section
from regqa.retrieval.build_cache import cache_files, oracle_chunks
from regqa.retrieval.calibrate_chunks import select_size
from regqa.retrieval.chunking import Chunk, ChunkLimitError, make_chunks
from regqa.retrieval.hybrid import Ranking
from regqa.translate.review_log import FIELDS, read_review_log

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_section_chunking_keeps_paragraphs_and_overlaps_only_whole_sentences() -> None:
    section: Section = {
        "section_id": "8.3",
        "title": "Review",
        "text": (
            "1. First steps apply to new students. Sign it. "
            "Every approved participant can send the advisor the form.\n\n"
            "2. File it separately."
        ),
        "page_start": 1,
        "page_end": 1,
    }

    def count(text: str) -> int:
        return len(text.split())

    header = "[Chapter {chapter} > Article {article}]"
    chunks = make_chunks([section], count, limit=17, overlap_ratio=0.15, header_template=header)
    assert len(chunks) == 3
    assert chunks[0]["text"].startswith("[Chapter 8 > Article 8.3] Review\n1.")
    assert chunks[0]["text"].endswith("Sign it.")
    assert "\nSign it. Every approved participant" in chunks[1]["text"]
    assert chunks[2]["text"].split("\n", 1)[1] == "2. File it separately."
    assert all(count(c["text"]) <= 17 for c in chunks)
    assert all(c["section_ids"] == ["8.3"] for c in chunks)
    assert not found(chunks[:2], {("8.3", "2")})
    assert oracle_chunks(chunks, {("8.3", "2")}, 2) == chunks[2:]
    with pytest.raises(ChunkLimitError, match="sentence exceeds"):
        make_chunks([section], count, limit=8, overlap_ratio=0.15, header_template=header)


def test_calibration_uses_paired_recall_and_prefers_the_smaller_tie() -> None:
    chosen, intervals = select_size(
        {128: [0.0] * 10, 256: [1.0] * 9 + [0.0], 512: [1.0] * 10}, seed=42, samples=2000
    )
    assert chosen == 256
    assert intervals[128] == [1.0, 1.0]
    assert intervals[256][0] == 0.0


def test_cache_keeps_scores_aligned_and_oracle_contains_only_dev_gold() -> None:
    question = validate_eval(FIXTURES / "mini_eval.jsonl", synthetic=True)[0]
    assert question.answerable
    gold = question.gold_spans[0].section_id
    ranked: list[Chunk] = [
        {"chunk_id": f"x:{i}", "section_ids": ["x"], "text": "Example context"} for i in range(4)
    ]
    ranked.append({"chunk_id": f"{gold}:1", "section_ids": [gold], "text": "Example evidence"})
    ranks = [Ranking(ranked, [1.0, 0.8, 0.6, 0.4, 0.2])]
    files = cache_files([question], ranks, k=5, top_k=4, synthetic=True)
    retrieval = json.loads(files["contexts.jsonl"])
    oracle = json.loads(files["contexts_oracle.jsonl"])
    assert retrieval["scores"][-1] == 0.2
    assert not retrieval["evidence_in_context"]
    assert oracle["evidence_in_context"] and oracle["chunks"] == ranked[-1:]
    metrics = retrieval_metrics([(question, ranked)])
    assert metrics["recall@4"] == 0 and metrics["recall@5"] == 1
    assert metrics["mrr@10"] == 0.2


def test_dev_only_loader_does_not_validate_test_labels(tmp_path: Path) -> None:
    path = tmp_path / "SYNTHETIC_FIXTURE.jsonl"
    dev = validate_eval(FIXTURES / "mini_eval.jsonl", synthetic=True)[0]
    path.write_text(dev.model_dump_json() + '\n{"split":"test"}\n', encoding="utf-8")
    assert validate_eval(path, split="dev", synthetic=True) == [dev]
    with pytest.raises(ValueError, match="Field required"):
        validate_eval(path, synthetic=True)


def test_review_log_rejects_incomplete_corrections(tmp_path: Path) -> None:
    path = tmp_path / "SYNTHETIC_FIXTURE.csv"
    review = ["8.3", "SYNTHETIC_FIXTURE", "2026-11-28", "corrected", "omission"]
    with path.open("w", encoding="utf-8", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(FIELDS)
        writer.writerow([*review, "source", "draft", "corrected", ""])
    assert read_review_log(path, {"8.3"})["8.3"]["error_type"] == "omission"
    with path.open("w", encoding="utf-8", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(FIELDS)
        writer.writerow([*review, "source", "draft", "", ""])
    with pytest.raises(ValueError, match="require all three excerpts"):
        read_review_log(path, {"8.3"})
