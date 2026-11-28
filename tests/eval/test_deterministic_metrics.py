"""Hand-computed unit tests for deterministic evaluation, stats, and reporting (§11, §13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scipy.stats import binomtest

from regqa.evaluation.abstention import evaluate_abstention_batch
from regqa.evaluation.citation import evaluate_citation_batch, score_citations
from regqa.evaluation.keyfact import evaluate_keyfact_batch, normalize_for_match, score_key_facts
from regqa.evaluation.report import build_full_report, write_report
from regqa.evaluation.stats import (
    bootstrap_ci,
    compare_systems_pairwise,
    exact_mcnemar,
    flip_analysis,
    holm_correction,
    paired_bootstrap_diff,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_keyfact_number_normalization_and_strict_scoring() -> None:
    assert "2 semesters" in normalize_for_match("Students may take two semesters.")
    assert "60%" in normalize_for_match("Weight is 60 percent.")
    assert "2" in normalize_for_match("GPA >= 2.0")

    eval_item = {
        "id": "q1",
        "answerable": True,
        "gold_answer": "Two consecutive semesters.",
        "key_facts": [["2 semesters", "two semesters"], ["consecutive"]],
    }
    good = {
        "qid": "q1",
        "parsed": {
            "answer_text": "Up to two semesters, consecutive.",
            "format_ok": True,
            "abstained": False,
        },
        "error": None,
    }
    res_good = score_key_facts(good, eval_item)
    assert res_good.strict_correct is True
    assert res_good.soft_recall == 1.0

    # Malformed output (format_ok=False) MUST score strict_correct=False even if facts match (§8)
    bad_fmt = {
        "qid": "q1",
        "parsed": {
            "answer_text": "Up to 2 semesters, consecutive.",
            "format_ok": False,
            "abstained": False,
        },
        "error": None,
    }
    res_bad = score_key_facts(bad_fmt, eval_item)
    assert res_bad.strict_correct is False
    assert res_bad.soft_recall == 1.0

    # Unanswerable question
    unans_item = {"id": "q2", "answerable": False, "gold_answer": "NOT_FOUND", "key_facts": []}
    pred_abstain = {
        "qid": "q2",
        "parsed": {"answer_text": "NOT_FOUND", "format_ok": True, "abstained": True},
        "error": None,
    }
    assert score_key_facts(pred_abstain, unans_item).strict_correct is True

    batch = evaluate_keyfact_batch([good, pred_abstain], {"q1": eval_item, "q2": unans_item})
    assert batch["strict_correctness"] == 1.0
    assert batch["format_ok_rate"] == 1.0


def test_citation_precision_recall_and_invalid_ids() -> None:
    eval_item = {
        "id": "q1",
        "answerable": True,
        "gold_spans": [{"section_id": "8.1"}, {"section_id": "8.4"}],
    }
    chunks = [
        {"chunk_id": "c1", "section_ids": ["8.1"], "text": "A"},
        {"chunk_id": "c2", "section_ids": ["4.2"], "text": "B"},
        {"chunk_id": "c3", "section_ids": ["8.4"], "text": "C"},
        {"chunk_id": "c4", "section_ids": ["1.1"], "text": "D"},
        {"chunk_id": "c5", "section_ids": ["9.1"], "text": "E"},  # 5th chunk not fed (top_k=4)
    ]
    pred = {"qid": "q1", "parsed": {"cited_ids": [1, 2, 5]}, "error": None}
    score = score_citations(pred, eval_item, chunks, top_k_context=4)
    assert score.valid_cited_ids == [1, 2]
    assert score.invalid_cited_ids == [5]
    assert score.has_invalid_citation is True
    # 1 relevant out of 3 cited (including invalid [5])
    assert score.precision == pytest.approx(1 / 3, abs=1e-4)
    # 1 gold section covered ("8.1") out of 2 ("8.1", "8.4")
    assert score.recall == pytest.approx(0.5)

    batch = evaluate_citation_batch(
        [pred], {"q1": eval_item}, {"q1": {"qid": "q1", "chunks": chunks}}, top_k_context=4
    )
    assert batch["invalid_citation_rate"] == 1.0
    assert batch["total_invalid_citations"] == 1


def test_abstention_confusion_matrix_and_evidence_split() -> None:
    eval_by_qid = {
        "u1": {"id": "u1", "answerable": False},
        "u2": {"id": "u2", "answerable": False},
        "a1": {"id": "a1", "answerable": True},
        "a2": {"id": "a2", "answerable": True},
        "a3": {"id": "a3", "answerable": True},
    }
    contexts_by_qid = {
        "a1": {"qid": "a1", "evidence_in_context": True},
        "a2": {"qid": "a2", "evidence_in_context": True},
        "a3": {"qid": "a3", "evidence_in_context": False},
    }
    preds = [
        {"qid": "u1", "parsed": {"abstained": True}, "error": None},   # TP
        {"qid": "u2", "parsed": {"abstained": False}, "error": None},  # FN (false answer)
        {"qid": "a1", "parsed": {"abstained": False}, "error": None},  # TN (ev=True)
        {"qid": "a2", "parsed": {"abstained": True}, "error": None},   # FP (ev=True)
        {"qid": "a3", "parsed": {"abstained": True}, "error": None},   # FP (ev=False)
    ]
    summary = evaluate_abstention_batch(preds, eval_by_qid, contexts_by_qid)
    assert (summary.tp, summary.fn, summary.fp, summary.tn) == (1, 1, 2, 1)
    assert summary.false_answer_rate == pytest.approx(0.5)
    assert summary.over_abstention_rate == pytest.approx(2 / 3, abs=1e-4)
    assert summary.over_abstention_with_evidence == pytest.approx(0.5)
    assert summary.over_abstention_without_evidence == pytest.approx(1.0)


def test_stats_bootstrap_mcnemar_holm_and_flips() -> None:
    ci1 = bootstrap_ci([1, 1, 0, 1, 0], n_resamples=1000, seed=42)
    ci2 = bootstrap_ci([1, 1, 0, 1, 0], n_resamples=1000, seed=42)
    assert ci1 == ci2
    assert ci1.mean == pytest.approx(0.6)

    a = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    mcn = exact_mcnemar(a, b)
    expected_p = float(binomtest(1, 9, p=0.5, alternative="two-sided").pvalue)
    assert mcn.n10_correct_to_wrong == 1
    assert mcn.n01_wrong_to_correct == 8
    assert mcn.p_value == pytest.approx(expected_p, abs=1e-5)

    assert holm_correction([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]

    flips = flip_analysis(
        a,
        b,
        qids=[f"q{i}" for i in range(10)],
        repeat_matrix_a=[[1, 1], [1, 0]],
    )
    assert flips.correct_to_wrong == 1
    assert flips.wrong_to_correct == 8
    assert flips.net_gain == 7
    assert flips.repeat_instability_a == pytest.approx(0.5)

    diff = paired_bootstrap_diff(a, b, n_resamples=1000, seed=42)
    assert diff.mean_diff == pytest.approx(0.7)
    assert diff.is_inconclusive is False

    pairs = compare_systems_pairwise(
        {
            "sysA": {f"q{i}": v for i, v in enumerate(a)},
            "sysB": {f"q{i}": v for i, v in enumerate(b)},
        },
        n_resamples=1000,
        seed=42,
    )
    assert len(pairs) == 1
    assert pairs[0]["mcnemar"]["p_value_holm"] == pairs[0]["mcnemar"]["p_value"]


def test_build_and_write_report_end_to_end(tmp_path: Path) -> None:
    pred_a = tmp_path / "pred_a.jsonl"
    pred_b = tmp_path / "pred_b.jsonl"
    rows_a = [
        {
            "qid": "fixture_01",
            "model": "qwen3-0.6b-4bit",
            "precision": "4bit",
            "raw_output": "Up to three consecutive terms. Sources: [1]",
            "parsed": {
                "answer_text": "Up to three consecutive terms.",
                "cited_ids": [1],
                "abstained": False,
                "format_ok": True,
            },
            "tokens_in": 120,
            "tokens_out": 10,
            "latency_s": 0.4,
            "peak_vram_mb": 550.0,
            "hardware": {"device": "cuda", "gpu_name": "NVIDIA GeForce RTX 3070"},
            "error": None,
        },
        {
            "qid": "fixture_09",
            "model": "qwen3-0.6b-4bit",
            "precision": "4bit",
            "raw_output": "NOT_FOUND",
            "parsed": {
                "answer_text": "NOT_FOUND",
                "cited_ids": [],
                "abstained": True,
                "format_ok": True,
            },
            "tokens_in": 120,
            "tokens_out": 2,
            "latency_s": 0.2,
            "peak_vram_mb": 550.0,
            "hardware": {"device": "cuda", "gpu_name": "NVIDIA GeForce RTX 3070"},
            "error": None,
        },
    ]
    rows_b = [{**r, "model": "qwen3-4b-4bit", "peak_vram_mb": 2600.0} for r in rows_a]
    pred_a.write_text("\n".join(json.dumps(r) for r in rows_a) + "\n", encoding="utf-8")
    pred_b.write_text("\n".join(json.dumps(r) for r in rows_b) + "\n", encoding="utf-8")

    report = build_full_report(
        [pred_a, pred_b],
        REPO_ROOT / "tests" / "fixtures" / "mini_eval.jsonl",
        n_resamples=1000,
        seed=42,
    )
    json_path, md_path = write_report(report, tmp_path / "out")
    assert json_path.is_file()
    assert md_path.is_file()
    md = md_path.read_text(encoding="utf-8")
    assert "Headline Quality Metrics" in md
    assert "Pairwise Comparisons" in md
