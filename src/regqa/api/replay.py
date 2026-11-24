"""Replay service for offline frontend development and demo mode.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §12.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from regqa.api.schemas import AskResponse, SourceCitation


# Sample canned mock data to ensure the frontend works out-of-the-box before experiments run
CANNED_SECTIONS: dict[str, dict[str, str]] = {
    "pasal_12": {
        "section_id": "pasal_12",
        "chapter": "Chapter 3",
        "article": "Article 12",
        "title": "Academic Leave (Cuti Akademik)",
        "text": (
            "1. Students may apply for academic leave after completing at least two semesters.\n"
            "2. Academic leave is granted for a maximum duration of two consecutive semesters.\n"
            "3. The period of academic leave is not counted towards the maximum study period limit."
        ),
    },
    "pasal_15": {
        "section_id": "pasal_15",
        "chapter": "Chapter 4",
        "article": "Article 15",
        "title": "Study Load and Evaluation (Beban Studi)",
        "text": (
            "1. The standard study load for an undergraduate degree program is 144 credit units.\n"
            "2. Students with a cumulative GPA of at least 3.00 may take up to 24 credit units in the following semester."
        ),
    },
}

CANNED_REPLAY_RESPONSES: list[dict[str, Any]] = [
    {
        "keywords": ["leave", "cuti", "semester"],
        "answer_text": "Students may take academic leave for a maximum duration of two consecutive semesters, provided they have completed at least two semesters.",
        "cited_ids": [1],
        "abstained": False,
        "format_ok": True,
        "sources": [
            {
                "passage_idx": 1,
                "section_id": "pasal_12",
                "chapter": "Chapter 3",
                "article": "Article 12",
                "text": "Academic leave is granted for a maximum duration of two consecutive semesters.",
            }
        ],
        "tokens_in": 312,
        "tokens_out": 38,
        "latency_s": 0.42,
        "peak_vram_mb": 4200.0,
    },
    {
        "keywords": ["parking", "parkir", "kantin", "dormitory", "unanswerable"],
        "answer_text": "NOT_FOUND",
        "cited_ids": [],
        "abstained": True,
        "format_ok": True,
        "sources": [],
        "tokens_in": 280,
        "tokens_out": 2,
        "latency_s": 0.15,
        "peak_vram_mb": 4150.0,
    },
]


class ReplayService:
    """Manages replay responses from saved logs or canned fixtures."""

    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self.runs_dir = Path(runs_dir)
        self.indexed_predictions: dict[str, dict[str, Any]] = {}
        self._load_saved_runs()

    def _load_saved_runs(self) -> None:
        """Scan runs directory for predictions.jsonl files if present."""
        if not self.runs_dir.is_dir():
            return

        for path in self.runs_dir.rglob("predictions.jsonl"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        key = f"{record.get('model')}:{record.get('qid')}"
                        self.indexed_predictions[key] = record
            except Exception:
                continue

    def get_section(self, section_id: str) -> dict[str, str] | None:
        """Retrieve regulation section text."""
        return CANNED_SECTIONS.get(section_id)

    def replay_query(
        self,
        question: str,
        model_id: str,
        precision: str = "4-bit",
        top_k: int = 4,
    ) -> AskResponse:
        """Provide a canned or saved replay response for a query."""
        q_lower = question.lower()

        # Match canned answers based on keywords
        matched = None
        for item in CANNED_REPLAY_RESPONSES:
            if any(kw in q_lower for kw in item["keywords"]):
                matched = item
                break

        if not matched:
            # Default fallback mock response
            matched = CANNED_REPLAY_RESPONSES[0]

        citations = [
            SourceCitation(
                passage_idx=s["passage_idx"],
                section_id=s["section_id"],
                chapter=s["chapter"],
                article=s["article"],
                text=s["text"],
            )
            for s in matched.get("sources", [])
        ]

        raw_output = matched["answer_text"]
        if not matched["abstained"] and matched.get("cited_ids"):
            sources_str = ", ".join(f"[{i}]" for i in matched["cited_ids"])
            raw_output = f"{raw_output}\nSources: {sources_str}"

        return AskResponse(
            question=question,
            model=model_id,
            precision=precision,
            mode="replay",
            answer_text=matched["answer_text"],
            cited_sources=citations,
            abstained=matched["abstained"],
            format_ok=matched["format_ok"],
            tokens_in=matched["tokens_in"],
            tokens_out=matched["tokens_out"],
            latency_s=matched["latency_s"],
            peak_vram_mb=matched.get("peak_vram_mb"),
            raw_output=raw_output,
        )
