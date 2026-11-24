"""Prompt building utilities and templates.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §8).
Consumers: Member A (local runs), Member C (hosted runs & API).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class PromptBuilder:
    """Builds standard chat prompts and hashes them for auditable runs."""

    def __init__(
        self,
        config_path: str | Path = "configs/prompts/rag_v1.yaml",
        system_prompt: str | None = None,
        user_template: str | None = None,
    ) -> None:
        if system_prompt and user_template:
            self.version = "custom"
            self.system_prompt = system_prompt.strip()
            self.user_template = user_template.strip()
        else:
            path = Path(config_path)
            if not path.is_file():
                # Fallback default if file not yet accessible
                self.version = "default_fallback"
                self.system_prompt = (
                    "You are a strict academic regulation assistant.\n"
                    "1. Answer only using the numbered context passages below.\n"
                    "2. Be concise and state the exact rule, number, duration, or condition.\n"
                    "3. End your answer with `Sources: [n], [m]` listing only the passage numbers you used.\n"
                    "4. If the context does not contain the answer, reply exactly: `NOT_FOUND`"
                )
                self.user_template = "Context passages:\n{context}\n\nQuestion: {question}"
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                self.version = data.get("version", "unknown")
                self.system_prompt = data["system_prompt"].strip()
                self.user_template = data["user_template"].strip()

    def format_passages(self, chunks: list[dict[str, Any]]) -> str:
        """Format chunks into 1-indexed numbered passages [1], [2], ...

        Each chunk is expected to have at least 'text', optionally 'header' or 'section_ids'.
        """
        if not chunks:
            return "(No context passages provided)"

        formatted: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "").strip()
            header = chunk.get("header")
            if not header and "section_ids" in chunk:
                # Build header if section_ids present
                sections = chunk["section_ids"]
                if isinstance(sections, list) and sections:
                    header = f"[{sections[0]}]"
            
            if header and not text.startswith("["):
                passage = f"[{idx}] {header}\n{text}"
            else:
                passage = f"[{idx}] {text}"
            formatted.append(passage)

        return "\n\n".join(formatted)

    def build_rag_messages(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Construct standard 2-turn chat messages for RAG generation."""
        context_str = self.format_passages(chunks)
        user_content = self.user_template.format(
            context=context_str,
            question=question.strip(),
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_closed_book_messages(self, question: str) -> list[dict[str, str]]:
        """Construct closed-book chat messages (for E0 baseline)."""
        system = (
            "You are an academic regulation assistant. "
            "Answer the question concisely based on academic regulations."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question.strip()},
        ]

    @staticmethod
    def compute_prompt_hash(messages: list[dict[str, str]]) -> str:
        """Compute stable SHA-256 hash of messages payload."""
        payload = json.dumps(messages, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
