"""JSON, JSONL, YAML, and SHA-256 checksum helper utilities.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §13).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write data to a JSON file, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single dictionary as a JSON line to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as JSON Lines."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read and parse a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write data to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_string_sha256(text: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
