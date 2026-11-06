"""Small, model-independent helpers for durable pipeline artifacts."""

import json
import logging
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a text artifact after durably flushing it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl(path: Path, *, logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    """Read dictionary rows from JSONL, skipping malformed rows with a warning."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                if logger is not None:
                    logger.warning(
                        "Skipping malformed JSONL line %d in %s: %s",
                        line_number,
                        path,
                        error,
                    )
                continue
            if isinstance(value, dict):
                rows.append(value)
            elif logger is not None:
                logger.warning("Skipping non-object JSONL line %d in %s", line_number, path)
    return rows
