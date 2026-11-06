"""
scoring.py (anomaly_detection)
================================
Lightweight, dependency-free (no torch/transformers/CLIP) helpers for parsing
a candidate image's `*_prob_data.txt` file and deriving the single continuous
per-image anomaly score used for image-level (continuous) AUROC.

Split out of rerun.py so that metrics.py can reuse the exact same scoring
logic without importing rerun.py's heavy VLM-loading dependencies (`models.py`
pulls in torch/transformers/clip) just to read a score off disk.

Score definition (mirrors the original notebook's `lowest_log_prob_getter`
"thresholding" section):
  - If any planner step was answered YES, the score is the confidence of the
    LOWEST-confidence YES step, divided by 100 (closer to 1.0 = more
    confidently anomalous).
  - Otherwise (every step answered NO), the score is 1 - the confidence of
    the LOWEST-confidence NO step, divided by 100 (closer to 1.0 = a
    borderline "good" call that could flip to anomalous on rerun).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_probability_file(path: str | Path) -> tuple[bool, str]:
    """Strictly validate one resumable ``*_prob_data.txt`` JSONL file.

    Metrics parsing is intentionally tolerant so one bad row does not abort a
    report. Resume must be stricter: every non-empty line must be valid JSON,
    every step must be unique and numeric, tokens must be YES/NO, confidence
    must be finite and within 0..100, and at least one usable step must exist.
    """
    probability_path = Path(path)
    if not probability_path.is_file():
        return False, "file is missing"
    if probability_path.stat().st_size == 0:
        return False, "file is empty"

    seen_steps: set[int] = set()
    try:
        lines = probability_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return False, f"file is unreadable: {error}"

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            return False, f"line {line_number} is invalid JSON: {error}"
        if not isinstance(entry, dict):
            return False, f"line {line_number} is not a JSON object"

        step_raw = entry.get("step")
        if not isinstance(step_raw, str) or not step_raw.isdigit():
            return False, f"line {line_number} has an invalid step"
        step = int(step_raw)
        if step in seen_steps:
            return False, f"line {line_number} duplicates step {step}"
        seen_steps.add(step)

        token = str(entry.get("token", "")).strip().lower()
        if token not in {"yes", "no"}:
            return False, f"line {line_number} has an invalid YES/NO token"
        try:
            confidence = float(str(entry["confidence"]).replace("%", ""))
        except (KeyError, TypeError, ValueError):
            return False, f"line {line_number} has an invalid confidence"
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 100.0:
            return False, f"line {line_number} confidence is outside 0..100"

    if not seen_steps:
        return False, "file contains no probability steps"
    return True, ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries. Malformed lines are skipped with a warning."""
    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                logger.warning("Skipping malformed JSONL line in %s: %s", path, error)
    return entries


def read_confidence_scores(probs_data_filepath: Path) -> tuple[dict[int, float], list[int]]:
    """Parses a *_prob_data.txt file into {step_number: confidence%} plus the list of steps answered YES."""
    confidence_scores: dict[int, float] = {}
    yes_steps: list[int] = []

    if not Path(probs_data_filepath).is_file():
        return confidence_scores, yes_steps

    for entry in read_jsonl(Path(probs_data_filepath)):
        step = entry.get("step")
        if not isinstance(step, str) or not step.isdigit():
            logger.warning("Skipping non-numeric step %r in %s", step, probs_data_filepath)
            continue

        try:
            step_number = int(step)
            confidence = float(str(entry["confidence"]).replace("%", ""))
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping invalid confidence entry in %s: %s", probs_data_filepath, entry)
            continue
        if not math.isfinite(confidence):
            logger.warning("Skipping non-finite confidence entry in %s: %s", probs_data_filepath, entry)
            continue

        confidence_scores[step_number] = confidence
        if str(entry.get("token", "")).lower() == "yes":
            yes_steps.append(step_number)

    return confidence_scores, yes_steps


def score_details(confidence_scores: dict[int, float], yes_steps: list[int]) -> dict[str, Any] | None:
    """Single per-image anomaly score: yes-step -> confidence/100, else 1 - (lowest no-step confidence)/100."""
    if not confidence_scores:
        return None

    if yes_steps:
        step_number = min(yes_steps, key=lambda step: confidence_scores[step])
        confidence = confidence_scores[step_number]
        return {"confidence": confidence, "step": step_number, "score": confidence / 100.0, "yes_step": True}

    step_number = min(confidence_scores, key=confidence_scores.get)
    confidence = confidence_scores[step_number]
    return {"confidence": confidence, "step": step_number, "score": (100.0 - confidence) / 100.0, "yes_step": False}


def lowest_log_prob_score(probs_data_filepath: str | Path) -> tuple[dict[int, float], dict[str, Any] | None]:
    """Reproduce the notebook's "thresholding" section score from one probability file."""
    confidence_scores, yes_steps = read_confidence_scores(Path(probs_data_filepath))
    return confidence_scores, score_details(confidence_scores, yes_steps)
