"""Small inference-result helpers shared by rerun execution paths."""

import logging
from pathlib import Path
from typing import Any

from ..config import resolve_portable_file

logger = logging.getLogger(__name__)


def load_planner_steps(planner_path: str | Path) -> list[str]:
    """Load one non-empty planner question per line."""
    project_root = Path(__file__).resolve().parents[2]
    planner_path = resolve_portable_file(
        planner_path,
        project_root,
        Path.cwd(),
    )
    if not planner_path.is_file():
        # Historical summaries store absolute planner paths.  When a run
        # moves between an SSHFS mount and the owning checkout, relocate the
        # stable repository-relative suffix instead of requiring JSONL
        # rewriting merely to resume inference.
        recorded = Path(planner_path)
        if "reference_images" in recorded.parts:
            index = recorded.parts.index("reference_images")
            relocated = project_root.joinpath(*recorded.parts[index:])
            if relocated.is_file():
                planner_path = relocated
    return [
        line.strip()
        for line in Path(planner_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_rerun_confidence(result: dict[str, Any], fallback: float) -> float:
    """Return log-probability confidence for the actual rerun decision."""
    decision_token = "YES" if result.get("is_anomaly") else "NO"
    for entry in result.get("probs_data") or []:
        if str(entry.get("token", "")).upper() == decision_token:
            try:
                return float(str(entry["confidence"]).replace("%", ""))
            except (KeyError, TypeError, ValueError):
                break
    logger.warning(
        "Rerun step produced no logprob-based confidence for its %s decision "
        "— keeping original confidence %.2f%%",
        decision_token,
        fallback,
    )
    return fallback


def secondary_probabilities(
    confidence_scores: dict[int, float], yes_steps: set[int]
) -> dict[int, float]:
    """Convert decision confidences into per-step YES probabilities."""
    return {
        step: (
            confidence / 100.0
            if step in yes_steps
            else 1.0 - confidence / 100.0
        )
        for step, confidence in confidence_scores.items()
    }
