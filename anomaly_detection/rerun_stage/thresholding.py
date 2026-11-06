"""Confidence-based step selection for binary and secondary reruns."""

import math
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

from ..scoring import read_confidence_scores, score_details
from ..secondary_classification import (
    SecondaryClassificationProfile,
    classify_yes_steps,
)


def lowest_confidence_steps(
    probs_data_filepath: str | Path,
    manual_threshold: bool = True,
    num_steps: int = 3,
) -> tuple[list[tuple[int, float]], dict[str, Any] | None, dict[int, float]]:
    """Return the lowest-confidence eligible steps and complete score data."""
    confidence_scores, yes_steps = read_confidence_scores(Path(probs_data_filepath))
    if not confidence_scores:
        return [], None, {}
    threshold = 99 if manual_threshold else 100
    eligible = {
        step: confidence
        for step, confidence in confidence_scores.items()
        if confidence < threshold
    }
    selected = sorted(eligible.items(), key=lambda item: item[1])[:num_steps]
    return selected, score_details(confidence_scores, yes_steps), confidence_scores


def global_secondary_rerun_score(
    profile: SecondaryClassificationProfile,
    confidence_scores: dict[int, float],
    *,
    num_steps: int,
) -> tuple[float, list[tuple[int, float]]]:
    """Return uncertainty and globally least-confident configured steps."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    missing = sorted(set(profile.referenced_steps) - set(confidence_scores))
    if missing:
        raise ValueError(
            f"Confidence data lacks secondary-classification step(s) {missing}."
        )
    invalid = sorted(
        step
        for step in profile.referenced_steps
        if (
            isinstance(confidence_scores[step], bool)
            or not isinstance(confidence_scores[step], (int, float))
            or not math.isfinite(float(confidence_scores[step]))
            or not 0.0 <= float(confidence_scores[step]) <= 100.0
        )
    )
    if invalid:
        raise ValueError(f"Invalid confidence for secondary step(s) {invalid}.")
    selected = sorted(
        ((step, float(confidence_scores[step])) for step in profile.referenced_steps),
        key=lambda item: (item[1], item[0]),
    )[:num_steps]
    if not selected:
        raise ValueError("Secondary-classification profile has no configured steps.")
    mean_confidence = sum(confidence for _, confidence in selected) / len(selected)
    return (100.0 - mean_confidence) / 100.0, selected


def class_aware_rerun_steps(
    profile: SecondaryClassificationProfile,
    yes_steps: list[int] | set[int] | tuple[int, ...],
    confidence_scores: dict[int, float],
    *,
    binary_score: float,
    binary_threshold: float,
    secondary_confidence_threshold: float,
    num_steps: int,
) -> tuple[list[tuple[int, float]], dict[str, bool]]:
    """Select uncertain steps according to their configured decision impact."""
    current = classify_yes_steps(profile, yes_steps)
    same_binary_impact, cross_binary_impact = _profile_decision_impact_steps(profile)
    same_binary_steps = [
        step for step in same_binary_impact if step in confidence_scores
    ]
    cross_binary_steps = [
        step for step in cross_binary_impact if step in confidence_scores
    ]
    secondary_lane = any(
        confidence_scores[step] < secondary_confidence_threshold
        for step in same_binary_steps
    )
    binary_lane = (
        current.binary_label == "good"
        and float(binary_score) >= float(binary_threshold)
    )
    eligible_steps: set[int] = set()
    if secondary_lane:
        eligible_steps.update(same_binary_steps)
    if binary_lane:
        eligible_steps.update(cross_binary_steps)
    selected = sorted(
        ((step, confidence_scores[step]) for step in eligible_steps),
        key=lambda item: item[1],
    )[:num_steps]
    return selected, {"secondary_lane": secondary_lane, "binary_lane": binary_lane}


@lru_cache(maxsize=32)
def _profile_decision_impact_steps(
    profile: SecondaryClassificationProfile,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Find steps that can affect class-only or cross-binary decisions."""
    steps = profile.referenced_steps
    same_binary: set[int] = set()
    cross_binary: set[int] = set()
    for flags in product((False, True), repeat=len(steps)):
        yes = {step for step, flag in zip(steps, flags) if flag}
        before = classify_yes_steps(profile, yes)
        for step in steps:
            after = classify_yes_steps(profile, yes.symmetric_difference({step}))
            if after.secondary_class == before.secondary_class:
                continue
            if after.binary_label == before.binary_label:
                same_binary.add(step)
            else:
                cross_binary.add(step)
    return tuple(sorted(same_binary)), tuple(sorted(cross_binary))
