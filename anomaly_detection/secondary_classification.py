"""Shared rule engine for optional step-driven secondary classification.

This module is deliberately model- and dataset-independent.  Inference,
rerun, mask generation, and metrics all use these functions so a configured
step rule has exactly one interpretation throughout the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from pipeline_core.model_output import normalize_step_decision_layout


PROFILE_FILENAME = "secondary_classification_profile.json"
PROFILE_SCHEMA_VERSION = 1
RULE_OPERATORS = ("any_yes", "all_yes", "any_no", "all_no")
_STEP_LINE_RE = re.compile(
    r"(?im)^[ \t]*\[step\s+(\d+)\][ \t]*(yes|no)[ \t]*(?:->|:)?[ \t]*(.*)$"
)


@dataclass(frozen=True)
class SecondaryClassRule:
    name: str
    binary_label: str
    any_yes: tuple[int, ...] = ()
    all_yes: tuple[int, ...] = ()
    any_no: tuple[int, ...] = ()
    all_no: tuple[int, ...] = ()
    localization_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class SecondaryClassificationProfile:
    prompt_template: str
    classes: tuple[SecondaryClassRule, ...]
    schema_version: int = PROFILE_SCHEMA_VERSION

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.classes)

    @property
    def referenced_steps(self) -> tuple[int, ...]:
        return tuple(sorted({step for rule in self.classes for step in _rule_steps(rule)}))


@dataclass(frozen=True)
class StepLog:
    answers: dict[int, bool]
    evidence: dict[int, str]

    @property
    def yes_steps(self) -> list[int]:
        return sorted(step for step, is_yes in self.answers.items() if is_yes)


@dataclass(frozen=True)
class SecondaryOutcome:
    secondary_class: str
    binary_label: str
    binary_prediction: int
    yes_steps: list[int]


def _parse_steps(value: Any, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "a list" if allow_empty else "a non-empty list"
        raise TypeError(f"{field} must be {requirement} of unique positive integers.")
    if any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in value):
        raise ValueError(f"{field} must contain only positive integers.")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicate step numbers.")
    return tuple(value)


def parse_profile(raw: Any, section: str = "secondary_classification") -> SecondaryClassificationProfile | None:
    """Parse and fully validate an optional secondary-classification profile."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"{section} must be a YAML mapping.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{section}.enabled must be true or false.")
    if not enabled:
        return None

    schema_version = raw.get("schema_version", PROFILE_SCHEMA_VERSION)
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"{section}.schema_version must be {PROFILE_SCHEMA_VERSION}, got {schema_version!r}."
        )
    prompt_template = raw.get("prompt_template", "decision_maker_secondary.j2")
    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise ValueError(f"{section}.prompt_template must be a non-empty filename.")
    prompt_path = Path(prompt_template.strip())
    if prompt_path.is_absolute() or ".." in prompt_path.parts or len(prompt_path.parts) != 1:
        raise ValueError(f"{section}.prompt_template must be a filename relative to prompts_dir.")

    classes_raw = raw.get("classes")
    if not isinstance(classes_raw, dict) or not classes_raw:
        raise ValueError(f"{section}.classes must be a non-empty mapping.")

    classes: list[SecondaryClassRule] = []
    for raw_name, class_raw in classes_raw.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{section}.classes contains an invalid class name: {raw_name!r}.")
        name = raw_name.strip()
        if not isinstance(class_raw, dict):
            raise TypeError(f"{section}.classes.{name} must be a mapping.")
        binary_label = class_raw.get("binary_label")
        if binary_label not in {"good", "anomaly"}:
            raise ValueError(
                f"{section}.classes.{name}.binary_label must be 'good' or 'anomaly'."
            )
        parsed: dict[str, tuple[int, ...]] = {}
        for operator in RULE_OPERATORS:
            if operator in class_raw:
                parsed[operator] = _parse_steps(
                    class_raw[operator], f"{section}.classes.{name}.{operator}"
                )
            else:
                parsed[operator] = ()
        if not any(parsed.values()):
            raise ValueError(f"{section}.classes.{name} must define at least one rule operator.")
        localization_steps = _parse_steps(
            class_raw.get("localization_steps", []),
            f"{section}.classes.{name}.localization_steps",
            allow_empty=True,
        )
        positive_steps = set(parsed["any_yes"]) | set(parsed["all_yes"])
        if not set(localization_steps).issubset(positive_steps):
            raise ValueError(
                f"{section}.classes.{name}.localization_steps must be drawn from "
                "that class's any_yes/all_yes steps."
            )
        classes.append(
            SecondaryClassRule(
                name=name,
                binary_label=binary_label,
                localization_steps=localization_steps,
                **parsed,
            )
        )

    profile = SecondaryClassificationProfile(
        prompt_template=prompt_template.strip(),
        classes=tuple(classes),
        schema_version=schema_version,
    )
    if not any(rule.binary_label == "anomaly" for rule in profile.classes):
        raise ValueError(f"{section}.classes must contain at least one binary anomaly class.")
    _validate_truth_table(profile, section)
    return profile


def _rule_steps(rule: SecondaryClassRule) -> tuple[int, ...]:
    return tuple(
        step
        for operator in RULE_OPERATORS
        for step in getattr(rule, operator)
    )


def match_rule(rule: SecondaryClassRule, yes_steps: set[int] | list[int] | tuple[int, ...]) -> bool:
    yes = set(yes_steps)
    return (
        (not rule.any_yes or bool(yes.intersection(rule.any_yes)))
        and (not rule.all_yes or set(rule.all_yes).issubset(yes))
        and (not rule.any_no or bool(set(rule.any_no) - yes))
        and (not rule.all_no or not yes.intersection(rule.all_no))
    )


def _validate_truth_table(profile: SecondaryClassificationProfile, section: str) -> None:
    steps = profile.referenced_steps
    for flags in product((False, True), repeat=len(steps)):
        yes = {step for step, flag in zip(steps, flags) if flag}
        matches = [rule.name for rule in profile.classes if match_rule(rule, yes)]
        if len(matches) != 1:
            answers = ", ".join(f"{step}={'YES' if step in yes else 'NO'}" for step in steps)
            raise ValueError(
                f"{section}.classes must match exactly one class for every configured "
                f"step combination; {answers or 'no configured steps'} matches {matches}."
            )


def classify_yes_steps(
    profile: SecondaryClassificationProfile,
    yes_steps: set[int] | list[int] | tuple[int, ...],
) -> SecondaryOutcome:
    yes = sorted(set(yes_steps))
    matches = [rule for rule in profile.classes if match_rule(rule, yes)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one secondary class for YES steps {yes}, got {[r.name for r in matches]}.")
    rule = matches[0]
    return SecondaryOutcome(
        secondary_class=rule.name,
        binary_label=rule.binary_label,
        binary_prediction=1 if rule.binary_label == "anomaly" else 0,
        yes_steps=yes,
    )


def parse_step_log(
    text: str,
    expected_steps: list[int] | tuple[int, ...] | set[int],
    *,
    evidence_required_steps: set[int] | None = None,
    fallback_evidence: Mapping[int, str] | None = None,
) -> StepLog:
    """Parse the model's final answer for every requested planner step.

    Models occasionally reason aloud and then restate a corrected full step
    log.  In that case the final occurrence of a step is the model's final
    answer.  Treating an earlier, superseded occurrence as fatal makes a
    recoverable formatting variation abort an entire dataset run.
    """
    text = normalize_step_decision_layout(text)
    expected = set(expected_steps)
    if not expected:
        raise ValueError("Secondary classification requires at least one expected planner step.")
    answers: dict[int, bool] = {}
    evidence: dict[int, str] = {}
    for number_text, answer_text, detail in _STEP_LINE_RE.findall(text):
        number = int(number_text)
        # Keep the last explicitly emitted value. This corresponds to the
        # corrected final block when the VLM revises an earlier step log.
        answers[number] = answer_text.lower() == "yes"
        evidence[number] = " ".join(detail.split()) if detail.strip() else ""
    missing = sorted(expected - answers.keys())
    unexpected = sorted(answers.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"Decision response has an incomplete secondary step log; missing={missing}, "
            f"unexpected={unexpected}."
        )
    # A structured final line can carry concrete, image-specific evidence
    # even when a checkpoint emits ``[STEP N] YES`` without repeating that
    # text on the step line.  The caller may supply that already-model-written
    # evidence only when it has established an unambiguous association.  This
    # is deliberately not a synthetic fallback: missing or ambiguous evidence
    # remains a contract violation.
    for step, detail in (fallback_evidence or {}).items():
        if (
            step in answers
            and answers[step]
            and not evidence.get(step)
            and isinstance(detail, str)
            and detail.strip()
        ):
            evidence[step] = " ".join(detail.split())

    required = evidence_required_steps or set()
    missing_evidence = sorted(step for step in required if answers.get(step) and not evidence.get(step))
    if missing_evidence:
        raise ValueError(f"YES step(s) {missing_evidence} require concrete localization evidence.")
    return StepLog(answers=answers, evidence=evidence)


def yes_probabilities(probability_rows: list[dict[str, Any]]) -> dict[int, float]:
    probabilities: dict[int, float] = {}
    for row in probability_rows:
        try:
            step = int(row["step"])
            token = str(row["token"]).strip().upper()
            confidence = float(str(row["confidence"]).replace("%", "")) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
        if token not in {"YES", "NO"} or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            continue
        if step in probabilities:
            raise ValueError(f"Probability data contains duplicate step {step}.")
        probabilities[step] = confidence if token == "YES" else 1.0 - confidence
    return probabilities


def score_classes(
    profile: SecondaryClassificationProfile,
    probabilities: Mapping[int, float],
) -> dict[str, float]:
    missing = sorted(set(profile.referenced_steps) - set(probabilities))
    if missing:
        raise ValueError(f"Probability data is missing configured secondary step(s): {missing}.")
    scores: dict[str, float] = {}
    for rule in profile.classes:
        clauses: list[float] = []
        if rule.any_yes:
            clauses.append(max(probabilities[step] for step in rule.any_yes))
        if rule.all_yes:
            clauses.append(min(probabilities[step] for step in rule.all_yes))
        if rule.any_no:
            clauses.append(max(1.0 - probabilities[step] for step in rule.any_no))
        if rule.all_no:
            clauses.append(min(1.0 - probabilities[step] for step in rule.all_no))
        scores[rule.name] = float(min(clauses))
    return scores


def binary_anomaly_score(
    profile: SecondaryClassificationProfile,
    class_scores: Mapping[str, float],
) -> float:
    return float(max(class_scores[rule.name] for rule in profile.classes if rule.binary_label == "anomaly"))


def sole_binary_anomaly_class(
    profile: SecondaryClassificationProfile,
) -> str | None:
    """Return the class that is exactly equivalent to binary anomaly, if any.

    A profile can map several secondary classes to ``anomaly``.  In that case
    no individual secondary mask is interchangeable with the binary mask.
    When there is exactly one, however, independently localizing that class
    creates two different pixel predictions for the same decision.
    """
    anomaly_classes = [
        rule.name for rule in profile.classes if rule.binary_label == "anomaly"
    ]
    return anomaly_classes[0] if len(anomaly_classes) == 1 else None


def localization_evidence_steps(
    profile: SecondaryClassificationProfile,
    class_name: str,
    yes_steps: set[int] | list[int] | tuple[int, ...],
) -> list[int]:
    rule = next((rule for rule in profile.classes if rule.name == class_name), None)
    if rule is None:
        raise ValueError(f"Unknown secondary class {class_name!r}.")
    yes = set(yes_steps)
    return [step for step in rule.localization_steps if step in yes]


def profile_to_dict(profile: SecondaryClassificationProfile) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for rule in profile.classes:
        payload: dict[str, Any] = {"binary_label": rule.binary_label}
        for operator in RULE_OPERATORS:
            values = getattr(rule, operator)
            if values:
                payload[operator] = list(values)
        payload["localization_steps"] = list(rule.localization_steps)
        classes[rule.name] = payload
    return {
        "enabled": True,
        "schema_version": profile.schema_version,
        "prompt_template": profile.prompt_template,
        "classes": classes,
    }


def profile_fingerprint(profile: SecondaryClassificationProfile) -> str:
    encoded = json.dumps(profile_to_dict(profile), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def profile_snapshot(profile: SecondaryClassificationProfile) -> dict[str, Any]:
    payload = profile_to_dict(profile)
    payload["fingerprint"] = profile_fingerprint(profile)
    return payload


def load_profile_snapshot(dataset_run_dir: Path) -> SecondaryClassificationProfile | None:
    path = Path(dataset_run_dir) / PROFILE_FILENAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    profile = parse_profile(raw, str(path))
    if profile is None:
        raise ValueError(f"Saved secondary profile is unexpectedly disabled: {path}")
    saved = raw.get("fingerprint")
    actual = profile_fingerprint(profile)
    if saved != actual:
        raise ValueError(f"Secondary profile fingerprint mismatch in {path}.")
    return profile


def binary_anomaly_rule_text(profile: SecondaryClassificationProfile) -> str:
    """Render anomaly-class step logic without exposing secondary class names."""
    expressions = []
    for rule in profile.classes:
        if rule.binary_label != "anomaly":
            continue
        clauses = []
        for operator in RULE_OPERATORS:
            steps = getattr(rule, operator)
            if not steps:
                continue
            rendered_steps = ", ".join(str(step) for step in steps)
            answer = "YES" if operator.endswith("yes") else "NO"
            if operator.startswith("any"):
                clauses.append(
                    f"at least one of steps {rendered_steps} is {answer}"
                )
            else:
                verb = "is" if len(steps) == 1 else "are"
                clauses.append(f"all of steps {rendered_steps} {verb} {answer}")
        expressions.append(" and ".join(clauses))
    return " or ".join(f"({expression})" for expression in expressions)
