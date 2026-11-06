"""
decision.py (anomaly_detection)
=================================
The VLM decision maker: given a reference image, a candidate image and a
planner's inspection steps, asks the VLM to answer each step and derive a
final good/anomaly decision, then parses that decision out of the response.

Behavior change (2026-08): decision and probability artifacts are written
through temporary files and atomic replacement. Probability data is committed
first and the decision file last, so resume never mistakes a half-written
pair left by interruption for completed inference.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ModelsConfig
from .hardware_monitor import record_pipeline_event
from .models import vlm
from .secondary_classification import (
    SecondaryClassificationProfile,
    binary_anomaly_rule_text,
    binary_anomaly_score,
    classify_yes_steps,
    localization_evidence_steps,
    parse_step_log,
    score_classes,
    yes_probabilities,
)
from .utils import build_prompt
from pipeline_core.model_output import normalize_step_decision_layout

logger = logging.getLogger(__name__)

DEFAULT_DECISION_PROMPT_TEMPLATE = "decision_maker_v2_original_compact.j2"
DECISION_FORMAT_MAX_ATTEMPTS = 2
DECISION_EVIDENCE_MAX_ATTEMPTS = 3


def resolve_decision_prompt_template(prompt_template: str | None) -> str:
    """Resolve an optional config/CLI override to the sole runtime default."""
    return prompt_template if prompt_template is not None else DEFAULT_DECISION_PROMPT_TEMPLATE


_STEP_LINE_RE = re.compile(r"(?im)^[ \t]*\[step\s+(\d+)\][ \t]*(yes|no)[ \t]*(?:->|:)?[ \t]*(.*)$")
_PLANNER_STEP_RE = re.compile(r"(?im)^[ \t]*step\s+(\d+)\s*:")
_AMBIGUOUS_STEP_CHOICE_RE = re.compile(
    r"(?is)\[step\s+(\d+)\][^\[]*?\bOR\s+\[step\s+\1\]"
)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text completely before replacing the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _step_evidence(text: str, decision: str) -> list[str]:
    """Descriptive text of every step whose decision matches (e.g. every
    "yes" step's reasoning), in step order, with the "[STEP N] yes ->"
    log-format prefix stripped so it reads as plain evidence text.

    NO steps carry no reason text by design (the main compact prompt asks for
    "[STEP N] NO" alone, to cut generation time), so this only ever finds
    real evidence for "yes"; callers must not treat an empty "no" result as
    an error."""
    return [
        detail.strip()
        for _, step_decision, detail in _STEP_LINE_RE.findall(text)
        if step_decision.lower() == decision and detail.strip()
    ]


def _strip_step_log_syntax(value: str) -> str:
    """Remove copied ``[STEP N] YES/NO ->`` markers from a final field."""
    cleaned = re.sub(
        r"(?i)\[step\s+\d+\]\s*(?:yes|no)\s*(?:->|:)?\s*",
        "",
        value,
    )
    return " ".join(cleaned.split())


def _label_from_steps(text: str) -> str | None:
    """Derive good/anomaly straight from the logged [STEP N] YES/NO answers,
    per the prompt's own Decision Rules (any YES -> Anomaly, all NO -> good).
    Returns None if no step lines were found to derive from."""
    step_answers = [answer.lower() for _, answer, _ in _STEP_LINE_RE.findall(text)]
    if not step_answers:
        return None
    return "anomaly" if "yes" in step_answers else "good"


def _planner_step_count(planner_steps: str) -> int | None:
    """Return the expected planner-step count when numbering is standard."""
    numbers = [int(number) for number in _PLANNER_STEP_RE.findall(planner_steps)]
    if numbers and numbers == list(range(1, len(numbers) + 1)):
        return len(numbers)
    return None


def _step_log_is_complete(text: str, expected_step_count: int | None) -> bool:
    """Check that every expected step appears exactly once in the response."""
    if expected_step_count is None or expected_step_count <= 0:
        return False
    answered_numbers = [int(number) for number, _, _ in _STEP_LINE_RE.findall(text)]
    return (
        len(answered_numbers) == expected_step_count
        and set(answered_numbers) == set(range(1, expected_step_count + 1))
    )


def _validate_probability_steps(
    probability_rows: list[dict[str, Any]], expected_step_count: int | None
) -> None:
    """Require exactly one confidence record for every numbered planner step."""
    if expected_step_count is None:
        return
    try:
        steps = [int(row["step"]) for row in probability_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Decision confidence rows contain an invalid step number.") from exc
    expected = list(range(1, expected_step_count + 1))
    if sorted(steps) != expected or len(steps) != len(set(steps)):
        raise ValueError(
            f"Decision confidence is incomplete; expected steps={expected}, got={steps}."
        )


def missing_yes_evidence_steps(text: str) -> list[int]:
    """Find generated YES steps that have no inline visual evidence."""
    normalized = normalize_step_decision_layout(text)
    return [
        int(number)
        for number, answer, detail in _STEP_LINE_RE.findall(normalized)
        if answer.lower() == "yes" and not detail.strip()
    ]


def validate_yes_step_evidence(text: str) -> None:
    """Require each positive decision to supply its own visual evidence."""
    missing = missing_yes_evidence_steps(text)
    if missing:
        raise ValueError(
            "YES steps lack concrete inline visual evidence: "
            f"{sorted(set(missing))}. Each YES must describe a visible anomaly."
        )


def _format_retry_prompt(
    planner_steps: str,
    error: Exception,
    expected_step_count: int | None,
) -> str:
    if expected_step_count is not None:
        required_step_keys = ", ".join(
            f"STEP {step}" for step in range(1, expected_step_count + 1)
        )
        step_requirement = (
            f"You must answer exactly these keys once each: {required_step_keys}."
        )
    else:
        step_requirement = "You must answer every planner step exactly once, in order."

    # This compact combined contract was hardware-tested on the exact LLaVA-OV
    # MVTec AD and LOCO failures that defeated the longer corrective prompt.
    # Do not add a forced minimum generation length here: on the failed LOCO
    # case it converted a legitimate early stop into a max-token repetition
    # loop instead of making the checkpoint answer the remaining steps.
    return (
        "Compare the candidate image against the reference image and answer "
        "every planner step in one response.\n"
        "A YES means that the anomaly described by that step is visible. A NO "
        "means it is not visible.\n"
        "Derive the final label only from your step answers: if ANY step is "
        "YES, the final label MUST be Anomaly; if ALL steps are NO, the final "
        "label MUST be good. Recount your YES answers before writing the "
        "final line. Do not label an all-NO response Anomaly.\n"
        f"{step_requirement}\n"
        "Do not omit a key, do not repeat a key, and do not repeat a conclusion.\n\n"
        f"PLANNER:\n{planner_steps.strip()}\n\n"
        "Return only one compact result block. Use `[STEP n] NO` for NO and "
        "`[STEP n] YES -> short visible evidence` for YES. Put one result per "
        "line. After the last step, write exactly one final line using either "
        "`Therefore, the final decision is: good` or "
        "`Therefore, the final decision is: Anomaly`. Do not add a summary or "
        "any other prose."
    )


def _format_evidence_retry_prompt(
    planner_steps: str,
    expected_step_count: int | None,
    missing_steps: list[int],
) -> str:
    """Give one last combined correction for bare YES answers, without copying them."""
    step_names = ", ".join(f"STEP {step}" for step in sorted(set(missing_steps)))
    return (
        _format_retry_prompt(
            planner_steps,
            ValueError("missing YES evidence"),
            expected_step_count,
        )
        + "\n\nIMPORTANT: Begin your answer with [STEP 1] and include every "
        "numbered step in order through the last step. Your previous answer "
        f"marked {step_names} YES without visual anomaly evidence. For each "
        "YES, append `->` plus a concrete abnormal difference and location. "
        "If no abnormal difference is visible for that step, answer NO. "
        "Do not invent evidence or copy the previous answer."
    )


def _rejected_response_path(decision_path: Path, attempt: int) -> Path:
    """Return a non-canonical audit path for one rejected VLM response."""
    decision_path = Path(decision_path)
    return (
        decision_path.parent
        / ".rejected_decisions"
        / f"{decision_path.name}.attempt_{attempt:02d}.txt"
    )


def _reason_from_steps(text: str, label: str) -> tuple[str, str]:
    """Synthesize a non-empty reason/details pair straight from the step log
    for `label`, for use whenever the model's own final-line reason can't be
    trusted: it was empty, or its label had to be corrected to match the step
    log. Never returns an empty string, so an all-NO "good" response always
    carries a real reason/details instead of nothing at all."""
    if label == "anomaly":
        evidence = _step_evidence(text, "yes")
        if evidence:
            # Real, image-specific evidence extracted from the model's own
            # step reasoning -- this is what gets localized, so no
            # boilerplate here.
            reason = " ".join(evidence)
            return reason, reason
        raise ValueError(
            "Anomaly response has no concrete visual evidence in either its "
            "YES step or structured final line."
        )

    no_steps = [num for num, answer, _ in _STEP_LINE_RE.findall(text) if answer.lower() == "no"]
    reason = (
        f"No anomalies found across all {len(no_steps)} inspected step(s); "
        "each inspected region matched the reference."
    )
    return reason, reason


def _canonicalize_final_line(text: str, label: str, reason: str, details: str) -> str:
    """Return the response with one clean, parseable final decision line."""
    display_label = "Anomaly" if label == "anomaly" else "good"
    canonical = f"{display_label}; reason={reason}; details={details}"
    structured_pattern = re.compile(
        r"(?im)^\s*<?(?:good|anomaly)>?\s*;\s*reason\s*=.*$"
    )
    if structured_pattern.search(text):
        return structured_pattern.sub(canonical, text.rstrip(), count=1)

    lines = text.rstrip().splitlines()
    compact_final_pattern = (
        r"\s*(?:Therefore,\s*the\s+final\s+decision\s+is:\s*)?"
        r"<?(?:good|anomaly)>?\s*"
    )
    if lines and re.fullmatch(compact_final_pattern, lines[-1], flags=re.IGNORECASE):
        lines[-1] = canonical
    else:
        lines.append(canonical)
    return "\n".join(lines)


def parse_final_decision(
    text: str,
    label_policy: str = "model",
    prefer_stated_details: bool = False,
    expected_step_count: int | None = None,
) -> tuple[str, str, str]:
    """
    Parse a VLM decision into its label, reason, and evidence details.

    The preferred output is ``<label>; reason=<summary>; details=<evidence>``.
    ``label_policy="model"`` (the default) preserves the VLM's explicit final
    label even when it contradicts the step log. The disagreement is logged
    for auditing and the downstream rerun stage therefore still sees the
    model's original good/anomaly decision. ``label_policy="steps"`` retains
    the legacy behavior where any YES forces anomaly and all-NO forces good.

    ``prefer_stated_details=True`` preserves a non-empty structured details
    field instead of replacing it with the concatenated YES-step evidence.
    This is opt-in for prompts whose final line is deliberately a compact,
    model-generated localization target.

    Other parsing safety nets:
    - If reason/details end up empty (e.g. an all-NO "good" response),
      they're derived from the model's OWN per-step evidence text
      (never a content-free placeholder), since for anomaly images this text
      is fed verbatim into the bounding-box/segmentation localization prompt
      (mask_generation.py) and must stay image-specific to localize well.
    - If the final line is truncated, an anomaly response can be recovered
      from any complete YES step because that decision is unambiguous. An
      all-NO response is recovered only when ``expected_step_count`` confirms
      that every planner step was answered.
    """
    if label_policy not in {"model", "steps"}:
        raise ValueError("label_policy must be either 'model' or 'steps'")

    text = normalize_step_decision_layout(text)

    ambiguous_choice = _AMBIGUOUS_STEP_CHOICE_RE.search(text)
    if ambiguous_choice is not None:
        raise ValueError(
            "Decision response emitted both alternatives for step "
            f"{ambiguous_choice.group(1)}; exactly one YES/NO answer is required."
        )

    steps_label = _label_from_steps(text)

    structured_match = re.search(
        r"(?im)^\s*<?(good|anomaly)>?\s*;\s*reason\s*=\s*(.*?)"
        r"(?:\s*;\s*details\s*=\s*(.*?))?\s*$",
        text,
    )
    if structured_match is not None:
        label = structured_match.group(1).lower()
        stated_reason = _strip_step_log_syntax(structured_match.group(2))
        stated_details = _strip_step_log_syntax(structured_match.group(3) or "")
        if steps_label is not None and steps_label != label:
            if label_policy == "steps":
                logger.warning(
                    "VLM final label '%s' contradicts its own step log (steps say '%s'); "
                    "overriding because decision.label_policy=steps.",
                    label, steps_label,
                )
                label = steps_label
                reason, details = _reason_from_steps(text, label)
                return label, reason, details
            logger.warning(
                "VLM final label '%s' contradicts its own step log (steps say '%s'); "
                "preserving the model's final label because decision.label_policy=model. "
                "The rerun stage may re-evaluate this prediction.",
                label, steps_label,
            )

        if label == "anomaly":
            yes_evidence = " ".join(_step_evidence(text, "yes")).strip()
            details = (
                stated_details
                if prefer_stated_details and stated_details
                else yes_evidence or stated_details or stated_reason
            )
            if not details:
                raise ValueError(
                    "Anomaly response has no concrete visual evidence in either its "
                    "YES step or structured final line."
                )
            reason = stated_reason or details
        elif not stated_reason or not stated_details:
            fallback_reason, fallback_details = _reason_from_steps(text, label)
            reason = stated_reason or fallback_reason
            details = stated_details or fallback_details
        else:
            reason, details = stated_reason, stated_details
        return label, reason, details

    final_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    match = re.fullmatch(
        r"(?:Therefore,\s*the\s+final\s+decision\s+is:\s*)?"
        r"<?(good|anomaly)>?",
        final_line,
        flags=re.IGNORECASE,
    )
    if match is None:
        # A truncated response can contain a complete YES step log but never
        # reach the mandatory final line.  In that case the prompt's own
        # decision rule is unambiguous: any YES means anomaly. An anomaly is
        # unambiguous as soon as any complete YES step exists.
        # An all-NO response needs the planner's expected count; otherwise we
        # cannot know whether generation stopped before a later step.
        if steps_label == "anomaly" or _step_log_is_complete(text, expected_step_count):
            reason, details = _reason_from_steps(text, steps_label)
            logger.warning(
                "VLM omitted its final decision line; derived label=%s from "
                "the available step log.",
                steps_label,
            )
            return steps_label, reason, details
        raise ValueError(f"Could not parse a final 'good'/'anomaly' label from VLM output:\n{text}")
    label = match.group(1).lower()
    if steps_label is not None and steps_label != label:
        if label_policy == "steps":
            logger.warning(
                "VLM final label '%s' contradicts its own step log (steps say '%s'); "
                "overriding because decision.label_policy=steps.",
                label, steps_label,
            )
            label = steps_label
        else:
            logger.warning(
                "VLM final label '%s' contradicts its own step log (steps say '%s'); "
                "preserving the model's final label because decision.label_policy=model. "
                "The rerun stage may re-evaluate this prediction.",
                label, steps_label,
            )

    if steps_label is not None:
        reason, details = _reason_from_steps(text, label)
    else:
        # Steps weren't in the expected "[STEP N] yes/no -> ..." format either.
        # Fall back to the full step-log text (everything before the bare
        # final label line) instead of a generic sentence, so localization
        # still receives the model's actual observations, not empty content.
        reason = text[: text.rfind(final_line)].strip() or final_line
        details = reason
    logger.warning(
        "VLM omitted structured '%s; reason=...; details=...' output; derived reason/details "
        "from the model's own step log instead of a placeholder.",
        label,
    )
    return label, reason, details


def decision_maker(
    modelcfg: ModelsConfig,
    prompts_dir: Path,
    vlmModel: "type[vlm]",
    ref_img: Image.Image,
    cand_img: Image.Image,
    planner_filepath: Path,
    decision_result_filepath: Path,
    probsdata_result_filepath: Path,
    text_base: Path | None = None,
    label_policy: str = "model",
    decision_prompt_template: str | None = None,
    secondary_profile: SecondaryClassificationProfile | None = None,
) -> dict[str, Any]:
    """
    Description - Runs the VLM decision maker for one candidate image against
    one reference image + planner, writes the decision text and per-step
    probability data to disk, and returns a summary dict.

    If text_base is provided, stores decision and probs_data paths relative
    to text_base.parent (dataset-directory-relative) for consistency with
    rerun.py's path resolution logic.
    """
    planner_steps = Path(planner_filepath).read_text(encoding="utf-8")
    expected_step_count = _planner_step_count(planner_steps)
    selected_prompt = (
        secondary_profile.prompt_template
        if secondary_profile else resolve_decision_prompt_template(decision_prompt_template)
    )
    prompt_kwargs: dict[str, Any] = {"planner_steps": planner_steps}
    if secondary_profile is not None:
        prompt_kwargs["binary_anomaly_rule_text"] = binary_anomaly_rule_text(secondary_profile)
    prompt = build_prompt(prompts_dir, selected_prompt, **prompt_kwargs)

    record_pipeline_event(
        "vlm_generate_start",
        decision_filepath=str(decision_result_filepath),
        planner_filepath=str(planner_filepath),
        image_count=2,
        model=modelcfg.vlm.modelname,
        decision_prompt_template=selected_prompt,
    )
    ablation_profile = os.environ.get("ANOMALY_ABLATION_PROFILE") == "1"
    vlm_call_time_sec = 0.0
    inference_time_sec = 0.0
    generated_token_count = 0
    timing_basis = "model_generate"
    generation_strategy = "combined"
    vlm_call_count = 0
    attempt_prompt = prompt
    step_log = None

    for attempt in range(1, DECISION_EVIDENCE_MAX_ATTEMPTS + 1):
        call_start = time.perf_counter()
        try:
            final_text, probs_data, generation_stats = vlmModel.run_vlm(
                modelcfg,
                [ref_img, cand_img],
                attempt_prompt,
            )
            vlm_call_count += 1
        except BaseException as exc:
            record_pipeline_event(
                "vlm_generate_python_failure",
                decision_filepath=str(decision_result_filepath),
                exception_type=type(exc).__name__,
                elapsed_sec=round(time.perf_counter() - call_start, 6),
                attempt=attempt,
            )
            raise
        attempt_call_time = time.perf_counter() - call_start
        vlm_call_time_sec += attempt_call_time
        if "generation_time_sec" in generation_stats:
            inference_time_sec += float(generation_stats["generation_time_sec"])
        else:
            inference_time_sec += attempt_call_time
            timing_basis = "vlm_call_fallback"
        if ablation_profile:
            generated_token_count += int(generation_stats["generated_token_count"])

        try:
            if secondary_profile is None:
                validate_yes_step_evidence(final_text)
            model_label, reason, details = parse_final_decision(
                final_text,
                label_policy=label_policy,
                expected_step_count=expected_step_count,
            )
            _validate_probability_steps(probs_data, expected_step_count)
            if secondary_profile is not None:
                if expected_step_count is None:
                    raise ValueError(
                        "Secondary classification requires consecutively numbered planner steps."
                    )
                evidence_steps = {
                    step
                    for rule in secondary_profile.classes
                    for step in rule.localization_steps
                }
                # Permit final-line evidence to repair exactly one bare,
                # localization-required YES.  With more than one such YES,
                # the final field cannot be assigned safely to a particular
                # step, so strict validation must still force a retry.
                normalized_final_text = normalize_step_decision_layout(final_text)
                bare_localization_yes = {
                    int(number)
                    for number, answer, detail in _STEP_LINE_RE.findall(normalized_final_text)
                    if (
                        answer.lower() == "yes"
                        and int(number) in evidence_steps
                        and not detail.strip()
                    )
                }
                final_evidence_fallback = (
                    {next(iter(bare_localization_yes)): details}
                    if len(bare_localization_yes) == 1 and details.strip()
                    else None
                )
                step_log = parse_step_log(
                    final_text,
                    list(range(1, expected_step_count + 1)),
                    evidence_required_steps=evidence_steps,
                    fallback_evidence=final_evidence_fallback,
                )
            break
        except ValueError as exc:
            rejected_path = _rejected_response_path(
                Path(decision_result_filepath), attempt
            )
            _atomic_write_text(rejected_path, final_text)
            missing_evidence = (
                missing_yes_evidence_steps(final_text)
                if secondary_profile is None else []
            )
            evidence_retry = (
                attempt == DECISION_FORMAT_MAX_ATTEMPTS
                and bool(missing_evidence)
            )
            if attempt >= DECISION_FORMAT_MAX_ATTEMPTS and not evidence_retry:
                logger.error(
                    "Invalid decision response after %d attempts (saved to %s):\n%s",
                    attempt,
                    rejected_path,
                    final_text,
                )
                raise
            logger.warning(
                "Decision response failed validation on attempt %d/%d: %s; "
                "raw response saved to %s; retrying with corrective feedback.",
                attempt,
                DECISION_EVIDENCE_MAX_ATTEMPTS if evidence_retry else DECISION_FORMAT_MAX_ATTEMPTS,
                exc,
                rejected_path,
            )
            record_pipeline_event(
                "vlm_generate_format_retry",
                decision_filepath=str(decision_result_filepath),
                attempt=attempt,
                error=str(exc),
                rejected_response_path=str(rejected_path),
            )
            attempt_prompt = (
                _format_evidence_retry_prompt(
                    planner_steps, expected_step_count, missing_evidence
                )
                if evidence_retry else _format_retry_prompt(
                    planner_steps, exc, expected_step_count
                )
            )

    generated_tokens_per_second = (
        generated_token_count / inference_time_sec if inference_time_sec > 0 else None
    )
    event_details: dict[str, Any] = {
        "decision_filepath": str(decision_result_filepath),
        "elapsed_sec": round(inference_time_sec, 6),
        "vlm_call_time_sec": round(vlm_call_time_sec, 6),
        "timing_basis": timing_basis,
        "generated_character_count": len(final_text),
        "probability_record_count": len(probs_data),
        "generation_attempts": attempt,
        "generation_strategy": generation_strategy,
        "vlm_call_count": vlm_call_count,
    }
    if ablation_profile:
        event_details.update(
            generated_token_count=generated_token_count,
            generated_tokens_per_second=(
                round(generated_tokens_per_second, 4)
                if generated_tokens_per_second is not None else None
            ),
        )
    record_pipeline_event(
        "vlm_generate_end",
        **event_details,
    )

    # Parsing and confidence validation completed before committing anything:
    # malformed output cannot leave artifacts that resume treats as complete.
    secondary_fields: dict[str, Any] = {}
    if secondary_profile is not None:
        assert step_log is not None
        outcome = classify_yes_steps(secondary_profile, step_log.yes_steps)
        probabilities = yes_probabilities(probs_data)
        class_scores = score_classes(secondary_profile, probabilities)
        label = outcome.binary_label
        if model_label != label:
            logger.warning(
                "VLM final label '%s' contradicts configured secondary step rules (%s); "
                "using the deterministic rule-derived label '%s'.",
                model_label,
                step_log.yes_steps,
                label,
            )
            relevant = localization_evidence_steps(
                secondary_profile, outcome.secondary_class, step_log.yes_steps
            )
            relevant_evidence = [step_log.evidence[step] for step in relevant if step in step_log.evidence]
            if label == "anomaly" and relevant_evidence:
                reason = details = " ".join(relevant_evidence)
            elif label == "good":
                reason = details = "The configured binary-anomaly step condition was not met."
        secondary_fields = {
            "yes_steps": outcome.yes_steps,
            "secondary_class": outcome.secondary_class,
            "secondary_class_scores": {
                name: round(score, 8) for name, score in class_scores.items()
            },
            "secondary_binary_score": round(
                binary_anomaly_score(secondary_profile, class_scores), 8
            ),
        }
    else:
        label = model_label
    final_text = _canonicalize_final_line(
        normalize_step_decision_layout(final_text), label, reason, details
    )
    probsdata_result_filepath = Path(probsdata_result_filepath)
    probability_text = "".join(
        json.dumps(entry, ensure_ascii=False) + "\n" for entry in probs_data
    )
    _atomic_write_text(probsdata_result_filepath, probability_text)

    if ablation_profile:
        performance_path = Path(decision_result_filepath).with_suffix(".performance.json")
        performance_payload = {
            "inference_time_sec": round(inference_time_sec, 6),
            "vlm_call_time_sec": round(vlm_call_time_sec, 6),
            "timing_basis": timing_basis,
            "generation_strategy": generation_strategy,
            "vlm_call_count": vlm_call_count,
            "generated_token_count": generated_token_count,
            "generated_tokens_per_second": (
                round(generated_tokens_per_second, 4)
                if generated_tokens_per_second is not None else None
            ),
        }
        _atomic_write_text(
            performance_path,
            json.dumps(performance_payload, indent=2, ensure_ascii=False),
        )

    # The decision is the pair's final commit marker. If interruption occurs
    # before this replace, resume sees no complete pair and reruns the image.
    decision_result_filepath = Path(decision_result_filepath)
    _atomic_write_text(decision_result_filepath, final_text)

    pred = 1 if label == "anomaly" else 0
    if ablation_profile:
        logger.info(
            "Decision for %s: label=%s pred=%d inference_time=%.3fs "
            "generated_tokens=%d tokens_per_second=%.2f",
            decision_result_filepath.name,
            label,
            pred,
            inference_time_sec,
            generated_token_count,
            generated_tokens_per_second or 0.0,
        )
    else:
        logger.info(
            "Decision for %s: label=%s pred=%d (inference_time=%.3fs)",
            decision_result_filepath.name, label, pred, inference_time_sec,
        )

    # Normalize paths: store relative to dataset directory (text_base.parent) if available
    stored_decision_path = str(decision_result_filepath)
    stored_probs_path = str(probsdata_result_filepath)

    if text_base is not None:
        try:
            stored_decision_path = str(decision_result_filepath.relative_to(text_base.parent))
            stored_probs_path = str(probsdata_result_filepath.relative_to(text_base.parent))
        except ValueError:
            # Fallback if relative_to fails (shouldn't happen in normal flow)
            pass

    result = {
        "label": label,
        "binary_prediction": pred,
        "decision_filepath": stored_decision_path,
        "probs_data_filepath": stored_probs_path,
        "inference_time_sec": round(inference_time_sec, 4),
        "vlm_call_time_sec": round(vlm_call_time_sec, 4),
        "timing_basis": timing_basis,
        "generation_strategy": generation_strategy,
        "vlm_call_count": vlm_call_count,
        "reason": reason,
        "details": details,
        **secondary_fields,
    }
    if ablation_profile:
        result.update(
            generated_token_count=generated_token_count,
            generated_tokens_per_second=(
                round(generated_tokens_per_second, 4)
                if generated_tokens_per_second is not None else None
            ),
        )
    return result
