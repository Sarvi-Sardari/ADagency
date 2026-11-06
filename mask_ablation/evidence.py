"""Load one collision-safe, rerun-aware XAI record per candidate image.

The anomaly pipeline has two evidence formats:

* first pass: step explanations in ``*.txt`` plus per-step YES/NO records;
* rerun: structured per-step JSON containing ``step``, ``reason`` and
  ``details`` but no new consolidated final summary.

This module overlays rerun outcomes instead of appending them.  Consequently
each source image occurs exactly once and a flipped image is attributed using
the actual rerun step that caused the flip.  No synthetic final reason is
invented.  ``final_reason_details`` mode merely joins the real reason/details
fields of the anomalous rerun step(s), which is the only faithful analogue of
the first-pass summary available in rerun artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

RERUN_RESULT_RELPATH = Path("rerun_results") / "rerun_result.jsonl"
EVIDENCE_MODES = {"yes_steps", "final_reason_details", "decision_output"}


@dataclass(frozen=True)
class EvidenceSample:
    image_path: Path
    reference_image_path: Path | None
    question: str
    output_text: str
    step: int
    predicted_token: str
    row_index: int
    decision_filepath: Path
    probs_data_filepath: Path | None
    planner_filepath: Path | None
    dataset_name: str
    label: str
    is_final_summary: bool
    original_yes_steps: tuple[int, ...]
    evidence_source: str

    def to_engine_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "reference_image_path": (
                str(self.reference_image_path) if self.reference_image_path else None
            ),
            "question": self.question,
            "output_text": self.output_text,
            "step": self.step,
            "predicted_token": self.predicted_token,
            "row_index": self.row_index,
            "decision_filepath": str(self.decision_filepath),
            "probs_data_filepath": (
                str(self.probs_data_filepath) if self.probs_data_filepath else ""
            ),
            "planner_filepath": (
                str(self.planner_filepath) if self.planner_filepath else ""
            ),
            "dataset_name": self.dataset_name,
            "label": self.label,
            "is_final_summary": self.is_final_summary,
            "original_yes_steps": list(self.original_yes_steps),
            "evidence_source": self.evidence_source,
        }


@dataclass(frozen=True)
class ImageEvidence:
    key: tuple[str, str, str]
    dataset_name: str
    label_folder: str
    source_label_folder: str
    img_name: str
    image_path: Path
    reference_image_path: Path | None
    is_anomaly: bool
    final_label: str
    rerun_status: str | None
    samples: tuple[EvidenceSample, ...]
    evidence_status: str

    @property
    def relative_mask_path(self) -> Path:
        directory = Path(self.label_folder)
        if self.source_label_folder != self.label_folder:
            directory /= self.source_label_folder
        return directory / f"{Path(self.img_name).stem}.png"

    def digest_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "image_path": str(self.image_path),
            "reference_image_path": (
                str(self.reference_image_path) if self.reference_image_path else None
            ),
            "is_anomaly": self.is_anomaly,
            "final_label": self.final_label,
            "rerun_status": self.rerun_status,
            "evidence_status": self.evidence_status,
            "samples": [sample.to_engine_dict() for sample in self.samples],
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning("Skipping malformed line %d in %s: %s", line_number, path, error)
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def image_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    raw_name = entry.get("img_name")
    if not raw_name:
        decision = entry.get("decision_filepath")
        if not decision:
            raise ValueError(f"Entry has neither img_name nor decision_filepath: {entry}")
        raw_name = Path(str(decision)).name.removesuffix(".txt")
    label_folder = str(entry.get("label_folder") or entry.get("label") or "unknown")
    source_label = str(entry.get("source_label_folder") or label_folder)
    return label_folder, source_label, str(raw_name)


def _resolve_artifact(
    raw_path: str | Path | None,
    *,
    dataset_run_dir: Path,
    project_root: Path,
) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        if not path.is_file():
            # Historical summaries can retain an old checkout/run root. Keep
            # the complete artifact-relative identity, never search by basename.
            for marker in ("rerun_results", "texts", "preproc"):
                if marker in path.parts:
                    index = path.parts.index(marker)
                    candidate = dataset_run_dir.joinpath(*path.parts[index:])
                    if candidate.is_file():
                        return candidate.resolve()
        return path.resolve()
    candidates = (
        dataset_run_dir / path,
        dataset_run_dir.parent / path,
        project_root / path,
        project_root.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _canonical_artifact_dir(
    base: Path, label_folder: str, source_label_folder: str, *, prefix: str = ""
) -> Path:
    directory = base / f"{prefix}{label_folder}"
    if source_label_folder != label_folder:
        directory /= source_label_folder
    return directory


def _resolve_candidate(
    entry: dict[str, Any],
    dataset_run_dir: Path,
    project_root: Path,
    key: tuple[str, str, str],
) -> Path:
    explicit = _resolve_artifact(
        entry.get("preprocess_cand_img_path"),
        dataset_run_dir=dataset_run_dir,
        project_root=project_root,
    )
    if explicit and explicit.is_file():
        return explicit

    label_folder, source_label, img_name = key
    fallback = _canonical_artifact_dir(
        dataset_run_dir / "preproc", label_folder, source_label, prefix="cand_"
    ) / f"processed_{img_name}"
    if fallback.is_file():
        return fallback.resolve()

    # Compatibility with the older hand-curated xai_test_sample summary,
    # whose rows point into several external run/dataset folders.
    decision = _resolve_artifact(
        entry.get("decision_filepath"),
        dataset_run_dir=dataset_run_dir,
        project_root=project_root,
    )
    dataset_name = str(entry.get("dataset") or dataset_run_dir.name)
    if decision:
        parts = decision.parts
        matching = [index for index, part in enumerate(parts) if part == dataset_name]
        if matching:
            actual_dataset_dir = Path(*parts[: matching[-1] + 1])
            legacy_label = "anomaly" if str(entry.get("label")) == "anomaly" else "good"
            legacy = (
                actual_dataset_dir
                / "preproc"
                / f"cand_{legacy_label}"
                / f"processed_{img_name}"
            )
            if legacy.is_file():
                return legacy.resolve()

    raise FileNotFoundError(
        f"Preprocessed candidate not found for {key!r}. Checked explicit path and {fallback}."
    )


def _resolve_reference(
    entry: dict[str, Any], dataset_run_dir: Path, project_root: Path
) -> Path | None:
    reference = _resolve_artifact(
        entry.get("preprocess_ref_img_path"),
        dataset_run_dir=dataset_run_dir,
        project_root=project_root,
    )
    return reference if reference and reference.is_file() else None


def _parse_step_tokens(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    tokens: dict[int, str] = {}
    for entry in read_jsonl(path):
        token = str(entry.get("token", "")).strip().upper()
        if token not in ("YES", "NO"):
            continue
        try:
            tokens[int(entry["step"])] = token
        except (KeyError, TypeError, ValueError):
            continue
    return tokens


def _parse_planner(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"Step\s+(\d+)\s*:\s*(.*?)(?=\n\s*Step\s+\d+\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {int(match.group(1)): match.group(2).strip() for match in pattern.finditer(text)}


def _parse_step_outputs(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\[?\s*STEP\s+(\d+)\s*\]?\s+(YES|NO)\s*(?:->|:)\s*"
        r"(.*?)(?=\n\s*\[?\s*STEP|\n\s*(?:Anomaly|Good)\s*;|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {int(match.group(1)): match.group(3).strip() for match in pattern.finditer(text)}


def _parse_final_fields(path: Path | None) -> tuple[str, str]:
    if path is None or not path.is_file():
        return "", ""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?:Anomaly|Good)\s*;\s*reason\s*=\s*(.*?)\s*;\s*details\s*=\s*(.*?)(?:\n|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _load_rerun_anomaly_steps(
    rerun: dict[str, Any], dataset_run_dir: Path, project_root: Path
) -> list[dict[str, Any]]:
    steps_path = _resolve_artifact(
        rerun.get("rerun_steps_filepath"),
        dataset_run_dir=dataset_run_dir,
        project_root=project_root,
    )
    if steps_path is None or not steps_path.is_file():
        return []
    try:
        value = json.loads(steps_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("Could not load rerun evidence %s: %s", steps_path, error)
        return []
    if not isinstance(value, list):
        return []
    return [step for step in value if isinstance(step, dict) and step.get("is_anomaly")]


def _question_for_steps(step_numbers: list[int], planner: dict[int, str]) -> str:
    """Build the context for a consolidated final reason/details target.

    Prefer the planner questions whose decisions were YES.  An anomalous final
    decision can occasionally coexist with all-NO step records; in that case
    the complete planner is still the real decision context and is more
    faithful than a generic invented question.  Use the generic fallback only
    when no planner text is available at all.
    """
    questions = [planner[number] for number in step_numbers if number in planner]
    if not questions:
        questions = [planner[number] for number in sorted(planner)]
    return " ".join(questions) or "What visual evidence supports the anomaly decision?"


def _base_samples(
    *,
    entry: dict[str, Any],
    row_index: int,
    key: tuple[str, str, str],
    image_path: Path,
    reference_path: Path | None,
    decision_path: Path,
    probs_path: Path | None,
    planner_path: Path | None,
    evidence_mode: str,
) -> list[EvidenceSample]:
    tokens = _parse_step_tokens(probs_path)
    planner = _parse_planner(planner_path)
    outputs = _parse_step_outputs(decision_path)
    yes_steps = sorted(step for step, token in tokens.items() if token == "YES")
    common = {
        "image_path": image_path,
        "reference_image_path": reference_path,
        "row_index": row_index,
        "decision_filepath": decision_path,
        "probs_data_filepath": probs_path,
        "planner_filepath": planner_path,
        "dataset_name": str(entry.get("dataset") or ""),
        "label": "anomaly",
        "predicted_token": "YES",
        "original_yes_steps": tuple(yes_steps),
        "evidence_source": "first_pass",
    }

    if evidence_mode == "yes_steps":
        return [
            EvidenceSample(
                question=planner[step],
                output_text=outputs[step],
                step=step,
                is_final_summary=False,
                **common,
            )
            for step in yes_steps
            if step in planner and step in outputs and outputs[step]
        ]

    if evidence_mode == "decision_output":
        output_text = decision_path.read_text(encoding="utf-8").strip()
        question = " ".join(planner[number] for number in sorted(planner))
        if not output_text:
            return []
        return [
            EvidenceSample(
                question=question or "Inspect the candidate image for anomalies.",
                output_text=output_text,
                step=999,
                is_final_summary=True,
                **common,
            )
        ]

    reason = str(entry.get("reason") or "").strip()
    details = str(entry.get("details") or "").strip()
    if not reason and not details:
        reason, details = _parse_final_fields(decision_path)
    output_text = " ".join(part for part in (reason, details) if part)
    if not output_text:
        return []
    return [
        EvidenceSample(
            question=_question_for_steps(yes_steps, planner),
            output_text=output_text,
            step=999,
            is_final_summary=True,
            **common,
        )
    ]


def _rerun_samples(
    *,
    entry: dict[str, Any],
    rerun: dict[str, Any],
    row_index: int,
    image_path: Path,
    reference_path: Path | None,
    decision_path: Path,
    probs_path: Path | None,
    planner_path: Path | None,
    dataset_run_dir: Path,
    project_root: Path,
    evidence_mode: str,
) -> list[EvidenceSample]:
    anomaly_steps = _load_rerun_anomaly_steps(rerun, dataset_run_dir, project_root)
    step_numbers = tuple(
        int(step["step_number"])
        for step in anomaly_steps
        if isinstance(step.get("step_number"), int)
    )
    common = {
        "image_path": image_path,
        "reference_image_path": reference_path,
        "row_index": row_index,
        "decision_filepath": decision_path,
        "probs_data_filepath": probs_path,
        "planner_filepath": planner_path,
        "dataset_name": str(entry.get("dataset") or ""),
        "label": "anomaly",
        "predicted_token": "YES",
        "original_yes_steps": step_numbers,
        "evidence_source": "rerun",
    }
    if evidence_mode == "yes_steps":
        samples: list[EvidenceSample] = []
        for index, step in enumerate(anomaly_steps):
            output_text = " ".join(
                str(step.get(field) or "").strip() for field in ("reason", "details")
            ).strip()
            if not output_text:
                continue
            samples.append(
                EvidenceSample(
                    question=str(step.get("step") or "").strip()
                    or "What visual evidence supports the anomaly decision?",
                    output_text=output_text,
                    step=int(step.get("step_number") or index + 1),
                    is_final_summary=False,
                    **common,
                )
            )
        return samples

    # Rerun stops at the first anomalous step, so it never emits a second
    # consolidated response.  For final/whole-output modes, concatenate only
    # the real anomalous rerun records; this preserves their provenance and
    # avoids borrowing the stale first-pass "good" summary.
    questions = [str(step.get("step") or "").strip() for step in anomaly_steps]
    parts: list[str] = []
    for step in anomaly_steps:
        parts.extend(
            str(step.get(field) or "").strip() for field in ("reason", "details")
        )
    output_text = " ".join(part for part in parts if part)
    if not output_text:
        return []
    return [
        EvidenceSample(
            question=" ".join(question for question in questions if question)
            or "What visual evidence supports the anomaly decision?",
            output_text=output_text,
            step=999,
            is_final_summary=True,
            **common,
        )
    ]


def build_image_evidence(
    *,
    run_dir: Path,
    dataset_name: str,
    include_rerun: bool,
    evidence_mode: str,
) -> list[ImageEvidence]:
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError(
            f"Unknown mask_generation_mode={evidence_mode!r}; "
            f"expected one of {sorted(EVIDENCE_MODES)}."
        )
    project_root = Path(__file__).resolve().parent.parent
    dataset_run_dir = run_dir / dataset_name
    entries = read_jsonl(dataset_run_dir / "all_data_summary.jsonl")

    reruns: dict[tuple[str, str, str], dict[str, Any]] = {}
    rerun_path = dataset_run_dir / RERUN_RESULT_RELPATH
    if include_rerun:
        if rerun_path.is_file():
            reruns = {image_key(entry): entry for entry in read_jsonl(rerun_path)}
        else:
            logger.warning(
                "[%s] include_rerun=true but %s is missing; using base predictions.",
                dataset_name,
                rerun_path,
            )

    seen: set[tuple[str, str, str]] = set()
    items: list[ImageEvidence] = []
    for row_index, entry in enumerate(entries):
        key = image_key(entry)
        if key in seen:
            raise ValueError(
                f"Duplicate image identity in {dataset_name}/all_data_summary.jsonl: {key!r}"
            )
        seen.add(key)
        image_path = _resolve_candidate(entry, dataset_run_dir, project_root, key)
        reference_path = _resolve_reference(entry, dataset_run_dir, project_root)
        decision_path = _resolve_artifact(
            entry.get("decision_filepath"),
            dataset_run_dir=dataset_run_dir,
            project_root=project_root,
        )
        if decision_path is None:
            raise ValueError(f"Missing decision_filepath for {key!r}")
        probs_path = _resolve_artifact(
            entry.get("probs_data_filepath"),
            dataset_run_dir=dataset_run_dir,
            project_root=project_root,
        )
        planner_path = _resolve_artifact(
            entry.get("planner_filepath"),
            dataset_run_dir=dataset_run_dir,
            project_root=project_root,
        )

        rerun = reruns.get(key)
        final_label = str((rerun or entry).get("label") or "good").strip().lower()
        is_anomaly = final_label == "anomaly"
        rerun_status = str(rerun.get("rerun_status")) if rerun else None
        samples: list[EvidenceSample] = []
        if is_anomaly:
            if rerun and rerun_status == "flipped_to_anomaly":
                samples = _rerun_samples(
                    entry=entry,
                    rerun=rerun,
                    row_index=row_index,
                    image_path=image_path,
                    reference_path=reference_path,
                    decision_path=decision_path,
                    probs_path=probs_path,
                    planner_path=planner_path,
                    dataset_run_dir=dataset_run_dir,
                    project_root=project_root,
                    evidence_mode=evidence_mode,
                )
            else:
                samples = _base_samples(
                    entry=entry,
                    row_index=row_index,
                    key=key,
                    image_path=image_path,
                    reference_path=reference_path,
                    decision_path=decision_path,
                    probs_path=probs_path,
                    planner_path=planner_path,
                    evidence_mode=evidence_mode,
                )

        if not is_anomaly:
            evidence_status = "predicted_good_zero_mask"
        elif samples:
            evidence_status = (
                "rerun_anomaly_evidence" if rerun_status == "flipped_to_anomaly" else "base_anomaly_evidence"
            )
        else:
            evidence_status = "anomaly_without_usable_evidence_zero_mask"
            logger.warning(
                "[%s] %s is predicted anomalous but has no usable %s evidence; "
                "the study will publish a zero mask rather than invent text.",
                dataset_name,
                key,
                evidence_mode,
            )

        items.append(
            ImageEvidence(
                key=key,
                dataset_name=dataset_name,
                label_folder=key[0],
                source_label_folder=key[1],
                img_name=key[2],
                image_path=image_path,
                reference_image_path=reference_path,
                is_anomaly=is_anomaly,
                final_label=final_label,
                rerun_status=rerun_status,
                samples=tuple(samples),
                evidence_status=evidence_status,
            )
        )

    return items


def evidence_digest(items: list[ImageEvidence]) -> str:
    payload = json.dumps(
        [item.digest_record() for item in items],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
