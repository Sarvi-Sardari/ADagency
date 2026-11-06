"""
object_crop.py (anomaly_detection)
==================================
Optional object-centred cropping step (added 2026-09).

Why this exists
---------------
Industrial frames often contain large amounts of background, fixtures, robot
parts, or neighbouring objects. When ``datasets[].object_crop.enabled`` is
true in the anomaly-detection config, every test image is first shown to the
VLM together with that dataset's object labels
(``prompts/object_labels/<dataset>_object_labels.txt`` by default). The VLM
returns ONE bounding box for the whole inspected object, the original image is
cropped to that box (plus ``padding_ratio``), and reference selection, the
planner-guided decision, rerun, and mask generation all operate on the crop.

Qwen3-VL and "native zoom"
--------------------------
Qwen3-VL does not crop images inside the network. Its official grounding
format is a JSON ``bbox_2d`` in coordinates RELATIVE to the supplied image on
a 0..1000 scale, and its "thinking with images" zoom-in is a tool call whose
crop is executed by host code. ``prompts/anomaly_detection_pipeline/
image_crop.j2`` therefore asks for exactly that native ``bbox_2d`` format and
this module performs the crop. The parser also accepts the pipeline's older
``box`` key so other VLM families remain usable.

Contracts
---------
- The model sees an aspect-preserving, *unpadded* downscale of the original
  image (bounded by ``models.vlm.image_processing`` maxima), so relative
  0..1000 coordinates map directly to original pixels.
- Malformed responses are retried with corrective feedback up to
  ``CROP_MAX_ATTEMPTS`` times; every rejected response is kept on disk.
- A well-formed "not found" answer (``found: false``, ``[]``, or null box) is
  NEVER converted into a full-image or empty crop. It raises
  ``ObjectNotFoundError`` with the model's reason so batch runs stop visibly and
  the live app can show "OBJECT NOT FOUND".
- Batch crops are cached per image as JSON, and the crop settings are frozen
  per dataset run in ``object_crop_profile.json`` so resume can never mix
  cropped and uncropped (or differently cropped) decisions in one run.
- Crop cost is measured like the decision (2026-09-25 follow-up):
  ``inference_time_sec`` sums the model-generation time of every crop attempt
  (``generation_time_sec`` from ``run_vlm``; whole-call time only as a
  fallback) and ``vlm_call_time_sec`` sums the full ``run_vlm`` calls. Both are
  cached with the box, so resumed images keep their original measurement, and
  are reported separately from the decision time in performance summaries.
  Caches written before this change carry no timing and report ``None``.

This module deliberately imports no torch/transformers code: the VLM is passed
in by the caller, keeping the file-only metrics path lightweight.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline_core.artifacts import atomic_write_text
from pipeline_core.imaging import _RESAMPLE
from pipeline_core.model_output import normalize_model_output
from pipeline_core.prompting import build_prompt

from .config import ModelsConfig, ObjectCropConfig, artifact_group_dir
from .hardware_monitor import record_pipeline_event
from .masking.backends import _normalize_box

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "object_crop_profile.json"
CACHE_SCHEMA_VERSION = 1
CROP_MAX_ATTEMPTS = 3
# A located object narrower/shorter than this many original pixels is treated
# as an implausible (malformed) answer rather than silently upscaled.
MIN_CROP_PIXELS = 8
# Name of the manifest written by anomaly_detection/crop_train_good.py next to train/good.
TRAIN_CROP_MANIFEST = "good_object_crop_manifest.json"


class ObjectNotFoundError(RuntimeError):
    """The VLM answered in the correct format that the object is not visible."""

    def __init__(
        self,
        message: str,
        *,
        labels: tuple[str, ...],
        reason: str,
        response: str,
        evidence_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.labels = labels
        self.reason = reason
        self.response = response
        self.evidence_path = evidence_path


@dataclass(frozen=True)
class ObjectCropResult:
    """One located object and its crop in ORIGINAL image pixel coordinates."""

    box_1000: tuple[float, float, float, float]
    box_pixels: tuple[int, int, int, int]  # (left, top, right, bottom), right/bottom exclusive
    source_size: tuple[int, int]            # (width, height) of the original image
    label: str
    response: str
    attempts: int
    # Summed over all attempts; None only for legacy caches without timing.
    inference_time_sec: float | None = None
    vlm_call_time_sec: float | None = None
    timing_basis: str | None = None

    def summary_record(self, metadata_path: str | None = None) -> dict[str, Any]:
        """Compact record stored in all_data_summary.jsonl / live audits.

        Metrics use ``box_pixels`` + ``source_size`` to place crop-geometry
        masks back onto the full-resolution ground truth; performance
        summaries average ``inference_time_sec`` as the crop cost.
        """
        record: dict[str, Any] = {
            "status": "cropped",
            "box_1000": [round(value, 3) for value in self.box_1000],
            "box_pixels": list(self.box_pixels),
            "source_size": list(self.source_size),
            "label": self.label,
            "attempts": self.attempts,
            "inference_time_sec": (
                round(self.inference_time_sec, 4) if self.inference_time_sec is not None else None
            ),
            "vlm_call_time_sec": (
                round(self.vlm_call_time_sec, 4) if self.vlm_call_time_sec is not None else None
            ),
            "timing_basis": self.timing_basis,
        }
        if metadata_path is not None:
            record["metadata_path"] = metadata_path
        return record


# ---------------------------------------------------------------------
# Prompt and response handling
# ---------------------------------------------------------------------


def prompt_sha256(prompts_dir: Path, template: str) -> str:
    """Hash the crop template bytes so edited prompts cannot silently resume."""
    return hashlib.sha256((Path(prompts_dir) / template).read_bytes()).hexdigest()


def render_crop_prompt(prompts_dir: Path, crop_cfg: ObjectCropConfig) -> str:
    return build_prompt(
        prompts_dir, crop_cfg.prompt_template, object_labels=list(crop_cfg.object_labels)
    )


def _decode_json(text: str) -> Any:
    cleaned = normalize_model_output(str(text))
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.I).strip()
    try:
        value: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        # Accept a single JSON object/array embedded in short prose.
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
        if match is None:
            raise ValueError(f"Object-crop response is not JSON: {text!r}") from None
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise ValueError(f"Object-crop response is not valid JSON: {text!r}") from error
    # Some VLMs JSON-encode the answer a second time; unwrap a bounded number.
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def _is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in {"false", "no"})


def parse_object_box(text: str) -> tuple[bool, list[float] | None, str, str]:
    """Parse an image_crop.j2 answer.

    Returns ``(found, box_1000, label, reason)``. ``found`` is False only for
    an explicit, well-formed "not visible" answer. Anything else that cannot
    be interpreted as a valid box raises ``ValueError`` (retryable).
    Several boxes (e.g. one per part) are merged into their enclosing box,
    which is exactly the "whole object" the prompt requests.
    """
    value = _decode_json(text)
    items: list[Any]
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        if not value:
            return False, None, "", "The model returned an empty detection list."
        items = value
    else:
        raise ValueError(f"Object-crop response must be a JSON object, got {type(value).__name__}")

    boxes: list[list[float]] = []
    labels: list[str] = []
    not_found_reasons: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Object-crop response list contains a non-object entry.")
        label = str(item.get("label") or "").strip()
        raw_box = next(
            (item[key] for key in ("bbox_2d", "box", "bbox") if item.get(key) is not None),
            None,
        )
        if _is_false(item.get("found")):
            not_found_reasons.append(str(item.get("reason") or "").strip())
            labels.append(label)
            continue
        if raw_box is None:
            raise ValueError("Object-crop response has neither a bbox_2d nor found=false.")
        box = _normalize_box(raw_box)
        if box is None:
            raise ValueError(f"Object-crop bbox_2d is not a valid [x1, y1, x2, y2] box: {raw_box!r}")
        boxes.append(box)
        labels.append(label)

    label = next((name for name in labels if name), "")
    if not boxes:
        reason = next((why for why in not_found_reasons if why), "")
        return False, None, label, reason
    merged = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    return True, merged, label, ""


def pixel_crop_box(
    box_1000: list[float] | tuple[float, ...],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Convert a relative 0..1000 box to padded, clamped original pixels."""
    x0, y0, x1, y1 = (float(value) for value in box_1000)
    left, right = x0 * width / 1000.0, x1 * width / 1000.0
    top, bottom = y0 * height / 1000.0, y1 * height / 1000.0
    pad_x = (right - left) * padding_ratio
    pad_y = (bottom - top) * padding_ratio
    crop = (
        max(0, int(math.floor(left - pad_x))),
        max(0, int(math.floor(top - pad_y))),
        min(width, int(math.ceil(right + pad_x))),
        min(height, int(math.ceil(bottom + pad_y))),
    )
    if crop[2] - crop[0] < MIN_CROP_PIXELS or crop[3] - crop[1] < MIN_CROP_PIXELS:
        raise ValueError(
            f"Located object box {crop} is smaller than {MIN_CROP_PIXELS} px; "
            "this is not a plausible whole-object crop."
        )
    return crop


def _model_view(image: Image.Image, modelcfg: ModelsConfig) -> Image.Image:
    """Aspect-preserving downscale WITHOUT padding (relative boxes stay exact)."""
    processing = modelcfg.vlm.image_processing
    max_w, max_h = processing.target_w[1], processing.target_h[1]
    scale = min(1.0, max_w / image.width, max_h / image.height)
    if scale >= 1.0:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, resample=_RESAMPLE[processing.resample])


def _retry_prompt(base_prompt: str, attempt: int, error: Exception, previous: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"FORMAT CORRECTION ATTEMPT {attempt - 1}: your previous answer could not be used "
        f"({str(error).splitlines()[0][:300]}). Reply with ONLY one JSON object of the form "
        '{"found": true, "bbox_2d": [x1, y1, x2, y2], "label": "<short product name>"} '
        "using 0-1000 relative coordinates with x1 < x2 and y1 < y2, or "
        '{"found": false, "bbox_2d": null, "label": "<short product name>", "reason": "<why>"} '
        "if the product is not visible. No prose, no Markdown.\n"
        f"Previous invalid answer: {json.dumps(previous[:2000])}"
    )


def locate_object(
    model: Any,
    modelcfg: ModelsConfig,
    image: Image.Image,
    crop_cfg: ObjectCropConfig,
    prompts_dir: Path,
    *,
    diagnostic_dir: Path | None = None,
    image_label: str = "the image",
) -> ObjectCropResult:
    """Ask the VLM for the whole-object box and return the padded crop.

    ``model`` is the loaded ``anomaly_detection.models.vlm`` class (or a test
    double exposing ``run_vlm``). Raises ``ObjectNotFoundError`` for a
    well-formed "not visible" answer and ``ValueError`` when every attempt was
    malformed. Both keep their evidence under ``diagnostic_dir`` when given.
    """
    if not crop_cfg.object_labels:
        raise ValueError(
            f"Object cropping needs object labels, but {crop_cfg.labels_file} is missing or empty."
        )
    rgb = image.convert("RGB")
    view = _model_view(rgb, modelcfg)
    base_prompt = render_crop_prompt(prompts_dir, crop_cfg)
    prompt = base_prompt
    previous_response = ""
    last_error: ValueError | None = None
    inference_time_sec = 0.0
    vlm_call_time_sec = 0.0
    timing_basis = "model_generate"

    for attempt in range(1, CROP_MAX_ATTEMPTS + 1):
        if attempt > 1 and last_error is not None:
            prompt = _retry_prompt(base_prompt, attempt, last_error, previous_response)
        record_pipeline_event(
            "object_crop_generate_start", image=image_label, attempt=attempt,
            model=modelcfg.vlm.modelname,
        )
        call_start = time.perf_counter()
        response, _, generation_stats = model.run_vlm(
            modelcfg,
            [view],
            prompt,
            require_step_context=False,
            require_confidence=False,
        )
        call_time = time.perf_counter() - call_start
        vlm_call_time_sec += call_time
        # Same convention as decision.py: pure generate() time when the
        # backend reports it, otherwise the whole call (flagged as fallback).
        if isinstance(generation_stats, dict) and "generation_time_sec" in generation_stats:
            inference_time_sec += float(generation_stats["generation_time_sec"])
        else:
            inference_time_sec += call_time
            timing_basis = "vlm_call_fallback"
        response = str(response)
        record_pipeline_event(
            "object_crop_generate_end", image=image_label, attempt=attempt,
            elapsed_sec=round(call_time, 6),
        )
        try:
            found, box_1000, label, reason = parse_object_box(response)
            if found:
                assert box_1000 is not None
                box_pixels = pixel_crop_box(box_1000, rgb.width, rgb.height, crop_cfg.padding_ratio)
                return ObjectCropResult(
                    box_1000=tuple(box_1000),  # type: ignore[arg-type]
                    box_pixels=box_pixels,
                    source_size=(rgb.width, rgb.height),
                    label=label,
                    response=response,
                    attempts=attempt,
                    inference_time_sec=inference_time_sec,
                    vlm_call_time_sec=vlm_call_time_sec,
                    timing_basis=timing_basis,
                )
        except ValueError as error:
            last_error = error
            previous_response = response
            if diagnostic_dir is not None:
                atomic_write_text(
                    Path(diagnostic_dir) / f"attempt_{attempt:02d}.json",
                    json.dumps(
                        {"attempt": attempt, "image": image_label, "error": str(error),
                         "response": response},
                        indent=2, ensure_ascii=False,
                    ),
                )
            logger.warning(
                "Invalid object-crop response for %s on attempt %d/%d: %s",
                image_label, attempt, CROP_MAX_ATTEMPTS, error,
            )
            continue

        # Well-formed "not found": deterministic decoding would repeat it, so
        # report it immediately instead of retrying or inventing a crop.
        evidence_path = None
        if diagnostic_dir is not None:
            evidence_path = Path(diagnostic_dir) / "object_not_found.json"
            atomic_write_text(
                evidence_path,
                json.dumps(
                    {"image": image_label, "labels": list(crop_cfg.object_labels),
                     "reason": reason, "response": response, "attempt": attempt},
                    indent=2, ensure_ascii=False,
                ),
            )
        record_pipeline_event("object_crop_not_found", image=image_label, reason=reason)
        message = (
            f"Object not found: the VLM could not locate the inspected object "
            f"({', '.join(crop_cfg.object_labels)}) in {image_label}. "
            f"Model reason: {reason or 'none given'}."
        )
        if evidence_path is not None:
            message += f" Evidence: {evidence_path}"
        raise ObjectNotFoundError(
            message,
            labels=crop_cfg.object_labels,
            reason=reason,
            response=response,
            evidence_path=evidence_path,
        )

    location = f"; inspect {diagnostic_dir}" if diagnostic_dir is not None else ""
    raise ValueError(
        f"Object-crop localization for {image_label} returned invalid output on all "
        f"{CROP_MAX_ATTEMPTS} attempts{location}."
    ) from last_error


# ---------------------------------------------------------------------
# Batch cache and per-run profile freezing
# ---------------------------------------------------------------------


def _cache_identity(
    image_name: str, source_size: tuple[int, int], crop_cfg: ObjectCropConfig,
    prompt_hash: str, model_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "image_name": image_name,
        "source_size": list(source_size),
        "labels": list(crop_cfg.object_labels),
        "padding_ratio": crop_cfg.padding_ratio,
        "prompt_template": crop_cfg.prompt_template,
        "prompt_sha256": prompt_hash,
        "model": model_name,
    }


def result_from_payload(payload: dict[str, Any]) -> ObjectCropResult:
    """Rebuild a cached result; raises on malformed content."""
    box_pixels = tuple(int(value) for value in payload["box_pixels"])
    source_size = tuple(int(value) for value in payload["source_size"])
    box_1000 = tuple(float(value) for value in payload["box_1000"])
    if len(box_pixels) != 4 or len(source_size) != 2 or len(box_1000) != 4:
        raise ValueError("cached crop has the wrong shape")
    left, top, right, bottom = box_pixels
    if not (0 <= left < right <= source_size[0] and 0 <= top < bottom <= source_size[1]):
        raise ValueError("cached crop lies outside its source image")

    def optional_time(key: str) -> float | None:
        value = payload.get(key)
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"cached crop has an invalid {key}")
        return number

    return ObjectCropResult(
        box_1000=box_1000,  # type: ignore[arg-type]
        box_pixels=box_pixels,  # type: ignore[arg-type]
        source_size=source_size,  # type: ignore[arg-type]
        label=str(payload.get("label", "")),
        response=str(payload.get("response", "")),
        attempts=int(payload.get("attempts", 1)),
        inference_time_sec=optional_time("inference_time_sec"),
        vlm_call_time_sec=optional_time("vlm_call_time_sec"),
        timing_basis=payload.get("timing_basis"),
    )


def load_or_locate_object_crop(
    model: Any,
    modelcfg: ModelsConfig,
    image_path: Path,
    crop_cfg: ObjectCropConfig,
    prompts_dir: Path,
    cache_path: Path,
) -> ObjectCropResult:
    """Reuse a valid cached crop for ``image_path`` or ask the VLM once.

    The cache is keyed by image name/size, labels, padding, prompt hash, and
    model. A missing/corrupt/stale cache is recomputed (never trusted).
    Rejected and not-found responses are kept in ``<cache stem>.diagnostics``.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    prompt_hash = prompt_sha256(prompts_dir, crop_cfg.prompt_template)
    identity = _cache_identity(
        image_path.name, (image.width, image.height), crop_cfg, prompt_hash,
        modelcfg.vlm.modelname,
    )
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in identity.items()):
                return result_from_payload(cached)
            logger.info("Object-crop cache is stale for %s; locating again.", image_path)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            logger.warning("Ignoring unreadable object-crop cache %s: %s", cache_path, error)

    result = locate_object(
        model, modelcfg, image, crop_cfg, prompts_dir,
        diagnostic_dir=cache_path.with_name(f"{cache_path.stem}.diagnostics"),
        image_label=str(image_path),
    )
    payload = {
        **identity,
        "source_image": str(image_path),
        "box_1000": list(result.box_1000),
        "box_pixels": list(result.box_pixels),
        "label": result.label,
        "attempts": result.attempts,
        "inference_time_sec": result.inference_time_sec,
        "vlm_call_time_sec": result.vlm_call_time_sec,
        "timing_basis": result.timing_basis,
        "response": result.response,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_write_text(cache_path, json.dumps(payload, indent=2, ensure_ascii=False))
    return result


def profile_snapshot(crop_cfg: ObjectCropConfig, prompts_dir: Path) -> dict[str, Any]:
    """Portable description of the crop method (no machine-local paths)."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "enabled": True,
        "labels": list(crop_cfg.object_labels),
        "padding_ratio": crop_cfg.padding_ratio,
        "prompt_template": crop_cfg.prompt_template,
        "prompt_sha256": prompt_sha256(prompts_dir, crop_cfg.prompt_template),
    }


def load_object_crop_profile(output_dataset_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dataset_dir) / PROFILE_FILENAME
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("enabled") is not True:
        raise ValueError(f"{path} is not a valid object-crop profile")
    return value


def _has_existing_decisions(output_dataset_dir: Path) -> bool:
    texts = Path(output_dataset_dir) / "texts"
    summary = Path(output_dataset_dir) / "all_data_summary.jsonl"
    return summary.exists() or (texts.is_dir() and any(texts.rglob("*_prob_data.txt")))


def ensure_object_crop_profile(
    output_dataset_dir: Path,
    crop_cfg: ObjectCropConfig | None,
    prompts_dir: Path,
) -> None:
    """Freeze crop settings for this dataset run; reject unsafe resume changes.

    Mirrors ``pipeline._ensure_secondary_profile``: toggling cropping, changing
    labels/padding, or editing image_crop.j2 inside an existing run directory
    would silently mix incompatible decisions, so it is refused.
    """
    path = Path(output_dataset_dir) / PROFILE_FILENAME
    enabled = crop_cfg is not None and crop_cfg.enabled
    if not enabled:
        if path.exists():
            raise RuntimeError(
                f"Object cropping is disabled, but this run was produced with object "
                f"cropping ({path}). Use a fresh run directory."
            )
        return
    assert crop_cfg is not None
    configured = profile_snapshot(crop_cfg, prompts_dir)
    if path.exists():
        saved = load_object_crop_profile(output_dataset_dir)
        if saved != configured:
            raise RuntimeError(
                f"The configured object-crop settings (labels, padding, or "
                f"{crop_cfg.prompt_template} content) differ from those saved in {path}. "
                "Use a fresh run directory."
            )
        return
    if _has_existing_decisions(output_dataset_dir):
        raise RuntimeError(
            f"Object cropping is enabled, but {output_dataset_dir} already contains "
            "uncropped detection results. Use a fresh run directory."
        )
    atomic_write_text(path, json.dumps(configured, indent=2) + "\n")


def validate_summary_crop_record(record: Any, *, cropping_enabled: bool) -> str | None:
    """Return an error message when a summary row's crop record is inconsistent."""
    if not cropping_enabled:
        return "has an object_crop record in an uncropped run" if record is not None else None
    if not isinstance(record, dict) or record.get("status") != "cropped":
        return "lacks a valid object_crop record"
    try:
        result_from_payload(record)
    except (KeyError, TypeError, ValueError) as error:
        return f"has an invalid object_crop record: {error}"
    return None


def object_crop_timing_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Crop-cost fields for a dataset ``performance_*.json`` (empty if uncropped).

    ``avg_vlm_inference_time_per_image_sec`` elsewhere stays the DECISION-only
    time so it remains comparable with uncropped runs. This adds:

    - ``avg_object_crop_time_per_image_sec``: model generation time of the
      crop call(s) per image (all attempts);
    - ``avg_vlm_inference_time_with_object_crop_per_image_sec``: decision +
      crop generation time, over images that have both measurements.

    Legacy crop records without timing are counted in
    ``num_images_with_object_crop`` but excluded from the averages.
    """
    cropped = [entry for entry in entries if isinstance(entry.get("object_crop"), dict)]
    if not cropped:
        return {}

    def number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result >= 0 else None

    timed = [
        (entry, number(entry["object_crop"].get("inference_time_sec")))
        for entry in cropped
    ]
    timed = [(entry, crop_time) for entry, crop_time in timed if crop_time is not None]
    crop_times = [crop_time for _, crop_time in timed]
    call_times = [
        value for entry, _ in timed
        if (value := number(entry["object_crop"].get("vlm_call_time_sec"))) is not None
    ]
    combined = [
        decision_time + crop_time
        for entry, crop_time in timed
        if (decision_time := number(entry.get("inference_time_sec"))) is not None and decision_time > 0
    ]
    bases = {str(entry["object_crop"].get("timing_basis")) for entry, _ in timed}
    return {
        "num_images_with_object_crop": len(cropped),
        "num_images_with_object_crop_timing": len(crop_times),
        "total_object_crop_time_sec": round(sum(crop_times), 4),
        "avg_object_crop_time_per_image_sec": (
            round(sum(crop_times) / len(crop_times), 4) if crop_times else None
        ),
        "total_object_crop_vlm_call_time_sec": round(sum(call_times), 4),
        "num_images_with_vlm_timing_with_object_crop": len(combined),
        "total_vlm_inference_time_with_object_crop_sec": round(sum(combined), 4),
        "avg_vlm_inference_time_with_object_crop_per_image_sec": (
            round(sum(combined) / len(combined), 4) if combined else None
        ),
        "object_crop_timing_basis": (
            next(iter(bases)) if len(bases) == 1 else ("mixed" if bases else None)
        ),
    }


def train_crop_manifest_warning(
    dataset_name: str, root_dir: Path, crop_cfg: ObjectCropConfig | None,
    prompts_dir: Path,
) -> str | None:
    """Explain a train/good vs. test-image crop mismatch, if any (advisory).

    Detection still runs: cropping references is an optional ad-hoc step
    (``python -m anomaly_detection.crop_train_good``), but a mismatch usually weakens CLIP reference
    retrieval and the reference/candidate comparison.
    """
    manifest_path = Path(root_dir) / "train" / TRAIN_CROP_MANIFEST
    enabled = crop_cfg is not None and crop_cfg.enabled
    if not enabled:
        if manifest_path.is_file():
            return (
                f"[{dataset_name}] train/good was object-cropped by anomaly_detection/crop_train_good.py, "
                "but object_crop is disabled for test images."
            )
        return None
    assert crop_cfg is not None
    if not manifest_path.is_file():
        return (
            f"[{dataset_name}] test images will be object-cropped, but train/good "
            "references are uncropped. Run `python -m anomaly_detection.crop_train_good` first if cropped "
            "references are intended."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = manifest.get("settings", {})
        current = profile_snapshot(crop_cfg, prompts_dir)
        differing = [
            key for key in ("labels", "padding_ratio", "prompt_sha256")
            if settings.get(key) != current[key]
        ]
    except (OSError, ValueError, AttributeError) as error:
        return f"[{dataset_name}] could not read {manifest_path}: {error}"
    if differing:
        return (
            f"[{dataset_name}] train/good was cropped with different settings "
            f"({', '.join(differing)}) than the configured test-image crop."
        )
    return None


def atomic_save_image(path: Path, image: Image.Image) -> None:
    """Save an image beside its final path, then atomically rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
    try:
        save_kwargs: dict[str, Any] = {}
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": 95, "subsampling": 0}
        image.save(temporary, **save_kwargs)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_cropped_candidate_png(
    source_image: Path,
    box_pixels: tuple[int, int, int, int],
    output_dataset_dir: Path,
    label_folder: str,
    source_label_folder: str,
) -> Path:
    """Save the original-resolution object crop, separate from letterboxed preproc."""
    source_image = Path(source_image)
    destination = artifact_group_dir(
        Path(output_dataset_dir) / "cropped_images", label_folder, source_label_folder
    ) / f"{source_image.name}.png"
    with Image.open(source_image) as opened:
        image = opened.convert("RGB")
    left, top, right, bottom = map(int, box_pixels)
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError(f"Invalid crop box for {source_image}: {box_pixels}")
    atomic_save_image(destination, image.crop((left, top, right, bottom)))
    return destination
