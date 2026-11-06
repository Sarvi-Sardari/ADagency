"""Standalone, extensible mask generation for completed anomaly runs.

Source-aware behavior change (2026-08)
---------------------------------------
Merged anomaly records are now identified by
``(label_folder, source_label_folder, img_name)`` instead of only
``(label_folder, img_name)``.  When source and display labels differ, masks
and coordinate caches include the source folder.  For example, the old,
colliding path ``masks/sam3/anomalies/000.png`` is now written as
``masks/sam3/anomalies/broken_large/000.png``. Non-merged groups retain the
old compact layout. Legacy runs whose preprocessed images already collided
are rejected with an actionable error; ``--overwrite`` cannot reconstruct
images that were overwritten during inference. A performance record is now
also emitted for every configured dataset, including datasets that performed
zero new work because all source-aware outputs were already valid.

The output contract is::

    <run_dir>/<dataset>/masks/<method_folder>/<label_folder>/[<source_label_folder>/]<image_stem>.png

where <method_folder> is:
  - "bounding_box"                                     for the "bounding_box" method
  - "<segmentation_model.model_name>"                  for the "segmentation" method
  - "bounding_box_<segmentation_model.model_name>"      for the "bounding_box_segmentation" method

`segmentation_model.model_name` comes from configs/model_configs.yaml (see
config.py's `SegmentationModelConfig`) -- so swapping the segmentation
backend (e.g. sam3 -> some future model) only requires editing that one
config section; nothing here or in mask_generation_configs.yaml needs to
change. See "Adding a new segmentation backend" further below.

Every method writes a grayscale anomaly score map in the same letterboxed
geometry used for anomaly-detection inference. ``metrics.py`` discovers
method directories dynamically (by listing <run_dir>/<dataset>/masks/*) and
removes that padding before pixel comparison.

Bounding-box coordinates (the VLM localization result used by both
"bounding_box" and "bounding_box_segmentation") are localized ONCE per image
and cached/reused, but a copy is written next to every method that actually
needs them, i.e. under
``<masks>/<method_folder>/coordinates_files/<label_folder>/[<source_label_folder>/]<image_stem>.json``
-- so each method's own folder is self-contained and human-inspectable.

Resumability / --overwrite semantics
-------------------------------------
``overwrite: false`` (the default) makes this script SAFE TO RE-RUN /
RESUME AT ANY TIME:
  - Per (dataset, method, image), if the final source-aware mask path already
    exists on disk AND is a valid, decodable image (see
    `_is_valid_score_map_png`), that image+method pair is treated as already
    done and skipped -- so an interrupted run just needs the same command
    run again. A truncated/corrupt PNG (e.g. left behind by a crash
    mid-``Image.save()``) is correctly treated as NOT done and regenerated,
    not silently accepted just because a file happens to exist at that path.
  - The bounding-box coordinates JSON described above is ALSO treated as a
    cache: if it already exists, parses as valid JSON with the expected
    shape, and its recorded `image`/`reasoning` still match, it is reused
    instead of asking the VLM again, even for images whose final mask PNG
    hasn't been generated yet (e.g. the SAM3 step of
    "bounding_box_segmentation" failed after localization succeeded). A
    corrupt/malformed cache file is treated as a cache miss (re-localized),
    not an error.
  - Every mask PNG and coordinates JSON is written via a temp-file +
    atomic-rename (`_atomic_write_text` / `_save_score_map`'s internal temp
    file), so a crash/segfault mid-write can never leave a half-written file
    sitting at the final path in the first place -- the two checks above are
    a second line of defense for files written before this hardening, or by
    an external process.
  - Each dataset+method's settings are fingerprinted (see
    `_method_signature`) into
    ``<run_dir>/<dataset>/mask_generation/methods/<method_folder>.json`` the
    first time it is generated. Re-running with the SAME settings and
    `overwrite=false` is a no-op check (fingerprint matches, nothing to
    invalidate). If a setting changed (e.g. a different segmentation model or
    confidence_threshold) and you run again WITHOUT --overwrite, generation
    stops with an error instead of silently mixing masks made under
    different settings.
``overwrite: true`` ignores all of the above: every requested image+method is
regenerated from scratch (existing masks and the coordinates cache are both
bypassed) and the fingerprint file is refreshed to match the current
settings.

Adding a new segmentation backend
-----------------------------------
1. Point `models.segmentation_model.model_name` (configs/model_configs.yaml)
   at your new backend's name.
2. Register a loader with `@register_segmentation_backend("<model_name>")`
   that takes a `SegmentationModelConfig` and returns a "processor" object.
3. If your processor exposes the same adapter shape SAM3 does
   (`set_image(image)`, `set_text_prompt(prompt, state)`,
   `add_geometric_prompt(box, label, state)`, each returning a dict with
   `"masks_logits"` / `"scores"`), the existing `generate_segmentation` /
   `generate_bounding_box_segmentation` generators work unmodified. Otherwise
   register your own generator function(s) with `@register_method(...)`.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from ..config import (
    MaskGenerationConfig,
    MaskGenerationDatasetEntry,
    ModelsConfig,
    SegmentationModelConfig,
    artifact_group_dir,
    load_models_config,
)
from ..secondary_classification import (
    load_profile_snapshot,
    localization_evidence_steps,
    profile_fingerprint,
    sole_binary_anomaly_class,
)
from pipeline_core.artifacts import atomic_write_text as _atomic_write_text
from pipeline_core.artifacts import read_jsonl
from pipeline_core.prompting import build_prompt as _build_prompt

logger = logging.getLogger(__name__)

RERUN_RESULT_RELPATH = Path("rerun_results") / "rerun_result.jsonl"
METHOD_METADATA_SUBDIR = Path("mask_generation") / "methods"
COORDINATES_SUBDIR_NAME = "coordinates_files"
# Keep the existing generator version for backward compatibility with safe,
# unmerged runs: their output paths and pixels did not change. Legacy MERGED
# runs are rejected separately by the preprocessed-path collision guard in
# _build_items(), so a global version bump would only force unnecessary
# --overwrite runs for LOCO/multi-class datasets whose labels were never merged.
GENERATOR_VERSION = 3
LOCALIZATION_MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------
# Process-wise progress-bar colors: one fixed color per canonical mask
# method (the "process"), reused across every dataset row so the same
# method is visually identifiable at a glance no matter how many datasets
# are running. Unknown/future method names fall back to a cycling palette
# assigned on first use.
# ---------------------------------------------------------------------
_METHOD_COLORS: dict[str, str] = {
    "bounding_box": "bright_cyan",
    "segmentation": "bright_magenta",
    "bounding_box_segmentation": "bright_yellow",
}
_FALLBACK_COLOR_CYCLE = itertools.cycle(["bright_green", "bright_blue", "orange3", "bright_red"])
_fallback_color_cache: dict[str, str] = {}


def _method_color(method: str) -> str:
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    if method not in _fallback_color_cache:
        _fallback_color_cache[method] = next(_FALLBACK_COLOR_CYCLE)
    return _fallback_color_cache[method]


class _PerTaskColorBarColumn(BarColumn):
    """A BarColumn whose fill color comes from each task's own `color` field,
    so different processes (methods) can be told apart at a glance even
    though they share one Progress display."""

    def render(self, task):  # type: ignore[override]
        bar = super().render(task)
        color = task.fields.get("color")
        if color:
            bar.complete_style = color
            bar.finished_style = color
        return bar


@dataclass(frozen=True)
class MethodSpec:
    name: str
    needs_boxes: bool
    needs_segmentation: bool
    generate: Callable[["GenerationItem", "Runtime"], np.ndarray]


@dataclass
class GenerationItem:
    dataset: MaskGenerationDatasetEntry
    dataset_run_dir: Path
    entry: dict[str, Any]
    img_name: str
    label_folder: str
    source_label_folder: str
    image_path: Path
    is_anomaly: bool
    reasoning: str
    pending_methods: set[str]
    secondary_class: str | None = None
    boxes: list[list[float]] | None = None
    box_labels: list[str] | None = None


METHOD_REGISTRY: dict[str, MethodSpec] = {}
# Segmentation backend loaders, keyed by SegmentationModelConfig.modelname
# (lower-cased). See module docstring "Adding a new segmentation backend".
SEGMENTATION_BACKENDS: dict[str, Callable[[SegmentationModelConfig], Any]] = {}


def register_method(name: str, *, needs_boxes: bool = False, needs_segmentation: bool = False):
    """Register a mask-generation method. New XAI/segmentation techniques
    only need a generator function and this decorator."""
    def decorator(function: Callable[[GenerationItem, "Runtime"], np.ndarray]):
        if name in METHOD_REGISTRY:
            raise ValueError(f"Mask method already registered: {name}")
        METHOD_REGISTRY[name] = MethodSpec(name, needs_boxes, needs_segmentation, function)
        return function
    return decorator


def register_segmentation_backend(model_name: str):
    """Register a segmentation-model loader under `model_name` (matched
    case-insensitively against models.segmentation_model.model_name)."""
    def decorator(loader: Callable[[SegmentationModelConfig], Any]):
        key = model_name.strip().lower()
        if key in SEGMENTATION_BACKENDS:
            raise ValueError(f"Segmentation backend already registered: {model_name}")
        SEGMENTATION_BACKENDS[key] = loader
        return loader
    return decorator


def _require_segmentation_model(models_cfg: ModelsConfig) -> SegmentationModelConfig:
    """Fetch models.segmentation_model, raising a clear, actionable error if
    a segmentation-based method was requested but it's missing."""
    if models_cfg.segmentation_model is None:
        raise ValueError(
            "A segmentation-based mask method was requested but "
            "models.segmentation_model is missing from configs/model_configs.yaml."
        )
    return models_cfg.segmentation_model


def _method_folder_name(models_cfg: ModelsConfig, method: str) -> str:
    """Maps a technique-level method key (from mask_generation_configs.yaml's
    `methods:` list) to the actual on-disk folder name under masks/. Plain
    bounding-box localization has no backend model, so its folder is the
    method key itself; segmentation-based methods are folder-named after
    whichever segmentation model is currently configured, so switching
    backends automatically renames future output without editing any
    dataset's `methods:` list."""
    if method == "segmentation":
        return _require_segmentation_model(models_cfg).modelname
    if method == "bounding_box_segmentation":
        return f"bounding_box_{_require_segmentation_model(models_cfg).modelname}"
    return method


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return read_jsonl(path, logger=logger)


def _image_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Return the collision-safe identity shared with rerun and metrics.

    ``source_label_folder`` was added before this generator-version change,
    but the fallback keeps genuinely old, non-merged summaries readable.
    """
    img_name = entry.get("img_name")
    if not img_name:
        decision_path = entry.get("decision_filepath")
        if not decision_path:
            raise ValueError(f"Result has neither img_name nor decision_filepath: {entry}")
        img_name = Path(decision_path).name.removesuffix(".txt")
    label_folder = str(entry["label_folder"])
    source_label_folder = str(entry.get("source_label_folder") or label_folder)
    return label_folder, source_label_folder, str(img_name)


def _artifact_relative_path(
    label_folder: str, source_label_folder: str, filename: str,
) -> Path:
    """Build a path using the same merge-only rule as inference and rerun."""
    return artifact_group_dir(
        Path(), label_folder, source_label_folder
    ) / filename


def _item_display_path(item: GenerationItem) -> str:
    """Human-readable identity matching the actual source-aware tree."""
    return _artifact_relative_path(
        item.label_folder, item.source_label_folder, item.img_name
    ).as_posix()


def _resolve_artifact(raw_path: str | Path, dataset_run_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        # Run summaries can have been produced through a mount on another
        # machine (for example a checkout at a different path on another PC) and later
        # consumed from the source checkout on pc1.  Preserve the suffix below
        # the current dataset directory when the original absolute prefix is
        # no longer available.
        parts = path.parts
        matches = [index for index, value in enumerate(parts) if value == dataset_run_dir.name]
        for index in reversed(matches):
            relocated = dataset_run_dir.joinpath(*parts[index + 1:])
            if relocated.is_file():
                return relocated.resolve()
        return path
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = (
        Path.cwd() / path,
        project_root / path,
        dataset_run_dir / path,
        dataset_run_dir.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _decision_reasoning(entry: dict[str, Any], dataset_run_dir: Path) -> str:
    summary_reasoning = " ".join(
        str(entry.get(field, "")).strip() for field in ("reason", "details")
        if str(entry.get(field, "")).strip()
    )
    if summary_reasoning:
        return summary_reasoning

    decision_path_raw = entry.get("decision_filepath")
    if not decision_path_raw:
        return "An anomaly was detected in this image."
    decision_path = _resolve_artifact(decision_path_raw, dataset_run_dir)
    if not decision_path.is_file():
        label_folder, source_label_folder, img_name = _image_key(entry)
        canonical_folder = artifact_group_dir(
            dataset_run_dir / "texts", label_folder, source_label_folder
        )
        # Also recognize the safe-but-duplicated label/label path produced by
        # a short-lived earlier implementation. New files always use the
        # canonical merge-only folder above.
        legacy_folder = dataset_run_dir / "texts" / label_folder / source_label_folder
        fallbacks = (
            canonical_folder / f"{img_name}.txt",
            legacy_folder / f"{img_name}.txt",
        )
        decision_path = next((path for path in fallbacks if path.is_file()), decision_path)
    if not decision_path.is_file():
        logger.warning("Decision text not found for localization: %s", decision_path)
        return "An anomaly was detected in this image."

    text = decision_path.read_text(encoding="utf-8")
    parts: list[str] = []
    for field in ("reason", "details"):
        match = re.search(rf"\b{field}\s*=\s*(.*?)(?=;\s*\w+\s*=|$)", text, flags=re.I | re.S)
        if match and match.group(1).strip():
            parts.append(match.group(1).strip())
    return " ".join(parts) or text.strip()


def _rerun_reasoning(rerun: dict[str, Any], dataset_run_dir: Path) -> str | None:
    raw_path = rerun.get("rerun_steps_filepath")
    if not raw_path:
        return None
    path = _resolve_artifact(raw_path, dataset_run_dir)
    if not path.is_file():
        return None
    try:
        steps = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    anomaly_steps = [step for step in steps if isinstance(step, dict) and step.get("is_anomaly")]
    parts = []
    for step in anomaly_steps:
        parts.extend(str(step.get(key, "")).strip() for key in ("step", "reason", "details"))
    return " ".join(part for part in parts if part) or None


_STEP_EVIDENCE_RE = re.compile(
    r"(?im)^[ \t]*\[step\s+(\d+)\][ \t]*(yes|no)[ \t]*(?:->|:)?[ \t]*(.*)$"
)


def _secondary_reasoning(
    entry: dict[str, Any], dataset_run_dir: Path, relevant_steps: list[int]
) -> str:
    """Return only evidence belonging to the selected class's YES steps."""
    decision_path_raw = entry.get("decision_filepath")
    if not decision_path_raw:
        raise ValueError("Secondary mask generation requires decision_filepath.")
    decision_path = _resolve_artifact(decision_path_raw, dataset_run_dir)
    text = decision_path.read_text(encoding="utf-8")
    evidence: dict[int, str] = {
        int(number): " ".join(detail.split())
        for number, answer, detail in _STEP_EVIDENCE_RE.findall(text)
        if answer.lower() == "yes" and detail.strip()
    }

    rerun = entry.get("_effective_rerun")
    if isinstance(rerun, dict) and rerun.get("rerun_steps_filepath"):
        rerun_path = _resolve_artifact(rerun["rerun_steps_filepath"], dataset_run_dir)
        steps = json.loads(rerun_path.read_text(encoding="utf-8"))
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("step_number"), int):
                continue
            number = step["step_number"]
            if not step.get("is_anomaly"):
                evidence.pop(number, None)
                continue
            detail = " ".join(
                str(step.get(field, "")).strip()
                for field in ("reason", "details")
                if str(step.get(field, "")).strip()
            )
            if detail:
                evidence[number] = detail

    missing = [step for step in relevant_steps if not evidence.get(step)]
    if missing:
        raise ValueError(
            f"Secondary class {entry.get('secondary_class')!r} lacks localization "
            f"evidence for effective YES step(s) {missing}."
        )
    return " ".join(evidence[step] for step in relevant_steps)


def _load_effective_entries(
    dataset_run_dir: Path, include_rerun: bool,
) -> list[tuple[dict[str, Any], bool, str]]:
    """Reads all_data_summary.jsonl and, when include_rerun is set, overlays
    each image's rerun_result.jsonl outcome (label + rerun-specific reasoning)
    so masks reflect the FINAL prediction, not just the first pass."""
    entries = _read_jsonl(dataset_run_dir / "all_data_summary.jsonl")
    reruns: dict[tuple[str, str, str], dict[str, Any]] = {}
    if include_rerun:
        rerun_path = dataset_run_dir / RERUN_RESULT_RELPATH
        if rerun_path.is_file():
            reruns = {_image_key(entry): entry for entry in _read_jsonl(rerun_path)}
        else:
            logger.warning("include_rerun=true but %s is missing; using base predictions", rerun_path)

    effective = []
    for entry in entries:
        rerun = reruns.get(_image_key(entry))
        effective_entry = dict(entry)
        if rerun is not None:
            for field in (
                "label", "binary_prediction", "yes_steps", "secondary_class",
                "secondary_class_scores", "secondary_binary_score",
            ):
                if field in rerun:
                    effective_entry[field] = rerun[field]
            effective_entry["_effective_rerun"] = rerun
        label = effective_entry.get("label")
        reasoning = _decision_reasoning(entry, dataset_run_dir)
        if rerun is not None and label == "anomaly":
            reasoning = _rerun_reasoning(rerun, dataset_run_dir) or reasoning
        effective.append((effective_entry, str(label).lower() == "anomaly", reasoning))
    return effective


def _output_path(item: GenerationItem, method: str, models_cfg: ModelsConfig) -> Path:
    folder_name = _method_folder_name(models_cfg, method)
    relative_path = _artifact_relative_path(
        item.label_folder, item.source_label_folder, f"{Path(item.img_name).stem}.png"
    )
    if item.secondary_class is not None:
        return (
            item.dataset_run_dir / "secondary_masks" / folder_name
            / item.secondary_class / relative_path
        )
    return item.dataset_run_dir / "masks" / folder_name / relative_path


def _coordinates_path(item: GenerationItem, folder_name: str) -> Path:
    """Public, per-method copy of an image's bounding-box coordinates, kept
    inside that method's own masks/<folder_name>/ directory for visibility."""
    relative_path = _artifact_relative_path(
        item.label_folder, item.source_label_folder, f"{Path(item.img_name).stem}.json"
    )
    if item.secondary_class is not None:
        return (
            item.dataset_run_dir / "secondary_masks" / folder_name
            / item.secondary_class / COORDINATES_SUBDIR_NAME / relative_path
        )
    return item.dataset_run_dir / "masks" / folder_name / COORDINATES_SUBDIR_NAME / relative_path


def _box_needing_methods(methods: set[str]) -> list[str]:
    return sorted(method for method in methods if METHOD_REGISTRY[method].needs_boxes)


def _method_signature(
    cfg: MaskGenerationConfig, models_cfg: ModelsConfig, dataset: MaskGenerationDatasetEntry, method: str,
    *, secondary_profile_fingerprint: str | None = None,
) -> dict[str, Any]:
    """A fingerprint of everything that affects a method's output pixels.
    Compared against the previously saved fingerprint (see
    `_prepare_method_metadata`) to detect setting changes across runs."""
    signature: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "method": method,
        "folder_name": _method_folder_name(models_cfg, method),
        "image_geometry": "letterboxed_preprocessed",
        "include_rerun": dataset.include_rerun,
        "domain": "secondary" if secondary_profile_fingerprint else "binary",
    }
    if secondary_profile_fingerprint:
        signature["secondary_profile_fingerprint"] = secondary_profile_fingerprint
    spec = METHOD_REGISTRY[method]
    if spec.needs_boxes:
        signature.update({
            "localizer_prompt": str(cfg.prompts_dir / "bounding_box_localizer.j2"),
        })
    if spec.needs_segmentation:
        seg = _require_segmentation_model(models_cfg)
        signature.update({
            "segmentation_model_name": seg.modelname,
            "segmentation_source": seg.source,
            "segmentation_device": seg.device,
            "segmentation_confidence_threshold": seg.confidence_threshold,
        })
    return signature


def _prepare_method_metadata(cfg: MaskGenerationConfig, models_cfg: ModelsConfig) -> None:
    """Prevent silent reuse when a method's prediction source/settings changed."""
    for dataset in cfg.datasets:
        dataset_run_dir = cfg.run_dir / dataset.name
        method_domains = [(method, False) for method in dataset.methods]
        method_domains.extend((method, True) for method in dataset.secondary_methods)
        secondary_profile = (
            load_profile_snapshot(dataset_run_dir)
            if dataset.generate_secondary_masks else None
        )
        if dataset.generate_secondary_masks and secondary_profile is None:
            raise ValueError(
                f"Secondary masks are enabled for {dataset.name}, but the run has no saved secondary profile."
            )
        for method, is_secondary in method_domains:
            if method not in METHOD_REGISTRY:
                raise ValueError(
                    f"No generator registered for method {method!r}. "
                    f"Available methods: {sorted(METHOD_REGISTRY)}"
                )
            signature = _method_signature(
                cfg, models_cfg, dataset, method,
                secondary_profile_fingerprint=(
                    profile_fingerprint(secondary_profile)
                    if is_secondary and secondary_profile else None
                ),
            )
            folder_name = _method_folder_name(models_cfg, method)
            metadata_root = (
                dataset_run_dir / "secondary_mask_generation" / "methods"
                if is_secondary else dataset_run_dir / METHOD_METADATA_SUBDIR
            )
            path = metadata_root / f"{folder_name}.json"
            if path.is_file() and not cfg.overwrite:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid mask-generation metadata: {path} ({error}). This looks like it was left "
                        "truncated by a crash. Use --overwrite to regenerate instead of guessing its intent."
                    ) from error
                if existing != signature:
                    raise ValueError(
                        f"Mask settings changed for {dataset.name}/{folder_name}. "
                        "Use --overwrite to regenerate instead of mixing incompatible masks."
                    )
            _atomic_write_text(path, json.dumps(signature, indent=2))
            logger.debug("[%s] %s fingerprint written to %s", dataset.name, folder_name, path)


def _save_score_map(path: Path, score_map: np.ndarray) -> None:
    array = np.asarray(score_map, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"A mask score map must be 2-D, got {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temp file + atomic rename (same reasoning as
    # `_atomic_write_text`) so a crash mid-save can never leave a
    # truncated/corrupt PNG at the final path that a resumed run would then
    # mistake for a completed mask (see `_is_valid_score_map_png`).
    tmp_path = path.with_name(f".{path.name}.tmp{os.getpid()}")
    Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(
        tmp_path,
        format="PNG",
    )
    os.replace(tmp_path, path)


def _is_valid_score_map_png(path: Path) -> bool:
    """
    Resumability check for a previously generated mask PNG -- used INSTEAD
    OF a bare `.is_file()` check so a truncated/corrupt file left behind by
    a crash (segfault mid-`Image.save()`, disk-full, etc.) is correctly
    treated as NOT done and regenerated, rather than silently accepted.
    Verifies the file is non-empty and that PIL can actually decode it.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


from .backends import (
    _boxes_to_mask,
    _normalize_box,
    _parse_boxes,
    _segmentation_output_to_map,
)


# ---------------------------------------------------------------------
# Segmentation backends (currently just SAM3 -- see module docstring
# "Adding a new segmentation backend" to add more).
# ---------------------------------------------------------------------

@register_segmentation_backend("sam3")
def _load_sam3_backend(seg_cfg: SegmentationModelConfig) -> Any:
    """Adapter loader for this repo's local `sam3` package. Returns a
    Sam3Processor exposing set_image / set_text_prompt / add_geometric_prompt."""
    if seg_cfg.repo_dir is None:
        raise ValueError(
            "models.segmentation_model.repo_dir is required for the 'sam3' backend "
            "(it points at the local sam3/ package checkout)."
        )
    repo_dir = str(seg_cfg.repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = str(seg_cfg.local_path) if seg_cfg.source == "local" else None
    logger.info(
        "Loading segmentation backend 'sam3' (source=%s, device=%s)", seg_cfg.source, seg_cfg.device,
    )
    model = build_sam3_image_model(
        device=seg_cfg.device,
        checkpoint_path=checkpoint_path,
        load_from_HF=seg_cfg.source == "huggingface",
    )
    return Sam3Processor(model, device=seg_cfg.device, confidence_threshold=seg_cfg.confidence_threshold)


def load_segmentation_processor(models_cfg: ModelsConfig) -> Any:
    """Load the configured segmentation backend for batch or single-image use."""
    seg_cfg = _require_segmentation_model(models_cfg)
    backend = SEGMENTATION_BACKENDS.get(seg_cfg.modelname.lower())
    if backend is None:
        raise ValueError(
            f"No segmentation backend registered for model_name={seg_cfg.modelname!r}. "
            f"Available backends: {sorted(SEGMENTATION_BACKENDS)}. "
            "Register one with @register_segmentation_backend(<model_name>)."
        )
    return backend(seg_cfg)


class Runtime:
    """Lazily owns heavyweight models and explicitly releases VRAM between phases."""

    def __init__(self, cfg: MaskGenerationConfig, models_cfg: ModelsConfig):
        self.cfg = cfg
        self.models_cfg = models_cfg
        self.vlm_model = None
        self.segmentation_processor = None

    def load_vlm(self):
        if self.vlm_model is None:
            from ..models import vlm
            logger.info("Loading VLM for bounding-box localization: %s", self.models_cfg.vlm.modelname)
            self.vlm_model = vlm.load(self.models_cfg)
        return self.vlm_model

    def unload_vlm(self) -> None:
        if self.vlm_model is not None:
            # Handles both local CUDA cleanup and managed vLLM server teardown.
            self.vlm_model.release()
        self.vlm_model = None
        self.clear_vram()

    def load_segmentation_model(self):
        if self.segmentation_processor is None:
            self.segmentation_processor = load_segmentation_processor(self.models_cfg)
        return self.segmentation_processor

    def unload_segmentation_model(self) -> None:
        self.segmentation_processor = None
        self.clear_vram()

    @staticmethod
    def clear_vram() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def localize(self, item: GenerationItem) -> tuple[list[list[float]], list[str]]:
        """Runs (or reuses a cached) VLM bounding-box localization for one
        image, then writes a coordinates JSON copy into EVERY currently
        pending box-needing method's own masks/<folder>/coordinates_files/
        directory (e.g. both "bounding_box" and "bounding_box_segmentation"
        when both are requested for this image)."""
        folder_names = [
            _method_folder_name(self.models_cfg, method)
            for method in _box_needing_methods(item.pending_methods)
        ]
        candidate_paths = [_coordinates_path(item, folder_name) for folder_name in folder_names]

        boxes: list[list[float]] | None = None
        labels: list[str] | None = None
        if not self.cfg.overwrite:
            for path in candidate_paths:
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as error:
                    # A crash mid-write can leave a truncated/corrupt cache
                    # file -- treat it as a cache MISS (re-localize) rather
                    # than raising, since this is just a performance cache,
                    # not a correctness-critical artifact.
                    logger.warning("Ignoring corrupt cached bounding-box coordinates at %s: %s", path, error)
                    continue
                if (
                    not isinstance(cached, dict)
                    or not isinstance(cached.get("boxes"), list)
                    or not isinstance(cached.get("labels"), list)
                ):
                    logger.warning("Ignoring malformed cached bounding-box coordinates at %s", path)
                    continue
                if cached.get("reasoning") == item.reasoning and cached.get("image") == str(item.image_path):
                    boxes, labels = cached.get("boxes", []), cached.get("labels", [])
                    logger.debug("Reusing cached bounding boxes from %s", path)
                    break

        if boxes is None:
            vlm_model = self.load_vlm()
            base_prompt = _build_prompt(
                self.cfg.prompts_dir, "bounding_box_localizer.j2", reasoning=item.reasoning
            )
            with Image.open(item.image_path) as source:
                image = source.convert("RGB")
                last_error: ValueError | None = None
                previous_response = ""
                for attempt in range(1, LOCALIZATION_MAX_ATTEMPTS + 1):
                    if attempt == 1:
                        prompt = base_prompt
                    else:
                        prompt = (
                            f"{base_prompt}\n\n"
                            f"FORMAT CORRECTION ATTEMPT {attempt - 1}: Your previous answer "
                            "could not be parsed. "
                            "Reply with ONLY a JSON array of objects with exactly the keys "
                            '"label" and "box". Do not quote the entire array and do not add '
                            "prose or Markdown. Localize visible reported anomalies; omit a "
                            "reported absence because an absent object has no box. Return [] "
                            "only if none of the reported anomalies is visibly localizable.\n"
                            f"Previous invalid answer: {json.dumps(previous_response)}"
                        )
                    response, _, _ = vlm_model.run_vlm(
                        self.models_cfg,
                        [image],
                        prompt,
                        require_step_context=False,
                        require_confidence=False,
                    )
                    try:
                        boxes, labels = _parse_boxes(response)
                        break
                    except ValueError as error:
                        last_error = error
                        previous_response = str(response)
                        response_dir = artifact_group_dir(
                            item.dataset_run_dir / "mask_generation" / "localization_responses",
                            item.label_folder,
                            item.source_label_folder,
                        )
                        response_path = response_dir / (
                            f"{item.img_name}.attempt_{attempt:02d}.json"
                        )
                        _atomic_write_text(
                            response_path,
                            json.dumps(
                                {
                                    "attempt": attempt,
                                    "image": str(item.image_path),
                                    "error": str(error),
                                    "response": str(response),
                                },
                                indent=2,
                                ensure_ascii=False,
                            ),
                        )
                        logger.warning(
                            "Invalid bounding-box response for %s/%s on attempt %d/%d; "
                            "saved to %s: %s",
                            item.dataset.name,
                            _item_display_path(item),
                            attempt,
                            LOCALIZATION_MAX_ATTEMPTS,
                            response_path,
                            error,
                        )
                else:
                    assert last_error is not None
                    raise ValueError(
                        "Bounding-box localization returned invalid output on all "
                        f"{LOCALIZATION_MAX_ATTEMPTS} attempts; inspect "
                        f"{item.dataset_run_dir / 'mask_generation' / 'localization_responses'}"
                    ) from last_error

        payload = json.dumps(
            {"image": str(item.image_path), "reasoning": item.reasoning, "boxes": boxes, "labels": labels},
            indent=2, ensure_ascii=False,
        )
        for path in candidate_paths:
            _atomic_write_text(path, payload)

        return boxes, labels


@register_method("bounding_box", needs_boxes=True)
def generate_bounding_box(item: GenerationItem, runtime: Runtime) -> np.ndarray:
    del runtime
    with Image.open(item.image_path) as image:
        width, height = image.size
    return _boxes_to_mask(item.boxes or [], width, height)


@register_method("segmentation", needs_segmentation=True)
def generate_segmentation(item: GenerationItem, runtime: Runtime) -> np.ndarray:
    """Full-image segmentation from a free-text prompt (the anomaly reasoning) -- no boxes needed."""
    processor = runtime.load_segmentation_model()
    with Image.open(item.image_path) as source:
        image = source.convert("RGB")
        state = processor.set_image(image)
        output = processor.set_text_prompt(prompt=item.reasoning, state=state)
        return _segmentation_output_to_map(output, image.height, image.width)


@register_method("bounding_box_segmentation", needs_boxes=True, needs_segmentation=True)
def generate_bounding_box_segmentation(item: GenerationItem, runtime: Runtime) -> np.ndarray:
    """Segmentation seeded with the localized bounding boxes as geometric prompts."""
    processor = runtime.load_segmentation_model()
    with Image.open(item.image_path) as source:
        image = source.convert("RGB")
        if not item.boxes:
            return np.zeros((image.height, image.width), dtype=np.float32)
        state = processor.set_image(image)
        output = state
        for x0, y0, x1, y1 in item.boxes:
            box = [
                (x0 + x1) / 2000.0,
                (y0 + y1) / 2000.0,
                (x1 - x0) / 1000.0,
                (y1 - y0) / 1000.0,
            ]
            output = processor.add_geometric_prompt(box=box, label=True, state=state)
        return _segmentation_output_to_map(output, image.height, image.width)


def _build_items(cfg: MaskGenerationConfig, models_cfg: ModelsConfig) -> list[GenerationItem]:
    items: list[GenerationItem] = []
    # A legacy merged run may contain distinct source records that point at
    # the same already-overwritten preprocessed PNG. Refuse that input even
    # with --overwrite: producing distinct filenames from the same wrong image
    # would look complete while remaining semantically corrupt.
    preprocessed_owners: dict[Path, tuple[str, str, str]] = {}
    for dataset in cfg.datasets:
        seen_identities: set[tuple[str, str, str]] = set()
        unknown = (set(dataset.methods) | set(dataset.secondary_methods)) - METHOD_REGISTRY.keys()
        if unknown:
            raise ValueError(f"No generator registered for methods: {sorted(unknown)}")
        dataset_run_dir = cfg.run_dir / dataset.name
        secondary_profile = (
            load_profile_snapshot(dataset_run_dir)
            if dataset.generate_secondary_masks else None
        )
        if dataset.generate_secondary_masks and secondary_profile is None:
            raise ValueError(
                f"Secondary masks are enabled for {dataset.name}, but the run has no saved secondary profile."
            )
        for entry, is_anomaly, reasoning in _load_effective_entries(
            dataset_run_dir, dataset.include_rerun
        ):
            label_folder, source_label_folder, img_name = _image_key(entry)
            raw_preprocessed_path = entry.get("preprocess_cand_img_path")
            image_path = (
                _resolve_artifact(raw_preprocessed_path, dataset_run_dir)
                if raw_preprocessed_path else Path()
            )
            if not image_path.is_file():
                fallback_dir = artifact_group_dir(
                    dataset_run_dir / "preproc",
                    label_folder,
                    source_label_folder,
                    prefix="cand_",
                )
                image_path = (fallback_dir / f"processed_{img_name}").with_suffix(".png")
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Preprocessed candidate image not found for "
                    f"{label_folder}/{source_label_folder}/{img_name}: {image_path}"
                )
            identity = (label_folder, source_label_folder, img_name)
            if identity in seen_identities:
                raise ValueError(
                    f"Duplicate image identity in {dataset.name}/all_data_summary.jsonl: "
                    f"{identity!r}. Each source image must occur exactly once."
                )
            seen_identities.add(identity)
            resolved_image_path = image_path.resolve()
            previous_owner = preprocessed_owners.get(resolved_image_path)
            if previous_owner is not None and previous_owner != identity:
                raise ValueError(
                    "Legacy merged-run collision detected: distinct records "
                    f"{previous_owner!r} and {identity!r} both reference {resolved_image_path}. "
                    "That image was already overwritten during inference and cannot be repaired "
                    "by mask --overwrite. Run anomaly detection again after the source-aware "
                    "preprocessing-path fix, then generate masks for the new run directory."
                )
            preprocessed_owners[resolved_image_path] = identity
            item = GenerationItem(
                dataset=dataset,
                dataset_run_dir=dataset_run_dir,
                entry=entry,
                img_name=img_name,
                label_folder=label_folder,
                source_label_folder=source_label_folder,
                image_path=image_path,
                is_anomaly=is_anomaly,
                reasoning=reasoning,
                pending_methods=set(),
            )
            # Resume decisions must use the exact same source-aware path as
            # writes; otherwise one merged class can make another look done.
            item.pending_methods = {
                method for method in dataset.methods
                if cfg.overwrite or not _is_valid_score_map_png(
                    _output_path(item, method, models_cfg)
                )
            }
            items.append(item)

            if secondary_profile is not None:
                secondary_class = entry.get("secondary_class")
                yes_steps = entry.get("yes_steps")
                if secondary_class not in secondary_profile.class_names:
                    raise ValueError(
                        f"{identity!r} has invalid secondary class {secondary_class!r}."
                    )
                if not isinstance(yes_steps, list):
                    raise ValueError(f"{identity!r} has invalid secondary yes_steps.")
                relevant_steps = localization_evidence_steps(
                    secondary_profile, secondary_class, yes_steps
                )
                class_rule = next(
                    rule for rule in secondary_profile.classes
                    if rule.name == secondary_class
                )
                if class_rule.localization_steps and not relevant_steps:
                    raise ValueError(
                        f"{identity!r} predicts localizable class {secondary_class!r} "
                        "without a relevant YES step."
                    )
                # A class with no localization steps (for example empty_clean)
                # deliberately produces no secondary mask file.
                if relevant_steps:
                    secondary_methods = set(dataset.secondary_methods)
                    if secondary_class == sole_binary_anomaly_class(secondary_profile):
                        # This class is exactly the binary anomaly decision.  Reuse
                        # methods already generated in masks/ instead of asking the
                        # localizer and segmenter a second, differently worded prompt.
                        secondary_methods.difference_update(dataset.methods)
                    secondary_item = GenerationItem(
                        dataset=dataset,
                        dataset_run_dir=dataset_run_dir,
                        entry=entry,
                        img_name=img_name,
                        label_folder=label_folder,
                        source_label_folder=source_label_folder,
                        image_path=image_path,
                        is_anomaly=True,
                        reasoning=_secondary_reasoning(
                            entry, dataset_run_dir, relevant_steps
                        ),
                        pending_methods=set(),
                        secondary_class=secondary_class,
                    )
                    secondary_item.pending_methods = {
                        method for method in secondary_methods
                        if cfg.overwrite or not _is_valid_score_map_png(
                            _output_path(secondary_item, method, models_cfg)
                        )
                    }
                    if secondary_item.pending_methods:
                        items.append(secondary_item)
    return items


def run_mask_generation(
    cfg: MaskGenerationConfig,
    console: Console | None = None,
    model_config_path: str | Path | None = None,
) -> None:
    """Generate requested masks, sharing localization work and heavyweight models.

    Progress is shown process-wise AND dataset-wise: one row per
    (mask method, dataset) combination, each method drawn in its own color
    (see `_method_color`) so multiple methods/datasets running in the same
    invocation stay easy to tell apart at a glance.
    """
    console = console or Console()
    # configs/model_configs.yaml is always used -- a cheap YAML parse, done up
    # front (not lazily inside Runtime) because folder-naming/path decisions
    # below need to know the segmentation model's name before any heavy model
    # is actually loaded.
    models_cfg = load_models_config(model_config_path or "configs/model_configs.yaml")
    logger.info(
        "Mask generation model config: vlm=%s, segmentation_model=%s",
        models_cfg.vlm.modelname,
        models_cfg.segmentation_model.modelname if models_cfg.segmentation_model else None,
    )

    # Build/validate every input before writing method metadata. In
    # particular, a legacy merged run must fail without modifying its v3
    # metadata merely because the user supplied --overwrite.
    items = _build_items(cfg, models_cfg)
    _prepare_method_metadata(cfg, models_cfg)
    runtime = Runtime(cfg, models_cfg)
    failures: list[str] = []

    # Snapshot per-(dataset, method) totals BEFORE any phase mutates
    # `item.pending_methods`, so progress bars know their full length upfront.
    box_totals: dict[str, int] = {}
    segmentation_totals: dict[tuple[str, str], int] = {}
    for item in items:
        if not item.is_anomaly:
            continue
        if any(METHOD_REGISTRY[m].needs_boxes for m in item.pending_methods):
            box_totals[item.dataset.name] = box_totals.get(item.dataset.name, 0) + 1
        for method in item.pending_methods:
            if METHOD_REGISTRY[method].needs_segmentation:
                key = (item.dataset.name, method)
                segmentation_totals[key] = segmentation_totals.get(key, 0) + 1

    # Always materialize zero masks for predicted-good images. This makes the
    # output complete and lets future consumers work without prediction logic.
    for item in items:
        if item.is_anomaly:
            continue
        with Image.open(item.image_path) as image:
            zero = np.zeros((image.height, image.width), dtype=np.float32)
        for method in item.pending_methods:
            _save_score_map(_output_path(item, method, models_cfg), zero)
        item.pending_methods.clear()

    # Rich's live redraws are not a reliable subprocess protocol when this
    # stage is launched by the ablation orchestrator. Emit newline-based,
    # method-specific counters so the parent can maintain one row per method
    # while the standalone per-dataset bars remain available.
    method_progress_totals = {
        "bounding_box": sum(box_totals.values()),
        "segmentation": sum(
            total for (dataset_name, method), total in segmentation_totals.items()
            if method == "segmentation"
        ),
        "bounding_box_segmentation": sum(
            total for (dataset_name, method), total in segmentation_totals.items()
            if method == "bounding_box_segmentation"
        ),
    }
    method_progress_completed = {method: 0 for method in method_progress_totals}

    def emit_method_progress(method: str, detail: str = "") -> None:
        total = method_progress_totals[method]
        if total <= 0:
            return
        method_progress_completed[method] = min(
            total,
            method_progress_completed[method] + 1,
        )
        logger.info(
            "Mask method progress: %s %d/%d · %s",
            method,
            method_progress_completed[method],
            total,
            detail or "working",
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        _PerTaskColorBarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )
    progress.start()
    localize_time_totals: dict[str, float] = {}
    segmentation_time_totals: dict[tuple[str, str], float] = {}
    try:
        box_tasks: dict[str, TaskID] = {}
        for dataset_name, total in box_totals.items():
            color = _method_color("bounding_box")
            box_tasks[dataset_name] = progress.add_task(
                f"[{color}]bounding_box[/{color}] \u00b7 {dataset_name}", total=total, color=color,
            )

        seg_tasks: dict[tuple[str, str], TaskID] = {}
        for (dataset_name, method), total in segmentation_totals.items():
            color = _method_color(method)
            folder_name = _method_folder_name(models_cfg, method)
            seg_tasks[(dataset_name, method)] = progress.add_task(
                f"[{color}]{folder_name}[/{color}] \u00b7 {dataset_name}", total=total, color=color,
            )

        for method, total in method_progress_totals.items():
            if total:
                logger.info(
                    "Mask method progress: %s 0/%d · preparing",
                    method,
                    total,
                )
            else:
                logger.info(
                    "Mask method progress: %s 1/1 · no pending work",
                    method,
                )

        # Localize once per image and reuse the boxes for both relevant methods.
        box_items = [
            item for item in items
            if item.is_anomaly and any(METHOD_REGISTRY[m].needs_boxes for m in item.pending_methods)
        ]
        if box_items:
            logger.info("Localizing anomalies for %d images", len(box_items))
            for index, item in enumerate(box_items, start=1):
                task_id = box_tasks[item.dataset.name]
                progress.update(task_id, description=(
                    f"[{_method_color('bounding_box')}]bounding_box[/{_method_color('bounding_box')}] "
                    f"\u00b7 {item.dataset.name}: {_item_display_path(item)}"
                ))
                logger.info(
                    "[boxes %d/%d] %s/%s",
                    index,
                    len(box_items),
                    item.dataset.name,
                    _item_display_path(item),
                )
                try:
                    localize_start = time.perf_counter()
                    item.boxes, item.box_labels = runtime.localize(item)
                    localize_time_totals[item.dataset.name] = (
                        localize_time_totals.get(item.dataset.name, 0.0) + (time.perf_counter() - localize_start)
                    )
                except Exception as error:
                    failed_methods = {
                        method for method in item.pending_methods if METHOD_REGISTRY[method].needs_boxes
                    }
                    failures.append(
                        f"{item.dataset.name}/{_item_display_path(item)} "
                        f"({', '.join(sorted(failed_methods))}): {error}"
                    )
                    logger.exception(
                        "Bounding-box localization failed for %s/%s",
                        item.dataset.name,
                        _item_display_path(item),
                    )
                    item.pending_methods.difference_update(failed_methods)
                    runtime.clear_vram()
                    progress.update(task_id, advance=1)
                    emit_method_progress(
                        "bounding_box",
                        detail=(
                            "failed · "
                            f"{item.dataset.name}/{_item_display_path(item)}"
                        ),
                    )
                    for failed_method in failed_methods:
                        if METHOD_REGISTRY[failed_method].needs_segmentation:
                            emit_method_progress(
                                failed_method,
                                detail=(
                                    "skipped after localization failure · "
                                    f"{item.dataset.name}/{_item_display_path(item)}"
                                ),
                            )
                    continue
                if "bounding_box" in item.pending_methods:
                    _save_score_map(
                        _output_path(item, "bounding_box", models_cfg),
                        generate_bounding_box(item, runtime),
                    )
                    item.pending_methods.remove("bounding_box")
                progress.update(task_id, advance=1)
                emit_method_progress(
                    "bounding_box",
                    detail=f"{item.dataset.name}/{_item_display_path(item)}",
                )
            runtime.unload_vlm()

        segmentation_items = [
            item for item in items
            if item.is_anomaly and any(METHOD_REGISTRY[m].needs_segmentation for m in item.pending_methods)
        ]
        if segmentation_items:
            runtime.load_segmentation_model()
            total = sum(len(item.pending_methods) for item in segmentation_items)
            completed = 0
            for item in segmentation_items:
                for method in tuple(item.pending_methods):
                    spec = METHOD_REGISTRY[method]
                    if not spec.needs_segmentation:
                        continue
                    completed += 1
                    folder_name = _method_folder_name(models_cfg, method)
                    task_id = seg_tasks[(item.dataset.name, method)]
                    color = _method_color(method)
                    progress.update(task_id, description=(
                        f"[{color}]{folder_name}[/{color}] \u00b7 "
                        f"{item.dataset.name}: {_item_display_path(item)}"
                    ))
                    logger.info(
                        "[segmentation %d/%d] %s/%s (%s)",
                        completed,
                        total,
                        item.dataset.name,
                        _item_display_path(item),
                        folder_name,
                    )
                    try:
                        gen_start = time.perf_counter()
                        score_map = spec.generate(item, runtime)
                        key = (item.dataset.name, method)
                        segmentation_time_totals[key] = segmentation_time_totals.get(key, 0.0) + (time.perf_counter() - gen_start)
                        _save_score_map(_output_path(item, method, models_cfg), score_map)
                    except Exception as error:
                        failures.append(
                            f"{item.dataset.name}/{_item_display_path(item)} "
                            f"({folder_name}): {error}"
                        )
                        logger.exception(
                            "%s failed for %s/%s",
                            folder_name,
                            item.dataset.name,
                            _item_display_path(item),
                        )
                        runtime.clear_vram()
                    item.pending_methods.remove(method)
                    progress.update(task_id, advance=1)
                    emit_method_progress(
                        method,
                        detail=(
                            f"{folder_name} · "
                            f"{item.dataset.name}/{_item_display_path(item)}"
                        ),
                    )
            runtime.unload_segmentation_model()
    finally:
        progress.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for dataset in cfg.datasets:
        dataset_name = dataset.name
        methods_perf: dict[str, dict[str, Any]] = {}
        localize_total = localize_time_totals.get(dataset_name)
        num_localized = box_totals.get(dataset_name, 0)
        if localize_total is not None and num_localized:
            methods_perf["localization"] = {
                "num_images": num_localized,
                "total_time_sec": round(localize_total, 4),
                "avg_time_per_image_sec": round(localize_total / num_localized, 4),
            }
        for (ds_name, method), total_time in segmentation_time_totals.items():
            if ds_name != dataset_name:
                continue
            count = segmentation_totals.get((ds_name, method), 0)
            if not count:
                continue
            methods_perf[method] = {
                "num_images": count,
                "total_time_sec": round(total_time, 4),
                "avg_time_per_image_sec": round(total_time / count, 4),
            }
        dataset_record_count = sum(1 for item in items if item.dataset.name == dataset_name)
        performance = {
            "dataset": dataset_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "records_verified": dataset_record_count,
            "new_work_performed": bool(methods_perf),
            "model": {
                "vlm": asdict(models_cfg.vlm),
                "segmentation_model": asdict(models_cfg.segmentation_model) if models_cfg.segmentation_model else None,
            },
            "methods": methods_perf,
        }
        performance_dir = cfg.run_dir / dataset_name / "mask_generation"
        performance_dir.mkdir(parents=True, exist_ok=True)
        performance_path = performance_dir / f"performance_{timestamp}.json"
        performance_path.write_text(json.dumps(performance, indent=2, default=str), encoding="utf-8")
        logger.info("[%s] Mask generation performance summary written to %s", dataset_name, performance_path)
        if not methods_perf:
            logger.info(
                "[%s] No new mask work was needed; all %d source-aware record paths were already valid.",
                dataset_name,
                dataset_record_count,
            )

    unresolved = [
        (
            item.dataset.name,
            item.source_label_folder,
            item.img_name,
            sorted(item.pending_methods),
        )
        for item in items if item.pending_methods
    ]
    if unresolved:
        raise RuntimeError(f"Some mask methods were not executed: {unresolved[:5]}")

    for dataset in cfg.datasets:
        count = sum(1 for item in items if item.dataset.name == dataset.name)
        folder_names = [_method_folder_name(models_cfg, method) for method in dataset.methods]
        logger.info("[%s] generated/verified %d images for methods %s", dataset.name, count, folder_names)
        logger.info("[%s] masks ready: %s", dataset.name, cfg.run_dir / dataset.name / "masks")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures[:20])
        raise RuntimeError(
            f"Mask generation finished with {len(failures)} failed image-method task(s). "
            f"Successful outputs are resumable; fix the errors and run again.\n{details}"
        )
