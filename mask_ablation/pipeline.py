"""Resumable orchestration for the XAI mask-ablation study.

Final score maps use the existing mask contract::

    <run_dir>/<dataset>/masks/<xai_model_method>/
        <label_folder>/[<source_label_folder>/]<image_stem>.png

This makes the current metrics stage discover Pixel AUROC/AUPRO columns for
every XAI arm automatically.  Rich diagnostics remain separate under
``<run_dir>/<dataset>/<diagnostics_subdir>/<xai_model_method>/``.

Resumability is per image, not per method.  Each final PNG is validated with
Pillow and published by temp-file + atomic rename as soon as the XAI core
finishes that image.  A method fingerprint prevents masks generated with
different evidence/model/optimizer settings from being mixed accidentally.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from anomaly_detection.hardware_monitor import record_pipeline_event
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import DatasetEntry, MaskAblationConfig, MethodVariant, ModelVariant
from .engine import run_single_combo
from .evidence import ImageEvidence, build_image_evidence, evidence_digest


logger = logging.getLogger(__name__)

GENERATOR_VERSION = 5
METHOD_METADATA_SUBDIR = Path("mask_generation") / "methods"
SMOOTHED_MASK_SUFFIX = "_smoothed"
SMOOTHED_MASK_SIGMA = 1.0
# Code-level switch for the optional raw-vs-smoothed comparison. Keep this
# disabled unless a future ablation explicitly needs smoothed masks again.
GENERATE_SMOOTHED_MASKS = False


def _matches_pilot_selector(item: ImageEvidence, selector: str) -> bool:
    """Disambiguate repeated defect filenames without using filesystem paths."""
    parts = selector.split("/")
    if len(parts) == 2 and parts[0] != item.source_label_folder:
        return False
    if len(parts) == 3 and tuple(parts[:2]) != (
        item.label_folder, item.source_label_folder
    ):
        return False
    return parts[-1] in {
        item.image_path.name, item.image_path.stem,
        item.img_name, Path(item.img_name).stem,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _save_score_map(path: Path, score_map: np.ndarray) -> None:
    array = np.asarray(score_map, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"XAI score map must be 2-D, got {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp{os.getpid()}")
    Image.fromarray(
        np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    ).save(temporary, format="PNG")
    os.replace(temporary, path)


def _prepare_published_score_map(
    score_map: np.ndarray, method: MethodVariant
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert the raw XAI score into the requested metrics-mask product.

    ``heatmap_hot_regions`` exactly follows the display heatmap's 5th/99th
    percentile normalization, then keeps only its yellow/red intensity range.
    The raw float map and color heatmap remain untouched in diagnostics.
    """
    array = np.nan_to_num(
        np.asarray(score_map, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0
    )
    array = np.clip(array, 0.0, 1.0)
    mode = str(method.params.get("published_mask_mode", "continuous"))
    if mode == "continuous":
        return array, {"published_mask_mode": mode}
    if mode != "heatmap_hot_regions":
        raise ValueError(
            f"Method {method.name!r}: unsupported published_mask_mode={mode!r}."
        )

    threshold = float(method.params.get("published_mask_threshold", 0.60))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Method {method.name!r}: published_mask_threshold must be in [0, 1]."
        )
    lo = float(np.percentile(array, 5))
    hi = float(np.percentile(array, 99))
    if hi - lo < 0.05:
        hi = lo + 0.05
    display_score = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    binary = (display_score >= threshold).astype(np.float32)
    return binary, {
        "published_mask_mode": mode,
        "published_mask_threshold": threshold,
        "display_percentile_low": lo,
        "display_percentile_high": hi,
        "published_white_fraction": float(binary.mean()),
    }


def _is_valid_score_map(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _method_output_dir(
    cfg: MaskAblationConfig,
    dataset: DatasetEntry,
    model: ModelVariant,
    method: MethodVariant,
) -> Path:
    return cfg.run_dir / dataset.name / "masks" / method.folder_name(model)


def _diagnostics_dir(
    cfg: MaskAblationConfig,
    dataset: DatasetEntry,
    model: ModelVariant,
    method: MethodVariant,
) -> Path:
    return (
        cfg.run_dir
        / dataset.name
        / cfg.diagnostics_subdir
        / method.folder_name(model)
    )


def _metadata_path(
    cfg: MaskAblationConfig,
    dataset: DatasetEntry,
    model: ModelVariant,
    method: MethodVariant,
) -> Path:
    return (
        cfg.run_dir
        / dataset.name
        / METHOD_METADATA_SUBDIR
        / f"{method.folder_name(model)}.json"
    )


def _signature(
    *,
    dataset: DatasetEntry,
    model: ModelVariant,
    method: MethodVariant,
    items: list[ImageEvidence],
) -> dict[str, Any]:
    return {
        "generator": "xai_mask_ablation",
        "generator_version": GENERATOR_VERSION,
        "method": method.name,
        "folder_name": method.folder_name(model),
        "model_name": model.name,
        "model_adapter": model.adapter,
        "model_source": model.source,
        "selected_model": model.selected_model,
        "method_args": method.params,
        "image_geometry": "letterboxed_preprocessed",
        "include_rerun": dataset.include_rerun,
        "evidence_digest": evidence_digest(items),
    }


def _prepare_metadata(
    *,
    cfg: MaskAblationConfig,
    dataset: DatasetEntry,
    model: ModelVariant,
    method: MethodVariant,
    signature: dict[str, Any],
) -> None:
    path = _metadata_path(cfg, dataset, model, method)
    mask_dir = _method_output_dir(cfg, dataset, model, method)
    if not cfg.overwrite and path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(
                f"Invalid XAI method fingerprint at {path}: {error}. "
                "Use --overwrite to regenerate this method."
            ) from error
        if previous != signature:
            has_published_masks = mask_dir.is_dir() and any(mask_dir.rglob("*.png"))
            if has_published_masks:
                raise ValueError(
                    f"XAI settings/evidence changed for {dataset.name}/"
                    f"{method.folder_name(model)}. Use --overwrite instead of mixing "
                    "incompatible masks in one method folder."
                )
            # A startup failure can leave only the fingerprint behind. With no
            # published masks there is nothing incompatible to mix, so replace
            # that stale metadata and let the same resumable command recover.
            logger.info("Replacing stale fingerprint after a mask-free failed run: %s", path)
    elif not cfg.overwrite and mask_dir.is_dir() and any(mask_dir.rglob("*.png")):
        raise ValueError(
            f"Existing masks at {mask_dir} have no XAI method fingerprint. "
            "Use a different output_name or pass --overwrite explicitly."
        )

    _atomic_write_text(path, json.dumps(signature, indent=2, ensure_ascii=False))


def _zero_map_for(item: ImageEvidence) -> np.ndarray:
    with Image.open(item.image_path) as image:
        width, height = image.size
    return np.zeros((height, width), dtype=np.float32)


def _backfill_smoothed_mask(
    *,
    diagnostics_dir: Path,
    smoothed_path: Path,
    item: ImageEvidence,
    method: MethodVariant,
) -> bool:
    """Create the smoothed comparison mask from an existing raw XAI map.

    This keeps the new raw-vs-smoothed comparison resumable: adding the
    comparison must not force an expensive LVLM attribution rerun when the
    original final raw ``.npy`` map and provenance already exist.
    """
    provenance_path = diagnostics_dir / "final_masks" / item.relative_mask_path.with_suffix(".json")
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        raw_path = Path(payload["core_output"]["raw_mask_path"])
        if not raw_path.is_file():
            return False
        score_map = np.load(raw_path).astype(np.float32)
        smoothed_score_map = gaussian_filter(score_map, sigma=SMOOTHED_MASK_SIGMA)
        smoothed_map, _ = _prepare_published_score_map(smoothed_score_map, method)
        _save_score_map(smoothed_path, smoothed_map)
        return True
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _write_image_provenance(
    diagnostics_dir: Path,
    item: ImageEvidence,
    final_mask_path: Path,
    *,
    core_output: dict[str, Any] | None,
) -> None:
    relative = item.relative_mask_path.with_suffix(".json")
    path = diagnostics_dir / "final_masks" / relative
    payload = {
        "key": item.key,
        "dataset": item.dataset_name,
        "image_path": str(item.image_path),
        "reference_image_path": (
            str(item.reference_image_path) if item.reference_image_path else None
        ),
        "final_prediction": item.final_label,
        "rerun_status": item.rerun_status,
        "evidence_status": item.evidence_status,
        "num_evidence_samples": len(item.samples),
        "evidence_sources": sorted({sample.evidence_source for sample in item.samples}),
        "published_mask_path": str(final_mask_path),
        "core_output": core_output,
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _validate_method_params(method: MethodVariant) -> str:
    if "include_rerun" in method.params:
        raise ValueError(
            f"Method {method.name!r}: include_rerun belongs at config/dataset level, "
            "not inside method args."
        )
    evidence_mode = str(method.params.get("mask_generation_mode", "yes_steps"))
    baseline_mode = method.params.get("baseline_mode", "blur")
    if baseline_mode not in ("blur", "reference"):
        raise ValueError(
            f"Method {method.name!r}: baseline_mode must be 'blur' or 'reference'."
        )
    reference_mode = method.params.get("reference_mode", "none")
    if reference_mode not in ("none", "context"):
        raise ValueError(
            f"Method {method.name!r}: reference_mode must be 'none' or 'context'."
        )
    published_mode = method.params.get("published_mask_mode", "continuous")
    if published_mode not in ("continuous", "heatmap_hot_regions"):
        raise ValueError(
            f"Method {method.name!r}: published_mask_mode must be "
            "'continuous' or 'heatmap_hot_regions'."
        )
    return evidence_mode


def _run_combo(
    *,
    cfg: MaskAblationConfig,
    dataset: DatasetEntry,
    model: ModelVariant,
    method: MethodVariant,
    items: list[ImageEvidence],
    signature_items: list[ImageEvidence] | None,
    progress: Progress,
    image_task_id: Any,
    optimisation_task_id: Any,
) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    folder_name = method.folder_name(model)
    record_pipeline_event(
        "xai_combo_start",
        dataset=dataset.name,
        model=model.name,
        method=method.name,
        method_folder=folder_name,
    )
    output_dir = _method_output_dir(cfg, dataset, model, method)
    smoothed_output_dir = output_dir.with_name(output_dir.name + SMOOTHED_MASK_SUFFIX)
    diagnostics_dir = _diagnostics_dir(cfg, dataset, model, method)
    # Fingerprint the complete evidence set even during a pilot. The pilot and
    # later full run perform identical calculations; only their image scope
    # differs, so the full run can safely resume the already-published masks.
    signature = _signature(
        dataset=dataset,
        model=model,
        method=method,
        items=signature_items if signature_items is not None else items,
    )
    _prepare_metadata(
        cfg=cfg,
        dataset=dataset,
        model=model,
        method=method,
        signature=signature,
    )

    resumed = 0
    zero_masks = 0
    generated = 0
    pending_attribution: list[ImageEvidence] = []
    item_by_path: dict[str, ImageEvidence] = {}
    image_started_at: dict[str, float] = {}
    image_timings: list[dict[str, Any]] = []

    for item in items:
        final_path = output_dir / item.relative_mask_path
        smoothed_path = smoothed_output_dir / item.relative_mask_path
        if not cfg.overwrite and _is_valid_score_map(final_path):
            smoothed_ready = True
            if GENERATE_SMOOTHED_MASKS:
                if not _is_valid_score_map(smoothed_path):
                    _backfill_smoothed_mask(
                        diagnostics_dir=diagnostics_dir,
                        smoothed_path=smoothed_path,
                        item=item,
                        method=method,
                    )
                smoothed_ready = _is_valid_score_map(smoothed_path)
            if smoothed_ready:
                resumed += 1
                image_timings.append({"key": item.key, "status": "resumed", "elapsed_sec": 0.0})
                progress.update(image_task_id, phase="resuming", image=item.image_path.name)
                progress.advance(image_task_id)
                continue
        if not item.is_anomaly or not item.samples:
            _save_score_map(final_path, _zero_map_for(item))
            if GENERATE_SMOOTHED_MASKS:
                _save_score_map(smoothed_path, _zero_map_for(item))
            _write_image_provenance(
                diagnostics_dir, item, final_path, core_output=None
            )
            zero_masks += 1
            image_timings.append({"key": item.key, "status": "zero_mask", "elapsed_sec": 0.0})
            progress.update(image_task_id, phase="zero mask", image=item.image_path.name)
            progress.advance(image_task_id)
            continue
        resolved = str(item.image_path.resolve())
        if resolved in item_by_path:
            raise ValueError(
                f"Two source identities resolve to the same candidate image: "
                f"{item_by_path[resolved].key!r} and {item.key!r} -> {resolved}. "
                "This is an unrecoverable legacy preprocessing collision."
            )
        item_by_path[resolved] = item
        pending_attribution.append(item)

    core_result: dict[str, Any] = {"num_samples": 0, "num_images": 0}
    if pending_attribution:
        samples = [
            sample.to_engine_dict()
            for item in pending_attribution
            for sample in item.samples
        ]

        def publish_final_mask(
            image_path: str,
            saved_output: dict[str, Any],
            core_metadata: dict[str, Any],
        ) -> None:
            nonlocal generated
            resolved = str(Path(image_path).resolve())
            item = item_by_path.get(resolved)
            if item is None:
                raise KeyError(f"XAI core returned an unexpected image path: {image_path}")
            raw_path = Path(saved_output["raw_mask_path"])
            score_map = np.load(raw_path).astype(np.float32)
            published_map, publication_metadata = _prepare_published_score_map(
                score_map, method
            )
            final_path = output_dir / item.relative_mask_path
            _save_score_map(final_path, published_map)
            smoothed_provenance: dict[str, Any] = {}
            if GENERATE_SMOOTHED_MASKS:
                smoothed_score_map = gaussian_filter(
                    score_map.astype(np.float32), sigma=SMOOTHED_MASK_SIGMA
                )
                smoothed_published_map, smoothed_metadata = _prepare_published_score_map(
                    smoothed_score_map, method
                )
                smoothed_path = smoothed_output_dir / item.relative_mask_path
                _save_score_map(smoothed_path, smoothed_published_map)
                smoothed_provenance = {
                    "smoothed_mask_path": str(smoothed_path),
                    "smoothed_mask_sigma": SMOOTHED_MASK_SIGMA,
                    "smoothed_publication": smoothed_metadata,
                }
            _write_image_provenance(
                diagnostics_dir,
                item,
                final_path,
                core_output={
                    **saved_output,
                    **core_metadata,
                    **publication_metadata,
                    **smoothed_provenance,
                },
            )
            generated += 1
            image_timings.append(
                {
                    "key": item.key,
                    "status": "generated",
                    "elapsed_sec": time.perf_counter()
                    - image_started_at.get(resolved, started),
                    "final_mask_path": str(final_path),
                }
            )
            progress.update(
                optimisation_task_id,
                phase="published",
                image=item.image_path.name,
                step="done",
                iteration="done",
                completed=1,
                total=1,
            )
            progress.update(image_task_id, phase="published", image=item.image_path.name)
            progress.advance(image_task_id)
            record_pipeline_event(
                "xai_image_published",
                dataset=dataset.name,
                model=model.name,
                method=method.name,
                image=item.image_path.name,
                image_key=item.key,
                final_mask_path=str(final_path),
            )

        last_telemetry_state: tuple[Any, ...] | None = None

        def update_core_progress(**fields: Any) -> None:
            """Receive status only; diagnostics still go to the invocation log."""
            nonlocal last_telemetry_state
            source_path = fields.pop("image_path", None)
            if source_path and fields.get("phase") == "preparing image":
                image_started_at.setdefault(str(Path(source_path).resolve()), time.perf_counter())
            telemetry_state = (
                fields.get("phase"),
                fields.get("image"),
                fields.get("step"),
                fields.get("mask_type"),
            )
            if telemetry_state != last_telemetry_state:
                record_pipeline_event(
                    "xai_progress",
                    dataset=dataset.name,
                    model=model.name,
                    method=method.name,
                    phase=fields.get("phase"),
                    image=fields.get("image"),
                    image_path=source_path,
                    step=fields.get("step"),
                    mask_type=fields.get("mask_type"),
                )
                last_telemetry_state = telemetry_state
            iteration_current = fields.pop("iteration_current", None)
            iteration_total = fields.pop("iteration_total", None)
            update: dict[str, Any] = dict(fields)
            if iteration_total is not None:
                update["total"] = int(iteration_total)
                update["completed"] = int(iteration_current or 0)
            elif fields.get("phase") in {"preparing image", "loading model"}:
                update["total"] = None
                update["completed"] = 0
            progress.update(optimisation_task_id, **update)
            progress.update(
                image_task_id,
                phase=fields.get("phase", "running"),
                image=fields.get("image", "--"),
            )

        core_result = run_single_combo(
            model_path=model.selected_model,
            adapter_name=model.adapter,
            output_dir=str(diagnostics_dir),
            param_overrides=method.params,
            samples=samples,
            final_mask_callback=publish_final_mask,
            progress_callback=update_core_progress,
        )

    missing = [
        str(output_dir / item.relative_mask_path)
        for item in items
        if not _is_valid_score_map(output_dir / item.relative_mask_path)
        or (
            GENERATE_SMOOTHED_MASKS
            and not _is_valid_score_map(
                smoothed_output_dir / item.relative_mask_path
            )
        )
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} final XAI mask(s) were not published for {dataset.name}/"
            f"{folder_name}; first missing path: {missing[0]}"
        )

    elapsed = time.perf_counter() - started
    record_pipeline_event(
        "xai_combo_end",
        dataset=dataset.name,
        model=model.name,
        method=method.name,
        generated=generated,
        resumed=resumed,
        zero_masks=zero_masks,
        elapsed_sec=elapsed,
    )
    per_image_performance_path = diagnostics_dir / "performance.json"
    _atomic_write_text(
        per_image_performance_path,
        json.dumps(
            {
                "dataset": dataset.name,
                "model": model.name,
                "method": method.name,
                "method_folder": folder_name,
                "elapsed_sec": elapsed,
                "avg_sec_per_image": elapsed / len(items) if items else None,
                "images": image_timings,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return {
        "dataset": dataset.name,
        "model": model.name,
        "method": method.name,
        "method_folder": folder_name,
        "status": "ok",
        "started_at": started_at,
        "num_images": len(items),
        "num_predicted_anomaly": sum(item.is_anomaly for item in items),
        "num_evidence_samples": sum(len(item.samples) for item in items),
        "num_generated": generated,
        "num_zero_masks": zero_masks,
        "num_resumed": resumed,
        "elapsed_sec": elapsed,
        "avg_sec_per_image": elapsed / len(items) if items else None,
        "output_dir": str(output_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "include_rerun": dataset.include_rerun,
        "core_result": core_result,
        "per_image_performance_path": str(per_image_performance_path),
        "error": "",
    }


def _write_summaries(
    cfg: MaskAblationConfig, records: list[dict[str, Any]], elapsed: float
) -> Path:
    summary_root = cfg.run_dir / cfg.diagnostics_subdir
    summary_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = summary_root / f"performance_{timestamp}.json"
    _atomic_write_text(
        json_path,
        json.dumps(
            {
                "run_dir": str(cfg.run_dir),
                "elapsed_sec": elapsed,
                "datasets": [asdict(dataset) for dataset in cfg.datasets],
                "models": [asdict(model) for model in cfg.models],
                "methods": [asdict(method) for method in cfg.methods],
                "overwrite": cfg.overwrite,
                "max_attribution_images_per_dataset": (
                    cfg.max_attribution_images_per_dataset
                ),
                "results": records,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
    )

    csv_path = summary_root / f"ablation_summary_{timestamp}.csv"
    fieldnames = [
        "dataset",
        "model",
        "method",
        "method_folder",
        "status",
        "num_images",
        "num_predicted_anomaly",
        "num_evidence_samples",
        "num_generated",
        "num_zero_masks",
        "num_resumed",
        "elapsed_sec",
        "avg_sec_per_image",
        "include_rerun",
        "output_dir",
        "diagnostics_dir",
        "started_at",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("XAI study summaries written to %s and %s", json_path, csv_path)
    return csv_path


def run_mask_ablation_study(
    cfg: MaskAblationConfig, console: Console | None = None
) -> Path:
    console = console or Console()
    combinations = [
        (dataset, model, method)
        for dataset in cfg.datasets
        for model in cfg.models
        for method in cfg.methods
    ]
    evidence_cache: dict[tuple[str, bool, str], list[ImageEvidence]] = {}
    records: list[dict[str, Any]] = []
    study_started = time.perf_counter()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[dim]{task.fields[phase]}[/]"),
        TextColumn("[cyan]{task.fields[mask_type]}[/]"),
        TextColumn("[dim]step {task.fields[step]}[/]"),
        TextColumn("[dim]iter {task.fields[iteration]}[/]"),
        TextColumn("[dim]{task.fields[image]}[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        # Rich otherwise replaces sys.stdout/sys.stderr with FileProxy objects
        # that replay every notebook diagnostic above the live progress row.
        # The CLI has already redirected both streams to the invocation log.
        redirect_stdout=False,
        redirect_stderr=False,
        # Method/image rows are live status only. Removing a task clears its
        # row, and stopping Progress leaves no completed-bar wall behind.
        transient=True,
    )
    active_dataset_name: str | None = None
    with progress:
        for dataset, model, method in combinations:
            if dataset.name != active_dataset_name:
                if active_dataset_name is not None:
                    console.print(f"[green]✓[/] [cyan]{active_dataset_name}[/] complete")
                active_dataset_name = dataset.name
                console.print(f"\n[bold cyan]Dataset: {dataset.name}[/]")

            image_task_id = progress.add_task(
                f"[blue]Images[/] · [yellow]{method.name}[/]",
                total=None,
                phase="starting",
                mask_type=str(method.params.get("segmentation_method", "dense")),
                step="--",
                iteration="--",
                image="--",
            )
            optimisation_task_id = progress.add_task(
                f"  [dim]↳[/] [magenta]{model.name}[/] · [yellow]{method.name}[/]",
                total=None,
                phase="waiting",
                mask_type=str(method.params.get("segmentation_method", "dense")),
                step="--",
                iteration="--",
                image="--",
            )
            try:
                evidence_mode = _validate_method_params(method)
                cache_key = (dataset.name, dataset.include_rerun, evidence_mode)
                if cache_key not in evidence_cache:
                    evidence_cache[cache_key] = build_image_evidence(
                        run_dir=cfg.run_dir,
                        dataset_name=dataset.name,
                        include_rerun=dataset.include_rerun,
                        evidence_mode=evidence_mode,
                    )
                all_items = evidence_cache[cache_key]
                items = all_items
                requested_pilot_images = cfg.pilot_images_by_dataset.get(dataset.name)
                if requested_pilot_images:
                    eligible = [
                        item for item in all_items if item.is_anomaly and item.samples
                    ]
                    selected_items: list[ImageEvidence] = []
                    for requested_name in requested_pilot_images:
                        matches = [
                            item
                            for item in eligible
                            if _matches_pilot_selector(item, requested_name)
                        ]
                        if not matches:
                            raise ValueError(
                                f"Configured pilot image {requested_name!r} was not "
                                f"found among anomalous images with usable "
                                f"{evidence_mode!r} evidence for {dataset.name}."
                            )
                        if len(matches) > 1:
                            identities = [item.key for item in matches]
                            raise ValueError(
                                f"Configured pilot image {requested_name!r} is "
                                f"ambiguous for {dataset.name}: {identities}. Use its "
                                "source/basename or label/source/basename identity."
                            )
                        selected_items.append(matches[0])
                    items = selected_items
                    logger.info(
                        "Explicit pilot scope | dataset=%s method=%s images=%s full=%d",
                        dataset.name,
                        method.name,
                        [item.image_path.name for item in items],
                        len(all_items),
                    )
                elif cfg.max_attribution_images_per_dataset is not None:
                    # Deterministic, useful pilot: select only images that
                    # genuinely require expensive attribution. A later
                    # ``--full`` run uses the same fingerprint and resumes
                    # these masks before processing the remaining images.
                    items = [
                        item for item in all_items if item.is_anomaly and item.samples
                    ][: cfg.max_attribution_images_per_dataset]
                    if not items:
                        raise ValueError(
                            f"Pilot found no anomalous images with usable "
                            f"{evidence_mode!r} evidence for {dataset.name}."
                        )
                    logger.info(
                        "Pilot scope | dataset=%s method=%s selected=%d full=%d",
                        dataset.name,
                        method.name,
                        len(items),
                        len(all_items),
                    )
                progress.update(
                    image_task_id,
                    total=len(items),
                    completed=0,
                    phase="resolving evidence",
                )
                record = _run_combo(
                    cfg=cfg,
                    dataset=dataset,
                    model=model,
                    method=method,
                    items=items,
                    signature_items=all_items,
                    progress=progress,
                    image_task_id=image_task_id,
                    optimisation_task_id=optimisation_task_id,
                )
            except Exception as error:  # one ablation arm must not erase completed peers
                record_pipeline_event(
                    "xai_combo_error",
                    dataset=dataset.name,
                    model=model.name,
                    method=method.name,
                    exception_type=type(error).__name__,
                    error=str(error),
                )
                logger.exception(
                    "XAI combo failed | dataset=%s model=%s method=%s",
                    dataset.name,
                    model.name,
                    method.name,
                )
                record = {
                    "dataset": dataset.name,
                    "model": model.name,
                    "method": method.name,
                    "method_folder": method.folder_name(model),
                    "status": "error",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "error": f"{type(error).__name__}: {error}",
                }
                records.append(record)
                if not cfg.continue_on_error:
                    _write_summaries(
                        cfg, records, time.perf_counter() - study_started
                    )
                    raise
                continue
            finally:
                # These are intentionally transient: the next method replaces
                # them instead of completed rows accumulating in the terminal.
                progress.remove_task(optimisation_task_id)
                progress.remove_task(image_task_id)
            records.append(record)

        if active_dataset_name is not None:
            console.print(f"[green]✓[/] [cyan]{active_dataset_name}[/] complete")

    summary_path = _write_summaries(
        cfg, records, time.perf_counter() - study_started
    )
    ok = sum(record.get("status") == "ok" for record in records)
    errors = len(records) - ok
    logger.info("XAI mask ablation complete | ok=%d error=%d", ok, errors)
    if errors and ok == 0:
        raise RuntimeError(
            f"Every XAI ablation combination failed; see {summary_path} and the log for details."
        )
    return summary_path
