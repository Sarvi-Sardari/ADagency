"""
metrics.py (anomaly_detection)
================================

Behavior change (2026-08): summary/rerun joins and predicted-mask lookups now
use ``(label_folder, source_label_folder, img_name)``. This matches the
source-aware mask layout and prevents merged defect classes with reused filenames
from sharing another class's prediction map.

Standalone metrics stage for the anomaly-detection pipeline — driven by
run_metrics.py + configs/metrics_configs.yaml. Run any time, on any completed
run_anomaly_detection.py output folder (optionally after a rerun.py pass too),
without needing the VLM or CLIP loaded.

For one dataset it:
  1. Reads <run_dir>/<dataset>/all_data_summary.jsonl (written by pipeline.py).
  2. Optionally folds in <run_dir>/<dataset>/rerun_results/rerun_result.jsonl
     (written by rerun.py) when `include_rerun: true` — this file always has
     ONE entry per image in all_data_summary.jsonl (whether or not that image
     actually qualified for rerun), so overriding binary_prediction/label/score
     from it is safe for every image, not just the ones that were reran.
  3. Groups images by the saved `label_folder` (for example, "good" and the
     configured merged "anomalies" group). The real test subfolder is kept as
     `source_label_folder` and is used for collision-safe mask lookup and
     per-defect-type ground truth lookup; no separate merge flag is needed.
  4. For every anomaly class (all label_folders other than "good"), computes
     accuracy/F1/confusion-matrix from the discrete binary_prediction, plus
     image-level AUROC and F1-max from the CONTINUOUS per-image anomaly score (see
     scoring.py) — a discrete 0/1 "AUROC" degenerates to a single ROC point,
     so the continuous score is used instead, per the original notebook's
     approach. F1-max also records its evaluation-selected threshold and
     finite-score coverage; ordinary F1 remains based on saved predictions.
     Adds an "overall" row when there is more than one anomaly class.
  5. If <run_dir>/<dataset>/masks/ exists (predicted anomaly maps from e.g.
      SAM3, bounding boxes, XAI — one subfolder per method) AND ground-truth
      masks resolve from `ground_truth_dir`, root_dir/ground_truth, or root_dir/mask_all,
     adds a "Pixel AUROC (<method>)" / "AUPRO (<method>)" column pair per
     masks/ subfolder found — fully dynamic, driven by whatever methods exist.
  6. Writes one CSV to <run_dir>/<dataset>/<output_subdir>/metrics_<timestamp>.csv
     (or metrics_w_rerun_<timestamp>.csv when include_rerun is set).

Behavior change (2026-09): a summary row may carry an ``object_crop`` record
(``box_pixels`` + ``source_size`` in original pixels) when the detection run
cropped each candidate to its located object. Such a row's predicted mask is
in crop geometry, so it is un-letterboxed with the CROP's aspect ratio and
pasted into an all-zero full-frame map at the crop position before comparison
with the full-frame ground truth. Pixels outside the crop score 0.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label as _connected_components
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score

from .config import MetricsDatasetEntry, artifact_group_dir
from .scoring import lowest_log_prob_score
from .secondary_classification import (
    load_profile_snapshot,
    profile_fingerprint,
    sole_binary_anomaly_class,
)

logger = logging.getLogger(__name__)

RERUN_RESULT_RELPATH = Path("rerun_results") / "rerun_result.jsonl"
_MASK_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------
# JSONL loading / joining (all_data_summary.jsonl <-> rerun_result.jsonl)
# ---------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                logger.warning("Skipping malformed JSONL line in %s: %s", path, error)
    return entries


def _img_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Return the collision-safe identity used by summary and rerun entries."""
    img_name = entry.get("img_name") or Path(entry["decision_filepath"]).name.removesuffix(".txt")
    label_folder = str(entry["label_folder"])
    source_label_folder = str(entry.get("source_label_folder") or label_folder)
    return (label_folder, source_label_folder, str(img_name))


def _resolve_probs_path(raw_path: str, dataset_run_dir: Path) -> Path:
    """Resolve a raw probs_data_filepath, checking both the dataset run directory and its parent."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (dataset_run_dir / path, dataset_run_dir.parent / path):
        if candidate.is_file():
            return candidate
    return dataset_run_dir / path


def load_summary_entries(summary_path: Path) -> list[dict[str, Any]]:
    """Loads <run_dir>/<dataset>/all_data_summary.jsonl as a list of dicts."""
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"all_data_summary.jsonl not found: {summary_path}")
    return _read_jsonl(summary_path)


def load_rerun_entries(rerun_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load rerun entries keyed by label, source label, and image name."""
    return {_img_key(entry): entry for entry in _read_jsonl(Path(rerun_path))}


# ---------------------------------------------------------------------
# Per-image record building (discrete prediction + continuous score)
# ---------------------------------------------------------------------

def _build_records(
    entries: list[dict[str, Any]],
    dataset_run_dir: Path,
    rerun_entries: dict[tuple[str, str, str], dict[str, Any]] | None,
    *,
    compute_image_auroc: bool = True,
) -> list[dict[str, Any]]:
    """
    Builds one record per image with: label_folder, img_name, ground truth,
    the (possibly rerun-overridden) discrete prediction/label, and the
    CONTINUOUS per-image anomaly score used for image AUROC.
    """
    records: list[dict[str, Any]] = []
    for entry in entries:
        key = _img_key(entry)
        rerun = (rerun_entries or {}).get(key)

        pred = entry["binary_prediction"]
        label = entry["label"]
        score: float | None = None

        secondary_class = entry.get("secondary_class")
        secondary_class_scores = entry.get("secondary_class_scores")
        secondary_binary_score = entry.get("secondary_binary_score")

        probs_path = entry.get("probs_data_filepath")
        if compute_image_auroc and secondary_binary_score is not None:
            score = float(secondary_binary_score)
        elif compute_image_auroc and probs_path:
            _, details = lowest_log_prob_score(_resolve_probs_path(probs_path, dataset_run_dir))
            if details is not None:
                score = details["score"]

        if rerun is not None:
            pred = rerun["binary_prediction"]
            label = rerun["label"]
            if compute_image_auroc and rerun.get("score") is not None:
                score = rerun["score"]
            for field in (
                "secondary_class", "secondary_class_scores", "secondary_binary_score"
            ):
                if field in rerun:
                    if field == "secondary_class":
                        secondary_class = rerun[field]
                    elif field == "secondary_class_scores":
                        secondary_class_scores = rerun[field]
                    else:
                        secondary_binary_score = rerun[field]

        records.append({
            "label_folder": key[0],
            "source_label_folder": key[1],
            "img_name": key[2],
            "gt": int(entry["gt_binary"]),
            "pred": int(pred),
            "label": label,
            "score": score,
            "secondary_class": secondary_class,
            "secondary_class_scores": secondary_class_scores,
            "secondary_binary_score": secondary_binary_score,
            "object_crop": entry.get("object_crop"),
        })
    return records


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["label_folder"], []).append(record)
    return groups


# ---------------------------------------------------------------------
# Image-level metrics (accuracy / f1 / confusion matrix / continuous AUROC)
# ---------------------------------------------------------------------

def _score_metrics(
    y_true: list[int],
    y_score: list[float | None],
    *,
    compute_image_auroc: bool,
    compute_f1_max: bool,
) -> dict[str, Any]:
    """Score metrics share finite-score coverage; ties use score >= threshold.

    F1-max selects a threshold on these evaluation labels, not a deployment
    threshold. The smallest threshold wins ties. Single-class sets are left
    unavailable, consistently with image AUROC.
    """
    result: dict[str, Any] = {}
    if not (compute_image_auroc or compute_f1_max):
        return result
    scored = [
        (gt, float(score)) for gt, score in zip(y_true, y_score)
        if score is not None and np.isfinite(float(score))
    ]
    if len(scored) < len(y_true):
        logger.warning(
            "%d/%d images had no finite anomaly score — excluded from score-based metrics.",
            len(y_true) - len(scored), len(y_true),
        )
    if compute_image_auroc:
        result["image_auroc"] = None
    if compute_f1_max:
        result.update(f1_max=None, f1_max_threshold=None, f1_max_n_scored=len(scored))
    if len({gt for gt, _ in scored}) != 2:
        return result
    truth, scores = zip(*scored)
    if compute_image_auroc:
        result["image_auroc"] = round(float(roc_auc_score(truth, scores)), 6)
    if compute_f1_max:
        precision, recall, thresholds = precision_recall_curve(truth, scores)
        denominator = precision[:-1] + recall[:-1]
        values = np.divide(
            2 * precision[:-1] * recall[:-1], denominator,
            out=np.zeros_like(denominator), where=denominator > 0,
        )
        best = int(np.argmax(values))
        result["f1_max"] = round(float(values[best]), 6)
        # Retain threshold precision so applying it reproduces the optimum.
        result["f1_max_threshold"] = float(thresholds[best])
    return result


def _row_metrics(
    good_records: list[dict[str, Any]],
    anomaly_records: list[dict[str, Any]],
    *,
    compute_image_auroc: bool = True,
    compute_f1_max: bool = True,
) -> dict[str, Any]:
    records = good_records + anomaly_records
    y_true = [r["gt"] for r in records]
    y_pred = [r["pred"] for r in records]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    row: dict[str, Any] = {
        "n (evaluated)": len(records),
        "pos (GT)": int(sum(y_true)),
        "neg (GT)": len(y_true) - int(sum(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }

    row.update(_score_metrics(
        y_true, [r["score"] for r in records],
        compute_image_auroc=compute_image_auroc, compute_f1_max=compute_f1_max,
    ))

    return row


# ---------------------------------------------------------------------
# Pixel-level metrics (Pixel AUROC / AUPRO) — only when masks/ + root_dir exist
# ---------------------------------------------------------------------

def discover_mask_methods(dataset_run_dir: Path) -> list[str]:
    """Returns the sorted names of every subfolder under <dataset_run_dir>/masks/
    (e.g. ["bounding_boxes", "sam3", "xai"]) — dynamically drives the pixel-metric
    columns, whatever prediction methods were actually generated for this run."""
    masks_dir = Path(dataset_run_dir) / "masks"
    if not masks_dir.is_dir():
        return []
    return sorted(p.name for p in masks_dir.iterdir() if p.is_dir())


def _validate_mask_prediction_source(
    dataset_run_dir: Path,
    mask_methods: list[str],
    include_rerun: bool,
) -> None:
    """Reject pixel metrics when mask and metric prediction sources differ.

    New mask-generation runs fingerprint ``include_rerun`` per method. Legacy
    or externally generated mask folders may not have this metadata and remain
    supported, but when a fingerprint exists it is authoritative.
    """
    metadata_dir = Path(dataset_run_dir) / "mask_generation" / "methods"
    for method in mask_methods:
        metadata_path = metadata_dir / f"{method}.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid mask-generation metadata: {metadata_path}: {error}") from error
        mask_include_rerun = metadata.get("include_rerun")
        if isinstance(mask_include_rerun, bool) and mask_include_rerun != include_rerun:
            requested = "rerun" if include_rerun else "base"
            generated = "rerun" if mask_include_rerun else "base"
            matching_flag = "--include-rerun" if mask_include_rerun else "--base-results"
            raise ValueError(
                f"Prediction-source mismatch for {Path(dataset_run_dir).name}/{method}: "
                f"the masks were generated from {generated} predictions, but metrics "
                f"requested {requested} predictions. Regenerate matching masks first, "
                f"or run metrics with {matching_flag}."
            )


def _resolve_gt_root(dataset_entry: MetricsDatasetEntry) -> Path | None:
    """Resolve an explicit mask directory or a known directory under the dataset root."""
    if dataset_entry.ground_truth_dir is not None:
        return dataset_entry.ground_truth_dir
    if dataset_entry.root_dir is None:
        return None

    for folder_name in ("ground_truth", "mask_all"):
        candidate = dataset_entry.root_dir / folder_name
        if candidate.is_dir():
            return candidate
    return None


def _find_gt_mask_files(gt_root: Path, label_folder: str, img_stem: str) -> list[Path]:
    """Find masks in label-nested, per-image nested, or shared flat layouts."""
    label_root = gt_root / label_folder
    search_root = label_root if label_root.is_dir() else gt_root

    nested_dir = search_root / img_stem
    if nested_dir.is_dir():
        return sorted(
            path for path in nested_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _MASK_IMAGE_EXTS
        )

    return sorted(
        path for path in search_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in _MASK_IMAGE_EXTS
        and (path.stem == img_stem or path.stem.startswith(f"{img_stem}_"))
    )


def _gt_mask_size(gt_root: Path) -> tuple[int, int] | None:
    """Return (height, width) from the first mask anywhere under the GT root."""
    for mask_file in sorted(gt_root.rglob("*")):
        if mask_file.is_file() and mask_file.suffix.lower() in _MASK_IMAGE_EXTS:
            with Image.open(mask_file) as im:
                return im.size[1], im.size[0]
    return None


def _load_gt_mask(gt_root: Path, label_folder: str, img_stem: str, target_h: int, target_w: int) -> np.ndarray:
    """Loads and OR-merges every ground-truth mask file for one image. Returns an
    all-zero mask for "good" images or when no ground-truth file is found."""
    if label_folder == "good":
        return np.zeros((target_h, target_w), dtype=np.uint8)

    combined = np.zeros((target_h, target_w), dtype=np.uint8)
    mask_files = _find_gt_mask_files(gt_root, label_folder, img_stem)
    if not mask_files:
        logger.warning(
            "No ground-truth mask found for %s/%s under %s — using an all-zero mask.",
            label_folder,
            img_stem,
            gt_root,
        )

    for mask_file in mask_files:
        mask_np = np.array(Image.open(mask_file).convert("L"))
        mask_resized = np.array(Image.fromarray(mask_np).resize((target_w, target_h), Image.NEAREST))
        combined = np.logical_or(combined, mask_resized > 0).astype(np.uint8)

    return combined


def _pred_score_from_pt(pt_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Loads a saved logit tensor ([H, W] or [N, H, W] for N candidate regions),
    resizes it (bilinear) to the ground-truth mask resolution, applies sigmoid,
    and (for the [N, H, W] case) takes the per-pixel max across regions."""
    import torch
    import torch.nn.functional as F

    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    if data.ndim == 2:
        data = data.unsqueeze(0)  # [1, H, W]

    resized = F.interpolate(data.unsqueeze(1).float(), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)
    combined_logits = resized.max(dim=0).values
    return torch.sigmoid(combined_logits).numpy().astype(np.float32)


def _method_uses_letterboxed_geometry(method_dir: Path) -> bool:
    """Read geometry written by mask_generation.py; absent metadata preserves
    the legacy assumption that a mask can be resized directly."""
    method_dir = Path(method_dir)
    if method_dir.parent.parent.name == "secondary_masks":
        # secondary_masks/<method>/<class>
        dataset_run_dir = method_dir.parent.parent.parent
        method_name = method_dir.parent.name
        metadata_path = (
            dataset_run_dir / "secondary_mask_generation" / "methods"
            / f"{method_name}.json"
        )
    else:
        metadata_path = (
            method_dir.parent.parent / "mask_generation" / "methods"
            / f"{method_dir.name}.json"
        )
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read mask geometry metadata: %s", metadata_path)
        return False
    return metadata.get("image_geometry") == "letterboxed_preprocessed"


def _crop_target_region(
    object_crop: dict[str, Any], target_h: int, target_w: int,
) -> tuple[int, int, int, int]:
    """Map an original-pixel crop box onto the ground-truth grid."""
    left, top, right, bottom = (int(value) for value in object_crop["box_pixels"])
    source_w, source_h = (int(value) for value in object_crop["source_size"])
    scale_x, scale_y = target_w / source_w, target_h / source_h
    x0 = max(0, min(target_w - 1, int(round(left * scale_x))))
    y0 = max(0, min(target_h - 1, int(round(top * scale_y))))
    x1 = max(x0 + 1, min(target_w, int(round(right * scale_x))))
    y1 = max(y0 + 1, min(target_h, int(round(bottom * scale_y))))
    return x0, y0, x1, y1


def _place_crop_score_map(
    array: np.ndarray, object_crop: dict[str, Any], target_h: int, target_w: int,
) -> np.ndarray:
    """Resize a crop-geometry map into its crop region of a zero full frame."""
    x0, y0, x1, y1 = _crop_target_region(object_crop, target_h, target_w)
    resized = Image.fromarray(array.astype(np.float32), mode="F").resize(
        (x1 - x0, y1 - y0), Image.BILINEAR
    )
    full = np.zeros((target_h, target_w), dtype=np.float32)
    full[y0:y1, x0:x1] = np.array(resized, dtype=np.float32)
    return full


def _align_image_score_map(
    array: np.ndarray, method_dir: Path, target_h: int, target_w: int,
    object_crop: dict[str, Any] | None = None,
) -> np.ndarray:
    """Align an image score map with the original ground-truth geometry.

    Pipeline-generated masks use the letterboxed VLM candidate image. Crop its
    symmetric padding first, then resize the remaining image content. Other
    mask producers retain the previous direct-resize behavior.

    With ``object_crop`` the letterboxed content has the crop's aspect ratio
    (not the full frame's) and is placed back at the crop position.
    """
    if object_crop is None and array.shape == (target_h, target_w):
        return array.astype(np.float32)

    source_h, source_w = array.shape
    if _method_uses_letterboxed_geometry(method_dir):
        if object_crop is not None:
            left, top, right, bottom = (int(value) for value in object_crop["box_pixels"])
            content_ref_w, content_ref_h = right - left, bottom - top
        else:
            content_ref_w, content_ref_h = target_w, target_h
        scale = min(source_w / content_ref_w, source_h / content_ref_h)
        content_w = max(1, min(source_w, round(content_ref_w * scale)))
        content_h = max(1, min(source_h, round(content_ref_h * scale)))
        left_pad = (source_w - content_w) // 2
        top_pad = (source_h - content_h) // 2
        array = array[top_pad:top_pad + content_h, left_pad:left_pad + content_w]

    if object_crop is not None:
        return _place_crop_score_map(array, object_crop, target_h, target_w)
    resized = Image.fromarray(array.astype(np.float32), mode="F").resize(
        (target_w, target_h), Image.BILINEAR
    )
    return np.array(resized, dtype=np.float32)


def _load_pred_mask(
    method_dir: Path,
    label_folder: str,
    source_label_folder: str,
    img_stem: str,
    target_h: int,
    target_w: int,
    is_predicted_anomaly: bool,
    object_crop: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Loads one image's predicted per-pixel anomaly score map for a given mask
    method. Returns an all-zero map when the image wasn't predicted anomalous
    (no mask needed). A predicted-anomaly image must have a persisted mask,
    including when that mask is intentionally all-zero; a missing file means
    mask generation is incomplete and is rejected instead of silently biasing
    pixel metrics toward zero.

    Supported file conventions under
    <method_dir>/<label_folder>/[<source_label_folder>/]:
      - "processed_<stem>.pt" / "<stem>.pt"  -> torch logit tensor (see
        _pred_score_from_pt); matches the SAM3/bounding-box convention already
        used elsewhere in this repo.
      - "processed_<stem>.<ext>" / "<stem>.<ext>" (png/jpg/jpeg/bmp) -> an
        already-computed grayscale probability map (0-255, normalized to 0-1).
      - "<stem>.npy" -> an already-computed float probability map.
    """
    if not is_predicted_anomaly:
        return np.zeros((target_h, target_w), dtype=np.float32)

    # Keep mask lookup identical to inference/mask-generation grouping. The
    # semantic class used to choose a planner never participates in this path.
    folder = artifact_group_dir(method_dir, label_folder, source_label_folder)

    for pt_name in (f"processed_{img_stem}.pt", f"{img_stem}.pt"):
        pt_path = folder / pt_name
        if pt_path.is_file():
            if object_crop is not None:
                x0, y0, x1, y1 = _crop_target_region(object_crop, target_h, target_w)
                full = np.zeros((target_h, target_w), dtype=np.float32)
                full[y0:y1, x0:x1] = _pred_score_from_pt(pt_path, y1 - y0, x1 - x0)
                return full
            return _pred_score_from_pt(pt_path, target_h, target_w)

    for ext in _MASK_IMAGE_EXTS:
        for name in (f"processed_{img_stem}{ext}", f"{img_stem}{ext}"):
            img_path = folder / name
            if img_path.is_file():
                arr = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
                return _align_image_score_map(
                    arr, method_dir, target_h, target_w, object_crop=object_crop
                )

    npy_path = folder / f"{img_stem}.npy"
    if npy_path.is_file():
        array = np.load(npy_path).astype(np.float32)
        if object_crop is not None:
            return _place_crop_score_map(array, object_crop, target_h, target_w)
        return array

    raise FileNotFoundError(
        "Predicted-anomaly mask is missing for "
        f"{label_folder}/{source_label_folder}/{img_stem} under {method_dir}. "
        "Mask generation must finish successfully before pixel metrics are calculated."
    )


def _progress_update(progress: Any | None, task_id: Any | None, **kwargs: Any) -> None:
    """Update the optional shared Rich task without coupling metrics to the CLI."""
    if progress is not None and task_id is not None:
        progress.update(task_id, **kwargs)


def _compute_aupro(
    gt_masks: np.ndarray,
    pred_scores: np.ndarray,
    num_thresholds: int = 200,
    fpr_limit: float = 0.3,
    *,
    progress: Any | None = None,
    task_id: Any | None = None,
    description: str = "Calculating AUPRO",
) -> float | None:
    """
    Per-region overlap (PRO) integrated up to fpr_limit and normalized to
    [0, 1] — same algorithm validated in this repo's own notebooks
    (scipy.ndimage.label per-image connected components, region overlap
    averaged per threshold, FPR from all ground-truth-normal pixels).
    """
    if gt_masks.sum() == 0:
        return None

    thresholds = np.concatenate((
        [np.inf],
        np.linspace(pred_scores.max(), pred_scores.min(), num_thresholds),
        [-np.inf],
    ))
    normal_pixels = gt_masks == 0
    normal_total = int(normal_pixels.sum())

    pros: list[float] = []
    fprs: list[float] = []
    _progress_update(
        progress,
        task_id,
        description=description,
        total=len(thresholds),
        completed=0,
    )
    for threshold in thresholds:
        pred_binary = pred_scores >= threshold

        region_overlaps: list[float] = []
        for gt_mask, pred_mask in zip(gt_masks, pred_binary):
            regions, n_regions = _connected_components(gt_mask)
            for region_id in range(1, n_regions + 1):
                region = regions == region_id
                region_overlaps.append(float(pred_mask[region].sum()) / float(region.sum()))

        pros.append(float(np.mean(region_overlaps)) if region_overlaps else 0.0)
        fp = int(np.logical_and(pred_binary, normal_pixels).sum())
        fprs.append(fp / normal_total if normal_total else 0.0)
        _progress_update(progress, task_id, advance=1)

    # Threshold sweeps often produce many points at the same FPR. Keep the
    # best overlap at each FPR, then interpolate the curve exactly at the
    # integration limit. Without this, a perfect binary map can incorrectly
    # integrate to zero because all of its useful points sit at FPR=0.
    best_pro_by_fpr: dict[float, float] = {}
    for fpr, pro in zip(fprs, pros):
        best_pro_by_fpr[fpr] = max(pro, best_pro_by_fpr.get(fpr, 0.0))
    sorted_fprs = np.array(sorted(best_pro_by_fpr), dtype=np.float64)
    sorted_pros = np.array(
        [best_pro_by_fpr[fpr] for fpr in sorted_fprs], dtype=np.float64
    )
    if sorted_fprs.size == 0:
        return None
    inside = sorted_fprs < fpr_limit
    x_values = list(sorted_fprs[inside])
    y_values = list(sorted_pros[inside])
    boundary_pro = float(np.interp(fpr_limit, sorted_fprs, sorted_pros))
    x_values.append(float(fpr_limit))
    y_values.append(boundary_pro)
    return round(float(auc(np.asarray(x_values), np.asarray(y_values)) / fpr_limit), 6)


def _pixel_metrics_for_group(
    records: list[dict[str, Any]], gt_root: Path, method_dir: Path, target_h: int, target_w: int,
    *,
    progress: Any | None = None,
    task_id: Any | None = None,
    progress_label: str = "Pixel metrics",
    compute_pixel_auroc: bool = True,
    compute_aupro: bool = True,
) -> tuple[float | None, float | None]:
    gt_stack: list[np.ndarray] = []
    pred_stack: list[np.ndarray] = []

    _progress_update(
        progress,
        task_id,
        description=f"{progress_label} · loading masks",
        total=len(records),
        completed=0,
    )
    for record in records:
        img_stem = Path(record["img_name"]).stem
        # Ground truth lives under the dataset's own fixed per-defect-type
        # directory layout regardless of merge_anomalies, so it must be
        # looked up by the image's REAL test subfolder, not by whatever
        # (possibly merged) label_folder this run grouped it under.
        gt_stack.append(_load_gt_mask(gt_root, record["source_label_folder"], img_stem, target_h, target_w))
        # Predicted masks use both the merged display label and the real source
        # label when they differ, matching mask_generation.py's v4 layout.
        pred_stack.append(_load_pred_mask(
            method_dir,
            record["label_folder"],
            record["source_label_folder"],
            img_stem,
            target_h,
            target_w,
            is_predicted_anomaly=(record["label"] == "anomaly"),
            object_crop=record.get("object_crop"),
        ))
        _progress_update(progress, task_id, advance=1)

    gt_arr = np.stack(gt_stack, axis=0)
    pred_arr = np.stack(pred_stack, axis=0)

    pixel_auroc = None
    if compute_pixel_auroc and len(np.unique(gt_arr)) == 2:
        _progress_update(
            progress,
            task_id,
            description=f"{progress_label} · calculating Pixel AUROC",
            total=None,
            completed=0,
        )
        pixel_auroc = round(float(roc_auc_score(gt_arr.reshape(-1), pred_arr.reshape(-1))), 6)

    pixel_aupro = (
        _compute_aupro(
            gt_arr,
            pred_arr,
            progress=progress,
            task_id=task_id,
            description=f"{progress_label} · calculating AUPRO",
        )
        if compute_aupro else None
    )
    _progress_update(
        progress,
        task_id,
        description=f"{progress_label} · complete",
        total=1,
        completed=1,
    )
    return pixel_auroc, pixel_aupro


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def _save_csv(
    output_dir: Path,
    dataset_name: str,
    mask_methods: list[str],
    rows: list[dict[str, Any]],
    include_rerun: bool,
    *,
    include_image_auroc: bool = True,
    include_f1_max: bool = True,
    include_pixel_auroc: bool = True,
    include_aupro: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "metrics_w_rerun" if include_rerun else "metrics"
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"

    fieldnames = ["dataset", "anomaly_class", "n (evaluated)", "pos (GT)", "neg (GT)", "tp", "fp", "tn", "fn", "accuracy", "f1"]
    if include_image_auroc:
        fieldnames.append("image_auroc")
    if include_f1_max:
        fieldnames.extend(("f1_max", "f1_max_threshold", "f1_max_n_scored"))
    for method in mask_methods:
        if include_pixel_auroc:
            fieldnames.append(f"Pixel AUROC ({method})")
        if include_aupro:
            fieldnames.append(f"AUPRO ({method})")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Accept both the historical compact internal keys and the newer
            # descriptive keys emitted directly by _row_metrics(). This keeps
            # CSV serialization stable if callers construct rows either way.
            csv_row = {"dataset": dataset_name, **row}
            for compact, descriptive in (
                ("n", "n (evaluated)"),
                ("pos", "pos (GT)"),
                ("neg", "neg (GT)"),
            ):
                if compact in csv_row:
                    csv_row[descriptive] = csv_row.pop(compact)
                elif descriptive not in csv_row:
                    raise ValueError(
                        f"Metrics row is missing both {compact!r} and {descriptive!r}: {row}"
                    )
            writer.writerow(csv_row)

    return csv_path


def discover_secondary_mask_methods(dataset_run_dir: Path) -> list[str]:
    root = Path(dataset_run_dir) / "secondary_masks"
    methods = {
        path.name for path in root.iterdir() if path.is_dir()
    } if root.is_dir() else set()
    # Metadata is written for configured secondary methods even when every
    # predicted mask channel can be represented by a binary-mask alias.  Keep
    # those methods discoverable without requiring duplicate mask files.
    metadata_root = Path(dataset_run_dir) / "secondary_mask_generation" / "methods"
    if metadata_root.is_dir():
        methods.update(path.stem for path in metadata_root.glob("*.json"))
    return sorted(methods)


def _validate_secondary_mask_prediction_source(
    dataset_run_dir: Path, methods: list[str], include_rerun: bool, fingerprint: str
) -> None:
    metadata_root = dataset_run_dir / "secondary_mask_generation" / "methods"
    for method in methods:
        path = metadata_root / f"{method}.json"
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("include_rerun") != include_rerun:
            raise ValueError(
                f"Secondary mask prediction source differs for {dataset_run_dir.name}/{method}."
            )
        if metadata.get("secondary_profile_fingerprint") != fingerprint:
            raise ValueError(
                f"Secondary mask profile differs for {dataset_run_dir.name}/{method}."
            )


def _secondary_gt_mask(
    mask_root: Path | None,
    class_name: str,
    img_stem: str,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    if mask_root is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    files = _find_gt_mask_files(mask_root, class_name, img_stem)
    if not files:
        raise FileNotFoundError(
            f"Ground-truth mask for secondary class {class_name!r}, image "
            f"{img_stem!r}, was not found under {mask_root}."
        )
    combined = np.zeros((target_h, target_w), dtype=np.uint8)
    for path in files:
        array = np.array(Image.open(path).convert("L"))
        resized = np.array(
            Image.fromarray(array).resize((target_w, target_h), Image.NEAREST)
        )
        combined = np.logical_or(combined, resized > 0).astype(np.uint8)
    return combined


def _secondary_truth_key(record: dict[str, Any]) -> str:
    """Identify a secondary GT image by its real test folder and image stem."""
    source = record.get("source_label_folder") or record["label_folder"]
    return f"{source}/{Path(record['img_name']).stem}"


def _secondary_actual_class(record: dict[str, Any], ground_truth: dict[str, str]) -> str:
    key = _secondary_truth_key(record)
    if key in ground_truth:
        return ground_truth[key]
    # Preserve direct helper callers and legacy datasets with unique stems.
    return ground_truth[Path(record["img_name"]).stem]


def _resolve_secondary_ground_truth(
    ground_truth: dict[str, str], records: list[dict[str, Any]],
) -> dict[str, str]:
    """Accept source/stem keys, or legacy bare stems only when unambiguous."""
    identities_by_stem: dict[str, set[tuple[str, str, str]]] = {}
    for record in records:
        stem = Path(record["img_name"]).stem
        identities_by_stem.setdefault(stem, set()).add(_img_key(record))
    resolved: dict[str, str] = {}
    owners: dict[str, tuple[str, str, str]] = {}
    for record in records:
        stem = Path(record["img_name"]).stem
        key = _secondary_truth_key(record)
        identity = _img_key(record)
        if key in owners and owners[key] != identity:
            raise ValueError(f"Secondary source identity {key!r} is duplicated across result groups.")
        owners[key] = identity
        if key in ground_truth:
            resolved[key] = ground_truth[key]
        elif stem in ground_truth:
            if len(identities_by_stem[stem]) > 1:
                raise ValueError(
                    f"Secondary ground truth is keyed only by image stem, but {stem!r} is duplicated; "
                    "use source_label_folder/image_stem keys for every colliding image."
                )
            resolved[key] = ground_truth[stem]
        else:
            raise ValueError(f"Secondary ground truth lacks image id {key!r} (legacy stem {stem!r}).")
    return resolved


def _secondary_pixel_metrics(
    records: list[dict[str, Any]],
    actual_classes: dict[str, str],
    target_class: str,
    gt_root: Path,
    method_class_dir: Path,
    target_h: int,
    target_w: int,
    *,
    compute_pixel_auroc: bool,
    compute_aupro: bool,
) -> tuple[float | None, float | None]:
    gt_stack: list[np.ndarray] = []
    pred_stack: list[np.ndarray] = []
    for record in records:
        stem = Path(record["img_name"]).stem
        actual = _secondary_actual_class(record, actual_classes)
        gt_stack.append(
            _secondary_gt_mask(
                gt_root if actual == target_class else None,
                target_class,
                stem,
                target_h,
                target_w,
            )
        )
        pred_stack.append(
            _load_pred_mask(
                method_class_dir,
                record["label_folder"],
                record["source_label_folder"],
                stem,
                target_h,
                target_w,
                is_predicted_anomaly=(record["secondary_class"] == target_class),
                object_crop=record.get("object_crop"),
            )
        )
    gt_array = np.stack(gt_stack)
    pred_array = np.stack(pred_stack)
    pixel_auroc = (
        round(float(roc_auc_score(gt_array.reshape(-1), pred_array.reshape(-1))), 6)
        if compute_pixel_auroc and len(np.unique(gt_array)) == 2 else None
    )
    pixel_aupro = _compute_aupro(gt_array, pred_array) if compute_aupro else None
    return pixel_auroc, pixel_aupro


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _run_secondary_metrics(
    dataset_run_dir: Path,
    dataset_entry: MetricsDatasetEntry,
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    compute_image_auroc: bool,
    compute_pixel_auroc: bool,
    compute_aupro: bool,
    compute_f1_max: bool = True,
) -> Path:
    settings = dataset_entry.secondary_metrics
    if settings is None:
        raise ValueError("Secondary metric settings are missing.")
    profile = load_profile_snapshot(dataset_run_dir)
    if profile is None:
        raise ValueError(
            f"Secondary metrics are enabled for {dataset_entry.name}, but its run has no saved profile."
        )

    raw_ground_truth = json.loads(settings.ground_truth_file.read_text(encoding="utf-8"))
    if not isinstance(raw_ground_truth, dict):
        raise ValueError(f"Secondary ground truth must be a JSON object: {settings.ground_truth_file}")
    ground_truth = {str(key): str(value) for key, value in raw_ground_truth.items()}
    valid_classes = set(profile.class_names)
    configured_mask_classes = set(settings.ground_truth_masks)
    if configured_mask_classes != valid_classes:
        raise ValueError(
            "secondary_metrics.ground_truth_masks keys must exactly match the "
            f"saved profile classes; expected={sorted(valid_classes)}, "
            f"got={sorted(configured_mask_classes)}."
        )
    unknown_labels = sorted(set(ground_truth.values()) - valid_classes)
    if unknown_labels:
        raise ValueError(f"Secondary ground truth contains unknown classes: {unknown_labels}")

    ground_truth = _resolve_secondary_ground_truth(ground_truth, records)
    for record in records:
        identity = (record["label_folder"], record["source_label_folder"], record["img_name"])
        if record.get("secondary_class") not in valid_classes:
            raise ValueError(f"Prediction for {identity!r} lacks a valid secondary class.")
        scores = record.get("secondary_class_scores")
        if not isinstance(scores, dict) or set(scores) != valid_classes:
            raise ValueError(f"Prediction for {identity!r} lacks complete secondary scores.")

    mask_methods = (
        discover_secondary_mask_methods(dataset_run_dir)
        if compute_pixel_auroc or compute_aupro else []
    )
    if mask_methods:
        _validate_secondary_mask_prediction_source(
            dataset_run_dir, mask_methods, dataset_entry.include_rerun,
            profile_fingerprint(profile),
        )

    target_size = None
    for class_name in profile.class_names:
        root = settings.ground_truth_masks.get(class_name)
        if root is not None and (target_size := _gt_mask_size(root)) is not None:
            break

    rows: list[dict[str, Any]] = []
    binary_equivalent_class = sole_binary_anomaly_class(profile)
    if binary_equivalent_class is not None:
        secondary_truth = [
            int(_secondary_actual_class(record, ground_truth) == binary_equivalent_class)
            for record in records
        ]
        secondary_predictions = [
            int(record["secondary_class"] == binary_equivalent_class)
            for record in records
        ]
        binary_truth = [int(record["gt"]) for record in records]
        binary_predictions = [int(record["pred"]) for record in records]
        if secondary_truth != binary_truth:
            raise ValueError(
                f"Secondary class {binary_equivalent_class!r} is the sole class mapped "
                "to binary anomaly, but its ground truth differs from binary ground truth."
            )
        if secondary_predictions != binary_predictions:
            raise ValueError(
                f"Secondary class {binary_equivalent_class!r} is the sole class mapped "
                "to binary anomaly, but its predictions differ from binary predictions."
            )
    for class_name in profile.class_names:
        y_true = [int(_secondary_actual_class(record, ground_truth) == class_name) for record in records]
        y_pred = [int(record["secondary_class"] == class_name) for record in records]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        row: dict[str, Any] = {
            "dataset": dataset_entry.name,
            "secondary_class": class_name,
            "n (evaluated)": len(records),
            "pos (GT)": int(sum(y_true)),
            "neg (GT)": len(records) - int(sum(y_true)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        }
        if compute_image_auroc or compute_f1_max:
            class_scores = [float(record["secondary_class_scores"][class_name]) for record in records]
            row.update(_score_metrics(
                y_true, class_scores, compute_image_auroc=compute_image_auroc,
                compute_f1_max=compute_f1_max,
            ))

        gt_root = settings.ground_truth_masks.get(class_name)
        if gt_root is not None and target_size is not None:
            target_h, target_w = target_size
            for method in mask_methods:
                method_class_dir = (
                    dataset_run_dir / "masks" / method
                    if (
                        class_name == binary_equivalent_class
                        and (dataset_run_dir / "masks" / method).is_dir()
                    )
                    else dataset_run_dir / "secondary_masks" / method / class_name
                )
                if method_class_dir == dataset_run_dir / "masks" / method:
                    logger.info(
                        "[%s] secondary class %s reuses binary %s masks",
                        dataset_entry.name,
                        class_name,
                        method,
                    )
                pixel_auroc, pixel_aupro = _secondary_pixel_metrics(
                    records, ground_truth, class_name, gt_root,
                    method_class_dir,
                    target_h, target_w,
                    compute_pixel_auroc=compute_pixel_auroc,
                    compute_aupro=compute_aupro,
                )
                if compute_pixel_auroc:
                    row[f"Pixel AUROC ({method})"] = pixel_auroc
                if compute_aupro:
                    row[f"AUPRO ({method})"] = pixel_aupro
        elif gt_root is None and target_size is not None:
            # A clean/all-zero class has no positive pixel target, so its own
            # Pixel AUROC/AUPRO is undefined. Report false-positive mask area
            # on its actual images instead.
            clean_records = [
                record for record in records
                if _secondary_actual_class(record, ground_truth) == class_name
            ]
            target_h, target_w = target_size
            for method in mask_methods:
                fractions: list[float] = []
                for record in clean_records:
                    predicted_class = record["secondary_class"]
                    if settings.ground_truth_masks.get(predicted_class) is None:
                        score_map = np.zeros((target_h, target_w), dtype=np.float32)
                    else:
                        predicted_method_dir = (
                            dataset_run_dir / "masks" / method
                            if (
                                predicted_class == binary_equivalent_class
                                and (dataset_run_dir / "masks" / method).is_dir()
                            )
                            else dataset_run_dir / "secondary_masks" / method / predicted_class
                        )
                        score_map = _load_pred_mask(
                            predicted_method_dir,
                            record["label_folder"], record["source_label_folder"],
                            Path(record["img_name"]).stem, target_h, target_w,
                            is_predicted_anomaly=True,
                            object_crop=record.get("object_crop"),
                        )
                    fractions.append(float(np.count_nonzero(score_map) / score_map.size))
                row[f"mean predicted mask fraction ({method})"] = (
                    round(float(np.mean(fractions)), 6) if fractions else None
                )
                row[f"max predicted mask fraction ({method})"] = (
                    round(float(np.max(fractions)), 6) if fractions else None
                )
                row[f"percent images with any predicted mask ({method})"] = (
                    round(float(np.mean(np.asarray(fractions) > 0) * 100.0), 6)
                    if fractions else None
                )
        rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_w_rerun" if dataset_entry.include_rerun else ""
    image_path = output_dir / f"secondary_image_metrics{suffix}_{timestamp}.csv"
    _write_dict_csv(image_path, rows)

    confusion_rows = []
    for actual_class in profile.class_names:
        actual_records = [
            record for record in records
            if _secondary_actual_class(record, ground_truth) == actual_class
        ]
        confusion_rows.append({
            "dataset": dataset_entry.name,
            "actual_class": actual_class,
            **{
                f"predicted_{predicted_class}": sum(
                    record["secondary_class"] == predicted_class for record in actual_records
                )
                for predicted_class in profile.class_names
            },
        })
    _write_dict_csv(
        output_dir / f"secondary_confusion_matrix{suffix}_{timestamp}.csv",
        confusion_rows,
    )
    logger.info("[%s] secondary metrics CSV written to %s", dataset_entry.name, image_path)
    return image_path


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def run_metrics_from_run_dir(
    run_dir: Path,
    dataset_entry: MetricsDatasetEntry,
    output_subdir: str,
    *,
    progress: Any | None = None,
    operation_task: Any | None = None,
    omitted_metrics: set[str] | None = None,
) -> Path | None:
    """
    Standalone entry point (see run_metrics.py + configs/metrics_configs.yaml):
    loads <run_dir>/<dataset_entry.name>/all_data_summary.jsonl, optionally
    merges rerun_results/rerun_result.jsonl, computes per-anomaly-class
    accuracy/F1/confusion-matrix/image-AUROC (+ Pixel AUROC/AUPRO when a
    masks/ folder and dataset_entry.root_dir are available), and writes one
    CSV to <run_dir>/<dataset_entry.name>/<output_subdir>/.
    """
    dataset_run_dir = Path(run_dir) / dataset_entry.name
    omitted_metrics = omitted_metrics or set()
    compute_image_auroc = "image-auroc" not in omitted_metrics
    compute_f1_max = "f1-max" not in omitted_metrics
    compute_pixel_auroc = "pixel-auroc" not in omitted_metrics
    compute_aupro = "aupro" not in omitted_metrics
    _progress_update(
        progress,
        operation_task,
        description=f"{dataset_entry.name} · loading summaries",
        total=None,
        completed=0,
    )
    entries = load_summary_entries(dataset_run_dir / "all_data_summary.jsonl")

    rerun_entries: dict[tuple[str, str, str], dict[str, Any]] | None = None
    if dataset_entry.include_rerun:
        rerun_path = dataset_run_dir / RERUN_RESULT_RELPATH
        if rerun_path.exists():
            rerun_entries = load_rerun_entries(rerun_path)
            logger.info("[%s] merged rerun results from %s", dataset_entry.name, rerun_path)
        else:
            logger.warning("[%s] include_rerun=true but %s does not exist — using base results only.", dataset_entry.name, rerun_path)

    records = _build_records(
        entries,
        dataset_run_dir,
        rerun_entries,
        compute_image_auroc=compute_image_auroc or compute_f1_max,
    )
    groups = _group_records(records)

    good_records = groups.get("good", [])
    if not good_records:
        logger.warning("[%s] No 'good' group found in all_data_summary.jsonl — skipping metrics.", dataset_entry.name)
        return None

    anomaly_group_names = sorted(name for name in groups if name != "good")
    if not anomaly_group_names:
        logger.warning("[%s] No anomaly classes found in all_data_summary.jsonl — skipping metrics.", dataset_entry.name)
        return None

    mask_methods = (
        discover_mask_methods(dataset_run_dir)
        if compute_pixel_auroc or compute_aupro else []
    )
    if mask_methods:
        _validate_mask_prediction_source(
            dataset_run_dir,
            mask_methods,
            include_rerun=dataset_entry.include_rerun,
        )
    compute_pixel = bool(mask_methods)
    gt_root: Path | None = None
    target_size: tuple[int, int] | None = None

    if compute_pixel:
        gt_root = _resolve_gt_root(dataset_entry)
        if gt_root is None:
            logger.warning(
                "[%s] masks/ found (%s), but no ground-truth mask directory was found. "
                "Set root_dir (auto-detects ground_truth/ or mask_all/) or ground_truth_dir.",
                dataset_entry.name,
                mask_methods,
            )
            compute_pixel = False

    if compute_pixel:
        target_size = _gt_mask_size(gt_root)
        if target_size is None:
            logger.warning("[%s] No ground-truth mask images found under %s — skipping pixel metrics.", dataset_entry.name, gt_root)
            compute_pixel = False
        else:
            logger.info("[%s] ground-truth masks resolved to %s", dataset_entry.name, gt_root)

    rows: list[dict[str, Any]] = []
    for group_name in anomaly_group_names:
        anomaly_records = groups[group_name]
        row = _row_metrics(
            good_records,
            anomaly_records,
            compute_image_auroc=compute_image_auroc,
            compute_f1_max=compute_f1_max,
        )
        row["anomaly_class"] = group_name
        if compute_pixel:
            target_h, target_w = target_size
            for method in mask_methods:
                pixel_auroc, pixel_aupro = _pixel_metrics_for_group(
                    good_records + anomaly_records, gt_root, dataset_run_dir / "masks" / method, target_h, target_w,
                    progress=progress,
                    task_id=operation_task,
                    progress_label=f"{dataset_entry.name} · {group_name} · {method}",
                    compute_pixel_auroc=compute_pixel_auroc,
                    compute_aupro=compute_aupro,
                )
                if compute_pixel_auroc:
                    row[f"Pixel AUROC ({method})"] = pixel_auroc
                if compute_aupro:
                    row[f"AUPRO ({method})"] = pixel_aupro
        rows.append(row)
        logger.info(
            "[%s] %s: acc=%.4f f1=%.4f image_auroc=%s",
            dataset_entry.name,
            group_name,
            row["accuracy"],
            row["f1"],
            row.get("image_auroc", "omitted"),
        )

    if len(anomaly_group_names) > 1:
        all_anomaly_records = [r for name in anomaly_group_names for r in groups[name]]
        overall = _row_metrics(
            good_records,
            all_anomaly_records,
            compute_image_auroc=compute_image_auroc,
            compute_f1_max=compute_f1_max,
        )
        overall["anomaly_class"] = "overall"
        if compute_pixel:
            target_h, target_w = target_size
            for method in mask_methods:
                pixel_auroc, pixel_aupro = _pixel_metrics_for_group(
                    good_records + all_anomaly_records, gt_root, dataset_run_dir / "masks" / method, target_h, target_w,
                    progress=progress,
                    task_id=operation_task,
                    progress_label=f"{dataset_entry.name} · overall · {method}",
                    compute_pixel_auroc=compute_pixel_auroc,
                    compute_aupro=compute_aupro,
                )
                if compute_pixel_auroc:
                    overall[f"Pixel AUROC ({method})"] = pixel_auroc
                if compute_aupro:
                    overall[f"AUPRO ({method})"] = pixel_aupro
        rows.append(overall)
        logger.info(
            "[%s] overall: acc=%.4f f1=%.4f image_auroc=%s",
            dataset_entry.name,
            overall["accuracy"],
            overall["f1"],
            overall.get("image_auroc", "omitted"),
        )

    csv_path = _save_csv(
        dataset_run_dir / output_subdir,
        dataset_entry.name,
        mask_methods if compute_pixel else [],
        rows,
        include_rerun=bool(rerun_entries),
        include_image_auroc=compute_image_auroc,
        include_f1_max=compute_f1_max,
        include_pixel_auroc=compute_pixel_auroc,
        include_aupro=compute_aupro,
    )
    logger.info("[%s] metrics CSV written to %s", dataset_entry.name, csv_path)
    if dataset_entry.secondary_metrics is not None:
        _run_secondary_metrics(
            dataset_run_dir,
            dataset_entry,
            records,
            dataset_run_dir / output_subdir,
            compute_image_auroc=compute_image_auroc,
            compute_f1_max=compute_f1_max,
            compute_pixel_auroc=compute_pixel_auroc,
            compute_aupro=compute_aupro,
        )
    _progress_update(
        progress,
        operation_task,
        description=f"{dataset_entry.name} · metrics saved",
        total=1,
        completed=1,
    )
    return csv_path
