"""Per-image reports for XAI mask-ablation runs.

This module deliberately lives outside ``anomaly_detection.metrics``.  The
normal metrics stage remains unchanged; these reports are a lightweight
companion used to identify the strongest image for each dataset/method.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware rank AUROC without a quadratic pairwise matrix.

    A pairwise ``positives[:, None] > negatives[None, :]`` implementation can
    allocate hundreds of gigabytes for ordinary image masks.  Ranking is
    mathematically equivalent and uses linear memory after sorting.
    """
    labels = np.asarray(y_true).reshape(-1).astype(bool)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _gt_files(root: Path, label: str, stem: str) -> list[Path]:
    folder = root / label
    search = folder if folder.is_dir() else root
    nested = search / stem
    if nested.is_dir():
        return sorted(p for p in nested.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    return sorted(p for p in search.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and (p.stem == stem or p.stem.startswith(stem + "_")))


def _gt_mask(root: Path, label: str, stem: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if label == "good":
        return np.zeros(shape, dtype=np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    for path in _gt_files(root, label, stem):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"))
        arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.NEAREST))
        out |= (arr > 0).astype(np.uint8)
    return out


def _mask_path(method_dir: Path, row: dict[str, Any]) -> Path:
    label = str(row.get("label_folder") or row.get("label") or "anomaly")
    source = str(row.get("source_label_folder") or label)
    stem = Path(str(row.get("img_name", ""))).stem
    folder = method_dir / label
    if source != label:
        folder = folder / source
    return folder / f"{stem}.png"


def _load_summary(dataset_dir: Path, include_rerun: bool) -> list[dict[str, Any]]:
    rows = []
    with (dataset_dir / "all_data_summary.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if include_rerun:
        rerun = dataset_dir / "rerun_results" / "rerun_result.jsonl"
        if rerun.is_file():
            updates = {}
            with rerun.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = (item.get("label_folder"), item.get("source_label_folder", item.get("label_folder")), item.get("img_name"))
                    updates[key] = item
            for row in rows:
                key = (row.get("label_folder"), row.get("source_label_folder", row.get("label_folder")), row.get("img_name"))
                if key in updates:
                    row.update({k: v for k, v in updates[key].items() if k in {"label_folder", "source_label_folder", "label", "binary_prediction"}})
    return rows


def write_per_image_reports(
    *, run_dir: Path, diagnostics_subdir: str, datasets: list[Any],
    models: list[Any], methods: list[Any], metrics_config_path: Path,
    console: Any | None = None,
) -> tuple[Path, Path] | None:
    """Write detailed per-image AUROC and best-image summaries.

    Images whose ground truth is all-zero (normal images) have an undefined
    single-image ROC and are emitted as ``--``; they are never selected as a
    best anomaly image.
    """
    try:
        cfg = json.loads("{}")
        import yaml
        cfg = yaml.safe_load(metrics_config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    roots = {str(item.get("name")): item for item in cfg.get("datasets", []) if isinstance(item, dict)}
    combos = [(dataset, model, method) for dataset in datasets for model in models for method in methods]
    total = 0
    cached: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        if dataset.name not in cached:
            cached[dataset.name] = _load_summary(run_dir / dataset.name, dataset.include_rerun)
        # Individual-image AUROC is meaningful only for anomalous images.
        # Dataset-level Pixel AUROC remains unchanged and still uses good and
        # anomalous images together in the normal metrics stage.
        total += sum(
            1 for row in cached[dataset.name]
            if str(row.get("label_folder", row.get("label", "good"))).lower() != "good"
        ) * len(models) * len(methods)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = run_dir / diagnostics_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    detail_path = out_root / f"pixel_auroc_per_image_{stamp}.csv"
    best_path = out_root / f"pixel_auroc_best_images_{stamp}.csv"
    detail_rows: list[dict[str, Any]] = []
    best: dict[tuple[str, str], dict[str, Any]] = {}
    progress = Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), transient=True, console=console) if console else None
    task = progress.add_task("Per-image Pixel AUROC", total=total) if progress else None
    if progress:
        progress.start()
    try:
        for dataset, model, method in combos:
            spec = roots.get(dataset.name, {})
            gt_raw = spec.get("ground_truth_dir") or spec.get("root_dir")
            if not gt_raw:
                continue
            gt_root = Path(gt_raw).expanduser()
            if not gt_root.is_absolute():
                gt_root = (metrics_config_path.parent.parent / gt_root).resolve()
            method_name = method.folder_name(model)
            method_dir = run_dir / dataset.name / "masks" / method_name
            for row in cached[dataset.name]:
                if str(row.get("label_folder", row.get("label", "good"))).lower() == "good":
                    continue
                image_name = str(row.get("img_name", ""))
                path = _mask_path(method_dir, row)
                score: float | None = None
                if path.is_file():
                    with Image.open(path) as im:
                        pred = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
                    gt = _gt_mask(gt_root, str(row.get("source_label_folder") or row.get("label_folder") or row.get("label", "good")), Path(image_name).stem, pred.shape)
                    if np.unique(gt).size == 2:
                        score = _roc_auc(gt.reshape(-1), pred.reshape(-1))
                detail_rows.append({"dataset": dataset.name, "model": model.name, "method": method.name, "method_folder": method_name, "image": image_name, "pixel_auroc": score if score is not None else "--", "mask_path": str(path)})
                if score is not None:
                    key = (dataset.name, method_name)
                    if key not in best or score > best[key]["pixel_auroc"]:
                        best[key] = {"dataset": dataset.name, "model": model.name, "method": method.name, "method_folder": method_name, "best_image": image_name, "pixel_auroc": score}
                if progress:
                    progress.update(task, advance=1, description=f"{dataset.name} · {method.name} · {image_name}")
    finally:
        if progress:
            progress.stop()
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "method", "method_folder", "image", "pixel_auroc", "mask_path"])
        writer.writeheader(); writer.writerows(detail_rows)
    with best_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "method", "method_folder", "best_image", "pixel_auroc"])
        writer.writeheader(); writer.writerows(best.values())
    return detail_path, best_path
