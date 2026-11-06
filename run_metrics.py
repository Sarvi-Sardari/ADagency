"""
run_metrics.py
===============
Standalone metrics/ROC-plot stage — driven entirely by
configs/metrics_configs.yaml.

Run this any time, on any completed run_anomaly_detection.py run folder,
without needing the VLM or CLIP loaded. Optionally folds in a
rerun_result.jsonl (written by the later manual rerun method) so that switching
`include_rerun` on/off in the YAML and re-running lets you compare
before/after-rerun metrics.

Usage
-----
  python run_metrics.py
  python run_metrics.py --metrics-config configs/metrics_configs.yaml
  python run_metrics.py --combine
  python run_metrics.py --combine --omit aupro
  python run_metrics.py --omit aupro pixel-auroc
  python run_metrics.py --combine --include-rerun
  python run_metrics.py --combine --base-results
  python run_metrics.py --include-secondary
  python run_metrics.py --binary-only
"""

import csv
import logging
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

PROJECT_ROOT = Path(__file__).resolve().parent

from anomaly_detection.config import load_metrics_config
from anomaly_detection.hardware_monitor import HardwareMonitor
from anomaly_detection.logging_utils import setup_logging
from anomaly_detection.metrics import run_metrics_from_run_dir
from cluster_deployment.config import load_deployment_config
from cluster_deployment.remote_monitor import build_monitor, monitored

CONFIG_DIR = PROJECT_ROOT / "configs"
logger = logging.getLogger(__name__)


def _as_number(value: str | None) -> float | None:
    """Parse a populated CSV cell as a number; blanks remain unavailable."""
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _combined_overall_row(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, object]:
    """Build one statistically meaningful run-level summary row.

    Each dataset contributes exactly once: its own ``overall`` row when it has
    multiple anomaly classes, otherwise its sole class row. Counts and
    confusion-matrix cells are additive. Accuracy and F1 are then recomputed
    from that summed confusion matrix. Other numeric metrics (AUROC/AUPRO and
    any future dynamic metrics) are macro-averaged over datasets with a value.
    Thus combined f1_max is the mean of dataset optima, not a pooled optimum;
    no shared f1_max_threshold is defined for that row.
    """
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "")), []).append(row)

    representatives: list[dict[str, str]] = []
    for dataset_rows in by_dataset.values():
        dataset_overall = next(
            (row for row in dataset_rows if str(row.get("anomaly_class", "")).lower() == "overall"),
            None,
        )
        representatives.append(dataset_overall or dataset_rows[0])

    result: dict[str, object] = {"dataset": "overall", "anomaly_class": "overall"}
    additive = ("n (evaluated)", "pos (GT)", "neg (GT)", "tp", "fp", "tn", "fn")
    if "f1_max_n_scored" in fieldnames:
        additive += ("f1_max_n_scored",)
    totals: dict[str, int] = {}
    for field in additive:
        totals[field] = int(sum(_as_number(row.get(field)) or 0 for row in representatives))
        result[field] = totals[field]

    confusion_total = totals["tp"] + totals["fp"] + totals["tn"] + totals["fn"]
    result["accuracy"] = (
        round((totals["tp"] + totals["tn"]) / confusion_total, 6)
        if confusion_total else ""
    )
    f1_denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    result["f1"] = round(2 * totals["tp"] / f1_denominator, 6) if f1_denominator else 0.0

    excluded = {"dataset", "anomaly_class", *additive, "accuracy", "f1"}
    for field in fieldnames:
        if field in excluded:
            continue
        if field == "f1_max_threshold":
            result[field] = ""
            continue
        values = [
            value for row in representatives
            if (value := _as_number(row.get(field))) is not None
        ]
        result[field] = round(sum(values) / len(values), 6) if values else ""
    return result


def _write_combined_csv(
    run_dir: Path,
    dataset_csv_paths: list[Path],
    include_rerun: bool,
) -> Path:
    """Combine per-dataset metric CSVs using the union of their columns.

    A method that is unavailable for one dataset simply has no corresponding
    keys in that dataset's rows; ``DictWriter`` therefore emits blank cells for
    those union columns in the combined file.
    """
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in dataset_csv_paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for fieldname in reader.fieldnames or []:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            rows.extend(reader)

    if not fieldnames:
        raise ValueError("Cannot combine metrics because no dataset CSV contained a header.")

    rows.append(_combined_overall_row(rows, fieldnames))

    prefix = "metrics_w_rerun" if include_rerun else "metrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(run_dir) / f"{prefix}_{timestamp}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _read_csv_union(paths: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    """Read CSV files while preserving the union of their columns."""
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for fieldname in reader.fieldnames or []:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            rows.extend(reader)
    return rows, fieldnames


def _secondary_overall_rows(
    rows: list[dict[str, str]], fieldnames: list[str]
) -> list[dict[str, object]]:
    """Build one cross-dataset overall row for each secondary class."""
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        class_name = str(row.get("secondary_class", "")).strip()
        if class_name:
            by_class.setdefault(class_name, []).append(row)

    overall_rows: list[dict[str, object]] = []
    additive = ("n (evaluated)", "pos (GT)", "neg (GT)", "tp", "fp", "tn", "fn")
    if "f1_max_n_scored" in fieldnames:
        additive += ("f1_max_n_scored",)
    excluded = {"dataset", "secondary_class", *additive, "accuracy", "f1"}
    for class_name, class_rows in by_class.items():
        result: dict[str, object] = {
            "dataset": "overall",
            "secondary_class": class_name,
        }
        totals = {
            field: int(sum(_as_number(row.get(field)) or 0 for row in class_rows))
            for field in additive
        }
        result.update(totals)
        confusion_total = totals["tp"] + totals["fp"] + totals["tn"] + totals["fn"]
        result["accuracy"] = (
            round((totals["tp"] + totals["tn"]) / confusion_total, 6)
            if confusion_total else ""
        )
        f1_denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
        result["f1"] = (
            round(2 * totals["tp"] / f1_denominator, 6)
            if f1_denominator else 0.0
        )
        for field in fieldnames:
            if field in excluded:
                continue
            if field == "f1_max_threshold":
                result[field] = ""
                continue
            values = [
                value for row in class_rows
                if (value := _as_number(row.get(field))) is not None
            ]
            result[field] = round(sum(values) / len(values), 6) if values else ""
        overall_rows.append(result)
    return overall_rows


def _write_combined_secondary_csv(
    run_dir: Path,
    dataset_csv_paths: list[Path],
    include_rerun: bool,
) -> Path:
    """Combine secondary class metrics and append an overall row per class."""
    rows, fieldnames = _read_csv_union(dataset_csv_paths)
    if not fieldnames:
        raise ValueError("Cannot combine secondary metrics because no CSV contained a header.")
    rows.extend(_secondary_overall_rows(rows, fieldnames))
    suffix = "_w_rerun" if include_rerun else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(run_dir) / f"secondary_image_metrics{suffix}_{timestamp}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_combined_secondary_confusion_csv(
    run_dir: Path,
    dataset_csv_paths: list[Path],
    include_rerun: bool,
) -> Path:
    """Combine secondary confusion matrices and sum each actual-class row."""
    rows, fieldnames = _read_csv_union(dataset_csv_paths)
    if not fieldnames:
        raise ValueError("Cannot combine secondary confusion matrices because no CSV contained a header.")
    predicted_fields = [field for field in fieldnames if field.startswith("predicted_")]
    by_actual: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        actual_class = str(row.get("actual_class", "")).strip()
        if actual_class:
            by_actual.setdefault(actual_class, []).append(row)
    for actual_class, class_rows in by_actual.items():
        rows.append({
            "dataset": "overall",
            "actual_class": actual_class,
            **{
                field: int(sum(_as_number(row.get(field)) or 0 for row in class_rows))
                for field in predicted_fields
            },
        })
    suffix = "_w_rerun" if include_rerun else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(run_dir) / f"secondary_confusion_matrix{suffix}_{timestamp}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def run_metrics_pipeline(
    metrics_config_path: str | None = None,
    run_dir_override: str | Path | None = None,
    combine: bool = False,
    omitted_metrics: set[str] | None = None,
    include_rerun_override: bool | None = None,
    include_secondary_override: bool | None = None,
) -> None:
    """
    Compute metrics/ROC plots for every dataset listed in
    configs/metrics_configs.yaml, reading straight from each dataset's
    existing run folder.
    """
    cfg_path = Path(metrics_config_path) if metrics_config_path else CONFIG_DIR / "metrics_configs.yaml"
    cfg = load_metrics_config(cfg_path, run_dir_override=run_dir_override)
    omitted_metrics = omitted_metrics or set()
    if include_rerun_override is not None:
        cfg = replace(
            cfg,
            datasets=[
                replace(dataset, include_rerun=include_rerun_override)
                for dataset in cfg.datasets
            ],
        )
    if include_secondary_override is False:
        cfg = replace(
            cfg,
            datasets=[replace(dataset, secondary_metrics=None) for dataset in cfg.datasets],
        )
    elif include_secondary_override is True and not any(
        dataset.secondary_metrics is not None for dataset in cfg.datasets
    ):
        raise ValueError(
            "--include-secondary was requested, but no dataset has enabled "
            "secondary_metrics configuration with ground-truth paths."
        )

    console = Console()
    # Keep detailed INFO diagnostics in the stage log.  The Rich progress
    # display below is the only routine terminal output; warnings and errors
    # remain visible so a failed metric stage is not silent.
    log_file = setup_logging(
        "metrics",
        log_dir=cfg.run_dir / "logs",
        console=console,
    )
    logger.info("Metrics config               : %s", cfg_path)
    logger.info("Run dir                      : %s", cfg.run_dir)
    logger.info("Datasets                     : %s", [(d.name) for d in cfg.datasets])
    logger.info("Datasets with include_rerun  : %s", [ d.name for d in cfg.datasets if d.include_rerun ])
    logger.info("Output subdir                : %s", cfg.output_subdir)
    logger.info("Combine dataset CSVs         : %s", combine)
    logger.info("Omitted optional metrics     : %s", sorted(omitted_metrics))
    logger.info(
        "Prediction-source CLI override: %s",
        "include rerun" if include_rerun_override is True
        else "base results" if include_rerun_override is False
        else "none (using YAML per dataset)",
    )
    logger.info("Log file                     : %s", log_file)

    dataset_csv_paths: list[Path] = []
    secondary_csv_paths: list[Path] = []
    secondary_confusion_paths: list[Path] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        dataset_task = progress.add_task("Datasets", total=len(cfg.datasets))

        for index, dataset in enumerate(cfg.datasets, start=1):
            # A fresh inner task per dataset makes its elapsed timer local to
            # that dataset instead of cumulative across the whole run.
            operation_task = progress.add_task(
                f"{dataset.name} · starting",
                total=None,
            )
            progress.update(
                dataset_task,
                description=f"Datasets: {index}/{len(cfg.datasets)} · {dataset.name}",
            )
            logger.info(
                "Metrics progress: %d/%d · %s preparing",
                index - 1,
                len(cfg.datasets),
                dataset.name,
            )
            logger.info("[%s] computing metrics …", dataset.name)
            csv_path = run_metrics_from_run_dir(
                run_dir=cfg.run_dir,
                dataset_entry=dataset,
                output_subdir=cfg.output_subdir,
                progress=progress,
                operation_task=operation_task,
                omitted_metrics=omitted_metrics,
            )
            if csv_path is not None:
                dataset_csv_paths.append(csv_path)
            if dataset.secondary_metrics is not None:
                suffix = "_w_rerun" if dataset.include_rerun else ""
                metrics_dir = cfg.run_dir / dataset.name / cfg.output_subdir
                secondary_path = max(
                    metrics_dir.glob(f"secondary_image_metrics{suffix}_*.csv"),
                    key=lambda path: (path.stat().st_mtime_ns, path.name),
                    default=None,
                )
                confusion_path = max(
                    metrics_dir.glob(f"secondary_confusion_matrix{suffix}_*.csv"),
                    key=lambda path: (path.stat().st_mtime_ns, path.name),
                    default=None,
                )
                if secondary_path is not None:
                    secondary_csv_paths.append(secondary_path)
                if confusion_path is not None:
                    secondary_confusion_paths.append(confusion_path)
            progress.update(dataset_task, advance=1)
            logger.info(
                "Metrics progress: %d/%d · %s complete",
                index,
                len(cfg.datasets),
                dataset.name,
            )
            progress.remove_task(operation_task)

    if combine:
        if not dataset_csv_paths:
            logger.warning("Combined metrics CSV was not written because no dataset produced metrics.")
        else:
            combined_path = _write_combined_csv(
                cfg.run_dir,
                dataset_csv_paths,
                include_rerun=any(dataset.include_rerun for dataset in cfg.datasets),
            )
            logger.info("Combined metrics CSV written to: %s", combined_path)
        if secondary_csv_paths:
            secondary_path = _write_combined_secondary_csv(
                cfg.run_dir,
                secondary_csv_paths,
                include_rerun=any(dataset.include_rerun for dataset in cfg.datasets),
            )
            logger.info("Combined secondary metrics CSV written to: %s", secondary_path)
        if secondary_confusion_paths:
            confusion_path = _write_combined_secondary_confusion_csv(
                cfg.run_dir,
                secondary_confusion_paths,
                include_rerun=any(dataset.include_rerun for dataset in cfg.datasets),
            )
            logger.info("Combined secondary confusion matrix written to: %s", confusion_path)

    logger.info("Metrics complete. Full log saved to: %s", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone metrics stage for anomaly-detection run folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics-config",
        default=None,
        metavar="PATH",
        help="Path to metrics_configs.yaml (default: configs/metrics_configs.yaml)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override run_dir from the metrics YAML.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Also write one combined CSV directly under run_dir. The union of "
            "all dataset columns is used, with blank cells for unavailable metrics, "
            "followed by a run-level overall row."
        ),
    )
    prediction_source = parser.add_mutually_exclusive_group()
    prediction_source.add_argument(
        "--include-rerun",
        action="store_true",
        help=(
            "Use rerun predictions for every configured dataset, overriding "
            "datasets[].include_rerun from the YAML."
        ),
    )
    secondary_mode = parser.add_mutually_exclusive_group()
    secondary_mode.add_argument(
        "--include-secondary",
        action="store_true",
        help="Run the secondary metrics enabled in the metrics YAML.",
    )
    secondary_mode.add_argument(
        "--binary-only",
        action="store_true",
        help="Skip secondary metrics even when they are enabled in the metrics YAML.",
    )
    prediction_source.add_argument(
        "--base-results",
        action="store_true",
        help=(
            "Use original all_data_summary predictions for every configured "
            "dataset, overriding datasets[].include_rerun from the YAML."
        ),
    )
    parser.add_argument(
        "--omit",
        nargs="+",
        choices=("aupro", "pixel-auroc", "image-auroc", "f1-max"),
        default=[],
        metavar="METRIC",
        help=(
            "Skip one or more optional calculations. Choices: aupro, "
            "pixel-auroc, image-auroc, f1-max. Example: --omit aupro pixel-auroc"
        ),
    )
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs" / "model_configs.yaml"),
        help="Shared deployment YAML used to select local/cluster telemetry",
    )
    parser.add_argument("--no-hardware-monitor", action="store_true", help="Disable hardware telemetry")
    parser.add_argument(
        "--hardware-monitor-interval", type=float, default=1.0, metavar="SECONDS",
        help="Hardware telemetry sampling interval",
    )
    args = parser.parse_args()
    include_rerun_override = (
        True if args.include_rerun
        else False if args.base_results
        else None
    )
    include_secondary_override = (
        True if args.include_secondary else False if args.binary_only else None
    )
    cfg_path = Path(args.metrics_config) if args.metrics_config else CONFIG_DIR / "metrics_configs.yaml"
    monitor_cfg = load_metrics_config(cfg_path, run_dir_override=args.run_dir)
    monitor = None
    if not args.no_hardware_monitor:
        telemetry_dir = monitor_cfg.run_dir / "logs" / f'hardware_telemetry_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        deployment = load_deployment_config(args.model_config)
        monitor = build_monitor(
            HardwareMonitor(
                output_dir=telemetry_dir,
                interval=args.hardware_monitor_interval,
                label="metrics",
            ),
            deployment,
            interval=args.hardware_monitor_interval,
            label="metrics",
            context={"pipeline": "metrics", "run_dir": str(monitor_cfg.run_dir)},
        )
        if not monitor.start():
            raise RuntimeError(f"Hardware telemetry failed to start: {telemetry_dir}")
        monitor.mark_event("pipeline_start", pipeline="metrics")

    with monitored(monitor):
        run_metrics_pipeline(
            str(cfg_path),
            run_dir_override=args.run_dir,
            combine=args.combine,
            omitted_metrics=set(args.omit),
            include_rerun_override=include_rerun_override,
            include_secondary_override=include_secondary_override,
        )
