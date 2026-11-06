"""CLI for the post-inference XAI mask-ablation study.

The command reads an existing anomaly-detection ``run_dir`` and publishes
metrics-compatible masks into that run.  It does not invoke or modify anomaly
detection, planner generation, rerun, standard mask generation, or metrics.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from mask_ablation.config import load_mask_ablation_config
from mask_ablation.logging_utils import (
    capture_terminal_output,
    open_terminal_stream,
    setup_logging,
)
from mask_ablation.pipeline import GENERATE_SMOOTHED_MASKS, run_mask_ablation_study
from mask_ablation.reports import write_per_image_reports
from anomaly_detection.hardware_monitor import HardwareMonitor


PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def _start_hardware_monitor(
    cfg: Any,
    *,
    interval: float,
    config_path: str,
    console: Console,
) -> HardwareMonitor:
    """Start the crash-resilient recorder before any XAI model is loaded."""
    telemetry_dir = (
        cfg.run_dir
        / cfg.diagnostics_subdir
        / f'hardware_telemetry_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    cpu_affinity = sorted(os.sched_getaffinity(0))
    monitor = HardwareMonitor(
        output_dir=telemetry_dir,
        interval=interval,
        label="xai_mask_ablation",
        watched_pid=os.getpid(),
        context={
            "pipeline": "xai_mask_ablation",
            "run_dir": str(cfg.run_dir),
            "config": str(Path(config_path).expanduser().resolve()),
            "datasets": [dataset.name for dataset in cfg.datasets],
            "models": [model.name for model in cfg.models],
            "methods": [method.name for method in cfg.methods],
            "cpu_affinity": cpu_affinity,
        },
    )
    if not monitor.start():
        raise RuntimeError(
            "XAI hardware telemetry failed to start; refusing to run without "
            f"crash diagnostics. See {telemetry_dir}. Use --no-hardware-monitor "
            "only when telemetry is intentionally disabled."
        )
    monitor.mark_event(
        "pipeline_start",
        pipeline="xai_mask_ablation",
        config=str(Path(config_path).expanduser().resolve()),
        cpu_affinity=cpu_affinity,
    )
    logger.info(
        "Hardware data: %s (%.2fs sampling; separate crash-resilient process)",
        telemetry_dir,
        interval,
    )
    console.print(f"[bold]Hardware telemetry:[/bold] {telemetry_dir}")
    return monitor


def _show_timing_summary(console: Console, summary_path: Path, log_file: Path) -> None:
    """Render one compact timing row per method across selected datasets."""
    with summary_path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    aggregated: dict[str, dict[str, Any]] = {}
    for record in records:
        method = record.get("method", "--")
        bucket = aggregated.setdefault(
            method,
            {
                "statuses": [],
                "generated": 0,
                "resumed": 0,
                "images": 0,
                "elapsed": 0.0,
            },
        )
        bucket["statuses"].append(record.get("status", "--"))
        bucket["generated"] += int(record.get("num_generated") or 0)
        bucket["resumed"] += int(record.get("num_resumed") or 0)
        bucket["images"] += int(record.get("num_images") or 0)
        bucket["elapsed"] += float(record.get("elapsed_sec") or 0.0)

    table = Table(title="XAI mask-ablation timing (all selected datasets)", expand=True)
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Masks", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("Avg / image", justify="right")
    for method, bucket in aggregated.items():
        statuses = bucket["statuses"]
        status = "ok" if statuses and all(value == "ok" for value in statuses) else "error"
        average = bucket["elapsed"] / bucket["images"] if bucket["images"] else None
        table.add_row(
            method,
            status,
            f"{bucket['generated']} new / {bucket['resumed']} resumed",
            f"{bucket['elapsed']:.1f}s",
            f"{average:.2f}s" if average is not None else "--",
        )
    console.print(table)
    console.print(f"[bold]Timing summary:[/bold] {summary_path}")
    console.print(f"[bold]Full log:[/bold] {log_file}")


def _run_metrics_quietly(
    cfg: Any, log_file: Path, generation_summary_path: Path, pixel_only: bool = False
) -> Path | None:
    """Run the existing metrics CLI with this study's run/dataset selection.

    This creates a small derived config under the XAI diagnostics directory;
    it does not change the user's metrics config or any metrics implementation.
    """
    with generation_summary_path.open(encoding="utf-8", newline="") as file:
        generation_rows = list(csv.DictReader(file))
    failed = [row for row in generation_rows if row.get("status") != "ok"]
    if failed:
        logger.error(
            "Skipping metrics because %d selected XAI dataset/method arm(s) failed; "
            "ranking partial method folders would be misleading.",
            len(failed),
        )
        return None

    source_path = PROJECT_ROOT / "configs" / "metrics_configs.yaml"
    if not source_path.is_file():
        logger.warning("No metrics config found at %s; skipping final XAI ranking.", source_path)
        return None
    with source_path.open(encoding="utf-8") as file:
        metrics_cfg = yaml.safe_load(file)
    if not isinstance(metrics_cfg, dict) or not isinstance(metrics_cfg.get("datasets"), list):
        logger.warning("Invalid metrics config at %s; skipping final XAI ranking.", source_path)
        return None

    selected = {dataset.name: dataset for dataset in cfg.datasets}
    datasets = []
    for item in metrics_cfg["datasets"]:
        if isinstance(item, dict) and item.get("name") in selected:
            entry = dict(item)
            entry["include_rerun"] = selected[entry["name"]].include_rerun
            datasets.append(entry)
    missing = sorted(set(selected) - {entry["name"] for entry in datasets})
    if missing:
        logger.warning("Metrics config lacks selected dataset(s) %s; skipping final XAI ranking.", missing)
        return None

    # The existing metrics stage discovers every directory under masks/. Older
    # XAI pilots may still be present there, so build a lightweight symlink
    # view containing only this study's configured method folders. This avoids
    # ranking stale/partial arms and avoids wasting AUPRO time on them, while
    # leaving the original run completely untouched.
    view_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metrics_view_root = (
        cfg.run_dir / cfg.diagnostics_subdir / "metrics_views" / view_stamp
    )
    for dataset in cfg.datasets:
        source_dataset = cfg.run_dir / dataset.name
        view_dataset = metrics_view_root / dataset.name
        view_dataset.mkdir(parents=True, exist_ok=True)
        (view_dataset / "all_data_summary.jsonl").symlink_to(
            source_dataset / "all_data_summary.jsonl"
        )
        source_rerun = source_dataset / "rerun_results"
        if source_rerun.is_dir():
            (view_dataset / "rerun_results").symlink_to(source_rerun, target_is_directory=True)
        source_metadata = source_dataset / "mask_generation"
        if source_metadata.is_dir():
            (view_dataset / "mask_generation").symlink_to(
                source_metadata, target_is_directory=True
            )
        view_masks = view_dataset / "masks"
        view_masks.mkdir()
        for model in cfg.models:
            for method in cfg.methods:
                folder_name = method.folder_name(model)
                source_method = source_dataset / "masks" / folder_name
                if not source_method.is_dir():
                    logger.error(
                        "Expected completed XAI mask folder is missing: %s",
                        source_method,
                    )
                    return None
                (view_masks / folder_name).symlink_to(
                    source_method, target_is_directory=True
                )
                if GENERATE_SMOOTHED_MASKS:
                    smoothed_method = source_dataset / "masks" / (
                        folder_name + "_smoothed"
                    )
                    if not smoothed_method.is_dir():
                        logger.error(
                            "Expected smoothed XAI mask folder is missing: %s",
                            smoothed_method,
                        )
                        return None
                    (view_masks / (folder_name + "_smoothed")).symlink_to(
                        smoothed_method, target_is_directory=True
                    )

    metrics_cfg["run_dir"] = str(metrics_view_root)
    metrics_cfg["datasets"] = datasets
    generated_config = cfg.run_dir / cfg.diagnostics_subdir / "metrics_xai_generated.yaml"
    generated_config.write_text(yaml.safe_dump(metrics_cfg, sort_keys=False), encoding="utf-8")
    started = time.time()
    metrics_command = [
        sys.executable,
        str(PROJECT_ROOT / "run_metrics.py"),
        "--metrics-config",
        str(generated_config),
        "--combine",
    ]
    if pixel_only:
        metrics_command.extend(["--omit", "aupro"])
    with log_file.open("a", encoding="utf-8", buffering=1) as stream:
        result = subprocess.run(
            metrics_command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode:
        logger.error("Metrics subprocess exited with code %d; see %s.", result.returncode, log_file)
        return None
    candidates = sorted(
        (
            path for path in metrics_view_root.glob("metrics*.csv")
            if path.stat().st_mtime >= started - 1
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        logger.warning("Metrics completed but produced no combined CSV; see %s.", log_file)
        return None
    return candidates[-1]


def _show_best_xai_metrics(
    console: Console, metrics_path: Path | None, cfg: Any
) -> None:
    if metrics_path is None:
        console.print("[yellow]XAI ranking unavailable; see the full log for the metrics-stage result.[/yellow]")
        return
    with metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    overall = next((row for row in rows if row.get("dataset") == "overall"), None)
    if overall is None:
        console.print(f"[yellow]No overall metrics row was written: {metrics_path}[/yellow]")
        return

    configured_xai_methods = {
        method.folder_name(model)
        for model in cfg.models
        for method in cfg.methods
    }
    if GENERATE_SMOOTHED_MASKS:
        configured_xai_methods |= {
            f"{method_name}_smoothed" for method_name in configured_xai_methods
        }
    ranked: list[tuple[str, float | None, float | None]] = []
    for column, value in overall.items():
        if not column.startswith("Pixel AUROC (xai_"):
            continue
        method = column.removeprefix("Pixel AUROC (").removesuffix(")")
        # Older pilot folders may still exist under masks/. Metrics discovers
        # them dynamically, but this study's ranking must include only the
        # methods selected in the current ablation configuration.
        if method not in configured_xai_methods:
            continue
        try:
            pixel_auroc = float(value) if value else None
        except ValueError:
            pixel_auroc = None
        try:
            aupro = float(overall.get(f"AUPRO ({method})", ""))
        except ValueError:
            aupro = None
        if pixel_auroc is not None or aupro is not None:
            ranked.append((method, pixel_auroc, aupro))
    if not ranked:
        console.print(f"[yellow]No XAI pixel metrics were available: {metrics_path}[/yellow]")
        return

    ranked.sort(key=lambda item: (item[2] is not None, item[2] or -1.0, item[1] or -1.0), reverse=True)
    best_method, best_pixel_auroc, best_aupro = ranked[0]
    table = Table(title="XAI mask ranking (overall)", expand=True)
    table.add_column("Method")
    table.add_column("Pixel AUROC", justify="right")
    table.add_column("AUPRO", justify="right")
    for method, pixel_auroc, aupro in ranked:
        table.add_row(
            method,
            f"{pixel_auroc:.4f}" if pixel_auroc is not None else "--",
            f"{aupro:.4f}" if aupro is not None else "--",
        )
    console.print(table)
    best_message = f"[bold green]Best XAI mask method:[/] {best_method}"
    if best_aupro is not None:
        best_message += f" (AUPRO {best_aupro:.4f})"
    console.print(best_message)
    console.print(f"[bold]Metrics summary:[/bold] {metrics_path}")


def _show_best_image_report(console: Console, report_paths: tuple[Path, Path] | None) -> None:
    if report_paths is None:
        return
    _, best_path = report_paths
    with best_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if rows:
        table = Table(title="Best image by dataset and XAI method", expand=True)
        for column in ("dataset", "method", "best_image", "pixel_auroc"):
            table.add_column(column.replace("_", " ").title(), justify="right" if column == "pixel_auroc" else "left")
        for row in rows:
            table.add_row(row["dataset"], row["method"], row["best_image"], f"{float(row['pixel_auroc']):.4f}")
        console.print(table)
    console.print(f"[bold]Per-image Pixel AUROC:[/bold] {report_paths[0]}")
    console.print(f"[bold]Best-image summary:[/bold] {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable XAI ablation masks for a completed anomaly-detection run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "mask_ablation_configs.yaml"),
        help="XAI mask-ablation YAML configuration",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Do not run the final Pixel AUROC/AUPRO ranking after mask generation",
    )
    parser.add_argument(
        "--pixel-only",
        action="store_true",
        help="Run final evaluation with Pixel AUROC only; omit AUPRO",
    )
    parser.add_argument("--run-dir", help="Override run_dir from YAML")
    parser.add_argument(
        "--dataset", action="append", help="Only run this configured dataset (repeatable)"
    )
    parser.add_argument(
        "--model", action="append", help="Only run this configured model (repeatable)"
    )
    parser.add_argument(
        "--method", action="append", help="Only run this configured method (repeatable)"
    )
    prediction_source = parser.add_mutually_exclusive_group()
    prediction_source.add_argument(
        "--include-rerun",
        action="store_true",
        help="Overlay rerun predictions/evidence for every selected dataset",
    )
    prediction_source.add_argument(
        "--base-results",
        action="store_true",
        help="Use only first-pass predictions/evidence",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate every selected final mask; required after changing a method signature",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Disable the YAML pilot limit/default selection and run the full "
            "configured dataset/method grid. Use only after checking pilot heatmaps."
        ),
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help=(
            "Capture step-level heatmaps/overlays/raw masks under run_dir/presentation. "
            "Use with --dataset, --method and a configured --pilot-images entry."
        ),
    )
    parser.add_argument(
        "--presentation-image",
        action="append",
        help="Image basename to capture (repeatable); limits presentation mode to these images.",
    )
    parser.add_argument(
        "--presentation-dir",
        help="Optional presentation output directory (default: <run_dir>/presentation).",
    )
    parser.add_argument(
        "--no-hardware-monitor",
        action="store_true",
        help="Disable the crash-resilient CPU/GPU/RAM/disk/kernel telemetry recorder.",
    )
    parser.add_argument(
        "--hardware-monitor-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Hardware telemetry sampling interval (minimum 0.2 seconds).",
    )
    args = parser.parse_args()

    include_rerun_override = None
    if args.include_rerun:
        include_rerun_override = True
    elif args.base_results:
        include_rerun_override = False

    try:
        cfg = load_mask_ablation_config(
            args.config,
            run_dir_override=args.run_dir,
            dataset_names=set(args.dataset) if args.dataset else None,
            model_names=set(args.model) if args.model else None,
            method_names=set(args.method) if args.method else None,
            include_rerun_override=include_rerun_override,
            overwrite_override=True if args.overwrite else None,
            full_run=args.full,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    if args.presentation:
        if not args.dataset or len(args.dataset) != 1:
            parser.error("--presentation requires exactly one --dataset")
        if not args.method:
            parser.error("--presentation requires --method")
        image_names = tuple(args.presentation_image or ())
        if not image_names:
            configured = cfg.pilot_images_by_dataset.get(args.dataset[0], ())
            image_names = tuple(configured)
        if not image_names:
            parser.error("--presentation requires --presentation-image or a pilot_images entry")
        presentation_root = Path(args.presentation_dir).expanduser().resolve() if args.presentation_dir else cfg.run_dir / "presentation"
        # Presentation is an opt-in diagnostic run.  It does not alter the
        # regular mask folders; the engine writes its intermediate products to
        # this separate root and the selected image scope keeps it small.
        cfg = replace(
            cfg,
            pilot_images_by_dataset={args.dataset[0]: image_names},
            methods=[
                replace(
                    method,
                    params={
                        **method.params,
                        "presentation_enabled": True,
                        "presentation_output_dir": str(presentation_root / args.dataset[0] / method.name),
                        # Keep the optimizer/fusion path identical to the
                        # normal run. Presentation frames are captured by
                        # the iteration callback from copies of the masks;
                        # enabling engine intermediate outputs would change
                        # phrase acceptance and final-map selection.
                        "save_intermediate_outputs": False,
                        "save_diagnostics": False,
                    },
                )
                for method in cfg.methods
            ],
            overwrite=True,
        )

    # Rich gets a duplicated terminal handle. stdout/stderr are redirected
    # below so notebook prints, transformers warnings and tqdm shard bars are
    # guaranteed to go to the log rather than leaking around this progress UI.
    terminal_stream = open_terminal_stream()
    console = Console(file=terminal_stream)
    log_file = setup_logging(
        "xai_mask_ablation",
        log_dir=cfg.run_dir / cfg.diagnostics_subdir / "logs",
        console=console,
    )
    hardware_monitor = None
    telemetry_stop_reason = "pipeline_finished"
    try:
        if not args.no_hardware_monitor:
            hardware_monitor = _start_hardware_monitor(
                cfg,
                interval=args.hardware_monitor_interval,
                config_path=args.config,
                console=console,
            )
        report_paths = None
        with capture_terminal_output(log_file):
            summary_path = run_mask_ablation_study(cfg, console=console)
            if args.no_evaluate or not cfg.evaluate_after_generation:
                metrics_path = None
            else:
                with console.status(
                    "[cyan]Evaluating Pixel AUROC and AUPRO for configured XAI methods…[/]"
                ):
                    metrics_path = _run_metrics_quietly(
                        cfg, log_file, summary_path, pixel_only=args.pixel_only
                    )
                # This is intentionally a companion report.  The existing
                # anomaly_detection metrics implementation and its CSV remain
                # unchanged; the report adds the image identity needed for
                # later visual/GIF selection.
                if metrics_path is not None and not args.no_evaluate:
                    report_paths = write_per_image_reports(
                        run_dir=cfg.run_dir,
                        diagnostics_subdir=cfg.diagnostics_subdir,
                        datasets=cfg.datasets,
                        models=cfg.models,
                        methods=cfg.methods,
                        metrics_config_path=PROJECT_ROOT / "configs" / "metrics_configs.yaml",
                        console=console,
                    )
        console.rule("[bold green]XAI mask ablation finished")
        _show_timing_summary(console, summary_path, log_file)
        _show_best_xai_metrics(console, metrics_path, cfg)
        _show_best_image_report(console, report_paths)
    except KeyboardInterrupt:
        telemetry_stop_reason = "pipeline_keyboard_interrupt"
        if hardware_monitor is not None:
            hardware_monitor.mark_event("pipeline_keyboard_interrupt")
        raise
    except BaseException as error:
        telemetry_stop_reason = "pipeline_exception"
        if hardware_monitor is not None:
            hardware_monitor.mark_event(
                "pipeline_exception", exception_type=type(error).__name__
            )
        raise
    finally:
        if hardware_monitor is not None:
            hardware_monitor.mark_event(
                "pipeline_main_exit", reason=telemetry_stop_reason
            )
            hardware_monitor.stop(reason=telemetry_stop_reason)
        terminal_stream.close()


if __name__ == "__main__":
    main()
