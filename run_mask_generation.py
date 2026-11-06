"""CLI entry point for standalone post-inference mask generation.

Behavior change (2026-08): merged records are source-aware in
resume checks and output paths. Legacy runs containing overwritten
preprocessed images fail before mask metadata or model work is performed;
they must be regenerated as a new anomaly-detection run. Normal resume never
modifies a dataset that already has its completed summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from rich.console import Console

from anomaly_detection.config import MASK_METHODS, load_mask_generation_config
from anomaly_detection.hardware_monitor import HardwareMonitor
from anomaly_detection.logging_utils import setup_logging
from anomaly_detection.mask_generation import run_mask_generation
from cluster_deployment.config import load_deployment_config
from cluster_deployment.remote_monitor import build_monitor, monitored
from cluster_deployment.server_manager import managed_server_lifecycle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pixel masks from an existing anomaly-detection run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "mask_generation_configs.yaml"),
        help="Mask-generation YAML configuration",
    )
    parser.add_argument("--run-dir", help="Override run_dir from YAML")
    parser.add_argument("--dataset", action="append", help="Only process this dataset (repeatable)")
    parser.add_argument(
        "--methods", nargs="+", metavar="METHOD",
        help=f"Override methods for selected datasets (built-ins: {', '.join(MASK_METHODS)})",
    )
    prediction_source = parser.add_mutually_exclusive_group()
    prediction_source.add_argument(
        "--include-rerun", action="store_true",
        help="Use rerun predictions for the selected datasets",
    )
    prediction_source.add_argument(
        "--base-results", action="store_true",
        help="Use the original all_data_summary predictions",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing masks and box caches")
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs" / "model_configs.yaml"),
        help="Shared model/deployment YAML configuration",
    )
    parser.add_argument("--no-hardware-monitor", action="store_true", help="Disable hardware telemetry")
    parser.add_argument(
        "--hardware-monitor-interval", type=float, default=1.0, metavar="SECONDS",
        help="Hardware telemetry sampling interval",
    )
    args = parser.parse_args()

    try:
        run_dir_override = (
            str(Path(args.run_dir).expanduser().resolve()) if args.run_dir else None
        )
        cfg = load_mask_generation_config(
            args.config,
            run_dir_override=run_dir_override,
            dataset_names=set(args.dataset) if args.dataset else None,
            methods_override=tuple(args.methods) if args.methods else None,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.include_rerun or args.base_results:
        cfg = replace(cfg, datasets=[
            replace(dataset, include_rerun=args.include_rerun) for dataset in cfg.datasets
        ])
    if args.overwrite:
        cfg = replace(cfg, overwrite=True)

    console = Console()
    # A non-interactive invocation is a child of the ablation orchestrator,
    # which consumes the INFO progress records from stdout. Interactive runs
    # keep INFO in the file so logging does not interfere with Rich's live UI.
    log_file = setup_logging(
        "mask_generation",
        log_dir=cfg.run_dir / "logs",
        console=console,
        console_level=logging.WARNING if sys.stdout.isatty() else logging.INFO,
    )
    logging.getLogger(__name__).info(
        "Mask generation: run_dir=%s datasets=%s log_file=%s", cfg.run_dir,
        [(dataset.name, list(dataset.methods)) for dataset in cfg.datasets], log_file,
    )
    monitor = None
    if not args.no_hardware_monitor:
        telemetry_dir = cfg.run_dir / "logs" / f'hardware_telemetry_mask_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        deployment = load_deployment_config(args.model_config)
        monitor = build_monitor(
            HardwareMonitor(
                output_dir=telemetry_dir,
                interval=args.hardware_monitor_interval,
                label="mask_generation",
            ),
            deployment,
            interval=args.hardware_monitor_interval,
            label="mask_generation",
            context={"pipeline": "mask_generation", "run_dir": str(cfg.run_dir)},
        )
        if not monitor.start():
            raise RuntimeError(f"Hardware telemetry failed to start: {telemetry_dir}")
        monitor.mark_event("pipeline_start", pipeline="mask_generation")

    with monitored(monitor), managed_server_lifecycle("mask_generation"):
        run_mask_generation(cfg, console=console, model_config_path=args.model_config)


if __name__ == "__main__":
    main()
