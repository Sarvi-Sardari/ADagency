"""
run_anomaly_detection.py
=========================
Entry point — orchestrates the online anomaly-detection pipeline across one
or more datasets, using planners produced by the offline planner-generation
pipeline (run_planner_generation.py).

Pipeline stages
---------------
  1. Load anomaly-detection dataset config + model config from configs/
  2. Initialise the CLIP encoder (reference-image selection / class matching)
  3. Load the VLM decision maker (with per-step YES/NO log-probabilities)
  4. For every dataset listed in anomaly_detection_configs.yaml:
       a. Embed all train/good reference images with CLIP
       b. For every test image: find its nearest reference image, resolve the
          planner (single-class -> "good" planner; multi-class -> CLIP
          text-matched class planner), run the VLM decision maker, and record
          the result
       c. Write all_data_summary.jsonl

Behavior change (2026-08)
-------------------------
Merged anomaly groups now retain ``source_label_folder`` in preprocessed
artifact paths. Same-named files from different physical defect folders no
longer overwrite one another. This changes only merged artifact paths; the
grouping and prediction behavior are unchanged. Normal resume never changes
a valid completed dataset. Before skipping it, resume validates the expected
image set and all decision, probability, planner, and preprocessed-image
artifacts. Invalid completed state raises without rewriting it; an older run
with collisions must be replaced by a new run. Multi-class names select
planners only and never become folders.

Metrics and low-confidence reruns are separate, standalone stages — run
them any time afterwards with:
  python run_metrics.py   (configs/metrics_configs.yaml)
  python -m anomaly_detection.rerun <outdir> <dataset_name>
rerun.py first finds the optimum threshold (disk-only, no VLM/CLIP needed),
then loads the VLM to rerun low-confidence "good" predictions, writing
<outdir>/<dataset_name>/rerun_results/rerun_result.jsonl. Use --threshold-only
to skip the VLM rerun stage.

Usage
-----
  # From the pipeline_project/ directory:
  python run_anomaly_detection.py

  # Explicit config paths:
  python run_anomaly_detection.py \\
      --dataset-config configs/anomaly_detection_configs.yaml \\
      --model-config   configs/model_configs.yaml

Why a separate dataset config from planner_generation?
--------------------------------------------------------
See configs/anomaly_detection_configs.yaml's header comment for the full
reasoning; in short: this pipeline (a) runs many datasets per invocation
instead of one, and (b) needs inference-only fields such as
`merge_anomalies`, `classes`, `class_examples`, and a planners source
directory that do not belong to the planner-generation run schema.
configs/model_configs.yaml IS reused as-is, since both pipelines load the
same VLM.

Expected project layout
-----------------------
  pipeline_project/
  ├── configs/
  │   ├── anomaly_detection_configs.yaml
  │   └── model_configs.yaml
  ├── datasets/
  │   └── <dataset_name>/
  │       ├── train/good/
  │       └── test/...
  ├── anomaly_detection/
  │   ├── config.py
  │   ├── decision.py
  │   ├── logging_utils.py
  │   ├── metrics.py            ← used by the standalone run_metrics.py
  │   ├── rerun.py               ← threshold selection + low-confidence VLM rerun
  │   ├── models.py
  │   ├── pipeline.py
  │   ├── reference_selector.py
  │   └── utils.py
  ├── prompts/
  │   └── anomaly_detection_pipeline/
  │       └── decision_maker_v2_original_compact.j2
  ├── outputs/                  ← planners (read) + runs (written), created automatically
  └── run_anomaly_detection.py  ← this file

The decision prompt can be overridden per invocation with
``--decision-prompt-template``; the ablation orchestrator uses that option to
select a different prompt for each manifested run.
"""

import sys
import os

# Must be set before torch (imported transitively by models.py below)
# initializes CUDA. expandable_segments reduces the allocator fragmentation
# that builds up across hundreds of variable-length generate() calls in a
# single long-running process -- a common cause of crashes (up to and
# including native segfaults with some custom CUDA kernels) that only
# surface deep into a run, never on the first few images.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set CUDA_LAUNCH_BLOCKING=1 in the shell only while reproducing a CUDA
# failure. Forcing it in normal runs serializes every kernel and materially
# reduces generation throughput.

import faulthandler
import logging
import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Column, Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from anomaly_detection.config import load_anomaly_detection_config, load_models_config
from anomaly_detection.decision import resolve_decision_prompt_template
from anomaly_detection.hardware_monitor import HardwareMonitor
from anomaly_detection.logging_utils import setup_logging
from anomaly_detection.models import Encoder, vlm
from anomaly_detection.pipeline import entire_dataset_run
from cluster_deployment.config import load_deployment_config
from cluster_deployment.remote_monitor import build_monitor
from cluster_deployment.server_manager import ServerManager

CONFIG_DIR = PROJECT_ROOT / "configs"
logger = logging.getLogger(__name__)


def run_anomaly_detection_pipeline(
    dataset_config_path: str | None = None,
    model_config_path: str | None = None,
    resume_dir: str | None = None,
    hardware_monitor_enabled: bool = True,
    hardware_monitor_interval: float = 1.0,
    decision_prompt_template: str | None = None,
) -> None:
    """
    Run the online anomaly-detection pipeline end-to-end for every dataset
    listed in the config.

    Parameters
    ----------
    dataset_config_path:
        Path to ``anomaly_detection_configs.yaml``. Defaults to
        ``pipeline_project/configs/anomaly_detection_configs.yaml``.
    model_config_path:
        Path to ``model_configs.yaml``. Defaults to
        ``pipeline_project/configs/model_configs.yaml``.
    resume_dir:
        Path to a PREVIOUS run's output directory (the timestamped folder
        under ``outputs/runs/``) to continue in-place instead of creating a
        new one. A dataset whose ``all_data_summary.jsonl`` exists is skipped
        only after its expected images and referenced artifacts validate.
        Invalid completed state raises without modification. Within an
        interrupted dataset that has no completed summary yet, valid per-image
        decision/probability files are reused instead of re-calling the VLM.
    hardware_monitor_enabled:
        Start the independent, read-only CPU/GPU/thermal/kernel telemetry
        recorder. Enabled by default.
    hardware_monitor_interval:
        Seconds between telemetry samples. Values below 0.2 are clamped by
        the recorder to protect pipeline performance.
    decision_prompt_template:
        Optional prompt filename relative to the configured prompts_dir. Used
        by the ablation orchestrator to select a different decision prompt per
        run without editing the shared anomaly-detection YAML.
    """
    ablation_profile = os.environ.get("ANOMALY_ABLATION_PROFILE") == "1"
    ds_cfg_path = Path(dataset_config_path) if dataset_config_path else CONFIG_DIR / "anomaly_detection_configs.yaml"
    mdl_cfg_path = Path(model_config_path) if model_config_path else CONFIG_DIR / "model_configs.yaml"

    datacfg = load_anomaly_detection_config(ds_cfg_path)
    if decision_prompt_template is not None:
        selected_prompt_path = Path(decision_prompt_template)
        if (
            not decision_prompt_template.strip()
            or selected_prompt_path.is_absolute()
            or ".." in selected_prompt_path.parts
        ):
            raise ValueError(
                "decision_prompt_template must be a non-empty filename relative to prompts_dir"
            )
        datacfg = replace(
            datacfg,
            decision_prompt_template=decision_prompt_template.strip(),
        )
    selected_prompt_template = (
        datacfg.secondary_classification.prompt_template
        if datacfg.secondary_classification
        else resolve_decision_prompt_template(datacfg.decision_prompt_template)
    )
    prompt_path = datacfg.prompts_dir / selected_prompt_template
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Decision prompt template not found: {prompt_path}")
    modelcfg = load_models_config(mdl_cfg_path)

    if resume_dir:
        run_dir = Path(resume_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume-dir does not exist: {run_dir}")
    else:
        run_timestamp = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_'
        run_dir = datacfg.output.runs_dir / run_timestamp
        run_dir.mkdir(parents=True, exist_ok=False)

    console = Console()
    log_file = setup_logging("anomaly_detection", log_dir=run_dir / "logs", console=console)

    # Enable BEFORE loading the model: if the process dies from a native
    # crash (e.g. SIGSEGV) during model.generate(), this dumps every
    # thread's Python-level stack to a dedicated file right before the OS
    # kills the process -- the only way to know which call was in flight,
    # since a hard native crash produces no Python traceback/exception at
    # all. The file handle is deliberately kept open for the process
    # lifetime (a crash can't run cleanup code).
    faulthandler_log_path = run_dir / "logs" / f'faulthandler_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    faulthandler_file = faulthandler_log_path.open("w", encoding="utf-8")
    faulthandler.enable(file=faulthandler_file, all_threads=True)

    console.rule("[bold cyan]Anomaly Detection Pipeline")
    logger.info("Dataset config : %s", ds_cfg_path)
    logger.info("Model config   : %s", mdl_cfg_path)
    logger.info("Datasets       : %s", [d.name for d in datacfg.datasets])
    logger.info("Decision prompt: %s", selected_prompt_template)
    logger.info("Log file       : %s", log_file)
    logger.info("Faulthandler   : %s (captures the Python stack of every thread on a native crash)", faulthandler_log_path)
    if resume_dir:
        logger.info("Resuming run   : %s", run_dir)

    hardware_monitor = None
    telemetry_stop_reason = "pipeline_finished"
    if hardware_monitor_enabled:
        telemetry_dir = run_dir / "logs" / f'hardware_telemetry_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        deployment_cfg = load_deployment_config(mdl_cfg_path)
        hardware_monitor = build_monitor(
            HardwareMonitor(
                output_dir=telemetry_dir,
                interval=hardware_monitor_interval,
                label="anomaly_detection",
            ),
            deployment_cfg,
            interval=hardware_monitor_interval,
            label="anomaly_detection",
            context={"pipeline": "anomaly_detection", "run_dir": str(run_dir)},
        )
        if hardware_monitor.start():
            logger.info("Hardware data  : %s (%.2fs sampling; separate crash-resilient process)", telemetry_dir, hardware_monitor_interval)
            console.print(f"[bold]Hardware telemetry:[/bold] {telemetry_dir}")
            hardware_monitor.mark_event(
                "pipeline_start",
                run_dir=str(run_dir),
                dataset_config=str(ds_cfg_path),
                model_config=str(mdl_cfg_path),
                datasets=[dataset.name for dataset in datacfg.datasets],
                resume=bool(resume_dir),
            )
        else:
            logger.warning("Hardware telemetry process failed to start; inference will continue without it. See %s", telemetry_dir)
            hardware_monitor = None

    try:
        if hardware_monitor:
            hardware_monitor.mark_event("encoder_load_start")
        Encoder.loadEncoder()
        if hardware_monitor:
            hardware_monitor.mark_event("encoder_load_end")
            hardware_monitor.mark_event("vlm_load_start", model=modelcfg.vlm.modelname)
        vlmModel = vlm.load(modelcfg)
        if hardware_monitor:
            hardware_monitor.mark_event("vlm_load_end", model=modelcfg.vlm.modelname)

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None, table_column=Column(ratio=1)),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]avg generation: {task.fields[avg_inference_sec]}"),
        ]
        # Object-crop cost column, only when some dataset crops (keeps the
        # historical layout for ordinary runs).
        if any(
            dataset.object_crop is not None and dataset.object_crop.enabled
            for dataset in datacfg.datasets
        ):
            progress_columns.append(TextColumn(
                "[dim]avg crop: {task.fields[avg_crop_sec]} · "
                "gen+crop: {task.fields[avg_with_crop_sec]}"
            ))
        if ablation_profile:
            progress_columns.append(TextColumn(
                "[dim]output: {task.fields[avg_output_tokens]} tok · "
                "{task.fields[generated_tokens_per_second]} tok/s"
            ))
        progress_columns.append(TimeElapsedColumn())
        with Progress(
            *progress_columns,
            console=console,
            expand=True,
            transient=True,
            auto_refresh=False,
        ) as progress:
            task = progress.add_task(
                f"[1/{len(datacfg.datasets)}] {datacfg.datasets[0].name}",
                total=len(datacfg.datasets),
                avg_inference_sec="--",
                avg_crop_sec="--",
                avg_with_crop_sec="--",
                avg_output_tokens="--",
                generated_tokens_per_second="--",
            )
            progress.refresh()
            dataset_performance = []

            for i, dataset in enumerate(datacfg.datasets, start=1):
                progress.update(
                    task,
                    description=f"[{i}/{len(datacfg.datasets)}] {dataset.name}",
                    avg_inference_sec="--",
                    avg_crop_sec="--",
                    avg_with_crop_sec="--",
                    avg_output_tokens="--",
                    generated_tokens_per_second="--",
                )
                progress.refresh()
                logger.info("[Dataset %d/%d] %s …", i, len(datacfg.datasets), dataset.name)
                if hardware_monitor:
                    hardware_monitor.mark_event(
                        "dataset_start",
                        dataset=dataset.name,
                        dataset_number=i,
                        dataset_count=len(datacfg.datasets),
                    )

                result = entire_dataset_run(datacfg=datacfg,
                    modelcfg=modelcfg,
                    vlmModel=vlmModel,
                    encoder=Encoder,
                    dataset=dataset,
                    run_dir=run_dir,
                    progress=progress,
                    console=console,
                )
                performance = result["performance"]
                dataset_performance.append(performance)
                if hardware_monitor:
                    hardware_monitor.mark_event(
                        "dataset_end",
                        dataset=dataset.name,
                        dataset_number=i,
                        num_images=performance.get("num_images"),
                        wall_time_sec=performance.get("total_wall_time_sec"),
                        vlm_inference_time_sec=performance.get("total_vlm_inference_time_sec"),
                        object_crop_time_sec=performance.get("total_object_crop_time_sec"),
                        total_generated_tokens=performance.get("total_generated_tokens"),
                        generated_tokens_per_second=performance.get("generated_tokens_per_second"),
                    )
                progress.update(
                    task,
                    avg_inference_sec=(
                        f"{performance['avg_vlm_inference_time_per_image_sec']:.2f}s"
                        if performance["avg_vlm_inference_time_per_image_sec"] is not None
                        else "--"
                    ),
                    avg_crop_sec=(
                        f"{performance['avg_object_crop_time_per_image_sec']:.2f}s"
                        if performance.get("avg_object_crop_time_per_image_sec") is not None
                        else "--"
                    ),
                    avg_with_crop_sec=(
                        f"{performance['avg_vlm_inference_time_with_object_crop_per_image_sec']:.2f}s"
                        if performance.get("avg_vlm_inference_time_with_object_crop_per_image_sec") is not None
                        else "--"
                    ),
                    avg_output_tokens=(
                        f"{performance['avg_generated_tokens_per_image']:.1f}"
                        if performance.get("avg_generated_tokens_per_image") is not None
                        else "--"
                    ),
                    generated_tokens_per_second=(
                        f"{performance['generated_tokens_per_second']:.1f}"
                        if performance.get("generated_tokens_per_second") is not None
                        else "--"
                    ),
                )

                progress.update(task, advance=1)
                progress.refresh()

    except KeyboardInterrupt:
        telemetry_stop_reason = "pipeline_keyboard_interrupt"
        if hardware_monitor:
            hardware_monitor.mark_event("pipeline_keyboard_interrupt")
        logger.warning("Interrupted by user. Partial artifacts, if any, are preserved in %s", run_dir)
        vlm.release()
        Encoder.model = None
        Encoder.preprocess = None
        console.print(f"[yellow]Interrupted. Partial artifacts are in:[/yellow] {run_dir}")
        return
    except Exception:
        telemetry_stop_reason = "pipeline_python_exception"
        if hardware_monitor:
            hardware_monitor.mark_event("pipeline_python_exception", exception_type=sys.exc_info()[0].__name__ if sys.exc_info()[0] else None)
        # Uncaught exceptions otherwise bypass the logging module entirely --
        # Python's default excepthook prints the traceback only to stderr,
        # which is why previous crashes left no trace in the log file.
        # logger.exception() writes the full traceback to BOTH the log file
        # and the console, so it survives even if the terminal scrollback is
        # gone. Re-raise afterwards so the process still exits non-zero and
        # behaves exactly as before for any calling script.
        logger.exception("Pipeline crashed with an unhandled exception. Partial artifacts, if any, are preserved in %s", run_dir)
        vlm.release()
        Encoder.model = None
        Encoder.preprocess = None
        console.print(f"[red]Crashed. Partial artifacts are in:[/red] {run_dir}")
        raise
    finally:
        # Keep telemetry alive through model/server teardown; failures here
        # must not hide the pipeline's original exception or completion.
        try:
            vlm.release()
        except Exception:
            logger.exception("VLM release failed during pipeline cleanup")
            ServerManager.stop(tag="anomaly_detection:cleanup")
        Encoder.model = None
        Encoder.preprocess = None
        if hardware_monitor:
            hardware_monitor.mark_event("pipeline_main_loop_exit", reason=telemetry_stop_reason)
            hardware_monitor.stop(reason=telemetry_stop_reason)
            telemetry_archive = hardware_monitor.output_dir.parent / f"{hardware_monitor.output_dir.name}.tar.gz"
            if telemetry_archive.is_file():
                logger.info("IT telemetry archive: %s", telemetry_archive)
                console.print(f"[bold]IT telemetry archive:[/bold] {telemetry_archive}")

    console.rule("[bold green]Anomaly detection complete")
    total_vlm_inference_time_sec = sum(
        item["total_vlm_inference_time_sec"] for item in dataset_performance
    )
    total_generated_tokens = sum(
        item.get("total_generated_tokens", 0) for item in dataset_performance
    )
    performance_summary: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": dataset_performance,
        "total_wall_time_sec": round(sum(item["total_wall_time_sec"] for item in dataset_performance), 4),
        "total_vlm_inference_time_sec": round(total_vlm_inference_time_sec, 4),
    }
    if ablation_profile:
        performance_summary.update(
            total_generated_tokens=total_generated_tokens,
            generated_tokens_per_second=(
                round(total_generated_tokens / total_vlm_inference_time_sec, 4)
                if total_vlm_inference_time_sec > 0 else None
            ),
        )
    # Object-crop cost (only when some dataset cropped). Decision-only totals
    # above are unchanged so cropped and uncropped runs stay comparable.
    crop_performance = [
        item for item in dataset_performance if "num_images_with_object_crop" in item
    ]
    if crop_performance:
        crop_images = sum(item["num_images_with_object_crop_timing"] for item in crop_performance)
        crop_total = sum(item["total_object_crop_time_sec"] for item in crop_performance)
        combined_images = sum(
            item["num_images_with_vlm_timing_with_object_crop"] for item in crop_performance
        )
        combined_total = sum(
            item["total_vlm_inference_time_with_object_crop_sec"] for item in crop_performance
        )
        performance_summary.update(
            total_object_crop_time_sec=round(crop_total, 4),
            avg_object_crop_time_per_image_sec=(
                round(crop_total / crop_images, 4) if crop_images else None
            ),
            total_vlm_inference_time_with_object_crop_sec=round(combined_total, 4),
            avg_vlm_inference_time_with_object_crop_per_image_sec=(
                round(combined_total / combined_images, 4) if combined_images else None
            ),
        )
    performance_summary_path = run_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    performance_summary_path.write_text(json.dumps(performance_summary, indent=2, default=str), encoding="utf-8")

    performance_table = Table(title="Model-generation timing by dataset", expand=True)
    performance_table.add_column("Dataset")
    performance_table.add_column("Images", justify="right")
    performance_table.add_column("Wall time", justify="right")
    performance_table.add_column("Total generation", justify="right")
    performance_table.add_column("Avg generation / image", justify="right")
    if crop_performance:
        performance_table.add_column("Avg crop / image", justify="right")
        performance_table.add_column("Avg generation + crop / image", justify="right")
    if ablation_profile:
        performance_table.add_column("Avg output", justify="right")
        performance_table.add_column("Tokens/s", justify="right")
    for item in dataset_performance:
        average = item["avg_vlm_inference_time_per_image_sec"]
        table_row = [
            item["dataset"],
            str(item["num_images"]),
            f"{item['total_wall_time_sec']:.2f}s",
            f"{item['total_vlm_inference_time_sec']:.2f}s",
            f"{average:.2f}s" if average is not None else "--",
        ]
        if crop_performance:
            for field in (
                "avg_object_crop_time_per_image_sec",
                "avg_vlm_inference_time_with_object_crop_per_image_sec",
            ):
                value = item.get(field)
                table_row.append(f"{value:.2f}s" if value is not None else "--")
        if ablation_profile:
            table_row.extend((
                (
                    f"{item['avg_generated_tokens_per_image']:.1f} tok"
                    if item.get("avg_generated_tokens_per_image") is not None else "--"
                ),
                (
                    f"{item['generated_tokens_per_second']:.1f}"
                    if item.get("generated_tokens_per_second") is not None else "--"
                ),
            ))
        performance_table.add_row(*table_row)
    console.print(performance_table)
    console.print(f"[bold]Timing summary saved to:[/bold] {performance_summary_path}")
    logger.info("Runs saved to     : %s", run_dir)
    logger.info("Full log saved to : %s", log_file)
    console.print(f"[bold]Runs saved to:[/bold] {run_dir}")
    console.print(f"[bold]Full log saved to:[/bold] {log_file}\n")


def main() -> None:
    """Parse CLI arguments and run anomaly detection."""
    parser = argparse.ArgumentParser(
        description="Online anomaly-detection pipeline (multi-dataset) using pre-generated planners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        metavar="PATH",
        help="Path to anomaly_detection_configs.yaml (default: configs/anomaly_detection_configs.yaml)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        metavar="PATH",
        help="Path to model_configs.yaml (default: configs/model_configs.yaml)",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        metavar="PATH",
        help=(
            "Continue a previous run in-place (e.g. "
            "'outputs/runs/2026-08-11 14:18:26_') instead of starting a new "
            "timestamped run directory. Valid completed datasets are skipped; "
            "their summary and referenced artifacts are checked first. "
            "Already-decided images inside an interrupted dataset are reused."
        ),
    )
    parser.add_argument(
        "--no-hardware-monitor",
        action="store_true",
        help="Disable the automatic read-only CPU/GPU/thermal/kernel telemetry recorder.",
    )
    parser.add_argument(
        "--hardware-monitor-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Hardware telemetry sampling interval (minimum 0.2 seconds).",
    )
    parser.add_argument(
        "--decision-prompt-template",
        default=None,
        metavar="FILENAME",
        help=(
            "Override decision.prompt_template with a filename relative to "
            "the anomaly config's prompts_dir."
        ),
    )
    args = parser.parse_args()
    run_anomaly_detection_pipeline(
        args.dataset_config,
        args.model_config,
        args.resume_dir,
        hardware_monitor_enabled=not args.no_hardware_monitor,
        hardware_monitor_interval=args.hardware_monitor_interval,
        decision_prompt_template=args.decision_prompt_template,
    )
