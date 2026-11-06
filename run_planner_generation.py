"""
run_planner_generation.py
=========================
Entry point — orchestrates the full offline planner-generation pipeline.

Pipeline stages
---------------
  1. Load dataset + model configs from  configs/
  2. Initialise the CLIP encoder                (reference image selection)
  3. Load VLM
  4. Select reference images                    (CLIP-based, reference.py)
  5. Resolve object names                       (VLM or cached .txt file)
  6. Generate per-image descriptions            (VLM, descriptor.py)
  7. Synthesise master descriptions             (LLM, descriptor.py)
  8. Generate planner steps                     (VLM, planner.py)

Usage
-----
  # From the pipeline_project/ directory:
  python run_planner_generation.py

  # Explicit config paths:
  python run_planner_generation.py \\
      --dataset-config configs/dataset_configs.yaml \\
      --model-config   configs/model_configs.yaml

Expected project layout
-----------------------
  pipeline_project/
  ├── configs/
  │   ├── dataset_configs.yaml
  │   └── model_configs.yaml
  ├── datasets/
  │   └── <dataset_name>/          ← dataset goes here
  │       ├── train/good/
  │       └── test/good/
  ├── planner_generation/
  │   ├── config.py
  │   ├── descriptor.py
  │   ├── models.py
  │   ├── object_namer.py
  │   ├── planner.py
  │   ├── reference.py
  │   └── utils.py
  ├── prompts/
  │   ├── description_text_summarizer.j2
  │   ├── normal_descriptor.j2
  │   ├── object_namer.j2
  │   ├── object_summarizer.j2
  │   └── planner.j2
  └── run_planner_generation.py    ← this file
"""

import json
import logging
import argparse
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Column
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

PROJECT_ROOT = Path(__file__).resolve().parent

from anomaly_detection.hardware_monitor import HardwareMonitor
from cluster_deployment.config import load_deployment_config
from cluster_deployment.server_manager import managed_server_lifecycle
from planner_generation.anomaly_rules import resolve_anomaly_rules
from planner_generation.config import load_dataset_config, load_models_config
from planner_generation.descriptor import master_description_generator, normal_description_generator
from planner_generation.logging_utils import setup_logging
from planner_generation.models import Encoder, llm, vlm
from planner_generation.object_namer import load_or_generate_object_names
from planner_generation.planner import planner_generator
from planner_generation.reference import reference_image_selector
from planner_generation.utils import clear_vram
from cluster_deployment.remote_monitor import build_monitor, monitored

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_DIR = PROJECT_ROOT / "configs"

STEPS = [
    "Initialising CLIP encoder",
    "Loading VLM",
    "Selecting reference images",
    "Resolving object names",
    "Generating image descriptions (VLM)",
    "Synthesising master descriptions (LLM)",
    "Resolving anomaly rules",
    "Generating planner steps (VLM)",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_planner_pipeline(
    dataset_config_path: str | None = None,
    model_config_path: str | None = None,
    hardware_monitor_enabled: bool = True,
    hardware_monitor_interval: float = 1.0,
) -> None:
    """
    Run the full offline planner-generation pipeline end-to-end.

    Parameters
    ----------
    dataset_config_path:
        Path to ``dataset_configs.yaml``.  Defaults to
        ``pipeline_project/configs/dataset_configs.yaml``.
    model_config_path:
        Path to ``model_configs.yaml``.  Defaults to
        ``pipeline_project/configs/model_configs.yaml``.
    """

    # ── 1. Load configs ───────────────────────────────────────────────────────
    ds_cfg_path  = Path(dataset_config_path) if dataset_config_path else CONFIG_DIR / "dataset_configs.yaml"
    mdl_cfg_path = Path(model_config_path)   if model_config_path   else CONFIG_DIR / "model_configs.yaml"

    datacfg  = load_dataset_config(ds_cfg_path)
    modelcfg = load_models_config(mdl_cfg_path)

    dataset_name   = datacfg.name
    is_multi_class = (datacfg.classes == "multi") #True when classes is 'multi'
    ref_folder     = datacfg.root_dir / "train" / "good"
    num_ref_imgs   = datacfg.reference_selection.number_of_images #For image description generation

    console = Console()
    log_file = setup_logging(
        dataset_name,
        log_dir=datacfg.output.reference_images_dir / "planner_generation_logs",
        console=console,
    )

    console.rule("[bold cyan]Planner Generation Pipeline")
    logger.info("Dataset config : %s", ds_cfg_path)
    logger.info("Model config   : %s", mdl_cfg_path)
    logger.info("Dataset  : %s", dataset_name)
    logger.info("Classes  : %s", datacfg.classes)
    logger.info("Ref imgs : %s per %s", num_ref_imgs, "class" if is_multi_class else "dataset")
    logger.info("Log file : %s", log_file)

    hardware_monitor = None
    if hardware_monitor_enabled:
        telemetry_dir = (
            datacfg.output.reference_images_dir
            / "planner_generation_logs"
            / f'hardware_telemetry_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        deployment_cfg = load_deployment_config(mdl_cfg_path)
        hardware_monitor = build_monitor(
            HardwareMonitor(
                output_dir=telemetry_dir,
                interval=hardware_monitor_interval,
                label="planner_generation",
            ),
            deployment_cfg,
            interval=hardware_monitor_interval,
            label="planner_generation",
            context={"pipeline": "planner_generation", "dataset": dataset_name},
        )
        if hardware_monitor.start():
            logger.info(
                "Hardware data  : %s (%.2fs sampling; separate crash-resilient process)",
                telemetry_dir, hardware_monitor_interval,
            )
            console.print(f"[bold]Hardware telemetry:[/bold] {telemetry_dir}")
            hardware_monitor.mark_event(
                "pipeline_start",
                dataset=dataset_name,
                dataset_config=str(ds_cfg_path),
                model_config=str(mdl_cfg_path),
            )
        else:
            logger.warning(
                "Hardware telemetry process failed to start; planner generation will "
                "continue without it. See %s", telemetry_dir,
            )
            hardware_monitor = None

    with monitored(hardware_monitor), managed_server_lifecycle("planner_generation"), Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        expand=True,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[1/{len(STEPS)}] {STEPS[0]}", total=len(STEPS))
        stage_durations: dict[str, float] = {}
        pipeline_start = time.perf_counter()

        # ── 2. Initialise CLIP encoder ────────────────────────────────────────
        logger.info("[Step 1/%d] %s …", len(STEPS), STEPS[0])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[0], step=1)
        stage_start = time.perf_counter()
        Encoder.loadEncoder()
        stage_durations[STEPS[0]] = time.perf_counter() - stage_start
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[0], step=1, duration_s=stage_durations[STEPS[0]])
        progress.update(task, advance=1, description=f"[2/{len(STEPS)}] {STEPS[1]}")

        # ── 3. Load VLM ────────────────────────────────────────────────────────
        logger.info("[Step 2/%d] %s …", len(STEPS), STEPS[1])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[1], step=2)
        stage_start = time.perf_counter()
        vlmModel = vlm.load(modelcfg)
        stage_durations[STEPS[1]] = time.perf_counter() - stage_start
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[1], step=2, duration_s=stage_durations[STEPS[1]])
        progress.update(task, advance=1, description=f"[3/{len(STEPS)}] {STEPS[2]}")

        # ── 4. Select reference images ────────────────────────────────────────
        logger.info("[Step 3/%d] %s …", len(STEPS), STEPS[2])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[2], step=3)
        stage_start = time.perf_counter()
        list_of_ref_img_filepath = reference_image_selector(dataconfig=datacfg,
                                                            modelcfg=modelcfg,
                                                            num_of_ref_img=num_ref_imgs,
                                                            ref_img_folder_path=ref_folder,
                                                            dataset_name=dataset_name,
                                                            is_class=is_multi_class,
                                                            class_labels_filepath=datacfg.class_labels_filepath if is_multi_class else None,
                                                            )
        stage_durations[STEPS[2]] = time.perf_counter() - stage_start
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[2], step=3, duration_s=stage_durations[STEPS[2]])
        logger.debug("Reference groups: %s", list_of_ref_img_filepath)
        progress.update(task, advance=1, description=f"[4/{len(STEPS)}] {STEPS[3]}")

        # ── 5. Resolve object names ───────────────────────────────────────────
        logger.info("[Step 4/%d] %s …", len(STEPS), STEPS[3])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[3], step=4)
        label_example_path = (datacfg.root_dir / "train" / "good" / datacfg.object_labels.label_example)
        # Pause the live progress display in case this step needs interactive
        # user input (label selection prompt) — input() and a live Rich display
        # don't mix well otherwise.
        progress.stop()
        stage_start = time.perf_counter()
        try:
            img_names = load_or_generate_object_names(
                datacfg=datacfg,
                modelcfg=modelcfg,
                vlmModel=vlmModel,
                label_example_img_path=label_example_path,
                PROJECT_ROOT=PROJECT_ROOT,
                console=console,
            )
            stage_durations[STEPS[3]] = time.perf_counter() - stage_start
        finally:
            progress.start()
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[3], step=4, duration_s=stage_durations[STEPS[3]])
        logger.info("Object names: %s", img_names)
        progress.update(task, advance=1, description=f"[5/{len(STEPS)}] {STEPS[4]}")

        # ── 6a. Generate per-image descriptions (VLM) ─────────────────────────
        logger.info("[Step 5/%d] %s …", len(STEPS), STEPS[4])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[4], step=5)
        stage_start = time.perf_counter()
        description_paths_by_class: dict[str, list[Path]] = {}
        preprocessed_image_paths_by_class: dict[str, list[Path]] = {}
        if is_multi_class:
            # list_of_ref_img_filepath is a list-of-lists: one inner list per class
            for class_label, img_path_group in zip(datacfg.class_labels, list_of_ref_img_filepath):
                description_paths, preprocessed_paths = normal_description_generator(
                    datacfg=datacfg,
                    modelcfg=modelcfg,
                    vlmModel=vlmModel,
                    list_of_ref_img_filepaths=img_path_group,
                    dataset_name=dataset_name,
                    is_class=True,
                    object_name_list=img_names,
                    class_label=class_label,
                )
                description_paths_by_class[class_label] = description_paths
                preprocessed_image_paths_by_class[class_label] = preprocessed_paths
        else:
            # list_of_ref_img_filepath is a flat list of paths
            description_paths, preprocessed_paths = normal_description_generator(
                datacfg=datacfg,
                modelcfg=modelcfg,
                vlmModel=vlmModel,
                list_of_ref_img_filepaths=list_of_ref_img_filepath,
                dataset_name=dataset_name,
                is_class=False,
                object_name_list=img_names,
            )
            description_paths_by_class["good"] = description_paths
            preprocessed_image_paths_by_class["good"] = preprocessed_paths
        logger.info(
            "Current-run description counts by class: %s",
            {label: len(paths) for label, paths in description_paths_by_class.items()},
        )
        stage_durations[STEPS[4]] = time.perf_counter() - stage_start
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[4], step=5, duration_s=stage_durations[STEPS[4]])
        progress.update(task, advance=1, description=f"[6/{len(STEPS)}] {STEPS[5]}")

        # ── 6b. Synthesise master descriptions (LLM) ──────────────────────────
        logger.info("[Step 6/%d] %s …", len(STEPS), STEPS[5])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[5], step=6)
        stage_start = time.perf_counter()
        clear_vram()
        llmModel=llm.load(modelcfg)
        master_description_paths_by_class = master_description_generator(
            datacfg=datacfg,
            modelcfg=modelcfg,
            llmModel=llmModel,
            dataset_name=dataset_name,
            img_names=img_names,
            description_paths_by_class=description_paths_by_class,
        )
        stage_durations[STEPS[5]] = time.perf_counter() - stage_start
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[5], step=6, duration_s=stage_durations[STEPS[5]])
        progress.update(task, advance=1, description=f"[7/{len(STEPS)}] {STEPS[6]}")

        # ── 6c. Resolve anomaly rules ──────────────────────────────────────────
        # Reviewed/approved rules are compiled here so planner generation always
        # consumes a single settled ANOMALY_RULES text — curation is not a
        # separate step the user runs on their own.
        logger.info("[Step 7/%d] %s …", len(STEPS), STEPS[6])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[6], step=7)
        stage_start = time.perf_counter()
        # Pause the live progress display in case this step needs interactive
        # user input (rule review / user-intent prompt), same as object naming.
        progress.stop()
        try:
            anomaly_rules_text, expected_rule_count = resolve_anomaly_rules(
                datacfg=datacfg,
                modelcfg=modelcfg,
                dataset_name=dataset_name,
                img_names=img_names,
                console=console,
            )
            stage_durations[STEPS[6]] = time.perf_counter() - stage_start
        finally:
            progress.start()
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[6], step=7, duration_s=stage_durations[STEPS[6]])
        logger.info(
            "Anomaly rules resolved (method=%s, expected_planner_steps=%s)",
            datacfg.anomaly_rules.method,
            expected_rule_count if expected_rule_count is not None else "legacy-untracked",
        )
        progress.update(task, advance=1, description=f"[8/{len(STEPS)}] {STEPS[7]}")

        # ── 6d. Generate planner steps (VLM) ──────────────────────────────────
        logger.info("[Step 8/%d] %s …", len(STEPS), STEPS[7])
        if hardware_monitor:
            hardware_monitor.mark_event("stage_start", stage=STEPS[7], step=8)
        stage_start = time.perf_counter()
        clear_vram()
        vlmModel=vlm.load(modelcfg)
        planner_generator(datacfg=datacfg,
                          modelcfg=modelcfg,
                          vlmModel=vlmModel,
                          dataset_name=dataset_name,
                          anomaly_rules_text=anomaly_rules_text,
                          master_description_paths_by_class=master_description_paths_by_class,
                          preprocessed_image_paths_by_class=preprocessed_image_paths_by_class,
                          expected_rule_count=expected_rule_count,
                          )
        stage_durations[STEPS[7]] = time.perf_counter() - stage_start
        if hardware_monitor:
            hardware_monitor.mark_event("stage_end", stage=STEPS[7], step=8, duration_s=stage_durations[STEPS[7]])
        progress.update(task, advance=1, description="Done")

    total_wall_time_sec = time.perf_counter() - pipeline_start
    performance = {
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "vlm": asdict(modelcfg.vlm),
            "llm": asdict(modelcfg.llm),
        },
        "total_wall_time_sec": round(total_wall_time_sec, 4),
        "stage_durations_sec": {name: round(duration, 4) for name, duration in stage_durations.items()},
    }
    performance_dir = datacfg.output.planners_dir / dataset_name / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    performance_path = performance_dir / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    performance_path.write_text(json.dumps(performance, indent=2, default=str), encoding="utf-8")
    logger.info("Performance summary written to %s", performance_path)

    console.rule(f"[bold green]Planner generated for {dataset_name}")
    logger.info("Planners saved to: %s", datacfg.output.planners_dir / dataset_name)
    logger.info("Full log saved to: %s", log_file)
    console.print(f"[bold]Planners saved to:[/bold] {datacfg.output.planners_dir / dataset_name}")
    console.print(f"[bold]Full log saved to:[/bold] {log_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline planner-generation pipeline for anomaly detection datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        metavar="PATH",
        help="Path to dataset_configs.yaml (default: configs/dataset_configs.yaml)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        metavar="PATH",
        help="Path to model_configs.yaml (default: configs/model_configs.yaml)",
    )
    parser.add_argument(
        "--no-hardware-monitor",
        action="store_true",
        help="Disable the crash-resilient hardware telemetry recorder.",
    )
    parser.add_argument(
        "--hardware-monitor-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Hardware telemetry sampling interval in seconds.",
    )
    args = parser.parse_args()
    run_planner_pipeline(
        args.dataset_config,
        args.model_config,
        hardware_monitor_enabled=not args.no_hardware_monitor,
        hardware_monitor_interval=args.hardware_monitor_interval,
    )
