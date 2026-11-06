#!/usr/bin/env python3
"""Run repeated anomaly-detection experiments without changing pipeline code.

The orchestrator deliberately treats every existing stage as a black-box CLI:

1. run_anomaly_detection.py for every repetition;
2. anomaly_detection/rerun.py for every repetition and dataset;
3. run_mask_generation.py for every repetition;
4. run_metrics.py for every repetition;
5. aggregate metrics, timings, and inference GPU telemetry across repetitions.

Run directories and stage completion are recorded in an atomic manifest under
the experiment's ``runs_dir``. Re-running the same command resumes incomplete
work. Existing pipeline artifacts retain their own finer-grained resume logic.

Example:

    python run_anomaly_detection_ablation.py \
        --config configs/anomaly_detection_ablation.example.yaml

    python run_anomaly_detection_ablation.py --config ... --runs 3
    python run_anomaly_detection_ablation.py --config ... --stages detection metrics
    python run_anomaly_detection_ablation.py --config ... --aggregate-only

The GPU report is intentionally scoped to anomaly-detection VLM inference. It
reuses the pipeline's existing telemetry and ``vlm_generate_start/end`` events,
so aggregation adds no sampler and no inference-time instrumentation overhead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from anomaly_detection.config import load_models_config
from anomaly_detection.hardware_monitor import HardwareMonitor
from pipeline_core.artifacts import atomic_write_text
from cluster_deployment.config import load_deployment_config
from cluster_deployment.remote_monitor import MonitoringError, build_monitor
from cluster_deployment.server_manager import EXTERNAL_SERVER_ENV, ServerManager

STAGE_ORDER = ("detection", "rerun", "masks", "metrics")
MANIFEST_NAME = "ablation_manifest.json"
SCHEMA_VERSION = 1
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class AblationError(RuntimeError):
    """A user-facing orchestration or artifact-validation failure."""


@dataclass(frozen=True)
class AblationConfig:
    source_path: Path
    experiment_name: str
    num_runs: int
    runs_dir: Path
    anomaly_config: Path
    model_config: Path
    mask_config: Path | None
    metrics_config: Path | None
    command_prefix: tuple[str, ...]
    # Optional per-run decision prompt sequence. One item applies to every
    # run; multiple items select item N for ablation run N.
    decision_prompt_sequence: tuple[str, ...]
    enabled_stages: tuple[str, ...]
    hardware_monitor_interval: float
    process_gpu_telemetry: bool
    generation_profile: bool
    rerun_num_steps: int
    rerun_policy: str
    mask_include_rerun: bool
    metrics_include_rerun: bool
    metrics_before_rerun: bool
    metrics_combine: bool
    metrics_omit: tuple[str, ...]
    std_ddof: int
    cooldown_seconds: float


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "learning"
    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class EtaTracker:
    """Adaptive wall-clock ETA based only on this experiment's observations."""

    def __init__(self, pending_keys: Sequence[str], samples: dict[str, list[float]]):
        self.pending_keys = list(pending_keys)
        self.samples = {key: list(values) for key, values in samples.items()}
        self.current_key: str | None = None
        self.current_label = "preparing"
        self.current_started: float | None = None
        self.current_fraction: float | None = None
        self.current_projected_duration: float | None = None

    @staticmethod
    def _generic_key(key: str) -> str:
        return key.split(":", 1)[0]

    def start(self, key: str, label: str) -> None:
        self.current_key = key
        self.current_label = label
        self.current_started = time.monotonic()
        self.current_fraction = None
        self.current_projected_duration = self.expected_duration(key)

    def observe_fraction(self, fraction: float) -> None:
        if self.current_started is None or not 0 < fraction <= 1:
            return
        # Progress displays can briefly redraw an older line. Never move the
        # estimator backwards within one child process.
        if self.current_fraction is not None and fraction < self.current_fraction:
            return
        self.current_fraction = fraction
        elapsed = time.monotonic() - self.current_started
        projection = elapsed / fraction
        if projection >= elapsed:
            self.current_projected_duration = projection

    def expected_duration(self, key: str) -> float | None:
        exact = self.samples.get(key, [])
        if exact:
            return statistics.fmean(exact)
        generic = self.samples.get(self._generic_key(key), [])
        if generic:
            return statistics.fmean(generic)
        if self.current_key == key and self.current_projected_duration is not None:
            return self.current_projected_duration
        return None

    def observe_output(self, line: str, dataset_counts: dict[str, int]) -> None:
        clean = _clean_line(line)
        if not clean or self.current_key is None:
            return
        stage = self._generic_key(self.current_key)
        if stage == "detection":
            total_images = sum(dataset_counts.values())
            if total_images <= 0:
                return
            dataset = next(
                (name for name in sorted(dataset_counts, key=len, reverse=True) if name in clean),
                None,
            )
            pairs = re.findall(r"\[(\d+)\s*/\s*(\d+)\]", clean)
            if dataset is not None and pairs:
                image_number, image_total = map(int, pairs[-1])
                if image_total == dataset_counts.get(dataset) and image_number <= image_total:
                    completed_before = 0
                    for name, count in dataset_counts.items():
                        if name == dataset:
                            break
                        completed_before += count
                    self.observe_fraction((completed_before + image_number) / total_images)
        elif stage == "rerun":
            pairs = re.findall(r"\[(\d+)\s*/\s*(\d+)\]", clean)
            if pairs:
                completed, total = map(int, pairs[-1])
                if total and completed <= total:
                    self.observe_fraction(completed / total)
        elif stage == "metrics":
            match = re.search(r"Datasets\s*\[(\d+)\s*/\s*(\d+)\]", clean)
            if match is None:
                match = re.search(
                    r"Metrics\s+progress\s*:\s*(\d+)\s*/\s*(\d+)",
                    clean,
                )
            if match:
                completed, total = map(int, match.groups())
                if total and completed <= total:
                    self.observe_fraction(completed / total)

    def finish(self, key: str, duration: float) -> None:
        if self.pending_keys:
            try:
                self.pending_keys.remove(key)
            except ValueError:
                pass
        if duration > 0:
            self.samples.setdefault(key, []).append(duration)
            generic = self._generic_key(key)
            if generic != key:
                self.samples.setdefault(generic, []).append(duration)
        self.current_key = None
        self.current_started = None
        self.current_fraction = None
        self.current_projected_duration = None

    def estimate(self) -> tuple[float | None, int]:
        estimate = 0.0
        unknown = 0
        current_elapsed = (
            time.monotonic() - self.current_started
            if self.current_started is not None else 0.0
        )
        keys = list(self.pending_keys)
        if self.current_key is not None and self.current_key not in keys:
            keys.insert(0, self.current_key)
        current_consumed = False
        for key in keys:
            expected = self.expected_duration(key)
            if expected is None:
                unknown += 1
                continue
            is_current = key == self.current_key and not current_consumed
            if is_current:
                estimate += max(0.0, expected - current_elapsed)
                current_consumed = True
            else:
                estimate += expected
        return (estimate if estimate or unknown == 0 else None), unknown

    def display_fields(self) -> tuple[str, str]:
        seconds, unknown = self.estimate()
        if seconds is None:
            return "learning", "learning"
        eta = _format_duration(seconds)
        finish_time = (
            datetime.now().astimezone() + timedelta(seconds=seconds)
        ).strftime("%a %d %b %H:%M")
        if unknown:
            return f"≥ {eta} (+{unknown} learning)", f"≥ {finish_time}"
        return eta, finish_time


class ProcessGpuSampler:
    """Low-overhead NVML sampling for exactly one pipeline PID.

    NVML exposes process memory and, on supported drivers, process SM/memory
    engine utilization. Power, temperature, clocks, and energy are not
    process-attributable and remain in the existing device-wide telemetry.
    """

    FIELDNAMES = (
        "timestamp_local", "timestamp_utc", "pid", "gpu_index",
        "process_gpu_memory_mib", "process_sm_utilization_pct",
        "process_memory_utilization_pct", "process_encoder_utilization_pct",
        "process_decoder_utilization_pct",
    )

    def __init__(self, path: Path, pid: int, gpu_index: int, interval: float):
        self.path = path
        self.pid = pid
        self.gpu_index = gpu_index
        self.interval = max(0.2, interval)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run,
            name=f"process-gpu-sampler-{self.pid}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval * 3))

    @staticmethod
    def _valid_memory(raw: Any, unavailable: Any) -> float | None:
        value = _number(raw)
        if value is None or raw == unavailable or value < 0 or value > 2**60:
            return None
        return value / 1024 / 1024

    def _run(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
            last_util_timestamp = int(time.time() * 1_000_000)
            with self.path.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=self.FIELDNAMES)
                writer.writeheader()
                file_obj.flush()
                while not self.stop_event.is_set():
                    now = datetime.now().astimezone()
                    row: dict[str, Any] = {
                        "timestamp_local": now.isoformat(timespec="milliseconds"),
                        "timestamp_utc": now.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
                        "pid": self.pid,
                        "gpu_index": self.gpu_index,
                    }
                    memories: list[float] = []
                    # Query compute and graphics contexts independently. Within
                    # each category, prefer v3 and use the legacy call only as
                    # a compatibility fallback.
                    for function_names in (
                        (
                            "nvmlDeviceGetComputeRunningProcesses_v3",
                            "nvmlDeviceGetComputeRunningProcesses",
                        ),
                        (
                            "nvmlDeviceGetGraphicsRunningProcesses_v3",
                            "nvmlDeviceGetGraphicsRunningProcesses",
                        ),
                    ):
                        for function_name in function_names:
                            function = getattr(pynvml, function_name, None)
                            if function is None:
                                continue
                            try:
                                processes = function(handle)
                            except pynvml.NVMLError:
                                continue
                            for process in processes:
                                if int(getattr(process, "pid", -1)) != self.pid:
                                    continue
                                memory = self._valid_memory(
                                    getattr(process, "usedGpuMemory", None), unavailable
                                )
                                if memory is not None:
                                    memories.append(memory)
                            break
                    if memories:
                        row["process_gpu_memory_mib"] = max(memories)

                    utilization_samples: list[Any] = []
                    process_utilization = getattr(pynvml, "nvmlDeviceGetProcessUtilization", None)
                    if process_utilization is not None:
                        try:
                            utilization_samples = process_utilization(handle, last_util_timestamp)
                        except pynvml.NVMLError:
                            utilization_samples = []
                    matching = [
                        sample for sample in utilization_samples
                        if int(getattr(sample, "pid", -1)) == self.pid
                    ]
                    if utilization_samples:
                        timestamps = [
                            int(getattr(sample, "timeStamp", 0))
                            for sample in utilization_samples
                            if int(getattr(sample, "timeStamp", 0)) > 0
                        ]
                        if timestamps:
                            last_util_timestamp = max(last_util_timestamp, max(timestamps))
                    if matching:
                        latest = max(matching, key=lambda sample: int(getattr(sample, "timeStamp", 0)))
                        for source, destination in (
                            ("smUtil", "process_sm_utilization_pct"),
                            ("memUtil", "process_memory_utilization_pct"),
                            ("encUtil", "process_encoder_utilization_pct"),
                            ("decUtil", "process_decoder_utilization_pct"),
                        ):
                            value = _number(getattr(latest, source, None))
                            if value is not None:
                                row[destination] = value

                    writer.writerow(row)
                    file_obj.flush()
                    self.stop_event.wait(self.interval)
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
        except Exception as exc:  # measurement must never stop an ablation
            self.error = f"{type(exc).__name__}: {exc}"
            self.path.with_suffix(".error.txt").write_text(self.error + "\n", encoding="utf-8")


class DetectionEventWatcher:
    """Read detection JSONL events without relying on child terminal redraws."""

    def __init__(
        self,
        logs_dir: Path,
        dataset_counts: dict[str, int],
        generation_profile: bool,
    ):
        self.logs_dir = logs_dir
        self.dataset_counts = dataset_counts
        self.generation_profile = generation_profile
        self.offsets: dict[Path, int] = {}
        self.completed: dict[str, set[str]] = defaultdict(set)
        self.inference_seconds: dict[str, dict[str, float]] = defaultdict(dict)
        self.generated_tokens: dict[str, dict[str, int]] = defaultdict(dict)
        self.detail = "loading models"

    @property
    def completed_images(self) -> int:
        return sum(len(images) for images in self.completed.values())

    @property
    def total_images(self) -> int:
        return sum(self.dataset_counts.values())

    def poll(self, eta_tracker: EtaTracker) -> str:
        paths = sorted(self.logs_dir.glob("hardware_telemetry_*/pipeline_events.jsonl"))
        for path in paths:
            try:
                with path.open(encoding="utf-8") as file_obj:
                    file_obj.seek(self.offsets.get(path, 0))
                    for raw in file_obj:
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        kind = event.get("event")
                        decision_filepath = event.get("decision_filepath")
                        dataset = _dataset_from_decision_path(
                            decision_filepath, tuple(self.dataset_counts)
                        )
                        if dataset is None or kind not in {
                            "vlm_generate_start", "vlm_generate_end",
                        }:
                            continue
                        identity = str(decision_filepath).replace("\\", "/")
                        if kind == "vlm_generate_end":
                            self.completed[dataset].add(identity)
                            elapsed = _number(event.get("elapsed_sec"))
                            if elapsed is not None and elapsed >= 0:
                                self.inference_seconds[dataset][identity] = elapsed
                            generated_tokens = _number(event.get("generated_token_count"))
                            if generated_tokens is not None and generated_tokens >= 0:
                                self.generated_tokens[dataset][identity] = int(generated_tokens)
                            completed = len(self.completed[dataset])
                            total = self.dataset_counts.get(dataset, 0)
                            eta_tracker.observe_output(
                                f"{dataset} [{completed}/{total}]", self.dataset_counts
                            )
                        else:
                            completed = len(self.completed[dataset])
                            total = self.dataset_counts.get(dataset, 0)
                        filename = Path(identity).name.removesuffix(".txt")
                        position = min(completed + (kind == "vlm_generate_start"), total)
                        timings = self.inference_seconds.get(dataset, {}).values()
                        average = (
                            f"{statistics.fmean(timings):.2f}s/image"
                            if timings else "learning"
                        )
                        token_values = self.generated_tokens.get(dataset, {})
                        average_tokens = (
                            f"{statistics.fmean(token_values.values()):.1f} tok"
                            if token_values else "--"
                        )
                        token_seconds = sum(
                            self.inference_seconds[dataset][image_identity]
                            for image_identity in token_values
                            if image_identity in self.inference_seconds[dataset]
                        )
                        tokens_per_second = (
                            f"{sum(token_values.values()) / token_seconds:.1f} tok/s"
                            if token_seconds > 0 else "--"
                        )
                        self.detail = (
                            f"{dataset} · image {position}/{total} · {filename}"
                            f" · avg generation {average}"
                        )
                        if self.generation_profile:
                            self.detail += (
                                f" · avg output {average_tokens} · {tokens_per_second}"
                            )
                    self.offsets[path] = file_obj.tell()
            except OSError:
                continue
        return self.detail


def _child_progress_from_output(line: str) -> tuple[int, int, str] | None:
    """Extract newline-based child progress for rerun, masks, and metrics."""
    clean = _clean_line(line)
    if not clean:
        return None
    patterns = (
        r"Mask\s+progress\s+\[(\d+)\s*/\s*(\d+)\]",
        r"\[(?:boxes|segmentation)\s+(\d+)\s*/\s*(\d+)\]",
        r"Datasets\s*\[(\d+)\s*/\s*(\d+)\]",
        r"Metrics\s+progress\s*:\s*(\d+)\s*/\s*(\d+)",
        r"\[(\d+)\s*/\s*(\d+)\]",
    )
    for pattern in patterns:
        matches = re.findall(pattern, clean, flags=re.IGNORECASE)
        if matches:
            current, total = map(int, matches[-1])
            if total > 0 and 0 <= current <= total:
                return current, total, clean[-120:]
    return None


def _mask_method_progress_from_output(
    line: str,
) -> tuple[str, int, int, str] | None:
    """Extract method-specific mask progress emitted by mask_generation.py."""
    clean = _clean_line(line)
    match = re.search(
        r"Mask\s+method\s+progress(?:\s*:\s*|\s*\[)"
        r"(bounding_box_segmentation|bounding_box|segmentation)\s+"
        r"(\d+)\s*/\s*(\d+)\]?",
        clean,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    method = match.group(1).lower()
    completed, total = int(match.group(2)), int(match.group(3))
    if total <= 0 or completed < 0 or completed > total:
        return None
    detail = clean[match.end():].lstrip(" ·-") or "working"
    return method, completed, total, detail[-120:]


def _resolve_path(value: Any, *, base: Path, field: str, required: bool = True) -> Path | None:
    if value is None:
        if required:
            raise AblationError(f"Missing required configuration field: {field}")
        return None
    if not isinstance(value, str) or not value.strip():
        raise AblationError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Pipeline configs are conventionally project-relative. A config kept
        # elsewhere can still use an absolute path without ambiguity.
        path = PROJECT_ROOT / path
    return path.resolve()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AblationError(f"{field} must be a YAML mapping")
    return value


def load_ablation_config(path: Path) -> AblationConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise AblationError(f"Ablation config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AblationError(f"Ablation config must contain a YAML mapping: {path}")

    experiment = _mapping(raw.get("experiment"), "experiment")
    configs = _mapping(raw.get("configs"), "configs")
    stages = _mapping(raw.get("stages"), "stages")
    rerun = _mapping(raw.get("rerun"), "rerun")
    masks = _mapping(raw.get("masks"), "masks")
    metrics = _mapping(raw.get("metrics"), "metrics")
    telemetry = _mapping(raw.get("telemetry"), "telemetry")
    aggregation = _mapping(raw.get("aggregation"), "aggregation")

    name = str(experiment.get("name", "")).strip()
    if not name:
        raise AblationError("experiment.name must be non-empty")
    num_runs = int(experiment.get("num_runs", 10))
    if num_runs < 1:
        raise AblationError("experiment.num_runs must be at least 1")

    runs_dir = _resolve_path(experiment.get("runs_dir"), base=path.parent, field="experiment.runs_dir")
    anomaly_config = _resolve_path(
        configs.get("anomaly_detection"), base=path.parent, field="configs.anomaly_detection"
    )
    model_config = _resolve_path(configs.get("model"), base=path.parent, field="configs.model")
    mask_config = _resolve_path(
        configs.get("mask_generation"), base=path.parent,
        field="configs.mask_generation", required=False,
    )
    metrics_config = _resolve_path(
        configs.get("metrics"), base=path.parent, field="configs.metrics", required=False,
    )

    command_prefix_raw = experiment.get("command_prefix", [])
    if not isinstance(command_prefix_raw, list) or not all(isinstance(item, str) for item in command_prefix_raw):
        raise AblationError("experiment.command_prefix must be a list of strings")

    prompt_sequence_raw = experiment.get("decision_prompt_sequence", [])
    if prompt_sequence_raw is None:
        prompt_sequence_raw = []
    if not isinstance(prompt_sequence_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in prompt_sequence_raw
    ):
        raise AblationError(
            "experiment.decision_prompt_sequence must be a list of non-empty strings"
        )
    prompt_sequence: tuple[str, ...] = tuple(item.strip() for item in prompt_sequence_raw)
    for template in prompt_sequence:
        template_path = Path(template)
        if template_path.is_absolute() or ".." in template_path.parts:
            raise AblationError(
                "experiment.decision_prompt_sequence entries must be filenames "
                "relative to prompts_dir"
            )

    enabled = tuple(stage for stage in STAGE_ORDER if bool(stages.get(stage, True)))
    if "masks" in enabled and mask_config is None:
        raise AblationError("configs.mask_generation is required when stages.masks is enabled")
    if "metrics" in enabled and metrics_config is None:
        raise AblationError("configs.metrics is required when stages.metrics is enabled")

    omit = metrics.get("omit", ["pixel-auroc", "aupro"])
    if not isinstance(omit, list) or not all(item in {"pixel-auroc", "aupro", "image-auroc"} for item in omit):
        raise AblationError("metrics.omit must contain only pixel-auroc, aupro, or image-auroc")

    std_ddof = int(aggregation.get("std_ddof", 1))
    if std_ddof not in {0, 1}:
        raise AblationError("aggregation.std_ddof must be 0 or 1")

    rerun_policy = str(rerun.get("policy", "binary-gate")).strip()
    if rerun_policy not in {"binary-gate", "class-aware", "global-secondary"}:
        raise AblationError(
            "rerun.policy must be binary-gate, class-aware, or global-secondary"
        )
    if rerun_policy == "global-secondary" and metrics_config is None:
        raise AblationError(
            "configs.metrics is required when rerun.policy is global-secondary "
            "because it supplies secondary class labels for threshold calibration"
        )

    metrics_before_rerun = bool(metrics.get("before_rerun", False))
    if metrics_before_rerun and not {"rerun", "masks", "metrics"}.issubset(enabled):
        raise AblationError(
            "metrics.before_rerun=true requires rerun, masks, and metrics stages"
        )

    return AblationConfig(
        source_path=path,
        experiment_name=name,
        num_runs=num_runs,
        runs_dir=runs_dir,
        anomaly_config=anomaly_config,
        model_config=model_config,
        mask_config=mask_config,
        metrics_config=metrics_config,
        command_prefix=tuple(command_prefix_raw),
        decision_prompt_sequence=prompt_sequence,
        enabled_stages=enabled,
        hardware_monitor_interval=max(0.2, float(telemetry.get("interval_seconds", 1.0))),
        process_gpu_telemetry=bool(telemetry.get("process_specific", True)),
        generation_profile=bool(telemetry.get("generation_profile", True)),
        rerun_num_steps=max(1, int(rerun.get("num_steps", 3))),
        rerun_policy=rerun_policy,
        mask_include_rerun=bool(masks.get("include_rerun", True)),
        metrics_include_rerun=bool(metrics.get("include_rerun", True)),
        metrics_before_rerun=metrics_before_rerun,
        metrics_combine=bool(metrics.get("combine", True)),
        metrics_omit=tuple(omit),
        std_ddof=std_ddof,
        cooldown_seconds=max(0.0, float(experiment.get("cooldown_seconds", 0.0))),
    )


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AblationError(f"{label} config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AblationError(f"{label} config must contain a YAML mapping: {path}")
    return payload


def dataset_names_from_config(path: Path) -> list[str]:
    payload = _load_yaml_mapping(path, "Anomaly-detection")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise AblationError(f"No datasets were configured in {path}")
    names: list[str] = []
    for index, entry in enumerate(datasets, start=1):
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise AblationError(f"datasets[{index}] in {path} has no valid name")
        name = str(entry["name"]).strip()
        if name in names:
            raise AblationError(f"Duplicate dataset name {name!r} in {path}")
        names.append(name)
    return names


def dataset_image_counts_from_config(path: Path) -> dict[str, int]:
    payload = _load_yaml_mapping(path, "Anomaly-detection")
    counts: dict[str, int] = {}
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for entry in payload.get("datasets", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        raw_root = entry.get("root_dir")
        if not name or not isinstance(raw_root, str):
            continue
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        test_dir = root / "test"
        count = 0
        if test_dir.is_dir():
            for group in test_dir.iterdir():
                if group.is_dir():
                    count += sum(
                        1 for image_path in group.iterdir()
                        if image_path.is_file() and image_path.suffix.lower() in image_suffixes
                    )
        counts[name] = count
    return counts


def gpu_index_from_model_config(path: Path) -> int | None:
    payload = _load_yaml_mapping(path, "Model")
    models = payload.get("models", {})
    vlm = models.get("vlm", {}) if isinstance(models, dict) else {}
    device = vlm.get("device", 0) if isinstance(vlm, dict) else 0
    if isinstance(device, int):
        return device if device >= 0 else None
    value = str(device).strip().lower()
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"cuda(?::(\d+))?", value)
    if match:
        return int(match.group(1) or 0)
    return None


def _configured_dataset_names(path: Path, label: str) -> list[str]:
    payload = _load_yaml_mapping(path, label)
    entries = payload.get("datasets")
    if not isinstance(entries, list):
        raise AblationError(f"{label} config has no datasets list: {path}")
    return [
        str(entry.get("name", "")).strip()
        for entry in entries if isinstance(entry, dict) and str(entry.get("name", "")).strip()
    ]


def validate_config_files(
    cfg: AblationConfig,
    stages: Sequence[str],
    dataset_names: Sequence[str],
    run_count: int,
) -> None:
    required = [(cfg.anomaly_config, "Anomaly-detection"), (cfg.model_config, "Model")]
    if "masks" in stages and cfg.mask_config is not None:
        required.append((cfg.mask_config, "Mask-generation"))
    if "metrics" in stages and cfg.metrics_config is not None:
        required.append((cfg.metrics_config, "Metrics"))
    for path, label in required:
        _load_yaml_mapping(path, label)
    if "detection" in stages and cfg.decision_prompt_sequence:
        if len(cfg.decision_prompt_sequence) > 1 and len(cfg.decision_prompt_sequence) < run_count:
            raise AblationError(
                "experiment.decision_prompt_sequence has fewer entries than the "
                f"requested runs ({len(cfg.decision_prompt_sequence)} < {run_count}). "
                "Use one entry to repeat a prompt, or provide one entry per run."
            )
        anomaly_payload = _load_yaml_mapping(cfg.anomaly_config, "Anomaly-detection")
        prompts_raw = anomaly_payload.get("prompts_dir", "prompts/anomaly_detection_pipeline")
        prompts_dir = Path(str(prompts_raw)).expanduser()
        if not prompts_dir.is_absolute():
            prompts_dir = PROJECT_ROOT / prompts_dir
        missing_prompts = [
            template for template in cfg.decision_prompt_sequence
            if not (prompts_dir / template).is_file()
        ]
        if missing_prompts:
            raise AblationError(
                "Decision prompt template(s) not found under "
                f"{prompts_dir}: {', '.join(missing_prompts)}"
            )
    if "detection" in stages:
        anomaly_payload = _load_yaml_mapping(cfg.anomaly_config, "Anomaly-detection")
        layout_problems: list[str] = []
        for entry in anomaly_payload.get("datasets", []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "<unnamed>"))
            raw_root = entry.get("root_dir")
            if not isinstance(raw_root, str) or not raw_root.strip():
                layout_problems.append(f"{name}: root_dir is missing")
                continue
            root = Path(raw_root).expanduser()
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            missing = [
                relative for relative in (Path("train/good"), Path("test"))
                if not (root / relative).is_dir()
            ]
            if missing:
                layout_problems.append(
                    f"{name}: {root} is missing {', '.join(str(item) for item in missing)}"
                )
        if layout_problems:
            raise AblationError(
                "Dataset preflight failed before model loading: " + "; ".join(layout_problems)
            )
    expected = set(dataset_names)
    for path, label, enabled in (
        (cfg.mask_config, "Mask-generation", "masks" in stages),
        (cfg.metrics_config, "Metrics", "metrics" in stages),
    ):
        if path is None or not enabled:
            continue
        configured = set(_configured_dataset_names(path, label))
        if configured != expected:
            missing = sorted(expected - configured)
            extra = sorted(configured - expected)
            raise AblationError(
                f"{label} datasets do not match the anomaly-detection config. "
                f"Missing: {missing or 'none'}; extra: {extra or 'none'}"
            )


def _config_fingerprint(
    cfg: AblationConfig,
    dataset_names: Sequence[str],
    *,
    include_generation_profile: bool = True,
    legacy_path_root: Path | None = None,
) -> str:
    """Fingerprint stage contents independently of the checkout mount path.

    legacy_path_root exists only to recognize manifests written before
    relative config keys were introduced. It hashes the current file bytes
    under their former absolute-path keys; no file is read from that root.
    """
    file_hashes: dict[str, str | None] = {}
    for path in (cfg.anomaly_config, cfg.model_config, cfg.mask_config, cfg.metrics_config):
        if path is not None:
            try:
                relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                # Stage configs are normally project-local. Keep a stable,
                # explicit key if an external config is ever supported.
                relative_path = Path("__external__") / path.name
            key = (
                str((legacy_path_root / relative_path).resolve())
                if legacy_path_root is not None
                else relative_path.as_posix()
            )
            file_hashes[key] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            )
    stable = {
        "experiment_name": cfg.experiment_name,
        "datasets": list(dataset_names),
        "files": file_hashes,
        "command_prefix": list(cfg.command_prefix),
        "hardware_monitor_interval": cfg.hardware_monitor_interval,
        "process_gpu_telemetry": cfg.process_gpu_telemetry,
        "rerun_num_steps": cfg.rerun_num_steps,
        "rerun_policy": cfg.rerun_policy,
        "mask_include_rerun": cfg.mask_include_rerun,
        "metrics_include_rerun": cfg.metrics_include_rerun,
        "metrics_before_rerun": cfg.metrics_before_rerun,
        "metrics_combine": cfg.metrics_combine,
        "metrics_omit": list(cfg.metrics_omit),
        "std_ddof": cfg.std_ddof,
    }
    # Keep the fingerprint identical for legacy ablations that do not use
    # prompt sequencing; add this field only when the new feature is active.
    if cfg.decision_prompt_sequence:
        stable["decision_prompt_sequence"] = list(cfg.decision_prompt_sequence)
    if include_generation_profile:
        stable["generation_profile"] = cfg.generation_profile
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_manifest_project_root(
    manifest: dict[str, Any],
    cfg: AblationConfig,
) -> Path | None:
    """Recover the prior checkout root from a legacy absolute config path."""
    raw_path = manifest.get("config_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    try:
        relative_config = cfg.source_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    prior_config = Path(raw_path)
    if len(prior_config.parts) < len(relative_config.parts):
        return None
    if prior_config.parts[-len(relative_config.parts):] != relative_config.parts:
        return None
    prior_root = prior_config
    for _ in relative_config.parts:
        prior_root = prior_root.parent
    return prior_root


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
    )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _start_ablation_diagnostics(
    cfg: AblationConfig,
    *,
    num_runs: int,
    stages: Sequence[str],
    aggregate_only: bool,
) -> HardwareMonitor | None:
    """Start an out-of-band observer that is never an ablation data source."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        cfg.runs_dir
        / "diagnostics"
        / f"ablation_{timestamp}_pid{os.getpid()}"
    )
    previous_events_file = os.environ.get("PIPELINE_TELEMETRY_EVENTS_FILE")
    try:
        monitor = HardwareMonitor(
            output_dir=output_dir,
            # Host/kernel events are streamed continuously; two-second numeric
            # polling bounds observer overhead while retaining useful precursors.
            interval=max(2.0, cfg.hardware_monitor_interval),
            label="anomaly_detection_ablation",
            watched_pid=os.getpid(),
            context={
                "purpose": "failure_diagnostics_only",
                "excluded_from_ablation_metrics": True,
                "experiment": cfg.experiment_name,
                "runs_dir": str(cfg.runs_dir),
                "config_path": str(cfg.source_path),
                "num_runs": num_runs,
                "stages": list(stages),
                "aggregate_only": aggregate_only,
                "orchestrator_pid": os.getpid(),
            },
            # A multi-day process-tree trace can be large. Keeping the original
            # files avoids an expensive duplicate compression pass at completion.
            archive_enabled=False,
        )
        # Also record the cluster's other nodes for the whole ablation. This is
        # the outermost run, so its guard suppresses duplicate remote recorders
        # in the stage subprocesses it spawns.
        monitor = build_monitor(
            monitor,
            load_deployment_config(cfg.model_config),
            interval=max(2.0, cfg.hardware_monitor_interval),
            label="anomaly_detection_ablation",
            context={
                "purpose": "failure_diagnostics_only",
                "experiment": cfg.experiment_name,
            },
        )
        started = monitor.start(wait_for_ready=False)
    except MonitoringError:
        # Required cluster telemetry is part of the run contract. Never turn
        # this into an unmonitored multi-day experiment.
        raise
    except Exception:
        return None
    finally:
        # HardwareMonitor normally exposes its event file to in-process model
        # code. The ablation observer is deliberately external: restoring the
        # environment prevents per-image diagnostic fsyncs in stage children,
        # preserving their timing and progress behavior.
        if previous_events_file is None:
            os.environ.pop("PIPELINE_TELEMETRY_EVENTS_FILE", None)
        else:
            os.environ["PIPELINE_TELEMETRY_EVENTS_FILE"] = previous_events_file
    if not started:
        return None
    _diagnostics_call(
        monitor,
        "mark_event",
        "ablation_diagnostics_started",
        diagnostics_dir=str(output_dir),
        excluded_from_ablation_metrics=True,
    )
    return monitor


def _diagnostics_call(
    monitor: HardwareMonitor | None,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke monitoring best-effort; observer failures never affect a stage."""

    if monitor is None:
        return None
    try:
        return getattr(monitor, method)(*args, **kwargs)
    except Exception:
        return None


@contextmanager
def _ablation_diagnostics_session(
    cfg: AblationConfig,
    *,
    num_runs: int,
    stages: Sequence[str],
    aggregate_only: bool,
    dry_run: bool,
    console: Console,
) -> Iterator[HardwareMonitor | None]:
    if dry_run:
        yield None
        return
    monitor = _start_ablation_diagnostics(
        cfg,
        num_runs=num_runs,
        stages=stages,
        aggregate_only=aggregate_only,
    )
    if monitor is None:
        console.print("[yellow]Diagnostics observer could not start; the ablation will continue.[/yellow]")
        yield None
        return

    console.print(f"[dim]Failure diagnostics: {monitor.output_dir}[/dim]")
    reason = "ablation_finished"
    try:
        yield monitor
    except BaseException as exc:
        reason = f"ablation_exception:{type(exc).__name__}"
        _diagnostics_call(
            monitor,
            "mark_event",
            "ablation_controller_exception",
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        raise
    finally:
        _diagnostics_call(monitor, "mark_event", "ablation_diagnostics_stopping", reason=reason)
        _diagnostics_call(monitor, "stop", reason=reason)


def _decision_prompt_for_run(cfg: AblationConfig, run_index: int) -> str | None:
    """Return the decision prompt override for a one-based ablation run."""
    sequence = cfg.decision_prompt_sequence
    if not sequence:
        return None
    if len(sequence) == 1:
        return sequence[0]
    if run_index > len(sequence):
        raise AblationError(
            f"No decision prompt configured for ablation run {run_index}; "
            f"sequence has {len(sequence)} entries."
        )
    return sequence[run_index - 1]


def _run_prompt_metadata(cfg: AblationConfig, run: dict[str, Any]) -> dict[str, str]:
    """Return prompt metadata only for prompt-sequence experiments."""
    if not cfg.decision_prompt_sequence:
        return {}
    prompt = run.get("decision_prompt_template")
    if not isinstance(prompt, str) or not prompt:
        prompt = _decision_prompt_for_run(cfg, int(run["index"])) or ""
    return {"decision_prompt_template": prompt}


def _group_fields(cfg: AblationConfig, *fields: str) -> tuple[str, ...]:
    """Add the prompt recipe to aggregate keys when recipes are mixed."""
    if cfg.decision_prompt_sequence:
        return ("decision_prompt_template", *fields)
    return tuple(fields)


def _new_run_record(
    index: int,
    runs_dir: Path,
    datasets: Sequence[str],
    decision_prompt_template: str | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    run_id = f"{timestamp}_ablation_{index:03d}"
    # A fast manifest extension can occur within the same second. The index is
    # part of the name, so every repetition remains unique and timestamped.
    run_dir = runs_dir / run_id
    record = {
        "index": index,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": _now(),
        "stages": {
            "detection": {"status": "pending"},
            "base_masks": {"status": "pending"},
            "base_metrics": {"status": "pending"},
            "rerun": {"datasets": {name: {"status": "pending"} for name in datasets}},
            "masks": {"status": "pending"},
            "metrics": {"status": "pending"},
        },
    }
    if decision_prompt_template is not None:
        record["decision_prompt_template"] = decision_prompt_template
    return record


def _replace_path_prefix(value: Any, old_root: Path, new_root: Path) -> Any:
    """Rebase manifest-owned absolute paths while retaining non-path strings."""
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, old_root, new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_path_prefix(item, old_root, new_root) for item in value]
    if not isinstance(value, str):
        return value
    old_text = str(old_root)
    if value == old_text:
        return str(new_root)
    prefix = f"{old_text}{os.sep}"
    if value.startswith(prefix):
        return f"{new_root}{os.sep}{value[len(prefix):]}"
    return value


def _canonicalize_manifest_run_dirs(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
) -> None:
    """Make persisted run paths portable across checkout and mount locations.

    A run is identified by ``run_id`` and always belongs immediately beneath
    the currently configured ``runs_dir``.  Older manifests stored the
    machine-specific absolute path and could therefore redirect a resume into
    an unrelated local directory on another host.
    """
    runs_root = cfg.runs_dir.resolve()
    for run in runs:
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or run_id in {".", ".."}
        ):
            raise AblationError(f"Manifest contains an invalid run_id: {run_id!r}")
        expected = (runs_root / run_id).resolve()
        if expected.parent != runs_root:
            raise AblationError(
                f"Manifest run_id escapes the configured runs directory: {run_id!r}"
            )

        recorded_raw = run.get("run_dir")
        recorded = (
            Path(recorded_raw).expanduser().resolve()
            if isinstance(recorded_raw, str) and recorded_raw
            else expected
        )
        if recorded != expected:
            rebased = _replace_path_prefix(run, recorded, expected)
            run.clear()
            run.update(rebased)
            history = run.setdefault("run_dir_history", [])
            if str(recorded) not in history:
                history.append(str(recorded))
        run["run_dir"] = str(expected)


def load_or_create_manifest(
    cfg: AblationConfig,
    dataset_names: Sequence[str],
    num_runs: int,
    *,
    dry_run: bool,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = cfg.runs_dir / MANIFEST_NAME
    fingerprint = _config_fingerprint(cfg, dataset_names)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise AblationError(f"Unsupported manifest schema in {manifest_path}")
        prior_fingerprint = manifest.get("config_fingerprint")
        legacy_roots = [PROJECT_ROOT]
        prior_root = _legacy_manifest_project_root(manifest, cfg)
        if prior_root is not None and prior_root != PROJECT_ROOT:
            legacy_roots.append(prior_root)
        legacy_fingerprints = {
            _config_fingerprint(cfg, dataset_names, legacy_path_root=root)
            for root in legacy_roots
        }
        if cfg.generation_profile:
            legacy_fingerprints.update(
                _config_fingerprint(
                    cfg,
                    dataset_names,
                    include_generation_profile=False,
                    legacy_path_root=root,
                )
                for root in legacy_roots
            )
        if prior_fingerprint in legacy_fingerprints:
            # Older manifests keyed the exact same stage file hashes by
            # absolute path. Migrate them after proving their contents match.
            manifest["config_fingerprint"] = fingerprint
            prior_fingerprint = fingerprint
        if prior_fingerprint != fingerprint:
            raise AblationError(
                "The ablation configuration or one of its stage configs changed since this "
                f"experiment began. Refusing to mix configurations in {cfg.runs_dir}. "
                "Use a new experiment.runs_dir for the changed ablation."
            )
        runs = manifest.setdefault("runs", [])
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": cfg.experiment_name,
            "created_at": _now(),
            "updated_at": _now(),
            "config_path": str(cfg.source_path),
            "config_fingerprint": fingerprint,
            "generation_profile": cfg.generation_profile,
            "datasets": list(dataset_names),
            "runs": [],
        }
        runs = manifest["runs"]

    for run in runs:
        run.setdefault("stages", {}).setdefault("base_masks", {"status": "pending"})
        run["stages"].setdefault("base_metrics", {"status": "pending"})

    _canonicalize_manifest_run_dirs(cfg, runs)

    manifest["generation_profile"] = cfg.generation_profile
    manifest["config_path"] = str(cfg.source_path)

    while len(runs) < num_runs:
        run_index = len(runs) + 1
        runs.append(
            _new_run_record(
                run_index,
                cfg.runs_dir,
                dataset_names,
                _decision_prompt_for_run(cfg, run_index),
            )
        )
    if not dry_run:
        cfg.runs_dir.mkdir(parents=True, exist_ok=True)
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    _atomic_json(path, manifest)


def _stage_record(run: dict[str, Any], stage: str, dataset: str | None = None) -> dict[str, Any]:
    stage_record = run["stages"][stage]
    if dataset is not None:
        return stage_record["datasets"][dataset]
    return stage_record


def _set_status(
    manifest_path: Path,
    manifest: dict[str, Any],
    run: dict[str, Any],
    stage: str,
    status: str,
    *,
    dataset: str | None = None,
    **details: Any,
) -> None:
    record = _stage_record(run, stage, dataset)
    if status in {"running", "complete"}:
        record.pop("error", None)
    record.update(status=status, updated_at=_now(), **details)
    save_manifest(manifest_path, manifest)


def _mark_running(
    manifest_path: Path,
    manifest: dict[str, Any],
    run: dict[str, Any],
    stage: str,
    *,
    dataset: str | None = None,
) -> None:
    record = _stage_record(run, stage, dataset)
    attempts = int(record.get("attempts", 0)) + 1
    _set_status(
        manifest_path, manifest, run, stage, "running",
        dataset=dataset, started_at=_now(), attempts=attempts,
    )


def _clean_line(line: str) -> str:
    return " ".join(ANSI_ESCAPE.sub("", line).replace("\r", " ").strip().split())


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    progress: Progress,
    task_id: Any,
    description: str,
    dry_run: bool,
    show_output: bool,
    eta_tracker: EtaTracker,
    dataset_counts: dict[str, int],
    process_gpu_path: Path | None = None,
    process_gpu_index: int | None = None,
    process_gpu_interval: float = 1.0,
    detection_events_root: Path | None = None,
    subprogress_total: int | None = None,
    subprogress_groups: dict[str, str] | None = None,
    generation_profile: bool = False,
    diagnostics_monitor: HardwareMonitor | None = None,
    extra_environment: dict[str, str] | None = None,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(command)
    if dry_run:
        progress.update(
            task_id,
            description=f"{description} · dry run",
            eta_text="ETA --",
            finish_text="finish --",
        )
        progress.console.print(f"[dim]DRY RUN:[/dim] {rendered}")
        return 0, 0.0

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTHONFAULTHANDLER", "1")
    if generation_profile:
        environment["ANOMALY_ABLATION_PROFILE"] = "1"
    else:
        environment.pop("ANOMALY_ABLATION_PROFILE", None)
    if extra_environment:
        environment.update(extra_environment)
    started = time.perf_counter()
    progress.update(task_id, description=description)
    subtask_ids: dict[str, Any] = {}
    subtask_id: Any | None = None
    if subprogress_groups:
        for group_key, group_label in subprogress_groups.items():
            subtask_ids[group_key] = progress.add_task(
                f"  ↳ {group_label} · preparing",
                total=None,
                elapsed_label="stage elapsed",
                eta_text="",
                finish_text="",
            )
    else:
        subtask_id = progress.add_task(
            f"  ↳ {description} · preparing",
            total=subprogress_total,
            elapsed_label="stage elapsed",
            eta_text="",
            finish_text="",
        )
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{_now()}] COMMAND: {rendered}\n")
        log_file.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            child_pgid = os.getpgid(process.pid)
            child_session = os.getsid(process.pid)
        except OSError:
            child_pgid = None
            child_session = None
        log_file.write(
            f"[{_now()}] CHILD: pid={process.pid} pgid={child_pgid} "
            f"session_id={child_session}\n"
        )
        log_file.flush()
        if diagnostics_monitor is not None:
            _diagnostics_call(
                diagnostics_monitor,
                "track_process",
                process.pid,
                "ablation_stage_child",
                description=description,
                command=list(command),
                cwd=str(cwd),
                stage_log=str(log_path),
                process_group_id=child_pgid,
                session_id=child_session,
            )
            _diagnostics_call(
                diagnostics_monitor,
                "mark_event",
                "stage_process_started",
                child_pid=process.pid,
                process_group_id=child_pgid,
                session_id=child_session,
                description=description,
                command=list(command),
                stage_log=str(log_path),
            )
        process_gpu_sampler = (
            ProcessGpuSampler(
                process_gpu_path,
                process.pid,
                process_gpu_index,
                process_gpu_interval,
            )
            if process_gpu_path is not None and process_gpu_index is not None else None
        )
        if process_gpu_sampler is not None:
            process_gpu_sampler.start()
        detection_watcher = (
            DetectionEventWatcher(
                detection_events_root, dataset_counts, generation_profile
            )
            if detection_events_root is not None else None
        )
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for child_line in process.stdout:
                    output_queue.put(child_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, name="ablation-child-output", daemon=True)
        reader.start()
        reader_finished = False
        last_refresh = 0.0
        try:
            while not (reader_finished and process.poll() is not None and output_queue.empty()):
                line: str | None = None
                try:
                    line = output_queue.get(timeout=0.5)
                    if line is None:
                        reader_finished = True
                except queue.Empty:
                    pass
                if line is not None:
                    log_file.write(line)
                    log_file.flush()
                    eta_tracker.observe_output(line, dataset_counts)
                    if subprogress_groups:
                        method_progress = _mask_method_progress_from_output(line)
                        if method_progress is not None:
                            group_key, completed, child_total, child_detail = method_progress
                            method_task_id = subtask_ids.get(group_key)
                            if method_task_id is not None:
                                group_label = subprogress_groups[group_key]
                                progress.update(
                                    method_task_id,
                                    completed=completed,
                                    total=child_total,
                                    description=(
                                        f"  ↳ {group_label} · {child_detail}"
                                    ),
                                )
                            # Segmentation is deliberately deferred until every
                            # image has been localized. Keep its zero-progress
                            # rows informative instead of leaving them at the
                            # misleading initial "preparing" state.
                            if group_key == "bounding_box":
                                waiting_detail = (
                                    "loading segmentation model"
                                    if completed >= child_total
                                    else f"waiting for localization ({completed}/{child_total})"
                                )
                                for waiting_key in (
                                    "segmentation",
                                    "bounding_box_segmentation",
                                ):
                                    waiting_task_id = subtask_ids.get(waiting_key)
                                    if waiting_task_id is None:
                                        continue
                                    waiting_task = next(
                                        (
                                            task for task in progress.tasks
                                            if task.id == waiting_task_id
                                        ),
                                        None,
                                    )
                                    if waiting_task is None or waiting_task.completed > 0:
                                        continue
                                    progress.update(
                                        waiting_task_id,
                                        description=(
                                            f"  ↳ {subprogress_groups[waiting_key]}"
                                            f" · {waiting_detail}"
                                        ),
                                    )
                    else:
                        child_progress = _child_progress_from_output(line)
                        if child_progress is not None and detection_watcher is None:
                            completed, child_total, child_detail = child_progress
                            assert subtask_id is not None
                            progress.update(
                                subtask_id,
                                completed=completed,
                                total=child_total,
                                description=f"  ↳ {child_detail}",
                            )
                    if show_output:
                        progress.console.print(line.rstrip())
                now = time.monotonic()
                if now - last_refresh >= 0.25:
                    event_detail = (
                        detection_watcher.poll(eta_tracker)
                        if detection_watcher is not None else eta_tracker.current_label
                    )
                    if detection_watcher is not None:
                        assert subtask_id is not None
                        progress.update(
                            subtask_id,
                            completed=detection_watcher.completed_images,
                            total=detection_watcher.total_images,
                            description=f"  ↳ {event_detail}",
                        )
                    eta, finish = eta_tracker.display_fields()
                    progress.update(
                        task_id,
                        description=description,
                        eta_text=f"ETA {eta}",
                        finish_text=f"finish {finish}",
                    )
                    last_refresh = now
            return_code = process.wait()
            reader.join(timeout=1.0)
            log_file.write(
                f"[{_now()}] CHILD EXIT: pid={process.pid} return_code={return_code} "
                f"signal={-return_code if return_code < 0 else None}\n"
            )
            log_file.flush()
            if diagnostics_monitor is not None:
                _diagnostics_call(
                    diagnostics_monitor,
                    "mark_event",
                    "stage_process_exited",
                    child_pid=process.pid,
                    return_code=return_code,
                    signal=(-return_code if return_code < 0 else None),
                    description=description,
                    duration_sec=round(time.perf_counter() - started, 6),
                    stage_log=str(log_path),
                )
        except KeyboardInterrupt:
            if diagnostics_monitor is not None:
                _diagnostics_call(
                    diagnostics_monitor,
                    "mark_event",
                    "stage_process_keyboard_interrupt",
                    child_pid=process.pid,
                    description=description,
                    stage_log=str(log_path),
                )
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            raise
        except BaseException as exc:
            if diagnostics_monitor is not None:
                _diagnostics_call(
                    diagnostics_monitor,
                    "mark_event",
                    "stage_process_observer_exception",
                    child_pid=process.pid,
                    description=description,
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    stage_log=str(log_path),
                )
            raise
        finally:
            if process_gpu_sampler is not None:
                process_gpu_sampler.stop()
                if process_gpu_sampler.error:
                    log_file.write(
                        f"[{_now()}] Process-specific GPU telemetry unavailable: "
                        f"{process_gpu_sampler.error}\n"
                    )
                    log_file.flush()
            if subtask_ids:
                for method_task_id in subtask_ids.values():
                    progress.remove_task(method_task_id)
            elif subtask_id is not None:
                progress.remove_task(subtask_id)
    return return_code, time.perf_counter() - started


def _base_command(cfg: AblationConfig, script: Path) -> list[str]:
    return [*cfg.command_prefix, sys.executable, str(script)]


def _module_command(cfg: AblationConfig, module: str) -> list[str]:
    return [*cfg.command_prefix, sys.executable, "-m", module]


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as file_obj:
        return sum(1 for line in file_obj if line.strip())


def validate_detection(run_dir: Path, datasets: Sequence[str]) -> None:
    problems: list[str] = []
    for dataset in datasets:
        summary = run_dir / dataset / "all_data_summary.jsonl"
        if _line_count(summary) == 0:
            problems.append(f"{dataset}: missing/empty all_data_summary.jsonl")
        performance = run_dir / dataset / "performance"
        if not performance.is_dir() or not any(performance.glob("performance_*.json")):
            problems.append(f"{dataset}: missing inference performance JSON")
    if problems:
        raise AblationError("Detection did not produce complete artifacts: " + "; ".join(problems))


def validate_rerun(run_dir: Path, dataset: str) -> None:
    base_count = _line_count(run_dir / dataset / "all_data_summary.jsonl")
    rerun_count = _line_count(run_dir / dataset / "rerun_results" / "rerun_result.jsonl")
    if base_count == 0 or rerun_count != base_count:
        raise AblationError(
            f"Rerun for {dataset} is incomplete: expected {base_count} entries, found {rerun_count}"
        )


def latest_file(paths: Iterable[Path]) -> Path | None:
    candidates = [path for path in paths if path.is_file()]
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name)) if candidates else None


def validate_masks(
    run_dir: Path,
    datasets: Sequence[str],
    performance_subdirs: Sequence[str | Path] = ("mask_generation",),
) -> dict[str, str]:
    """Require a parseable, dataset-matching mask completion record."""
    outputs: dict[str, str] = {}
    for dataset in datasets:
        searched_dirs = [run_dir / dataset / subdir for subdir in performance_subdirs]
        path = None
        for performance_dir in searched_dirs:
            if performance_dir.is_dir():
                path = latest_file(performance_dir.glob("performance_*.json"))
            if path is not None:
                break
        if path is None:
            raise AblationError(
                f"Mask performance output is missing for {dataset} under "
                + " or ".join(str(directory) for directory in searched_dirs)
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AblationError(
                f"Mask performance output is invalid for {dataset}: {path} ({error})"
            ) from error
        if not isinstance(payload, dict) or payload.get("dataset") != dataset:
            raise AblationError(
                f"Mask performance output does not describe {dataset}: {path}"
            )
        records_verified = payload.get("records_verified")
        if (
            isinstance(records_verified, bool)
            or not isinstance(records_verified, int)
            or records_verified <= 0
        ):
            raise AblationError(
                f"Mask performance output has no verified records for {dataset}: {path}"
            )
        outputs[dataset] = str(path)
    return outputs


def _metrics_output_subdir(metrics_config: Path) -> str:
    payload = _load_yaml_mapping(metrics_config, "Metrics")
    output = payload.get("output", {})
    if not isinstance(output, dict):
        return "metrics"
    value = str(output.get("subdir", "metrics")).strip()
    return value or "metrics"


def validate_metrics(
    run_dir: Path,
    datasets: Sequence[str],
    include_rerun: bool,
    output_subdir: str,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    prefix = "metrics_w_rerun_" if include_rerun else "metrics_"
    for dataset in datasets:
        metrics_dir = run_dir / dataset / output_subdir
        path = latest_file(metrics_dir.glob(f"{prefix}*.csv")) if metrics_dir.is_dir() else None
        if path is None:
            raise AblationError(f"Metrics CSV was not produced for {dataset} under {metrics_dir}")
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                first_row = next(reader, None)
        except (OSError, csv.Error) as error:
            raise AblationError(
                f"Metrics CSV is invalid for {dataset}: {path} ({error})"
            ) from error
        if not reader.fieldnames or first_row is None:
            raise AblationError(f"Metrics CSV is empty for {dataset}: {path}")
        if first_row.get("dataset") != dataset:
            raise AblationError(
                f"Metrics CSV does not describe {dataset}: {path}"
            )
        outputs[dataset] = str(path)
    return outputs


def write_metrics_override(source: Path, destination: Path, run_dir: Path) -> Path:
    payload = _load_yaml_mapping(source, "Metrics")
    payload["run_dir"] = str(run_dir.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def _run_detection(
    cfg: AblationConfig,
    datasets: Sequence[str],
    run: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    progress: Progress,
    task_id: Any,
    run_count: int,
    eta_tracker: EtaTracker,
    dataset_counts: dict[str, int],
    diagnostics_monitor: HardwareMonitor | None,
    *,
    dry_run: bool,
    show_output: bool,
) -> None:
    run_label = f"Run {run['index']}/{run_count}"
    decision_prompt_template = _decision_prompt_for_run(cfg, int(run["index"]))
    prompt_label = decision_prompt_template or "config prompt"
    detection_label = f"{run_label} · detection · {prompt_label}"
    record = _stage_record(run, "detection")
    if record.get("status") == "complete":
        validate_detection(Path(run["run_dir"]), datasets)
        progress.update(task_id, advance=1, description=f"{detection_label} · resumed")
        return
    run_dir = Path(run["run_dir"])
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        _mark_running(manifest_path, manifest, run, "detection")
    command = _base_command(cfg, PROJECT_ROOT / "run_anomaly_detection.py") + [
        "--dataset-config", str(cfg.anomaly_config),
        "--model-config", str(cfg.model_config),
        "--resume-dir", str(run_dir),
        "--hardware-monitor-interval", str(cfg.hardware_monitor_interval),
    ]
    if decision_prompt_template is not None:
        command.extend(["--decision-prompt-template", decision_prompt_template])
    log_path = cfg.runs_dir / "orchestrator_logs" / f"run_{run['index']:03d}" / "detection.log"
    operation_key = "detection"
    eta_tracker.start(operation_key, datasets[0] if datasets else "starting")
    attempt = int(_stage_record(run, "detection").get("attempts", 1))
    process_gpu_path = (
        run_dir / "logs" / f"process_gpu_detection_attempt_{attempt:02d}.csv"
        if cfg.process_gpu_telemetry else None
    )
    try:
        code, duration = run_command(
            command, cwd=PROJECT_ROOT, log_path=log_path, progress=progress,
            task_id=task_id, description=detection_label,
            dry_run=dry_run, show_output=show_output,
            eta_tracker=eta_tracker, dataset_counts=dataset_counts,
            process_gpu_path=process_gpu_path,
            process_gpu_index=gpu_index_from_model_config(cfg.model_config),
            process_gpu_interval=cfg.hardware_monitor_interval,
            detection_events_root=run_dir / "logs",
            subprogress_total=sum(dataset_counts.values()),
            generation_profile=cfg.generation_profile,
            diagnostics_monitor=diagnostics_monitor,
        )
        if code != 0:
            raise AblationError(f"Detection exited with status {code}; see {log_path}")
        if not dry_run:
            validate_detection(run_dir, datasets)
            _set_status(
                manifest_path, manifest, run, "detection", "complete",
                completed_at=_now(), duration_sec=round(duration, 4), log_path=str(log_path),
                process_gpu_telemetry_path=(str(process_gpu_path) if process_gpu_path else None),
            )
        eta_tracker.finish(operation_key, duration)
    except Exception as exc:
        if not dry_run:
            _set_status(manifest_path, manifest, run, "detection", "failed", error=str(exc), log_path=str(log_path))
        raise
    progress.update(task_id, advance=1, description=f"{detection_label} · complete")


def _require_detection(run: dict[str, Any], datasets: Sequence[str]) -> Path:
    run_dir = Path(run["run_dir"])
    validate_detection(run_dir, datasets)
    return run_dir


def _prepare_invalid_rerun_recovery(
    manifest_path: Path,
    manifest: dict[str, Any],
    run: dict[str, Any],
    dataset: str,
    error: Exception,
    *,
    persist: bool = True,
) -> None:
    """Preserve corrupt rerun evidence and make its dependants runnable again.

    Older manifests could say ``complete`` solely because the child exited
    zero.  If strict validation disproves that claim, retaining ``complete``
    makes every future resume fail at the same point.  Downgrade just that
    dataset and invalidate only downstream products which consumed it.
    """
    run_dir = Path(run["run_dir"])
    source = run_dir / dataset / "rerun_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_dir = (
        run_dir / ".ablation" / "invalid_artifacts"
        / f"rerun_{dataset}_{timestamp}"
    )
    if persist and source.is_dir():
        shutil.copytree(source, evidence_dir)

    reason = str(error)
    rerun_record = _stage_record(run, "rerun", dataset)
    rerun_record.update(
        status="pending",
        artifact_validation_error=reason,
        invalid_artifacts_backup=(
            str(evidence_dir) if persist and source.is_dir() else None
        ),
        invalidated_at=_now(),
    )

    masks_record = _stage_record(run, "masks")
    affected = set(masks_record.get("repair_datasets", []))
    affected.add(dataset)
    masks_record.update(
        status="pending",
        force_overwrite=True,
        repair_datasets=sorted(affected),
        invalidated_by=f"rerun:{dataset}",
        invalidated_at=_now(),
    )
    metrics_record = _stage_record(run, "metrics")
    metrics_record.update(
        status="pending",
        invalidated_by=f"rerun:{dataset}",
        invalidated_at=_now(),
    )
    if persist:
        save_manifest(manifest_path, manifest)


def _run_rerun_dataset(
    cfg: AblationConfig,
    datasets: Sequence[str],
    run: dict[str, Any],
    dataset: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    progress: Progress,
    task_id: Any,
    run_count: int,
    eta_tracker: EtaTracker,
    dataset_counts: dict[str, int],
    diagnostics_monitor: HardwareMonitor | None,
    *,
    dry_run: bool,
    show_output: bool,
) -> None:
    run_label = f"Run {run['index']}/{run_count}"
    record = _stage_record(run, "rerun", dataset)
    if record.get("status") == "complete":
        run_dir = _require_detection(run, datasets)
        try:
            validate_rerun(run_dir, dataset)
        except AblationError as error:
            _prepare_invalid_rerun_recovery(
                manifest_path, manifest, run, dataset, error,
                persist=not dry_run,
            )
        else:
            progress.update(task_id, advance=1, description=f"{run_label} · rerun · {dataset} · resumed")
            return
    run_dir = Path(run["run_dir"]) if dry_run else _require_detection(run, datasets)
    if not dry_run:
        _mark_running(manifest_path, manifest, run, "rerun", dataset=dataset)
    attempt = int(_stage_record(run, "rerun", dataset).get("attempts", 1))
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    process_gpu_path = (
        run_dir / "logs" / f"process_gpu_rerun_{safe_dataset}_attempt_{attempt:02d}.csv"
        if cfg.process_gpu_telemetry else None
    )
    inference_events_path = (
        run_dir / "logs" / f"rerun_inference_{safe_dataset}_attempt_{attempt:02d}_events.jsonl"
    )
    command = _module_command(cfg, "anomaly_detection.rerun") + [
        str(run_dir), dataset,
        "--num-steps", str(cfg.rerun_num_steps),
        "--secondary-policy", cfg.rerun_policy,
        "--model-config", str(cfg.model_config),
        # The ablation owns one continuous local+cluster recorder and already
        # has a dedicated rerun GPU sampler/event stream.
        "--no-hardware-monitor",
    ]
    if cfg.rerun_policy == "global-secondary":
        command.extend(["--metrics-config", str(cfg.metrics_config)])
    log_path = cfg.runs_dir / "orchestrator_logs" / f"run_{run['index']:03d}" / f"rerun_{dataset}.log"
    operation_key = f"rerun:{dataset}"
    eta_tracker.start(operation_key, dataset)
    try:
        code, duration = run_command(
            command, cwd=PROJECT_ROOT, log_path=log_path, progress=progress,
            task_id=task_id, description=f"{run_label} · rerun · {dataset}",
            dry_run=dry_run, show_output=show_output,
            eta_tracker=eta_tracker, dataset_counts=dataset_counts,
            subprogress_total=dataset_counts.get(dataset),
            generation_profile=cfg.generation_profile,
            diagnostics_monitor=diagnostics_monitor,
            process_gpu_path=process_gpu_path,
            process_gpu_index=gpu_index_from_model_config(cfg.model_config),
            process_gpu_interval=cfg.hardware_monitor_interval,
            extra_environment={
                "PIPELINE_TELEMETRY_EVENTS_FILE": str(inference_events_path),
            },
        )
        if code != 0:
            raise AblationError(f"Rerun exited with status {code}; see {log_path}")
        if not dry_run:
            validate_rerun(run_dir, dataset)
            _set_status(
                manifest_path, manifest, run, "rerun", "complete", dataset=dataset,
                completed_at=_now(), duration_sec=round(duration, 4), log_path=str(log_path),
                process_gpu_telemetry_path=(str(process_gpu_path) if process_gpu_path else None),
                inference_events_path=str(inference_events_path),
            )
        eta_tracker.finish(operation_key, duration)
    except Exception as exc:
        if not dry_run:
            _set_status(
                manifest_path, manifest, run, "rerun", "failed", dataset=dataset,
                error=str(exc), log_path=str(log_path),
            )
        raise
    progress.update(task_id, advance=1, description=f"{run_label} · rerun · {dataset} · complete")


def _run_masks(
    cfg: AblationConfig,
    datasets: Sequence[str],
    run: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    progress: Progress,
    task_id: Any,
    run_count: int,
    eta_tracker: EtaTracker,
    dataset_counts: dict[str, int],
    diagnostics_monitor: HardwareMonitor | None,
    *,
    dry_run: bool,
    show_output: bool,
    stage_name: str = "masks",
    include_rerun: bool | None = None,
) -> None:
    use_rerun = cfg.mask_include_rerun if include_rerun is None else include_rerun
    stage_label = "base masks" if stage_name == "base_masks" else "masks"
    run_label = f"Run {run['index']}/{run_count}"
    record = _stage_record(run, stage_name)
    if record.get("status") == "complete":
        run_dir = _require_detection(run, datasets)
        if use_rerun:
            for dataset in datasets:
                validate_rerun(run_dir, dataset)
        performance_subdirs = (
            ("base_evaluation/mask_generation", "mask_generation")
            if stage_name == "base_masks" else ("mask_generation",)
        )
        validate_masks(run_dir, datasets, performance_subdirs)
        progress.update(task_id, advance=1, description=f"{run_label} · {stage_label} · resumed")
        return
    run_dir = Path(run["run_dir"]) if dry_run else _require_detection(run, datasets)
    assert cfg.mask_config is not None
    if use_rerun and not dry_run:
        for dataset in datasets:
            validate_rerun(run_dir, dataset)
    if not dry_run:
        _mark_running(manifest_path, manifest, run, stage_name)
    command = _base_command(cfg, PROJECT_ROOT / "run_mask_generation.py") + [
        "--config", str(cfg.mask_config), "--run-dir", str(run_dir),
        "--model-config", str(cfg.model_config),
        "--no-hardware-monitor",
        "--include-rerun" if use_rerun else "--base-results",
    ]
    repair_datasets = tuple(record.get("repair_datasets", ()))
    if record.get("force_overwrite"):
        command.append("--overwrite")
        for dataset in repair_datasets:
            command.extend(["--dataset", dataset])
    log_path = cfg.runs_dir / "orchestrator_logs" / f"run_{run['index']:03d}" / f"{stage_name}.log"
    operation_key = stage_name
    eta_tracker.start(operation_key, "starting")
    try:
        code, duration = run_command(
            command, cwd=PROJECT_ROOT, log_path=log_path, progress=progress,
            task_id=task_id, description=f"{run_label} · {stage_label}",
            dry_run=dry_run, show_output=show_output,
            eta_tracker=eta_tracker, dataset_counts=dataset_counts,
            subprogress_groups={
                "bounding_box": "bounding_box localization (shared)",
                "segmentation": "segmentation",
                "bounding_box_segmentation": "bounding_box_segmentation",
            },
            generation_profile=cfg.generation_profile,
            diagnostics_monitor=diagnostics_monitor,
        )
        if code != 0:
            raise AblationError(f"Mask generation exited with status {code}; see {log_path}")
        if not dry_run:
            validate_masks(run_dir, datasets)
            record.pop("force_overwrite", None)
            record.pop("repair_datasets", None)
            _set_status(
                manifest_path, manifest, run, stage_name, "complete",
                completed_at=_now(), duration_sec=round(duration, 4), log_path=str(log_path),
            )
        eta_tracker.finish(operation_key, duration)
    except Exception as exc:
        if not dry_run:
            _set_status(manifest_path, manifest, run, stage_name, "failed", error=str(exc), log_path=str(log_path))
        raise
    progress.update(task_id, advance=1, description=f"{run_label} · {stage_label} · complete")


def _run_metrics(
    cfg: AblationConfig,
    datasets: Sequence[str],
    run: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    progress: Progress,
    task_id: Any,
    run_count: int,
    eta_tracker: EtaTracker,
    dataset_counts: dict[str, int],
    diagnostics_monitor: HardwareMonitor | None,
    *,
    dry_run: bool,
    show_output: bool,
    stage_name: str = "metrics",
    include_rerun: bool | None = None,
) -> None:
    use_rerun = cfg.metrics_include_rerun if include_rerun is None else include_rerun
    stage_label = "base metrics" if stage_name == "base_metrics" else "metrics"
    run_label = f"Run {run['index']}/{run_count}"
    record = _stage_record(run, stage_name)
    if record.get("status") == "complete":
        run_dir = _require_detection(run, datasets)
        if use_rerun:
            for dataset in datasets:
                validate_rerun(run_dir, dataset)
        assert cfg.metrics_config is not None
        validate_metrics(
            run_dir,
            datasets,
            use_rerun,
            _metrics_output_subdir(cfg.metrics_config),
        )
        progress.update(task_id, advance=1, description=f"{run_label} · {stage_label} · resumed")
        return
    run_dir = Path(run["run_dir"]) if dry_run else _require_detection(run, datasets)
    assert cfg.metrics_config is not None
    if use_rerun and not dry_run:
        for dataset in datasets:
            validate_rerun(run_dir, dataset)
    override_path = run_dir / ".ablation" / "metrics_config.yaml"
    if not dry_run:
        write_metrics_override(cfg.metrics_config, override_path, run_dir)
        _mark_running(manifest_path, manifest, run, stage_name)
    command = _base_command(cfg, PROJECT_ROOT / "run_metrics.py") + [
        "--metrics-config", str(override_path),
        "--model-config", str(cfg.model_config),
        "--no-hardware-monitor",
        "--include-rerun" if use_rerun else "--base-results",
    ]
    if cfg.metrics_combine:
        command.append("--combine")
    if cfg.metrics_omit:
        command.extend(["--omit", *cfg.metrics_omit])
    log_path = cfg.runs_dir / "orchestrator_logs" / f"run_{run['index']:03d}" / f"{stage_name}.log"
    operation_key = stage_name
    eta_tracker.start(operation_key, datasets[0] if datasets else "starting")
    try:
        code, duration = run_command(
            command, cwd=PROJECT_ROOT, log_path=log_path, progress=progress,
            task_id=task_id, description=f"{run_label} · {stage_label}",
            dry_run=dry_run, show_output=show_output,
            eta_tracker=eta_tracker, dataset_counts=dataset_counts,
            subprogress_total=len(datasets),
            generation_profile=cfg.generation_profile,
            diagnostics_monitor=diagnostics_monitor,
        )
        if code != 0:
            raise AblationError(f"Metrics exited with status {code}; see {log_path}")
        if not dry_run:
            outputs = validate_metrics(
                run_dir,
                datasets,
                use_rerun,
                _metrics_output_subdir(cfg.metrics_config),
            )
            _set_status(
                manifest_path, manifest, run, stage_name, "complete",
                completed_at=_now(), duration_sec=round(duration, 4),
                log_path=str(log_path), outputs=outputs,
            )
        eta_tracker.finish(operation_key, duration)
    except Exception as exc:
        if not dry_run:
            _set_status(manifest_path, manifest, run, stage_name, "failed", error=str(exc), log_path=str(log_path))
        raise
    progress.update(task_id, advance=1, description=f"{run_label} · {stage_label} · complete")


def _archive_base_mask_artifacts(run_dir: Path, datasets: Sequence[str]) -> None:
    """Move base-prediction masks aside before rerun-aware mask generation.

    Metrics CSVs stay in their normal timestamped locations (their base names
    do not collide with ``metrics_w_rerun``).  Pixel maps and fingerprints are
    namespaced so both prediction sources remain auditable.
    """
    artifact_names = (
        "masks", "mask_generation", "secondary_masks", "secondary_mask_generation",
    )
    for dataset in datasets:
        dataset_dir = run_dir / dataset
        archive_dir = dataset_dir / "base_evaluation"
        for name in artifact_names:
            source = dataset_dir / name
            destination = archive_dir / name
            if not source.exists():
                continue
            if destination.exists():
                # The base artifact was already archived by an earlier
                # invocation.  A source with the same name is the rerun-aware
                # artifact subsequently generated in the normal location.
                # Leave both intact when resuming; moving or rejecting the
                # source here would either destroy final results or make a
                # fully completed run impossible to resume.
                continue
            archive_dir.mkdir(parents=True, exist_ok=True)
            source.replace(destination)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AblationError(f"Expected a JSON object in {path}")
    return payload


def _latest_valid_json_file(paths: Iterable[Path]) -> Path | None:
    """Return the newest readable JSON-object artifact, if one exists.

    Performance summaries are supplementary timing data.  A crashed remote
    filesystem can leave the last one as a zero-byte file after all primary
    inference, masks, and metrics have already completed.  Do not let that
    optional artifact make aggregation of the completed run impossible.
    """
    candidates = sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for path in candidates:
        try:
            _read_json(path)
        except (AblationError, OSError, UnicodeError, json.JSONDecodeError):
            continue
        return path
    return None


def _resolve_run_artifact(raw_path: Any, run_dir: Path, dataset: str) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [
        PROJECT_ROOT / path,
        run_dir / path,
        run_dir / dataset / path,
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _rerun_artifact_timing(run_dir: Path, dataset: str) -> dict[str, Any]:
    """Recover all rerun VLM timings, including work done before a resume.

    A rerun performance JSON describes only work performed by that invocation.
    The incrementally preserved per-image step JSON files retain every actual
    inference time, so they are the authoritative source for an interrupted
    then resumed repetition.
    """
    result_path = run_dir / dataset / "rerun_results" / "rerun_result.jsonl"
    if not result_path.is_file():
        return {}
    num_entries = 0
    num_images_rerun = 0
    step_times: list[float] = []
    step_tokens: list[int] = []
    token_timing_sec = 0.0
    with result_path.open(encoding="utf-8") as file_obj:
        for raw in file_obj:
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            num_entries += 1
            step_path = _resolve_run_artifact(entry.get("rerun_steps_filepath"), run_dir, dataset)
            if step_path is None:
                continue
            try:
                steps = json.loads(step_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(steps, list):
                continue
            image_times = [
                number for step in steps if isinstance(step, dict)
                if (number := _number(step.get("inference_time_sec"))) is not None
            ]
            if image_times:
                num_images_rerun += 1
                step_times.extend(image_times)
            for step in steps:
                if not isinstance(step, dict):
                    continue
                tokens = _number(step.get("generated_token_count"))
                timing = _number(step.get("inference_time_sec"))
                if tokens is not None and timing is not None and timing > 0:
                    step_tokens.append(int(tokens))
                    token_timing_sec += timing
    total = sum(step_times)
    total_tokens = sum(step_tokens)
    return {
        "rerun_num_entries_checked": num_entries,
        "rerun_num_images_rerun": num_images_rerun,
        "rerun_num_rerun_steps": len(step_times),
        "rerun_total_rerun_step_inference_time_sec": total,
        "rerun_avg_rerun_step_inference_time_sec": total / len(step_times) if step_times else None,
        "rerun_total_generated_tokens": total_tokens,
        "rerun_avg_generated_tokens_per_step": (
            total_tokens / len(step_tokens) if step_tokens else None
        ),
        "rerun_generated_tokens_per_second": (
            total_tokens / token_timing_sec if token_timing_sec > 0 else None
        ),
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as file_obj:
        for raw in file_obj:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _artifact_image_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    label = str(entry.get("label_folder", ""))
    return (
        label,
        str(entry.get("source_label_folder") or label),
        str(entry.get("img_name", "")),
    )


def _image_timing_rows(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    """Return detection, rerun, and combined inference timing per image.

    The persisted summary and rerun-step artifacts are authoritative, so this
    also reconstructs image-level timing for reruns completed before live
    rerun telemetry was introduced.
    """
    output: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        for dataset in datasets:
            detections = _jsonl_rows(run_dir / dataset / "all_data_summary.jsonl")
            rerun_entries = {
                _artifact_image_key(entry): entry
                for entry in _jsonl_rows(
                    run_dir / dataset / "rerun_results" / "rerun_result.jsonl"
                )
            }
            for entry in detections:
                label, source, img_name = _artifact_image_key(entry)
                base = {
                    "run_index": run["index"],
                    "run_id": run["run_id"],
                    "run_dir": str(run_dir),
                    **_run_prompt_metadata(cfg, run),
                    "dataset": dataset,
                    "label_folder": label,
                    "source_label_folder": source,
                    "img_name": img_name,
                    "image_identity": "/".join((dataset, label, source, img_name)),
                }
                detection_time = _number(entry.get("inference_time_sec"))
                detection_basis = str(entry.get("timing_basis", "legacy_vlm_call"))
                output.append({
                    **base,
                    "phase": "detection",
                    "inference_time_sec": detection_time if detection_time is not None else "",
                    "timing_basis": detection_basis,
                    "rerun_step_count": 0,
                    "timing_availability": (
                        "recorded" if detection_time is not None else "not_recorded"
                    ),
                })

                # Optional object-crop call made before the decision; its own
                # phase keeps detection timing comparable with uncropped runs.
                crop_record = entry.get("object_crop")
                crop_time = (
                    _number(crop_record.get("inference_time_sec"))
                    if isinstance(crop_record, dict) else None
                )
                if isinstance(crop_record, dict):
                    output.append({
                        **base,
                        "phase": "object_crop",
                        "inference_time_sec": crop_time if crop_time is not None else "",
                        "timing_basis": str(crop_record.get("timing_basis") or "not_recorded"),
                        "rerun_step_count": 0,
                        "timing_availability": (
                            "recorded" if crop_time is not None else "not_recorded"
                        ),
                    })
                rerun_entry = rerun_entries.get((label, source, img_name), {})
                step_path = _resolve_run_artifact(
                    rerun_entry.get("rerun_steps_filepath"), run_dir, dataset
                )
                rerun_times: list[float] = []
                rerun_bases: set[str] = set()
                if step_path is not None:
                    try:
                        steps = json.loads(step_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        steps = []
                    if isinstance(steps, list):
                        rerun_times = [
                            value for step in steps if isinstance(step, dict)
                            if (value := _number(step.get("inference_time_sec"))) is not None
                        ]
                        rerun_bases = {
                            str(step.get("timing_basis", "legacy_vlm_call"))
                            for step in steps if isinstance(step, dict)
                            and _number(step.get("inference_time_sec")) is not None
                        }
                rerun_time = sum(rerun_times) if rerun_times else None
                rerun_basis = (
                    next(iter(rerun_bases)) if len(rerun_bases) == 1 else "mixed"
                )
                if rerun_time is not None:
                    output.append({
                        **base,
                        "phase": "rerun",
                        "inference_time_sec": rerun_time,
                        "timing_basis": rerun_basis,
                        "rerun_step_count": len(rerun_times),
                        "timing_availability": "recorded",
                    })
                combined = (
                    detection_time + (rerun_time or 0.0) + (crop_time or 0.0)
                    if detection_time is not None else None
                )
                output.append({
                    **base,
                    "phase": "combined",
                    "inference_time_sec": combined if combined is not None else "",
                    "timing_basis": (
                        detection_basis
                        if rerun_time is None or detection_basis == rerun_basis
                        else "mixed"
                    ),
                    "rerun_step_count": len(rerun_times),
                    "timing_availability": (
                        "recorded" if combined is not None else "not_recorded"
                    ),
                })
    return output


def _image_value_rollups(
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    value_field: str,
    ddof: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        values = [
            value for row in group
            if (value := _number(row.get(value_field))) is not None
        ]
        mean, std = _mean_std(values, ddof)
        result: dict[str, Any] = dict(zip(group_fields, key))
        first = group[0]
        for field in ("run_id", "run_dir"):
            if field not in result and field in first:
                result[field] = first[field]
        result.update({
            "image_count": len(values),
            f"{value_field}_total": sum(values) if values else "",
            f"{value_field}_mean_per_image": mean if mean is not None else "",
            f"{value_field}_std_across_images": std if std is not None else "",
        })
        output.append(result)
    return output


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], preferred: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for field in preferred:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Sequence[float], ddof: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) <= ddof:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - ddof)
    return mean, math.sqrt(max(0.0, variance))


def _aggregate_numeric_rows(
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    ddof: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)

    output: list[dict[str, Any]] = []
    excluded = {*group_fields, "run_index", "run_id", "run_dir"}
    for key in sorted(grouped):
        group = grouped[key]
        result: dict[str, Any] = dict(zip(group_fields, key))
        result["runs"] = len({str(row.get("run_index", "")) for row in group})
        numeric_fields: list[str] = []
        for row in group:
            for field, value in row.items():
                if field not in excluded and _number(value) is not None and field not in numeric_fields:
                    numeric_fields.append(field)
        for field in numeric_fields:
            values = [number for row in group if (number := _number(row.get(field))) is not None]
            mean, std = _mean_std(values, ddof)
            result[f"{field}_mean"] = round(mean, 8) if mean is not None else ""
            result[f"{field}_std"] = round(std, 8) if std is not None else ""
            result[f"{field}_n"] = len(values)
        output.append(result)
    return output


def _metric_rows(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
    datasets: Sequence[str],
    *,
    stage_name: str = "metrics",
    include_rerun: bool | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    use_rerun = cfg.metrics_include_rerun if include_rerun is None else include_rerun
    prefix = "metrics_w_rerun_" if use_rerun else "metrics_"
    output_subdir = _metrics_output_subdir(cfg.metrics_config) if cfg.metrics_config is not None else "metrics"
    for run in runs:
        run_dir = Path(run["run_dir"])
        for dataset in datasets:
            configured_output = _stage_record(run, stage_name).get("outputs", {}).get(dataset)
            path = Path(configured_output) if configured_output else latest_file(
                (run_dir / dataset / output_subdir).glob(f"{prefix}*.csv")
            )
            if path is None or not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as file_obj:
                for row in csv.DictReader(file_obj):
                    output.append({
                        "run_index": run["index"], "run_id": run["run_id"],
                        "run_dir": str(run_dir),
                        **_run_prompt_metadata(cfg, run),
                        **row,
                    })

        combined_candidates = list(run_dir.glob(f"{prefix}*.csv"))
        if not use_rerun:
            combined_candidates = [
                path for path in combined_candidates
                if not path.name.startswith("metrics_w_rerun_")
            ]
        combined = latest_file(combined_candidates)
        if combined is not None:
            with combined.open(newline="", encoding="utf-8") as file_obj:
                overall = next(
                    (row for row in csv.DictReader(file_obj) if str(row.get("dataset", "")).lower() == "overall"),
                    None,
                )
            if overall:
                overall["dataset"] = "__experiment__"
                output.append({
                    "run_index": run["index"], "run_id": run["run_id"],
                    "run_dir": str(run_dir),
                    **_run_prompt_metadata(cfg, run),
                    **overall,
                })
    return output


def _secondary_metric_file(directory: Path, include_rerun: bool) -> Path | None:
    """Return the latest secondary image-metrics CSV for one directory."""
    if not directory.is_dir():
        return None
    if include_rerun:
        candidates = directory.glob("secondary_image_metrics_w_rerun_*.csv")
    else:
        candidates = (
            path for path in directory.glob("secondary_image_metrics_*.csv")
            if not path.name.startswith("secondary_image_metrics_w_rerun_")
        )
    return latest_file(candidates)


def _secondary_metric_rows(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    """Collect per-class secondary rows plus each run's class-overall rows."""
    output: list[dict[str, Any]] = []
    output_subdir = (
        _metrics_output_subdir(cfg.metrics_config)
        if cfg.metrics_config is not None else "metrics"
    )
    for run in runs:
        run_dir = Path(run["run_dir"])
        metadata = {
            "run_index": run["index"],
            "run_id": run["run_id"],
            "run_dir": str(run_dir),
            **_run_prompt_metadata(cfg, run),
        }
        for dataset in datasets:
            path = _secondary_metric_file(
                run_dir / dataset / output_subdir,
                cfg.metrics_include_rerun,
            )
            if path is None:
                continue
            with path.open(newline="", encoding="utf-8") as file_obj:
                for row in csv.DictReader(file_obj):
                    output.append({**metadata, **row})

        combined = _secondary_metric_file(run_dir, cfg.metrics_include_rerun)
        if combined is not None:
            with combined.open(newline="", encoding="utf-8") as file_obj:
                for row in csv.DictReader(file_obj):
                    if str(row.get("dataset", "")).lower() != "overall":
                        continue
                    row["dataset"] = "__experiment__"
                    output.append({**metadata, **row})
    return output


def _performance_rows(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        for dataset in datasets:
            path = latest_file((run_dir / dataset / "performance").glob("performance_*.json"))
            if path is None:
                continue
            perf = _read_json(path)
            num_images = _number(perf.get("num_images"))
            row: dict[str, Any] = {
                "run_index": run["index"], "run_id": run["run_id"],
                "run_dir": str(run_dir),
                **_run_prompt_metadata(cfg, run),
                "dataset": dataset,
                "detection_num_images": perf.get("num_images"),
                "detection_total_wall_time_sec": perf.get("total_wall_time_sec"),
                "detection_total_vlm_inference_time_sec": perf.get("total_vlm_inference_time_sec"),
                "detection_avg_vlm_inference_time_per_image_sec": perf.get("avg_vlm_inference_time_per_image_sec"),
                "detection_num_images_with_token_stats": perf.get("num_images_with_token_stats"),
                "detection_total_generated_tokens": perf.get("total_generated_tokens"),
                "detection_avg_generated_tokens_per_image": perf.get("avg_generated_tokens_per_image"),
                "detection_generated_tokens_per_second": perf.get("generated_tokens_per_second"),
                "detection_timing_basis": perf.get("timing_basis", "legacy_vlm_call"),
                "detection_attempts": _stage_record(run, "detection").get("attempts", 1),
            }
            wall = _number(perf.get("total_wall_time_sec"))
            if wall is not None and num_images:
                row["detection_wall_time_per_image_sec"] = wall / num_images
            # Object-crop cost, present only for runs with object_crop enabled
            # (absent keys keep historical aggregate CSV columns unchanged).
            for field in (
                "num_images_with_object_crop_timing",
                "total_object_crop_time_sec",
                "avg_object_crop_time_per_image_sec",
                "total_vlm_inference_time_with_object_crop_sec",
                "avg_vlm_inference_time_with_object_crop_per_image_sec",
            ):
                if field in perf:
                    row[f"detection_{field}"] = perf[field]

            rerun_path = _latest_valid_json_file(
                (run_dir / dataset / "rerun_results").glob("performance_*.json")
            )
            if rerun_path is not None:
                rerun = _read_json(rerun_path)
                row["rerun_latest_attempt_wall_time_sec"] = rerun.get("total_wall_time_sec")
            row.update(_rerun_artifact_timing(run_dir, dataset))
            rerun_record = _stage_record(run, "rerun", dataset)
            row["rerun_attempts"] = rerun_record.get("attempts", 1)

            mask_path = _latest_valid_json_file(
                (run_dir / dataset / "mask_generation").glob("performance_*.json")
            )
            mask_model_time = 0.0
            if mask_path is not None:
                mask = _read_json(mask_path)
                row["mask_records_verified"] = mask.get("records_verified")
                methods = mask.get("methods", {})
                if isinstance(methods, dict):
                    for method, values in methods.items():
                        if not isinstance(values, dict):
                            continue
                        safe_method = re.sub(r"[^A-Za-z0-9]+", "_", str(method)).strip("_").lower()
                        for field in ("num_images", "total_time_sec", "avg_time_per_image_sec"):
                            row[f"mask_{safe_method}_{field}"] = values.get(field)
                        total = _number(values.get("total_time_sec"))
                        if total is not None:
                            mask_model_time += total
            row["mask_attempts"] = _stage_record(run, "masks").get("attempts", 1)
            row["metrics_attempts"] = _stage_record(run, "metrics").get("attempts", 1)
            row["timing_has_resumed_stage"] = int(any(
                (_number(row.get(field)) or 1) > 1
                for field in ("detection_attempts", "rerun_attempts", "mask_attempts")
            ))

            detection_model = _number(perf.get("total_vlm_inference_time_sec")) or 0.0
            detection_model += _number(perf.get("total_object_crop_time_sec")) or 0.0
            rerun_model = _number(row.get("rerun_total_rerun_step_inference_time_sec")) or 0.0
            if num_images:
                row["all_model_stages_time_per_test_image_sec"] = (
                    detection_model + rerun_model + mask_model_time
                ) / num_images
            output.append(row)
    return output


def _parse_time(value: str) -> float | None:
    """Parse telemetry timestamps without letting a damaged row abort aggregation."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, TypeError, ValueError):
        return None


def _dataset_from_decision_path(raw_path: Any, datasets: Sequence[str]) -> str | None:
    if not isinstance(raw_path, str):
        return None
    normalized = raw_path.replace("\\", "/")
    for dataset in sorted(datasets, key=len, reverse=True):
        if f"/{dataset}/texts/" in f"/{normalized.lstrip('/')}":
            return dataset
    return None


def _inference_intervals(
    events_path: Path,
    datasets: Sequence[str],
) -> dict[str, list[tuple[str, float, float]]]:
    starts: dict[str, list[float]] = defaultdict(list)
    intervals: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    with events_path.open(encoding="utf-8") as file_obj:
        for raw in file_obj:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            decision_filepath = event.get("decision_filepath")
            dataset = _dataset_from_decision_path(decision_filepath, datasets)
            if dataset is None or kind not in {"vlm_generate_start", "vlm_generate_end"}:
                continue
            identity = str(decision_filepath).replace("\\", "/")
            timestamp = event.get("timestamp_utc") or event.get("timestamp_local")
            if not isinstance(timestamp, str):
                continue
            moment = _parse_time(timestamp)
            if moment is None:
                continue
            if kind == "vlm_generate_start":
                starts[identity].append(moment)
            elif starts[identity]:
                start = starts[identity].pop()
                if moment > start:
                    intervals[dataset].append((identity, start, moment))
    return intervals


def _samples_for_intervals(
    telemetry_path: Path,
    intervals: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, dict[str, float]]]]:
    output: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    ordered = {dataset: sorted(spans) for dataset, spans in intervals.items()}
    positions = {dataset: 0 for dataset in ordered}
    with telemetry_path.open(newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            raw_timestamp = row.get("timestamp_utc") or row.get("timestamp_local")
            if not raw_timestamp:
                continue
            timestamp = _parse_time(raw_timestamp)
            if timestamp is None:
                continue
            values = {
                "device_gpu_utilization_pct": _number(row.get("gpu0.utilization.gpu")),
                "device_gpu_memory_utilization_pct": _number(row.get("gpu0.utilization.memory")),
                "device_gpu_memory_used_mib": _number(row.get("gpu0.memory.used")),
                "device_gpu_power_w": _number(row.get("gpu0.power.draw")),
                "device_gpu_temperature_c": _number(row.get("gpu0.temperature.gpu")),
            }
            numeric_values = {key: value for key, value in values.items() if value is not None}
            if not numeric_values:
                continue
            for dataset, spans in ordered.items():
                position = positions[dataset]
                while position < len(spans) and timestamp > spans[position][1]:
                    position += 1
                positions[dataset] = position
                if position < len(spans) and spans[position][0] <= timestamp <= spans[position][1]:
                    output[dataset].append((timestamp, numeric_values))
    return output


def _process_gpu_samples_for_intervals(
    logs_dir: Path,
    intervals: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, dict[str, float]]]]:
    output: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    ordered = {dataset: sorted(spans) for dataset, spans in intervals.items()}
    for path in sorted(logs_dir.glob("process_gpu_detection_attempt_*.csv")):
        positions = {dataset: 0 for dataset in ordered}
        with path.open(newline="", encoding="utf-8") as file_obj:
            for row in csv.DictReader(file_obj):
                raw_timestamp = row.get("timestamp_utc") or row.get("timestamp_local")
                if not raw_timestamp:
                    continue
                timestamp = _parse_time(raw_timestamp)
                if timestamp is None:
                    continue
                values = {
                    field: number for field in (
                        "process_gpu_memory_mib",
                        "process_sm_utilization_pct",
                        "process_memory_utilization_pct",
                        "process_encoder_utilization_pct",
                        "process_decoder_utilization_pct",
                    ) if (number := _number(row.get(field))) is not None
                }
                if not values:
                    continue
                for dataset, spans in ordered.items():
                    position = positions[dataset]
                    while position < len(spans) and timestamp > spans[position][1]:
                        position += 1
                    positions[dataset] = position
                    if position < len(spans) and spans[position][0] <= timestamp <= spans[position][1]:
                        output[dataset].append((timestamp, values))
    return output


def _gpu_rows(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        combined_samples: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
        combined_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
        telemetry_dirs = sorted((run_dir / "logs").glob("hardware_telemetry_*")) if (run_dir / "logs").is_dir() else []
        telemetry_attempts: list[
            tuple[Path, dict[str, list[tuple[str, float, float]]]]
        ] = []
        for telemetry_dir in telemetry_dirs:
            events_path = telemetry_dir / "pipeline_events.jsonl"
            telemetry_path = telemetry_dir / "telemetry.csv"
            if not events_path.is_file() or not telemetry_path.is_file():
                continue
            telemetry_attempts.append(
                (telemetry_path, _inference_intervals(events_path, datasets))
            )

        # Resume works per image. The same decision path therefore identifies
        # the same image across attempts. Keep its latest completed VLM call:
        # images skipped on resume retain their earlier interval, while an
        # image rerun after an interrupted/invalid result replaces the older
        # interval instead of being counted twice.
        latest_by_image: dict[tuple[str, str], tuple[float, int, float]] = {}
        for attempt_index, (_, records) in enumerate(telemetry_attempts):
            for dataset, spans in records.items():
                for identity, start, end in spans:
                    key = (dataset, identity)
                    previous = latest_by_image.get(key)
                    if previous is None or end > previous[0]:
                        latest_by_image[key] = (end, attempt_index, start)

        for attempt_index, (telemetry_path, records) in enumerate(telemetry_attempts):
            selected: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for dataset, spans in records.items():
                for identity, start, end in spans:
                    winner = latest_by_image.get((dataset, identity))
                    if winner == (end, attempt_index, start):
                        selected[dataset].append((start, end))
            samples = _samples_for_intervals(telemetry_path, selected)
            for dataset, spans in selected.items():
                combined_intervals[dataset].extend(spans)
            for dataset, values in samples.items():
                combined_samples[dataset].extend(values)

        process_samples = _process_gpu_samples_for_intervals(
            run_dir / "logs", combined_intervals
        ) if (run_dir / "logs").is_dir() else {}

        for dataset in datasets:
            samples = sorted(combined_samples.get(dataset, []), key=lambda item: item[0])
            intervals = combined_intervals.get(dataset, [])
            if not samples and not intervals:
                continue
            row: dict[str, Any] = {
                "run_index": run["index"], "run_id": run["run_id"],
                "run_dir": str(run_dir),
                **_run_prompt_metadata(cfg, run),
                "dataset": dataset,
                "inference_intervals": len(intervals),
                "telemetry_samples": len(samples),
                "measured_inference_duration_sec": sum(end - start for start, end in intervals),
            }
            for field in (
                "device_gpu_utilization_pct", "device_gpu_memory_utilization_pct",
                "device_gpu_memory_used_mib", "device_gpu_power_w", "device_gpu_temperature_c",
            ):
                values = [sample[field] for _, sample in samples if field in sample]
                mean, within_std = _mean_std(values, 0)
                row[f"{field}_avg"] = mean if mean is not None else ""
                row[f"{field}_sample_std"] = within_std if within_std is not None else ""
                row[f"{field}_peak"] = max(values) if values else ""

            pid_samples = process_samples.get(dataset, [])
            row["process_telemetry_samples"] = len(pid_samples)
            for field in (
                "process_gpu_memory_mib", "process_sm_utilization_pct",
                "process_memory_utilization_pct", "process_encoder_utilization_pct",
                "process_decoder_utilization_pct",
            ):
                values = [sample[field] for _, sample in pid_samples if field in sample]
                mean, within_std = _mean_std(values, 0)
                row[f"{field}_avg"] = mean if mean is not None else ""
                row[f"{field}_sample_std"] = within_std if within_std is not None else ""
                row[f"{field}_peak"] = max(values) if values else ""

            # Energy is estimated per inference call from sampled mean power
            # over that call's exact event duration. This avoids integrating
            # model loading, CLIP, idle gaps, and resumed-dataset checks.
            energy_wh = 0.0
            powered_intervals = 0
            for start, end in intervals:
                power = [
                    sample["device_gpu_power_w"] for timestamp, sample in samples
                    if start <= timestamp <= end and "device_gpu_power_w" in sample
                ]
                if power:
                    energy_wh += statistics.fmean(power) * (end - start) / 3600.0
                    powered_intervals += 1
            row["device_gpu_energy_wh"] = energy_wh if powered_intervals else ""
            perf_path = latest_file((run_dir / dataset / "performance").glob("performance_*.json"))
            if perf_path is not None:
                num_images = _number(_read_json(perf_path).get("num_images"))
                if num_images and powered_intervals:
                    row["device_gpu_energy_wh_per_image"] = energy_wh / num_images
            output.append(row)
    return output


PROCESS_GPU_FIELDS = (
    "process_gpu_memory_mib",
    "process_sm_utilization_pct",
    "process_memory_utilization_pct",
    "process_encoder_utilization_pct",
    "process_decoder_utilization_pct",
)


def _rerun_intervals(events_path: Path) -> list[tuple[str, int, float, float]]:
    starts: dict[tuple[str, int], list[float]] = defaultdict(list)
    intervals: list[tuple[str, int, float, float]] = []
    for event in _jsonl_rows(events_path):
        kind = event.get("event")
        identity = event.get("image_identity")
        step = int(_number(event.get("step_number")) or 0)
        timestamp = event.get("timestamp_utc") or event.get("timestamp_local")
        if (
            not isinstance(identity, str)
            or not isinstance(timestamp, str)
            or kind not in {"rerun_vlm_generate_start", "rerun_vlm_generate_end"}
        ):
            continue
        key = (identity, step)
        moment = _parse_time(timestamp)
        if moment is None:
            continue
        if kind == "rerun_vlm_generate_start":
            starts[key].append(moment)
        elif starts[key]:
            start = starts[key].pop()
            if moment > start:
                intervals.append((identity, step, start, moment))
    return intervals


def _samples_by_identity(
    paths: Sequence[Path],
    intervals: Sequence[tuple[str, float, float]],
) -> dict[str, list[dict[str, float]]]:
    """Associate sampler rows with non-overlapping image inference intervals."""
    output: dict[str, list[dict[str, float]]] = defaultdict(list)
    ordered = sorted(intervals, key=lambda item: item[1])
    for path in paths:
        if not path.is_file():
            continue
        position = 0
        with path.open(newline="", encoding="utf-8") as file_obj:
            for row in csv.DictReader(file_obj):
                raw_timestamp = row.get("timestamp_utc") or row.get("timestamp_local")
                if not raw_timestamp:
                    continue
                timestamp = _parse_time(raw_timestamp)
                if timestamp is None:
                    continue
                while position < len(ordered) and timestamp > ordered[position][2]:
                    position += 1
                if position >= len(ordered):
                    break
                identity, start, end = ordered[position]
                if start <= timestamp <= end:
                    values = {
                        field: value for field in PROCESS_GPU_FIELDS
                        if (value := _number(row.get(field))) is not None
                    }
                    if values:
                        output[identity].append(values)
    return output


def _gpu_sample_metrics(samples: Sequence[dict[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {"process_telemetry_samples": len(samples)}
    for field in PROCESS_GPU_FIELDS:
        values = [sample[field] for sample in samples if field in sample]
        mean, sample_std = _mean_std(values, 0)
        result[f"{field}_avg"] = mean if mean is not None else ""
        result[f"{field}_sample_std"] = sample_std if sample_std is not None else ""
        result[f"{field}_peak"] = max(values) if values else ""
    return result


def _decision_metadata(run_dir: Path, dataset: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for entry in _jsonl_rows(run_dir / dataset / "all_data_summary.jsonl"):
        label, source, img_name = _artifact_image_key(entry)
        metadata = {
            "dataset": dataset,
            "label_folder": label,
            "source_label_folder": source,
            "img_name": img_name,
            "image_identity": "/".join((dataset, label, source, img_name)),
        }
        raw = str(entry.get("decision_filepath", "")).replace("\\", "/")
        if raw:
            # Decision paths stored in per-dataset summaries are commonly
            # relative (for example texts/good/00001.png.txt). Prefixing the
            # dataset prevents same-named images in another dataset from
            # overwriting or matching this metadata record.
            output[f"{dataset}/{raw.lstrip('/')}"] = metadata
    return output


def _metadata_for_decision(
    identity: str, metadata: dict[str, dict[str, str]]
) -> dict[str, str] | None:
    normalized = identity.replace("\\", "/")
    for suffix, value in metadata.items():
        if normalized == suffix or normalized.endswith("/" + suffix.lstrip("/")):
            return value
    return None


def _gpu_image_rows(
    cfg: AblationConfig,
    runs: Sequence[dict[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    """Official PID-attributed GPU statistics per image and inference phase."""
    output: list[dict[str, Any]] = []
    timing_rows = _image_timing_rows(cfg, runs, datasets)
    timing_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timing_rows:
        timing_by_run[str(row["run_dir"])].append(row)

    for run in runs:
        run_dir = Path(run["run_dir"])
        logs_dir = run_dir / "logs"
        base_by_identity = {
            str(row["image_identity"]): row
            for row in timing_by_run.get(str(run_dir), [])
            if row.get("phase") == "detection"
        }
        rerun_expected = {
            str(row["image_identity"])
            for row in timing_by_run.get(str(run_dir), [])
            if row.get("phase") == "rerun"
        }

        # Detection: select the latest completed interval for each image across
        # resume attempts, exactly like the legacy dataset-level calculation.
        detection_candidates: list[tuple[str, float, float]] = []
        decision_meta: dict[str, dict[str, str]] = {}
        for dataset in datasets:
            decision_meta.update(_decision_metadata(run_dir, dataset))
        for telemetry_dir in sorted(logs_dir.glob("hardware_telemetry_*")):
            events_path = telemetry_dir / "pipeline_events.jsonl"
            if not events_path.is_file():
                continue
            records = _inference_intervals(events_path, datasets)
            for spans in records.values():
                detection_candidates.extend(spans)
        latest_detection: dict[str, tuple[float, float]] = {}
        for decision_identity, start, end in detection_candidates:
            metadata = _metadata_for_decision(decision_identity, decision_meta)
            if metadata is None:
                continue
            image_identity = metadata["image_identity"]
            previous = latest_detection.get(image_identity)
            if previous is None or end > previous[1]:
                latest_detection[image_identity] = (start, end)
        detection_intervals = [
            (identity, span[0], span[1]) for identity, span in latest_detection.items()
        ]
        detection_samples = _samples_by_identity(
            sorted(logs_dir.glob("process_gpu_detection_attempt_*.csv")),
            detection_intervals,
        )

        # Rerun: each image may contain several VLM step intervals. Select the
        # latest completed copy per image/step across resumed attempts.
        latest_rerun_steps: dict[tuple[str, int], tuple[float, float]] = {}
        for events_path in sorted(logs_dir.glob("rerun_inference_*_events.jsonl")):
            for identity, step, start, end in _rerun_intervals(events_path):
                key = (identity, step)
                previous = latest_rerun_steps.get(key)
                if previous is None or end > previous[1]:
                    latest_rerun_steps[key] = (start, end)
        rerun_intervals = [
            (identity, span[0], span[1])
            for (identity, _), span in latest_rerun_steps.items()
        ]
        rerun_duration_by_image: dict[str, float] = defaultdict(float)
        for identity, start, end in rerun_intervals:
            rerun_duration_by_image[identity] += end - start
        rerun_samples = _samples_by_identity(
            sorted(logs_dir.glob("process_gpu_rerun_*_attempt_*.csv")),
            rerun_intervals,
        )

        for identity, timing_base in sorted(base_by_identity.items()):
            base_fields = (
                "run_index", "run_id", "run_dir", "dataset", "label_folder",
                "source_label_folder", "img_name", "image_identity",
            )
            if cfg.decision_prompt_sequence:
                base_fields = ("decision_prompt_template", *base_fields)
            base = {field: timing_base[field] for field in base_fields}
            detection_values = detection_samples.get(identity, [])
            rerun_values = rerun_samples.get(identity, [])
            detection_span = latest_detection.get(identity)
            detection_duration = (
                detection_span[1] - detection_span[0] if detection_span else None
            )
            rerun_duration = rerun_duration_by_image.get(identity)
            output.append({
                **base, "phase": "detection",
                "gpu_availability": "recorded" if detection_values else "not_recorded",
                "measured_inference_duration_sec": (
                    detection_duration if detection_duration is not None else ""
                ),
                **_gpu_sample_metrics(detection_values),
            })
            if identity in rerun_expected:
                output.append({
                    **base, "phase": "rerun",
                    "gpu_availability": "recorded" if rerun_values else "not_recorded_historical",
                    "measured_inference_duration_sec": rerun_duration or "",
                    **_gpu_sample_metrics(rerun_values),
                })
            combined_values = [*detection_values, *rerun_values]
            combined_complete = bool(detection_values) and (
                identity not in rerun_expected or bool(rerun_values)
            )
            output.append({
                **base, "phase": "combined",
                "gpu_availability": (
                    "recorded" if combined_complete else
                    "partial_detection_only" if detection_values else "not_recorded"
                ),
                "measured_inference_duration_sec": (
                    detection_duration + (rerun_duration or 0.0)
                    if detection_duration is not None else ""
                ),
                **_gpu_sample_metrics(combined_values),
            })
    return output


def _gpu_image_rollups(
    rows: Sequence[dict[str, Any]], group_fields: Sequence[str], ddof: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("gpu_availability") == "recorded":
            grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        result: dict[str, Any] = dict(zip(group_fields, key))
        for field in ("run_id", "run_dir"):
            if field not in result and field in group[0]:
                result[field] = group[0][field]
        result["image_count"] = len(group)
        for field in PROCESS_GPU_FIELDS:
            source = f"{field}_avg"
            values = [
                value for row in group if (value := _number(row.get(source))) is not None
            ]
            mean, std = _mean_std(values, ddof)
            result[f"{field}_mean_per_image"] = mean if mean is not None else ""
            result[f"{field}_std_across_images"] = std if std is not None else ""
            result[f"{field}_image_n"] = len(values)
        output.append(result)
    return output


def _representative_metric_rows(
    summary: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        key = (
            str(row.get("decision_prompt_template", "")),
            str(row.get("dataset", "")),
        )
        by_dataset[key].append(row)
    representatives: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in by_dataset.items():
        overall = next(
            (row for row in rows if str(row.get("anomaly_class", "")).lower() == "overall"),
            None,
        )
        if overall is not None:
            representatives[key] = overall
        elif len(rows) == 1:
            representatives[key] = rows[0]
    return representatives


def aggregate(cfg: AblationConfig, manifest: dict[str, Any], num_runs: int) -> dict[str, Path]:
    runs = manifest.get("runs", [])[:num_runs]
    datasets = list(manifest.get("datasets", []))
    if not runs:
        raise AblationError("There are no manifested runs to aggregate")

    metrics_raw = _metric_rows(cfg, runs, datasets)
    base_metrics_raw = (
        _metric_rows(
            cfg, runs, datasets, stage_name="base_metrics", include_rerun=False
        )
        if cfg.metrics_before_rerun else []
    )
    secondary_metrics_raw = _secondary_metric_rows(cfg, runs, datasets)
    timing_raw = _performance_rows(cfg, runs, datasets)
    gpu_raw = _gpu_rows(cfg, runs, datasets)
    gpu_image_raw = _gpu_image_rows(cfg, runs, datasets)
    gpu_image_dataset_runs = _gpu_image_rollups(
        gpu_image_raw, _group_fields(cfg, "run_index", "dataset", "phase"), cfg.std_ddof
    )
    gpu_image_run_summary = _gpu_image_rollups(
        gpu_image_raw, _group_fields(cfg, "run_index", "phase"), cfg.std_ddof
    )
    gpu_image_dataset_summary = _aggregate_numeric_rows(
        gpu_image_dataset_runs, _group_fields(cfg, "dataset", "phase"), cfg.std_ddof
    )
    gpu_image_experiment_summary = _aggregate_numeric_rows(
        gpu_image_run_summary, _group_fields(cfg, "phase"), cfg.std_ddof
    )
    image_timing_raw = _image_timing_rows(cfg, runs, datasets)
    image_timing_dataset_runs = _image_value_rollups(
        image_timing_raw,
        _group_fields(cfg, "run_index", "dataset", "phase"),
        "inference_time_sec",
        cfg.std_ddof,
    )
    image_timing_run_summary = _image_value_rollups(
        image_timing_raw,
        _group_fields(cfg, "run_index", "phase"),
        "inference_time_sec",
        cfg.std_ddof,
    )
    image_timing_dataset_summary = _aggregate_numeric_rows(
        image_timing_dataset_runs, _group_fields(cfg, "dataset", "phase"), cfg.std_ddof
    )
    image_timing_experiment_summary = _aggregate_numeric_rows(
        image_timing_run_summary, _group_fields(cfg, "phase"), cfg.std_ddof
    )
    metrics_summary = _aggregate_numeric_rows(
        metrics_raw, _group_fields(cfg, "dataset", "anomaly_class"), cfg.std_ddof
    )
    base_metrics_summary = _aggregate_numeric_rows(
        base_metrics_raw, _group_fields(cfg, "dataset", "anomaly_class"), cfg.std_ddof
    )
    secondary_metrics_summary = _aggregate_numeric_rows(
        secondary_metrics_raw,
        _group_fields(cfg, "dataset", "secondary_class"),
        cfg.std_ddof,
    )
    timing_summary = _aggregate_numeric_rows(
        timing_raw, _group_fields(cfg, "dataset"), cfg.std_ddof
    )
    gpu_summary = _aggregate_numeric_rows(
        gpu_raw, _group_fields(cfg, "dataset"), cfg.std_ddof
    )

    outputs = {
        "metrics_runs": cfg.runs_dir / "metrics_all_runs.csv",
        "metrics_summary": cfg.runs_dir / "metrics_mean_std.csv",
        "timing_runs": cfg.runs_dir / "timing_all_runs.csv",
        "timing_summary": cfg.runs_dir / "timing_mean_std.csv",
        "gpu_runs": cfg.runs_dir / "gpu_detection_inference_all_runs.csv",
        "gpu_summary": cfg.runs_dir / "gpu_detection_inference_mean_std.csv",
        "gpu_image": cfg.runs_dir / "gpu_inference_per_image.csv",
        "gpu_image_dataset_runs": cfg.runs_dir / "gpu_inference_dataset_runs.csv",
        "gpu_image_run_summary": cfg.runs_dir / "gpu_inference_run_summary.csv",
        "gpu_image_dataset_summary": cfg.runs_dir / "gpu_inference_dataset_mean_std.csv",
        "gpu_image_experiment_summary": cfg.runs_dir / "gpu_inference_experiment_mean_std.csv",
        "image_timing": cfg.runs_dir / "inference_timing_per_image.csv",
        "image_timing_dataset_runs": cfg.runs_dir / "inference_timing_dataset_runs.csv",
        "image_timing_run_summary": cfg.runs_dir / "inference_timing_run_summary.csv",
        "image_timing_dataset_summary": cfg.runs_dir / "inference_timing_dataset_mean_std.csv",
        "image_timing_experiment_summary": cfg.runs_dir / "inference_timing_experiment_mean_std.csv",
        "overall": cfg.runs_dir / "ablation_overall_mean_std.csv",
    }
    if base_metrics_raw:
        outputs.update({
            "base_metrics_runs": cfg.runs_dir / "metrics_before_rerun_all_runs.csv",
            "base_metrics_summary": cfg.runs_dir / "metrics_before_rerun_mean_std.csv",
        })
    if secondary_metrics_raw:
        outputs.update({
            "secondary_metrics_runs": cfg.runs_dir / "secondary_metrics_all_runs.csv",
            "secondary_metrics_summary": cfg.runs_dir / "secondary_metrics_mean_std.csv",
        })
    _write_csv(outputs["metrics_runs"], metrics_raw, ("run_index", "run_id", "run_dir", "dataset", "anomaly_class"))
    _write_csv(outputs["metrics_summary"], metrics_summary, ("dataset", "anomaly_class", "runs"))
    if base_metrics_raw:
        _write_csv(
            outputs["base_metrics_runs"], base_metrics_raw,
            ("run_index", "run_id", "run_dir", "dataset", "anomaly_class"),
        )
        _write_csv(
            outputs["base_metrics_summary"], base_metrics_summary,
            ("dataset", "anomaly_class", "runs"),
        )
    if secondary_metrics_raw:
        _write_csv(
            outputs["secondary_metrics_runs"],
            secondary_metrics_raw,
            ("run_index", "run_id", "run_dir", "dataset", "secondary_class"),
        )
        _write_csv(
            outputs["secondary_metrics_summary"],
            secondary_metrics_summary,
            ("dataset", "secondary_class", "runs"),
        )
    _write_csv(outputs["timing_runs"], timing_raw, ("run_index", "run_id", "run_dir", "dataset"))
    _write_csv(outputs["timing_summary"], timing_summary, ("dataset", "runs"))
    _write_csv(outputs["gpu_runs"], gpu_raw, ("run_index", "run_id", "run_dir", "dataset"))
    _write_csv(outputs["gpu_summary"], gpu_summary, ("dataset", "runs"))
    _write_csv(
        outputs["gpu_image"], gpu_image_raw,
        ("run_index", "run_id", "run_dir", "dataset", "label_folder",
         "source_label_folder", "img_name", "image_identity", "phase",
         "gpu_availability"),
    )
    _write_csv(
        outputs["gpu_image_dataset_runs"], gpu_image_dataset_runs,
        ("run_index", "run_id", "run_dir", "dataset", "phase"),
    )
    _write_csv(
        outputs["gpu_image_run_summary"], gpu_image_run_summary,
        ("run_index", "run_id", "run_dir", "phase"),
    )
    _write_csv(
        outputs["gpu_image_dataset_summary"], gpu_image_dataset_summary,
        ("dataset", "phase", "runs"),
    )
    _write_csv(
        outputs["gpu_image_experiment_summary"], gpu_image_experiment_summary,
        ("phase", "runs"),
    )
    _write_csv(
        outputs["image_timing"], image_timing_raw,
        ("run_index", "run_id", "run_dir", "dataset", "label_folder",
         "source_label_folder", "img_name", "image_identity", "phase"),
    )
    _write_csv(
        outputs["image_timing_dataset_runs"], image_timing_dataset_runs,
        ("run_index", "run_id", "run_dir", "dataset", "phase"),
    )
    _write_csv(
        outputs["image_timing_run_summary"], image_timing_run_summary,
        ("run_index", "run_id", "run_dir", "phase"),
    )
    _write_csv(
        outputs["image_timing_dataset_summary"], image_timing_dataset_summary,
        ("dataset", "phase", "runs"),
    )
    _write_csv(
        outputs["image_timing_experiment_summary"], image_timing_experiment_summary,
        ("phase", "runs"),
    )

    metric_representatives = _representative_metric_rows(metrics_summary)
    timing_by_dataset = {
        (str(row.get("decision_prompt_template", "")), str(row.get("dataset"))): row
        for row in timing_summary
    }
    gpu_by_dataset = {
        (str(row.get("decision_prompt_template", "")), str(row.get("dataset"))): row
        for row in gpu_summary
    }
    overall_rows: list[dict[str, Any]] = []
    prompt_keys = (
        sorted({
            _run_prompt_metadata(cfg, run).get("decision_prompt_template", "")
            for run in runs
        })
        if cfg.decision_prompt_sequence else [""]
    )
    for prompt in prompt_keys:
        for dataset in datasets:
            row: dict[str, Any] = {"dataset": dataset}
            if cfg.decision_prompt_sequence:
                row["decision_prompt_template"] = prompt
            for prefix, source in (
                ("metrics", metric_representatives.get((prompt, dataset), {})),
                ("timing", timing_by_dataset.get((prompt, dataset), {})),
                ("gpu_detection_inference", gpu_by_dataset.get((prompt, dataset), {})),
            ):
                for field, value in source.items():
                    if field not in {"dataset", "anomaly_class", "decision_prompt_template"}:
                        row[f"{prefix}_{field}"] = value
            overall_rows.append(row)
        experiment_metric = metric_representatives.get((prompt, "__experiment__"))
        if experiment_metric:
            row = {"dataset": "__experiment__"}
            if cfg.decision_prompt_sequence:
                row["decision_prompt_template"] = prompt
            for field, value in experiment_metric.items():
                if field not in {"dataset", "anomaly_class", "decision_prompt_template"}:
                    row[f"metrics_{field}"] = value
            overall_rows.append(row)
    _write_csv(outputs["overall"], overall_rows, ("dataset",))
    return outputs


def _operation_count(
    stages: Sequence[str], run_count: int, dataset_count: int,
    *, metrics_before_rerun: bool = False,
) -> int:
    total = 0
    if "detection" in stages:
        total += run_count
    if "rerun" in stages:
        total += run_count * dataset_count
    if "masks" in stages:
        total += run_count
    if "metrics" in stages:
        total += run_count
    if metrics_before_rerun:
        total += 2 * run_count
    return total


def _eta_state(
    runs: Sequence[dict[str, Any]],
    stages: Sequence[str],
    datasets: Sequence[str],
    *,
    metrics_before_rerun: bool = False,
) -> tuple[list[str], dict[str, list[float]]]:
    pending: list[str] = []
    samples: dict[str, list[float]] = defaultdict(list)

    def register(key: str, record: dict[str, Any]) -> None:
        duration = _number(record.get("duration_sec"))
        if record.get("status") == "complete":
            if duration is not None and duration > 0:
                samples[key].append(duration)
                generic = key.split(":", 1)[0]
                if generic != key:
                    samples[generic].append(duration)
        else:
            pending.append(key)

    for run in runs:
        if "detection" in stages:
            register("detection", _stage_record(run, "detection"))
        if "rerun" in stages:
            for dataset in datasets:
                register(f"rerun:{dataset}", _stage_record(run, "rerun", dataset))
        if "masks" in stages:
            register("masks", _stage_record(run, "masks"))
        if "metrics" in stages:
            register("metrics", _stage_record(run, "metrics"))
        if metrics_before_rerun:
            register("base_masks", _stage_record(run, "base_masks"))
            register("base_metrics", _stage_record(run, "base_metrics"))
    return pending, dict(samples)


def run_ablation(
    cfg: AblationConfig,
    *,
    num_runs: int,
    stages: Sequence[str],
    aggregate_only: bool,
    dry_run: bool,
    show_output: bool,
    console: Console,
) -> None:
    datasets = dataset_names_from_config(cfg.anomaly_config)
    dataset_counts = dataset_image_counts_from_config(cfg.anomaly_config)
    validate_config_files(cfg, stages, datasets, num_runs)
    manifest_path, manifest = load_or_create_manifest(cfg, datasets, num_runs, dry_run=dry_run)
    runs = manifest["runs"][:num_runs]

    console.print(
        f"[bold]Ablation:[/bold] {cfg.experiment_name} · {num_runs} runs · "
        f"{len(datasets)} datasets · stages: {', '.join(stages) if stages else 'aggregation only'}"
    )
    console.print(f"[bold]Runs directory:[/bold] {cfg.runs_dir}")
    if not show_output:
        console.print(f"[dim]Child output is saved under {cfg.runs_dir / 'orchestrator_logs'}.[/dim]")

    total = 1 if aggregate_only else _operation_count(
        stages, len(runs), len(datasets),
        metrics_before_rerun=cfg.metrics_before_rerun,
    ) + 1
    pending_keys, eta_samples = _eta_state(
        runs, stages, datasets,
        metrics_before_rerun=cfg.metrics_before_rerun,
    )
    eta_tracker = EtaTracker(pending_keys, eta_samples)
    with _ablation_diagnostics_session(
        cfg,
        num_runs=num_runs,
        stages=stages,
        aggregate_only=aggregate_only,
        dry_run=dry_run,
        console=console,
    ) as diagnostics_monitor, Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None), TaskProgressColumn(),
        TextColumn("[dim]{task.fields[elapsed_label]}"), TimeElapsedColumn(),
        TextColumn("[cyan]{task.fields[eta_text]}"),
        TextColumn("[dim]{task.fields[finish_text]}"),
        console=console, expand=True,
    ) as progress:
        initial_eta, initial_finish = eta_tracker.display_fields()
        task = progress.add_task(
            "Preparing",
            total=total,
            elapsed_label="overall elapsed",
            eta_text=f"ETA {initial_eta}",
            finish_text=f"finish {initial_finish}",
        )
        try:
            if not aggregate_only:
                # Stages are deliberately batched globally: all first runs,
                # then every rerun, then masks, then metrics.
                # Detection and rerun use the same VLM. In cluster mode the
                # orchestrator owns one server across all child processes,
                # avoiding a 32B cold reload for every run/dataset. Mask
                # generation is excluded because it must release vLLM before
                # loading its local segmentation model on the controller GPU.
                shared_server_started = False
                try:
                    if not dry_run and ({"detection", "rerun"} & set(stages)):
                        shared_models = load_models_config(cfg.model_config)
                        if shared_models.vlm.server_launch is not None:
                            _diagnostics_call(
                                diagnostics_monitor,
                                "mark_event",
                                "shared_vllm_server_start",
                                model=shared_models.vlm.api_model_name,
                            )
                            ServerManager.ensure_running(
                                shared_models.vlm,
                                "ABLATION_VLM",
                                shared_models.deployment,
                            )
                            os.environ[EXTERNAL_SERVER_ENV] = "1"
                            shared_server_started = True
                            _diagnostics_call(
                                diagnostics_monitor,
                                "mark_event",
                                "shared_vllm_server_ready",
                                model=shared_models.vlm.api_model_name,
                            )
                    if "detection" in stages:
                        for run in runs:
                            _run_detection(
                                cfg, datasets, run, manifest_path, manifest, progress, task,
                                len(runs), eta_tracker, dataset_counts, diagnostics_monitor,
                                dry_run=dry_run, show_output=show_output,
                            )
                            if cfg.cooldown_seconds and run is not runs[-1] and not dry_run:
                                progress.update(task, description=f"Cooling down for {cfg.cooldown_seconds:g}s")
                                time.sleep(cfg.cooldown_seconds)
                    if cfg.metrics_before_rerun:
                        # Pixel metrics require masks, and mask generation must
                        # run without the shared VLM server occupying the GPU.
                        os.environ.pop(EXTERNAL_SERVER_ENV, None)
                        if shared_server_started:
                            ServerManager.stop(tag="ablation_base_evaluation")
                            shared_server_started = False
                        for run in runs:
                            _run_masks(
                                cfg, datasets, run, manifest_path, manifest, progress, task,
                                len(runs), eta_tracker, dataset_counts, diagnostics_monitor,
                                dry_run=dry_run, show_output=show_output,
                                stage_name="base_masks", include_rerun=False,
                            )
                            _run_metrics(
                                cfg, datasets, run, manifest_path, manifest, progress, task,
                                len(runs), eta_tracker, dataset_counts, diagnostics_monitor,
                                dry_run=dry_run, show_output=show_output,
                                stage_name="base_metrics", include_rerun=False,
                            )
                            if not dry_run:
                                _archive_base_mask_artifacts(Path(run["run_dir"]), datasets)

                        if not dry_run and "rerun" in stages:
                            shared_models = load_models_config(cfg.model_config)
                            if shared_models.vlm.server_launch is not None:
                                ServerManager.ensure_running(
                                    shared_models.vlm, "ABLATION_VLM", shared_models.deployment
                                )
                                os.environ[EXTERNAL_SERVER_ENV] = "1"
                                shared_server_started = True
                    if "rerun" in stages:
                        for run in runs:
                            for dataset in datasets:
                                _run_rerun_dataset(
                                    cfg, datasets, run, dataset, manifest_path, manifest, progress, task,
                                    len(runs), eta_tracker, dataset_counts, diagnostics_monitor,
                                    dry_run=dry_run, show_output=show_output,
                                )
                finally:
                    os.environ.pop(EXTERNAL_SERVER_ENV, None)
                    if shared_server_started:
                        _diagnostics_call(
                            diagnostics_monitor,
                            "mark_event",
                            "shared_vllm_server_stop",
                        )
                        ServerManager.stop(tag="ablation_detection_rerun_complete")
                        _diagnostics_call(
                            diagnostics_monitor,
                            "mark_event",
                            "shared_vllm_server_stopped",
                        )
                if "masks" in stages:
                    for run in runs:
                        _run_masks(
                            cfg, datasets, run, manifest_path, manifest, progress, task,
                            len(runs), eta_tracker, dataset_counts, diagnostics_monitor,
                            dry_run=dry_run, show_output=show_output,
                        )
                if "metrics" in stages:
                    for run in runs:
                        _run_metrics(
                            cfg, datasets, run, manifest_path, manifest, progress, task,
                            len(runs), eta_tracker, dataset_counts, diagnostics_monitor,
                            dry_run=dry_run, show_output=show_output,
                        )

            progress.update(task, description="Aggregating metrics, timing, and GPU telemetry")
            if dry_run:
                progress.update(task, advance=1, description="Dry run complete")
                return
            outputs = aggregate(cfg, manifest, num_runs)
            manifest["aggregation"] = {
                "status": "complete", "completed_at": _now(),
                "num_runs_requested": num_runs,
                "outputs": {name: str(path) for name, path in outputs.items()},
            }
            save_manifest(manifest_path, manifest)
            progress.update(task, advance=1, description="Ablation complete")
        except KeyboardInterrupt:
            if not dry_run:
                manifest["last_interrupted_at"] = _now()
                save_manifest(manifest_path, manifest)
            raise

    console.print("[bold green]Ablation orchestration complete.[/bold green]")
    console.print(f"Metrics summary: {cfg.runs_dir / 'metrics_mean_std.csv'}")
    if "secondary_metrics_summary" in outputs:
        console.print(f"Secondary summary: {outputs['secondary_metrics_summary']}")
    console.print(f"Timing summary : {cfg.runs_dir / 'timing_mean_std.csv'}")
    console.print(f"GPU summary    : {cfg.runs_dir / 'gpu_detection_inference_mean_std.csv'}")
    console.print(f"Overall summary: {cfg.runs_dir / 'ablation_overall_mean_std.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable multi-run anomaly-detection ablation orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Ablation orchestrator YAML")
    parser.add_argument("--runs", type=int, default=None, help="Override experiment.num_runs")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_ORDER, default=None,
        help="Run only these stages, in canonical pipeline order",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=STAGE_ORDER, default=[],
        help="Skip selected configured stages",
    )
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild aggregate CSV files")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without executing them")
    parser.add_argument(
        "--show-child-output", action="store_true",
        help="Echo child pipeline output in addition to writing per-stage logs",
    )
    parser.add_argument(
        "--generation-profile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Record generated-token counts and throughput; overrides "
            "telemetry.generation_profile"
        ),
    )
    parser.add_argument(
        "--prediction-source", choices=("rerun", "base"), default=None,
        help=(
            "Override masks.include_rerun and metrics.include_rerun together. "
            "Use 'base' when skipping rerun but still generating masks/metrics."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    try:
        cfg = load_ablation_config(args.config)
        if args.generation_profile is not None:
            cfg = replace(cfg, generation_profile=args.generation_profile)
        if args.prediction_source is not None:
            use_rerun = args.prediction_source == "rerun"
            cfg = replace(
                cfg,
                mask_include_rerun=use_rerun,
                metrics_include_rerun=use_rerun,
            )
        num_runs = args.runs if args.runs is not None else cfg.num_runs
        if num_runs < 1:
            raise AblationError("--runs must be at least 1")
        selected = tuple(args.stages) if args.stages is not None else cfg.enabled_stages
        skipped = set(args.skip)
        stages = tuple(stage for stage in STAGE_ORDER if stage in selected and stage not in skipped)
        run_ablation(
            cfg,
            num_runs=num_runs,
            stages=stages,
            aggregate_only=args.aggregate_only,
            dry_run=args.dry_run,
            show_output=args.show_child_output,
            console=console,
        )
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Re-run the same command to resume.[/yellow]")
        return 130
    except (AblationError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        console.print(f"[bold red]Ablation failed:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
