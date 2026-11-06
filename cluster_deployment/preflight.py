"""Validate local/cluster deployment without loading a model.

Use this after starting Ray and before a long planner/ablation run::

    python -m cluster_deployment.preflight --model-config configs/model_configs.yaml

Cluster mode checks live Ray GPU resources, executes a tiny inspection task in
every Ray GPU worker environment, verifies matching Ray/vLLM/PyTorch/CUDA
versions and model paths, and confirms passwordless SSH plus the telemetry
Python interpreter on every remote machine.  No GPU model is loaded.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .cluster_runtime import (
    inspect_cluster_runtimes,
    resolve_cluster_plan,
    resolve_remote_monitor_nodes,
)
from .config import apply_model_profile, load_model_config_bundle
from .remote_monitor import SSH_BASE_OPTIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _selected_model_paths(raw: dict, mode: str) -> tuple[str, ...]:
    models = raw.get("models") or {}
    paths: list[str] = []
    for name in ("vlm", "llm"):
        if name not in models:
            continue
        selected = apply_model_profile(models[name], mode, f"models.{name}")
        if selected.get("source") != "local":
            continue
        value = selected.get("local_path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"models.{name}.local_path is required for source=local.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        paths.append(str(path.resolve()))
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only deployment/Ray/vLLM/telemetry preflight"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_configs.yaml",
    )
    parser.add_argument(
        "--telemetry-smoke-seconds",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "After preflight succeeds, actually start local+remote recorders, "
            "sample for this many seconds, stop them, and copy remote data to /tmp"
        ),
    )
    args = parser.parse_args()
    config_path = args.model_config.expanduser().resolve()
    raw, deployment = load_model_config_bundle(config_path)
    model_paths = _selected_model_paths(raw, deployment.mode)

    if deployment.mode == "local":
        missing = [path for path in model_paths if not Path(path).is_dir()]
        if missing:
            raise SystemExit("Local model directories are missing: " + ", ".join(missing))
        print(json.dumps({"mode": "local", "model_paths": model_paths, "status": "ready"}, indent=2))
        return 0

    plan = resolve_cluster_plan(deployment)
    runtimes = inspect_cluster_runtimes(deployment, plan, model_paths=model_paths)
    remote_nodes = resolve_remote_monitor_nodes(deployment, ray_nodes=plan.nodes)
    ssh_results: list[dict[str, object]] = []
    for node in remote_nodes:
        command = (
            f"test -x {shlex.quote(node.python)} && "
            f"{shlex.quote(node.python)} --version && "
            "nvidia-smi --query-gpu=name --format=csv,noheader"
        )
        result = subprocess.run(
            ["ssh", *SSH_BASE_OPTIONS, *node.ssh_options, node.target, command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ssh_results.append(
            {
                "name": node.name,
                "host": node.host,
                "ok": result.returncode == 0,
                "output": (result.stdout or result.stderr).strip(),
            }
        )
    failed_ssh = [item["host"] for item in ssh_results if not item["ok"]]
    report = {
        "mode": "cluster",
        "ray_address": plan.ray_address,
        "available_gpu_count": plan.available_gpu_count,
        "requested_gpu_count": plan.requested_gpu_count,
        "tensor_parallel_size": plan.tensor_parallel_size,
        "pipeline_parallel_size": plan.pipeline_parallel_size,
        "ray_nodes": [
            {"host": node.host, "gpus": node.gpu_count, "cpus": node.cpu_count}
            for node in plan.active_gpu_nodes
        ],
        "worker_runtimes": [
            {
                "host": runtime.host,
                "python": runtime.python_executable,
                "ray": runtime.ray_version,
                "vllm": runtime.vllm_version,
                "torch": runtime.torch_version,
                "torch_cuda": runtime.torch_cuda_version,
                "gpus": runtime.gpu_names,
                "model_paths": runtime.model_paths,
            }
            for runtime in runtimes
        ],
        "remote_telemetry_ssh": ssh_results,
        "status": (
            "not_ready"
            if failed_ssh and deployment.remote_monitoring.required
            else "ready"
        ),
    }
    if failed_ssh and deployment.remote_monitoring.required:
        print(json.dumps(report, indent=2))
        print(
            "Passwordless SSH/telemetry preflight failed for: "
            + ", ".join(str(item) for item in failed_ssh),
            file=sys.stderr,
        )
        return 1
    if args.telemetry_smoke_seconds > 0:
        if args.telemetry_smoke_seconds < 1:
            raise SystemExit("--telemetry-smoke-seconds must be at least 1 second.")
        from anomaly_detection.hardware_monitor import HardwareMonitor
        from .remote_monitor import build_monitor

        telemetry_dir = Path("/tmp") / f'pipeline_cluster_telemetry_smoke_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        monitor = build_monitor(
            HardwareMonitor(
                output_dir=telemetry_dir,
                interval=0.5,
                label="cluster_preflight_smoke",
                archive_enabled=False,
            ),
            deployment,
            interval=0.5,
            label="cluster_preflight_smoke",
            context={"pipeline": "cluster_preflight", "purpose": "telemetry_smoke_test"},
        )
        if not monitor.start():
            raise SystemExit("Telemetry smoke monitor failed to start.")
        try:
            monitor.mark_event("smoke_test_start")
            time.sleep(args.telemetry_smoke_seconds)
            monitor.mark_event("smoke_test_end")
        finally:
            monitor.stop(reason="smoke_test_finished")
        report["telemetry_smoke"] = {
            "seconds": args.telemetry_smoke_seconds,
            "output_directory": str(telemetry_dir),
            "status": "passed",
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
