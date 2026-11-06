"""Lifecycle manager for a single auto-managed vLLM API server.

Planner generation swaps VLM -> LLM -> VLM.  Other stages need only the VLM.
All use this same manager so server startup, Ray topology validation, readiness
checks, logging, and teardown have identical behavior.

Ray nodes are intentionally not managed here.  Operators start the Ray head
and workers before the pipeline; this class validates the live cluster and
starts/stops only ``vllm serve`` on the machine running the pipeline.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

from .cluster_runtime import (
    ClusterPlan,
    inspect_cluster_runtimes,
    render_server_command,
    resolve_cluster_plan,
)
from .config import DeploymentConfig

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent / "logs" / "vllm_server_logs"
EXTERNAL_SERVER_ENV = "PIPELINE_VLLM_SERVER_MANAGED_EXTERNALLY"


def _record_event(event: str, **details: Any) -> None:
    """Best-effort event recording without importing a pipeline package."""

    raw_path = os.environ.get("PIPELINE_TELEMETRY_EVENTS_FILE")
    if not raw_path:
        return
    payload = {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "event": event,
        **details,
    }
    try:
        with Path(raw_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        logger.debug("Could not record telemetry event %s", event, exc_info=True)


class ServerManager:
    """Process-wide owner of at most one managed ``vllm serve`` process."""

    _process: subprocess.Popen[Any] | None = None
    _log_file: Any = None
    _current_model: str | None = None
    _served_model_name: str | None = None
    _base_url: str | None = None
    _command: tuple[str, ...] | None = None
    _plan: ClusterPlan | None = None
    _lock = threading.RLock()
    _atexit_registered = False
    _runtime_preflight_cache: set[tuple[str, str, tuple[str, ...]]] = set()

    @classmethod
    def ensure_running(
        cls,
        model_cfg: Any,
        tag: str,
        deployment: DeploymentConfig | None,
    ) -> None:
        """Start or swap the configured server and wait until it is ready."""

        launch = getattr(model_cfg, "server_launch", None)
        if launch is None:
            # A base_url without server_launch is an explicitly external server.
            return
        if os.environ.get(EXTERNAL_SERVER_ENV) == "1":
            cls._verify_external_server(model_cfg, tag)
            return
        if deployment is None or not deployment.is_cluster:
            raise RuntimeError(f"[{tag}] Managed vLLM servers require deployment.mode=cluster.")

        with cls._lock:
            plan = resolve_cluster_plan(deployment)
            target_model = model_cfg.selected_model
            served_model_name = model_cfg.api_model_name
            target_url = model_cfg.base_url.rstrip("/")
            command_list = render_server_command(
                launch.command,
                plan,
                model=target_model,
                served_model_name=served_model_name,
            )
            executable = shutil.which(command_list[0])
            if executable is None:
                sibling = Path(sys.executable).resolve().parent / command_list[0]
                if sibling.is_file() and os.access(sibling, os.X_OK):
                    executable = str(sibling)
            if executable is None:
                raise RuntimeError(
                    f"[{tag}] Cannot find server executable {command_list[0]!r} in PATH "
                    f"or beside the active Python interpreter {sys.executable}."
                )
            command_list[0] = executable
            command = tuple(
                command_list
            )
            if (
                cls._process is not None
                and cls._process.poll() is None
                and cls._current_model == target_model
                and cls._served_model_name == served_model_name
                and cls._base_url == target_url
                and cls._command == command
            ):
                logger.info("[%s] vLLM server already running (pid=%s)", tag, cls._process.pid)
                return

            cls.stop(tag=f"{tag}:swap")
            cls._start(model_cfg, tag, deployment, plan, command)

    @classmethod
    def _verify_external_server(cls, model_cfg: Any, tag: str) -> None:
        """Validate a server owned by a parent ablation orchestrator."""

        base_url = model_cfg.base_url.rstrip("/")
        try:
            response = requests.get(f"{base_url}/models", timeout=30)
            response.raise_for_status()
            available = [item.get("id") for item in response.json().get("data", [])]
        except Exception as exc:
            raise RuntimeError(
                f"[{tag}] Parent-managed vLLM server at {base_url} is unavailable: {exc}"
            ) from exc
        if model_cfg.api_model_name not in available:
            raise RuntimeError(
                f"[{tag}] Parent-managed server does not expose "
                f"{model_cfg.api_model_name!r}; available={available}."
            )

    @classmethod
    def stop(cls, tag: str = "cleanup") -> None:
        """Stop the managed server process group; external servers are untouched."""

        with cls._lock:
            process = cls._process
            if process is None:
                return
            _record_event(
                "vllm_server_stop_start",
                tag=tag,
                server_pid=process.pid,
                model=cls._served_model_name,
            )
            logger.info(
                "[%s] Stopping managed vLLM server pid=%s model=%s",
                tag,
                process.pid,
                cls._served_model_name,
            )
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=60)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        logger.error("[%s] vLLM process group did not exit after SIGKILL", tag)
            if cls._log_file is not None:
                try:
                    cls._log_file.close()
                except OSError:
                    pass
            _record_event(
                "vllm_server_stopped",
                tag=tag,
                server_pid=process.pid,
                return_code=process.poll(),
            )
            cls._clear_state()
            # Ray actors and CUDA allocations can lag behind frontend exit.
            time.sleep(2)

    @classmethod
    def _clear_state(cls) -> None:
        cls._process = None
        cls._log_file = None
        cls._current_model = None
        cls._served_model_name = None
        cls._base_url = None
        cls._command = None
        cls._plan = None

    @classmethod
    def _start(
        cls,
        model_cfg: Any,
        tag: str,
        deployment: DeploymentConfig,
        plan: ClusterPlan,
        command: tuple[str, ...],
    ) -> None:
        launch = model_cfg.server_launch
        cls._validate_cluster_runtimes(model_cfg, deployment, plan, tag)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"vllm_serve_{tag.lower()}_{stamp}.log"

        env = os.environ.copy()
        env.update(launch.env)
        env.setdefault("RAY_ADDRESS", deployment.cluster.ray_address)
        log_file = log_path.open("w", encoding="utf-8")
        _record_event(
            "vllm_server_start",
            tag=tag,
            model=model_cfg.selected_model,
            served_model_name=model_cfg.api_model_name,
            command=list(command),
            requested_gpu_count=plan.requested_gpu_count,
            available_gpu_count=plan.available_gpu_count,
            tensor_parallel_size=plan.tensor_parallel_size,
            pipeline_parallel_size=plan.pipeline_parallel_size,
            nodes=[
                {"host": node.host, "gpus": node.gpu_count, "cpus": node.cpu_count}
                for node in plan.active_gpu_nodes
            ],
            log_path=str(log_path),
        )
        logger.info(
            "[%s] Starting vLLM with %d/%d Ray GPU(s), TP=%d PP=%d; log=%s",
            tag,
            plan.requested_gpu_count,
            plan.available_gpu_count,
            plan.tensor_parallel_size,
            plan.pipeline_parallel_size,
            log_path,
        )
        try:
            process = subprocess.Popen(
                list(command),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise

        # Store state immediately so every failure path can terminate it.
        cls._process = process
        cls._log_file = log_file
        cls._current_model = model_cfg.selected_model
        cls._served_model_name = model_cfg.api_model_name
        cls._base_url = model_cfg.base_url.rstrip("/")
        cls._command = command
        cls._plan = plan
        if not cls._atexit_registered:
            atexit.register(cls.stop, "atexit")
            cls._atexit_registered = True

        deadline = time.monotonic() + launch.ready_timeout
        session = requests.Session()
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"[{tag}] vLLM exited during startup with code {process.returncode}. "
                        f"See {log_path}."
                    )
                try:
                    response = session.get(f"{cls._base_url}/models", timeout=5)
                    if response.ok:
                        available = [
                            item.get("id") for item in response.json().get("data", [])
                        ]
                        if cls._served_model_name in available:
                            _record_event(
                                "vllm_server_ready",
                                tag=tag,
                                server_pid=process.pid,
                                model=cls._served_model_name,
                            )
                            logger.info(
                                "[%s] vLLM ready pid=%s model=%s",
                                tag,
                                process.pid,
                                cls._served_model_name,
                            )
                            return
                except (requests.RequestException, ValueError, KeyError):
                    pass
                time.sleep(5)
            raise TimeoutError(
                f"[{tag}] vLLM did not serve {cls._served_model_name!r} within "
                f"{launch.ready_timeout:.0f}s. See {log_path}."
            )
        except Exception:
            cls.stop(tag=f"{tag}:startup_failed")
            raise
        finally:
            session.close()

    @classmethod
    def _validate_cluster_runtimes(
        cls,
        model_cfg: Any,
        deployment: DeploymentConfig,
        plan: ClusterPlan,
        tag: str,
    ) -> None:
        """Verify the actual Python/Ray/vLLM/Torch worker environments."""
        model_path = Path(model_cfg.selected_model)
        cache_key = (
            deployment.cluster.ray_address,
            str(model_path),
            tuple(node.node_id for node in plan.active_gpu_nodes),
        )
        if cache_key in cls._runtime_preflight_cache:
            return
        paths = (str(model_path),) if model_path.is_absolute() else ()
        runtimes = inspect_cluster_runtimes(
            deployment,
            plan,
            model_paths=paths,
        )
        logger.info(
            "[%s] Ray runtime preflight passed: %s",
            tag,
            "; ".join(
                f"{runtime.host} gpu={','.join(runtime.gpu_names) or 'unknown'} "
                f"vllm={runtime.vllm_version} torch={runtime.torch_version}"
                for runtime in runtimes
            ),
        )
        cls._runtime_preflight_cache.add(cache_key)


@contextmanager
def managed_server_lifecycle(tag: str = "pipeline") -> Iterator[None]:
    """Guarantee cleanup of any server started inside the context."""

    try:
        yield
    finally:
        ServerManager.stop(tag=tag)
