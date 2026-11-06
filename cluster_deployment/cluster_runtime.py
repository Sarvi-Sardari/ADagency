"""Read-only Ray discovery and vLLM topology planning.

Ray itself is managed outside the pipeline (``ray start`` on each machine).
This module attaches briefly as a driver, reads the live resource table, and
disconnects.  It never starts/stops Ray nodes or submits GPU work.
"""
from __future__ import annotations

import ipaddress
import logging
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from .config import DeploymentConfig, RemoteNodeConfig

logger = logging.getLogger(__name__)


class ClusterConfigurationError(RuntimeError):
    """The live Ray cluster cannot satisfy the configured topology."""


@dataclass(frozen=True)
class RayNode:
    node_id: str
    host: str
    gpu_count: int
    cpu_count: int
    alive: bool = True


@dataclass(frozen=True)
class ClusterPlan:
    ray_address: str
    nodes: tuple[RayNode, ...]
    available_gpu_count: int
    requested_gpu_count: int
    tensor_parallel_size: int
    pipeline_parallel_size: int

    @property
    def active_gpu_nodes(self) -> tuple[RayNode, ...]:
        return tuple(node for node in self.nodes if node.alive and node.gpu_count > 0)


@dataclass(frozen=True)
class NodeRuntime:
    node_id: str
    host: str
    python_executable: str
    python_version: str
    ray_version: str | None
    vllm_version: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    gpu_names: tuple[str, ...]
    model_paths: dict[str, bool]


def _normalise_host(value: str) -> str:
    value = value.strip()
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        try:
            return socket.gethostbyname(value)
        except OSError:
            return value


def local_host_candidates(configured_host: str | None = None) -> set[str]:
    """Return names/IPs that identify the machine running this process."""

    candidates = {"127.0.0.1", "localhost", socket.gethostname(), socket.getfqdn()}
    if configured_host:
        candidates.add(configured_host)
    for name in tuple(candidates):
        try:
            candidates.add(socket.gethostbyname(name))
        except OSError:
            continue
    return {_normalise_host(item) for item in candidates if item}


def _ray_nodes_from_runtime(address: str) -> list[dict[str, Any]]:
    """Fetch Ray's structured node table without retaining a driver."""

    try:
        import ray
    except ImportError as exc:
        raise ClusterConfigurationError(
            "Cluster mode requires Ray in the active Python environment."
        ) from exc

    already_initialised = ray.is_initialized()
    try:
        if not already_initialised:
            ray.init(
                address=address,
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                log_to_driver=False,
            )
        return list(ray.nodes())
    except Exception as exc:
        raise ClusterConfigurationError(
            f"Could not connect to Ray at {address!r}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if not already_initialised:
            try:
                ray.shutdown()
            except Exception:
                logger.debug("Ray driver shutdown failed", exc_info=True)


def parse_ray_nodes(records: Iterable[dict[str, Any]]) -> tuple[RayNode, ...]:
    """Convert ``ray.nodes()`` records into stable, testable node objects."""

    nodes: list[RayNode] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        resources = record.get("Resources") or {}
        host = str(record.get("NodeManagerAddress") or "").strip()
        if not host:
            continue
        gpu_count = max(0, int(math.floor(float(resources.get("GPU", 0) or 0))))
        cpu_count = max(0, int(math.floor(float(resources.get("CPU", 0) or 0))))
        nodes.append(
            RayNode(
                node_id=str(record.get("NodeID") or record.get("NodeManagerHostname") or host),
                host=_normalise_host(host),
                gpu_count=gpu_count,
                cpu_count=cpu_count,
                alive=bool(record.get("Alive", False)),
            )
        )
    return tuple(sorted(nodes, key=lambda node: (node.host, node.node_id)))


def discover_ray_nodes(deployment: DeploymentConfig) -> tuple[RayNode, ...]:
    if not deployment.is_cluster:
        return ()
    nodes = parse_ray_nodes(_ray_nodes_from_runtime(deployment.cluster.ray_address))
    active = tuple(node for node in nodes if node.alive)
    if not active:
        raise ClusterConfigurationError(
            f"Ray at {deployment.cluster.ray_address!r} reports no active nodes."
        )
    return active


def resolve_cluster_plan(
    deployment: DeploymentConfig,
    *,
    nodes: tuple[RayNode, ...] | None = None,
) -> ClusterPlan:
    """Validate the configured GPU limit and calculate vLLM TP/PP sizes."""

    if not deployment.is_cluster:
        raise ClusterConfigurationError("A Ray cluster plan was requested in local mode.")
    nodes = discover_ray_nodes(deployment) if nodes is None else nodes
    gpu_nodes = tuple(node for node in nodes if node.alive and node.gpu_count > 0)
    available = sum(node.gpu_count for node in gpu_nodes)
    if available < 1:
        raise ClusterConfigurationError("The active Ray cluster advertises no GPU resources.")

    requested = deployment.cluster.gpu_count or available
    if requested > available:
        raise ClusterConfigurationError(
            f"deployment.cluster.gpu_count requests {requested} GPU(s), but Ray "
            f"currently advertises only {available} across {len(gpu_nodes)} active GPU node(s)."
        )

    tensor = deployment.cluster.tensor_parallel_size
    if requested % tensor:
        raise ClusterConfigurationError(
            f"Requested GPU count {requested} is not divisible by tensor_parallel_size={tensor}."
        )
    pipeline = requested // tensor
    return ClusterPlan(
        ray_address=deployment.cluster.ray_address,
        nodes=tuple(nodes),
        available_gpu_count=available,
        requested_gpu_count=requested,
        tensor_parallel_size=tensor,
        pipeline_parallel_size=pipeline,
    )


def resolve_remote_monitor_nodes(
    deployment: DeploymentConfig,
    *,
    ray_nodes: tuple[RayNode, ...] | None = None,
) -> tuple[RemoteNodeConfig, ...]:
    """Resolve one SSH recorder for every active remote Ray machine.

    Explicit ``nodes`` entries act as per-host overrides.  With Ray discovery,
    newly joined machines inherit the remote-monitor defaults, so adding a
    third PC does not require a code change.
    """

    monitoring = deployment.remote_monitoring
    if not monitoring.enabled:
        return ()
    if not deployment.is_cluster and not monitoring.monitor_in_local_mode:
        return ()

    explicit = {_normalise_host(node.host): node for node in monitoring.nodes}
    if not monitoring.discover_from_ray:
        return tuple(explicit.values())

    ray_nodes = discover_ray_nodes(deployment) if ray_nodes is None else ray_nodes
    local_hosts = local_host_candidates(deployment.cluster.local_host)
    resolved: list[RemoteNodeConfig] = []
    for ray_node in ray_nodes:
        if not ray_node.alive or ray_node.gpu_count < 1:
            continue
        host = _normalise_host(ray_node.host)
        if host in local_hosts:
            continue
        override = explicit.get(host)
        if override is not None:
            resolved.append(override)
            continue
        if not monitoring.default_user:
            raise ClusterConfigurationError(
                f"Ray discovered remote GPU node {host}, but no explicit telemetry "
                "node or deployment.remote_monitoring.default_user is configured."
            )
        resolved.append(
            RemoteNodeConfig(
                name=f"ray_{host.replace('.', '_').replace(':', '_')}",
                host=host,
                user=monitoring.default_user,
                python=monitoring.default_python,
                remote_output_root=monitoring.default_remote_output_root,
                monitor_script=monitoring.default_monitor_script,
                ssh_options=monitoring.ssh_options,
            )
        )
    return tuple(resolved)


def render_server_command(
    command: Iterable[str],
    plan: ClusterPlan,
    *,
    model: str,
    served_model_name: str,
) -> list[str]:
    """Expand documented placeholders in a vLLM launch command."""

    replacements = {
        "{pipeline_parallel_size}": str(plan.pipeline_parallel_size),
        "{tensor_parallel_size}": str(plan.tensor_parallel_size),
        "{gpu_count}": str(plan.requested_gpu_count),
        "{ray_address}": plan.ray_address,
        "{model}": model,
        "{served_model_name}": served_model_name,
    }
    rendered: list[str] = []
    for raw in command:
        value = raw
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        rendered.append(value)
    return rendered


def _inspect_node_runtime(model_paths: tuple[str, ...]) -> dict[str, Any]:
    """Runs inside a Ray worker on one specific node."""

    def package_version(name: str) -> str | None:
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:
            return None

    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
    except Exception:
        torch_version = None
        torch_cuda_version = None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_names = tuple(line.strip() for line in output.stdout.splitlines() if line.strip())
    except Exception:
        gpu_names = ()
    return {
        "host": socket.gethostbyname(socket.gethostname()),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "ray_version": package_version("ray"),
        "vllm_version": package_version("vllm"),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "gpu_names": gpu_names,
        "model_paths": {path: os.path.isdir(path) for path in model_paths},
    }


def inspect_cluster_runtimes(
    deployment: DeploymentConfig,
    plan: ClusterPlan,
    *,
    model_paths: Iterable[str] = (),
) -> tuple[NodeRuntime, ...]:
    """Run a read-only environment/path check inside every Ray GPU node."""

    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError as exc:
        raise ClusterConfigurationError("Cluster preflight requires Ray.") from exc

    already_initialised = ray.is_initialized()
    try:
        if not already_initialised:
            ray.init(
                address=deployment.cluster.ray_address,
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                log_to_driver=False,
            )
        inspector = ray.remote(num_cpus=0)(_inspect_node_runtime)
        targets = plan.active_gpu_nodes
        refs = [
            inspector.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node.node_id,
                    soft=False,
                )
            ).remote(tuple(model_paths))
            for node in targets
        ]
        payloads = ray.get(refs, timeout=120)
    except Exception as exc:
        raise ClusterConfigurationError(
            f"Ray node runtime preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if not already_initialised:
            try:
                ray.shutdown()
            except Exception:
                logger.debug("Ray driver shutdown failed", exc_info=True)

    runtime_rows: list[NodeRuntime] = []
    for node, payload in zip(plan.active_gpu_nodes, payloads):
        payload_host = str(payload.get("host") or "")
        try:
            is_loopback = ipaddress.ip_address(payload_host).is_loopback
        except ValueError:
            is_loopback = payload_host in {"localhost", ""}
        runtime_rows.append(NodeRuntime(
            node_id=node.node_id,
            host=node.host if is_loopback else payload_host,
            python_executable=str(payload.get("python_executable") or ""),
            python_version=str(payload.get("python_version") or ""),
            ray_version=payload.get("ray_version"),
            vllm_version=payload.get("vllm_version"),
            torch_version=payload.get("torch_version"),
            torch_cuda_version=payload.get("torch_cuda_version"),
            gpu_names=tuple(payload.get("gpu_names") or ()),
            model_paths=dict(payload.get("model_paths") or {}),
        ))
    runtimes = tuple(runtime_rows)
    _validate_runtime_compatibility(runtimes)
    return runtimes


def _validate_runtime_compatibility(runtimes: tuple[NodeRuntime, ...]) -> None:
    if not runtimes:
        raise ClusterConfigurationError("No Ray GPU runtimes were inspected.")
    missing_packages = [
        runtime.host
        for runtime in runtimes
        if not runtime.ray_version or not runtime.vllm_version or not runtime.torch_version
    ]
    if missing_packages:
        raise ClusterConfigurationError(
            "Ray/vLLM/PyTorch is missing from Ray worker environments on: "
            + ", ".join(missing_packages)
        )
    for field_name in ("ray_version", "vllm_version", "torch_version", "torch_cuda_version"):
        versions = {getattr(runtime, field_name) for runtime in runtimes}
        if len(versions) > 1:
            details = ", ".join(
                f"{runtime.host}={getattr(runtime, field_name)}" for runtime in runtimes
            )
            raise ClusterConfigurationError(
                f"Cluster runtime mismatch for {field_name}: {details}"
            )
    missing_paths = [
        f"{runtime.host}:{path}"
        for runtime in runtimes
        for path, exists in runtime.model_paths.items()
        if not exists
    ]
    if missing_paths:
        raise ClusterConfigurationError(
            "Model directories are missing inside Ray worker environments: "
            + ", ".join(missing_paths)
        )
