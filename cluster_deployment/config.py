"""Deployment, Ray-cluster, and remote-telemetry configuration.

The regular model file contains only the deployment selector::

    deployment:
      mode: local
      cluster_config: model_cluster_configs.yaml

When ``mode`` is ``local``, the cluster file is not opened or validated.
When it is ``cluster``, deployment details and model server overrides are
loaded from that separate file and merged over the ordinary model entries.

Cluster mode deliberately does *not* start or stop Ray.  Ray is a machine
service and should be started on the head and worker PCs before a pipeline
run.  The pipeline discovers the live Ray nodes, validates the requested GPU
topology, starts/stops only its vLLM frontend, and records telemetry on every
active Ray machine.

Legacy inline ``local_profile`` and ``cluster_profile`` mappings remain
supported so older experiment configurations continue to load.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_MODES = {"local", "cluster"}


@dataclass(frozen=True)
class ClusterConfig:
    """How a pipeline should consume an already-running Ray cluster.

    ``gpu_count=None`` means use every GPU advertised by active Ray nodes.
    Setting an integer limits vLLM to that many GPUs (Ray may choose any
    suitable subset when more GPUs are active).  ``tensor_parallel_size=1``
    is the appropriate default for one Ethernet-connected GPU per PC; the
    remaining requested GPUs become pipeline-parallel stages.
    """

    ray_address: str = "auto"
    local_host: str | None = None
    gpu_count: int | None = None
    tensor_parallel_size: int = 1


@dataclass(frozen=True)
class RemoteNodeConfig:
    """SSH overrides for one Ray machine.

    Most installations only need ``host``.  Missing values inherit the
    defaults from :class:`RemoteMonitoringConfig`.
    """

    name: str
    host: str
    user: str
    python: str
    remote_output_root: str
    monitor_script: str | None = None
    ssh_options: tuple[str, ...] = ()

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


@dataclass(frozen=True)
class RemoteMonitoringConfig:
    enabled: bool = False
    required: bool = True
    discover_from_ray: bool = True
    monitor_in_local_mode: bool = False
    sync_monitor_script: bool = True
    default_user: str = ""
    default_python: str = "/usr/bin/python3"
    default_remote_output_root: str = "/tmp/pipeline_telemetry"
    default_monitor_script: str | None = None
    ssh_options: tuple[str, ...] = ()
    nodes: tuple[RemoteNodeConfig, ...] = ()


@dataclass(frozen=True)
class DeploymentConfig:
    mode: str
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    remote_monitoring: RemoteMonitoringConfig = field(
        default_factory=RemoteMonitoringConfig
    )

    @property
    def is_cluster(self) -> bool:
        return self.mode == "cluster"


def _optional_nonempty_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string when provided.")
    value = value.strip()
    return value or None


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings.")
    return tuple(value)


def _parse_cluster_config(raw: Any) -> ClusterConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("deployment.cluster must be a YAML mapping.")

    ray_address = raw.get("ray_address", "auto")
    if not isinstance(ray_address, str) or not ray_address.strip():
        raise ValueError("deployment.cluster.ray_address must be a non-empty string.")

    gpu_count_raw = raw.get("gpu_count", "auto")
    if gpu_count_raw in (None, "auto"):
        gpu_count = None
    elif isinstance(gpu_count_raw, bool) or not isinstance(gpu_count_raw, int):
        raise TypeError("deployment.cluster.gpu_count must be 'auto' or a positive integer.")
    elif gpu_count_raw < 1:
        raise ValueError("deployment.cluster.gpu_count must be greater than zero.")
    else:
        gpu_count = gpu_count_raw

    tensor_parallel_size = raw.get("tensor_parallel_size", 1)
    if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int):
        raise TypeError("deployment.cluster.tensor_parallel_size must be a positive integer.")
    if tensor_parallel_size < 1:
        raise ValueError("deployment.cluster.tensor_parallel_size must be greater than zero.")

    return ClusterConfig(
        ray_address=ray_address.strip(),
        local_host=_optional_nonempty_string(
            raw.get("local_host"), "deployment.cluster.local_host"
        ),
        gpu_count=gpu_count,
        tensor_parallel_size=tensor_parallel_size,
    )


def _required_default_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _parse_remote_node(
    raw: Any,
    index: int,
    *,
    default_user: str,
    default_python: str,
    default_remote_output_root: str,
    default_monitor_script: str | None,
    default_ssh_options: tuple[str, ...],
) -> RemoteNodeConfig:
    section = f"deployment.remote_monitoring.nodes[{index}]"
    if not isinstance(raw, dict):
        raise TypeError(f"{section} must be a YAML mapping.")

    host = _required_default_string(raw.get("host"), f"{section}.host")
    name_raw = raw.get("name", host)
    name = _required_default_string(name_raw, f"{section}.name")
    user = _required_default_string(
        raw.get("user", default_user), f"{section}.user"
    )
    python = _required_default_string(
        raw.get("python", default_python), f"{section}.python"
    )
    remote_output_root = _required_default_string(
        raw.get("remote_output_root", default_remote_output_root),
        f"{section}.remote_output_root",
    )
    monitor_script = _optional_nonempty_string(
        raw.get("monitor_script", default_monitor_script),
        f"{section}.monitor_script",
    )
    node_options = _string_tuple(raw.get("ssh_options"), f"{section}.ssh_options")

    return RemoteNodeConfig(
        name=name,
        host=host,
        user=user,
        python=python,
        remote_output_root=remote_output_root,
        monitor_script=monitor_script,
        ssh_options=(*default_ssh_options, *node_options),
    )


def _parse_remote_monitoring(raw: Any) -> RemoteMonitoringConfig:
    if raw is None:
        return RemoteMonitoringConfig()
    if not isinstance(raw, dict):
        raise TypeError("deployment.remote_monitoring must be a YAML mapping.")

    for key in (
        "enabled",
        "required",
        "discover_from_ray",
        "monitor_in_local_mode",
        "sync_monitor_script",
    ):
        if key in raw and not isinstance(raw[key], bool):
            raise TypeError(f"deployment.remote_monitoring.{key} must be true or false.")

    default_user = _optional_nonempty_string(
        raw.get("default_user"), "deployment.remote_monitoring.default_user"
    ) or ""
    default_python = _required_default_string(
        raw.get("default_python", "/usr/bin/python3"),
        "deployment.remote_monitoring.default_python",
    )
    default_remote_output_root = _required_default_string(
        raw.get("default_remote_output_root", "/tmp/pipeline_telemetry"),
        "deployment.remote_monitoring.default_remote_output_root",
    )
    default_monitor_script = _optional_nonempty_string(
        raw.get("default_monitor_script"),
        "deployment.remote_monitoring.default_monitor_script",
    )
    ssh_options = _string_tuple(
        raw.get("ssh_options"), "deployment.remote_monitoring.ssh_options"
    )

    nodes_raw = raw.get("nodes", []) or []
    if not isinstance(nodes_raw, list):
        raise TypeError("deployment.remote_monitoring.nodes must be a list.")
    nodes = tuple(
        _parse_remote_node(
            node,
            index,
            default_user=default_user,
            default_python=default_python,
            default_remote_output_root=default_remote_output_root,
            default_monitor_script=default_monitor_script,
            default_ssh_options=ssh_options,
        )
        for index, node in enumerate(nodes_raw)
    )

    enabled = raw.get("enabled", False)
    discover_from_ray = raw.get("discover_from_ray", True)
    if enabled and discover_from_ray and not default_user and not nodes:
        raise ValueError(
            "deployment.remote_monitoring.default_user is required when Ray "
            "discovery is enabled and no explicit nodes are configured."
        )
    if enabled and not discover_from_ray and not nodes:
        raise ValueError(
            "Remote monitoring is enabled but neither Ray discovery nor explicit nodes are configured."
        )

    return RemoteMonitoringConfig(
        enabled=enabled,
        required=raw.get("required", True),
        discover_from_ray=discover_from_ray,
        monitor_in_local_mode=raw.get("monitor_in_local_mode", False),
        sync_monitor_script=raw.get("sync_monitor_script", True),
        default_user=default_user,
        default_python=default_python,
        default_remote_output_root=default_remote_output_root,
        default_monitor_script=default_monitor_script,
        ssh_options=ssh_options,
        nodes=nodes,
    )


def parse_deployment_config(raw: Any) -> DeploymentConfig:
    """Parse and validate the optional ``deployment`` mapping."""

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("deployment must be a YAML mapping.")

    mode = raw.get("mode", "local")
    if not isinstance(mode, str) or mode.strip() not in VALID_MODES:
        raise ValueError(
            f"deployment.mode must be one of {sorted(VALID_MODES)}, got {mode!r}."
        )

    return DeploymentConfig(
        mode=mode.strip(),
        cluster=_parse_cluster_config(raw.get("cluster")),
        remote_monitoring=_parse_remote_monitoring(raw.get("remote_monitoring")),
    )


def apply_model_profile(
    raw_model: Any,
    deployment_mode: str,
    section: str,
) -> dict[str, Any]:
    """Return one model mapping with its selected deployment profile applied.

    Common keys remain the defaults.  ``local_profile`` or
    ``cluster_profile`` may override any of them.  Profile keys themselves
    are removed before the normal model parser sees the mapping.
    """

    if not isinstance(raw_model, dict):
        raise TypeError(f"{section} must be a YAML mapping.")
    profile_key = f"{deployment_mode}_profile"
    profile = raw_model.get(profile_key, {})
    if profile is None:
        profile = {}
    if not isinstance(profile, dict):
        raise TypeError(f"{section}.{profile_key} must be a YAML mapping.")
    merged = {
        key: value
        for key, value in raw_model.items()
        if key not in {"local_profile", "cluster_profile"}
    }
    merged.update(profile)
    return merged


def load_deployment_config(config_path: str | Path) -> DeploymentConfig:
    """Resolve deployment settings selected by a model YAML file.

    A referenced cluster file is deliberately ignored in local mode.
    """

    _, deployment = load_model_config_bundle(config_path)
    return deployment


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in: {path}")
    return data


def _merge_model_sections(
    base_models: Any,
    cluster_models: Any,
) -> dict[str, Any]:
    """Merge cluster-only model overrides over the ordinary model entries."""

    if not isinstance(base_models, dict):
        raise TypeError("models must be a YAML mapping.")
    if cluster_models is None:
        return dict(base_models)
    if not isinstance(cluster_models, dict):
        raise TypeError("models in the cluster configuration must be a YAML mapping.")

    merged = dict(base_models)
    for model_name, override in cluster_models.items():
        if not isinstance(override, dict):
            raise TypeError(
                f"models.{model_name} in the cluster configuration must be a YAML mapping."
            )
        base = merged.get(model_name, {})
        if not isinstance(base, dict):
            raise TypeError(f"models.{model_name} must be a YAML mapping.")
        merged[model_name] = {**base, **override}
    return merged


def load_model_config_bundle(
    config_path: str | Path,
) -> tuple[dict[str, Any], DeploymentConfig]:
    """Load model data plus the deployment selected by it.

    The returned model mapping is unchanged in local mode. In cluster mode,
    ``models`` overrides from ``deployment.cluster_config`` are merged over
    it. This gives every downstream entry point one authoritative resolution
    path and prevents local runs from depending on SSH/Ray configuration.
    """

    path = Path(config_path).expanduser().resolve()
    data = _read_yaml_mapping(path)
    selector = data.get("deployment") or {}
    if not isinstance(selector, dict):
        raise TypeError("deployment must be a YAML mapping.")

    mode = selector.get("mode", "local")
    # Validate the selector before deciding whether another file may be read.
    selected = parse_deployment_config({"mode": mode})
    cluster_config_value = selector.get("cluster_config")

    if not selected.is_cluster:
        # New split-file format: cluster details must have zero influence on a
        # local run. Preserve legacy inline local telemetry only when no split
        # cluster file was configured.
        deployment = (
            selected
            if cluster_config_value is not None
            else parse_deployment_config(selector)
        )
        return data, deployment

    if cluster_config_value is None:
        # Backward compatibility for old all-in-one model configurations.
        return data, parse_deployment_config(selector)
    if not isinstance(cluster_config_value, str) or not cluster_config_value.strip():
        raise ValueError("deployment.cluster_config must be a non-empty path string.")

    cluster_path = Path(cluster_config_value).expanduser()
    if not cluster_path.is_absolute():
        cluster_path = path.parent / cluster_path
    cluster_data = _read_yaml_mapping(cluster_path.resolve())
    cluster_deployment = cluster_data.get("deployment") or {}
    if not isinstance(cluster_deployment, dict):
        raise TypeError("deployment in the cluster configuration must be a YAML mapping.")

    deployment = parse_deployment_config({"mode": "cluster", **cluster_deployment})
    effective = dict(data)
    effective["models"] = _merge_model_sections(
        data.get("models"), cluster_data.get("models")
    )
    # Expose the resolved mapping to callers that inspect the effective file.
    effective["deployment"] = {"mode": "cluster", **cluster_deployment}
    return effective, deployment
