"""One crash-resilient hardware recorder per participating machine.

The local recorder remains the event authority.  Every active remote Ray GPU
node runs the exact same ``hardware_monitor.py`` in standalone mode, writing
to that machine's local disk.  At shutdown its directory is copied beneath
the local recorder and paired with the controller event stream plus a clock
offset measurement for cross-machine correlation.

Remote monitoring uses non-interactive SSH by design.  Configure SSH keys
before a long run; a password prompt hidden behind an orchestration process is
not reliable or recoverable.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .cluster_runtime import ClusterConfigurationError, resolve_remote_monitor_nodes

logger = logging.getLogger(__name__)

SSH_BASE_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-o", "StrictHostKeyChecking=accept-new",
)
REMOTE_ACTIVE_ENV = "PIPELINE_REMOTE_TELEMETRY_ACTIVE"


class MonitoringError(RuntimeError):
    """Required local or remote telemetry could not be established."""


class RemoteHardwareMonitor:
    """Manage one standalone recorder over SSH."""

    def __init__(
        self,
        node: Any,
        *,
        interval: float,
        label: str,
        context: dict[str, Any] | None,
        sync_monitor_script: bool,
    ) -> None:
        self.node = node
        self.interval = interval
        self.label = label
        self.context = context or {}
        self.sync_monitor_script = sync_monitor_script
        self.remote_dir: str | None = None
        self.remote_pid: int | None = None
        self.started = False
        self.clock_offset_seconds: float | None = None
        self.clock_round_trip_seconds: float | None = None

    @property
    def ssh_options(self) -> tuple[str, ...]:
        return (*SSH_BASE_OPTIONS, *self.node.ssh_options)

    def _ssh(self, command: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ssh", *self.ssh_options, self.node.target, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _copy_to(self, local_path: Path, remote_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "scp", "-q", *self.ssh_options,
                str(local_path), f"{self.node.target}:{remote_path}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _measure_clock(self) -> None:
        started = time.time()
        result = self._ssh("date +%s.%N", timeout=15)
        finished = time.time()
        if result.returncode != 0:
            return
        try:
            remote_time = float(result.stdout.strip())
        except ValueError:
            return
        midpoint = (started + finished) / 2.0
        self.clock_offset_seconds = remote_time - midpoint
        self.clock_round_trip_seconds = finished - started

    def start(self) -> bool:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        remote_dir = (
            f"{self.node.remote_output_root.rstrip('/')}"
            f"/hardware_telemetry_{self.label}_{stamp}"
        )
        try:
            mkdir = self._ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("[telemetry:%s] SSH setup failed: %s", self.node.name, exc)
            return False
        if mkdir.returncode != 0:
            logger.error(
                "[telemetry:%s] Cannot create %s: %s",
                self.node.name,
                remote_dir,
                (mkdir.stderr or mkdir.stdout).strip(),
            )
            return False

        if self.sync_monitor_script:
            local_script = (
                Path(__file__).resolve().parent
                / "anomaly_detection"
                / "hardware_monitor.py"
            )
            remote_script = f"{remote_dir}/hardware_monitor.py"
            copied = self._copy_to(local_script, remote_script)
            if copied.returncode != 0:
                logger.error(
                    "[telemetry:%s] Could not copy recorder: %s",
                    self.node.name,
                    (copied.stderr or copied.stdout).strip(),
                )
                return False
        else:
            if not self.node.monitor_script:
                logger.error(
                    "[telemetry:%s] sync_monitor_script=false requires monitor_script.",
                    self.node.name,
                )
                return False
            remote_script = self.node.monitor_script

        self._measure_clock()
        context_json = json.dumps(
            {
                **self.context,
                "node": self.node.name,
                "host": self.node.host,
                "role": "remote_ray_node",
                "clock_offset_seconds": self.clock_offset_seconds,
                "clock_round_trip_seconds": self.clock_round_trip_seconds,
            },
            ensure_ascii=False,
            default=str,
        )
        # A background process can keep an SSH exec channel open even after
        # stdout/stderr are redirected because the remote non-interactive shell
        # may continue waiting for that job.  ``setsid -f`` performs a real
        # detach.  The detached child records its PID before exec'ing Python so
        # shutdown still targets the recorder itself rather than a launcher.
        pid_file = f"{remote_dir}/monitor.pid"
        recorder_command = (
            f"printf '%s\\n' \"$$\" > {shlex.quote(pid_file)}; "
            f"exec {shlex.quote(self.node.python)} {shlex.quote(remote_script)} "
            f"--output {shlex.quote(remote_dir)} "
            f"--standalone --no-archive "
            f"--interval {shlex.quote(str(self.interval))} "
            f"--label {shlex.quote(f'{self.label}_{self.node.name}')} "
            f"--context-json {shlex.quote(context_json)}"
        )
        remote_cmd = (
            f"cd {shlex.quote(remote_dir)} && "
            f"nohup setsid -f sh -c {shlex.quote(recorder_command)} "
            f"</dev/null > {shlex.quote(remote_dir + '/remote_stdio.log')} 2>&1 && "
            f"attempt=0; while test ! -s {shlex.quote(pid_file)} && "
            f"test \"$attempt\" -lt 50; do "
            f"sleep 0.1; attempt=$((attempt + 1)); done; "
            f"cat {shlex.quote(pid_file)}"
        )
        try:
            result = self._ssh(remote_cmd)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("[telemetry:%s] Recorder launch failed: %s", self.node.name, exc)
            return False
        if result.returncode != 0:
            logger.error(
                "[telemetry:%s] Recorder launch failed: %s",
                self.node.name,
                (result.stderr or result.stdout).strip(),
            )
            return False
        try:
            remote_pid = int(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            logger.error(
                "[telemetry:%s] Recorder returned no PID: %r",
                self.node.name,
                result.stdout,
            )
            return False

        self.remote_dir = remote_dir
        self.remote_pid = remote_pid
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = self._ssh(
                f"test -f {shlex.quote(remote_dir + '/.ready')} && "
                f"kill -0 {remote_pid} 2>/dev/null",
                timeout=15,
            )
            if status.returncode == 0:
                self.started = True
                logger.info(
                    "[telemetry:%s] Recorder ready pid=%s dir=%s",
                    self.node.name,
                    remote_pid,
                    remote_dir,
                )
                return True
            time.sleep(1)

        logger.error(
            "[telemetry:%s] Recorder did not become ready; inspect %s:%s/remote_stdio.log",
            self.node.name,
            self.node.host,
            remote_dir,
        )
        self._ssh(f"kill -TERM {remote_pid} 2>/dev/null", timeout=15)
        return False

    def stop(self, local_destination: Path, controller_events: Path, reason: str) -> None:
        if not self.started or self.remote_dir is None or self.remote_pid is None:
            return
        stop_command = (
            f"printf %s {shlex.quote(reason)} > "
            f"{shlex.quote(self.remote_dir + '/.stop_reason')} && "
            f"kill -TERM {self.remote_pid} 2>/dev/null"
        )
        try:
            self._ssh(stop_command, timeout=20)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                result = self._ssh(
                    f"test -f {shlex.quote(self.remote_dir + '/summary.json')} || "
                    f"! kill -0 {self.remote_pid} 2>/dev/null",
                    timeout=15,
                )
                if result.returncode == 0:
                    break
                time.sleep(0.5)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("[telemetry:%s] Stop confirmation failed: %s", self.node.name, exc)
        self._fetch(local_destination, controller_events)

    def _fetch(self, local_destination: Path, controller_events: Path) -> None:
        destination = Path(local_destination) / f"remote_{self.node.name}"
        destination.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "scp", "-r", "-q", *self.ssh_options,
                f"{self.node.target}:{self.remote_dir}", str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            logger.error(
                "[telemetry:%s] Copy-back failed; data remains at %s:%s: %s",
                self.node.name,
                self.node.host,
                self.remote_dir,
                (result.stderr or result.stdout).strip(),
            )
            return

        if controller_events.is_file():
            shutil.copy2(controller_events, destination / "controller_events.jsonl")
        correlation = {
            "node": self.node.name,
            "host": self.node.host,
            "remote_directory": self.remote_dir,
            "clock_offset_seconds_remote_minus_controller": self.clock_offset_seconds,
            "clock_measurement_round_trip_seconds": self.clock_round_trip_seconds,
            "controller_events": "controller_events.jsonl",
        }
        (destination / "correlation.json").write_text(
            json.dumps(correlation, indent=2), encoding="utf-8"
        )
        logger.info("[telemetry:%s] Copied telemetry to %s", self.node.name, destination)


class MultiNodeMonitor:
    """Local recorder plus one standalone recorder per remote Ray machine."""

    def __init__(
        self,
        local_monitor: Any,
        *,
        remote_nodes: tuple[Any, ...],
        interval: float,
        label: str,
        context: dict[str, Any] | None,
        required: bool,
        sync_monitor_script: bool,
    ) -> None:
        self.local = local_monitor
        self.required = required
        self.remotes = [
            RemoteHardwareMonitor(
                node,
                interval=interval,
                label=label,
                context=context,
                sync_monitor_script=sync_monitor_script,
            )
            for node in remote_nodes
        ]
        self._started = False

    @property
    def output_dir(self) -> Path:
        return self.local.output_dir

    def start(self, **kwargs: Any) -> bool:
        if not self.local.start(**kwargs):
            return False
        os.environ[REMOTE_ACTIVE_ENV] = "1"
        failed: list[str] = []
        for remote in self.remotes:
            try:
                if remote.start():
                    self.local.mark_event(
                        "remote_telemetry_started",
                        node=remote.node.name,
                        host=remote.node.host,
                        remote_dir=remote.remote_dir,
                        remote_pid=remote.remote_pid,
                        clock_offset_seconds=remote.clock_offset_seconds,
                    )
                else:
                    failed.append(remote.node.name)
                    self.local.mark_event(
                        "remote_telemetry_start_failed", node=remote.node.name
                    )
                    if remote.remote_dir is not None:
                        # Preserve startup logs (wrong interpreter, permissions,
                        # missing nvidia-smi, etc.) before rejecting the run.
                        remote._fetch(self.local.output_dir, self.local.events_path)
            except Exception:
                logger.exception("[telemetry:%s] Unexpected startup failure", remote.node.name)
                failed.append(remote.node.name)
        if failed and self.required:
            for remote in self.remotes:
                if remote.started:
                    remote.stop(
                        self.local.output_dir,
                        self.local.events_path,
                        "required_peer_start_failed",
                    )
            self.local.stop(reason="required_remote_telemetry_start_failed")
            os.environ.pop(REMOTE_ACTIVE_ENV, None)
            raise MonitoringError(
                "Required telemetry failed on: " + ", ".join(sorted(failed))
            )
        self._started = True
        return True

    def mark_event(self, event: str, **details: Any) -> None:
        # The UTC controller event stream is copied beside every remote trace
        # at shutdown, avoiding an SSH round trip for every image event.
        self.local.mark_event(event, **details)

    def track_process(self, pid: int, role: str, **details: Any) -> None:
        self.local.track_process(pid, role, **details)

    def stop(self, reason: str = "pipeline_finished") -> None:
        if not self._started:
            return
        for remote in self.remotes:
            try:
                remote.stop(self.local.output_dir, self.local.events_path, reason)
            except Exception:
                logger.exception("[telemetry:%s] Unexpected shutdown failure", remote.node.name)
        manifest = {
            "schema_version": 1,
            "reason": reason,
            "local": {"telemetry_directory": str(self.local.output_dir)},
            "remote_nodes": [
                {
                    "name": remote.node.name,
                    "host": remote.node.host,
                    "started": remote.started,
                    "remote_pid": remote.remote_pid,
                    "remote_directory": remote.remote_dir,
                    "copied_under": str(
                        self.local.output_dir / f"remote_{remote.node.name}"
                    ),
                    "clock_offset_seconds_remote_minus_controller": remote.clock_offset_seconds,
                    "clock_measurement_round_trip_seconds": remote.clock_round_trip_seconds,
                }
                for remote in self.remotes
            ],
        }
        try:
            (self.local.output_dir / "cluster_telemetry_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.exception("Could not write cluster telemetry manifest")
        os.environ.pop(REMOTE_ACTIVE_ENV, None)
        self.local.stop(reason=reason)
        self._started = False


def build_monitor(
    local_monitor: Any,
    deployment_cfg: Any,
    *,
    interval: float = 1.0,
    label: str = "pipeline",
    context: dict[str, Any] | None = None,
) -> Any:
    """Wrap a local monitor with recorders for all active remote Ray nodes."""

    monitoring = deployment_cfg.remote_monitoring
    if not monitoring.enabled:
        return local_monitor
    if not deployment_cfg.is_cluster and not monitoring.monitor_in_local_mode:
        logger.info("Local mode selected; remote cluster telemetry is disabled.")
        return local_monitor
    if os.environ.get(REMOTE_ACTIVE_ENV) == "1":
        logger.info("A parent orchestrator already owns remote telemetry.")
        return local_monitor
    try:
        nodes = resolve_remote_monitor_nodes(deployment_cfg)
    except ClusterConfigurationError:
        if monitoring.required:
            raise
        logger.warning("Could not discover optional remote telemetry nodes", exc_info=True)
        return local_monitor
    if not nodes:
        if monitoring.required and deployment_cfg.is_cluster:
            # A one-node Ray cluster legitimately has no remote node.
            logger.info("Ray has no remote GPU nodes; using the local recorder only.")
        return local_monitor
    return MultiNodeMonitor(
        local_monitor,
        remote_nodes=nodes,
        interval=interval,
        label=label,
        context=context,
        required=monitoring.required,
        sync_monitor_script=monitoring.sync_monitor_script,
    )


@contextmanager
def monitored(monitor: Any) -> Iterator[None]:
    reason = "pipeline_finished"
    try:
        yield
    except KeyboardInterrupt:
        reason = "pipeline_keyboard_interrupt"
        if monitor is not None:
            monitor.mark_event(reason)
        raise
    except BaseException as exc:
        reason = "pipeline_python_exception"
        if monitor is not None:
            monitor.mark_event(reason, exception_type=type(exc).__name__)
        raise
    finally:
        if monitor is not None:
            monitor.stop(reason=reason)
