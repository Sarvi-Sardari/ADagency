"""Crash-resilient host telemetry for the anomaly-detection pipeline.

The monitor runs in a separate process so samples already flushed to disk
survive a Python/CUDA native crash in the pipeline.  It intentionally uses
only the Python standard library and read-only Linux/NVIDIA interfaces.

Outputs are written to one timestamped telemetry directory:

* telemetry.csv                one wide, machine-readable sample per interval
* pipeline_events.jsonl        model/dataset/image boundaries for correlation
* kernel_before_pipeline.log   complete boot kernel log at monitor startup
* kernel_during_pipeline.log   live kernel messages emitted during the run
* kernel_after_pipeline.log    complete boot kernel log at monitor shutdown
* system_snapshot.txt          kernel, CPU, firmware, modules and package state
* nvidia_smi_start.txt/end.txt detailed NVIDIA state when available
* metadata.json                sensor inventory and availability/limitations
* summary.json                 min/max/mean telemetry and detected kernel faults
* monitor.log                  monitor errors and lifecycle messages

No hardware settings are changed.  The recorder cannot sample sub-millisecond
voltage transients, and it can only record voltage/DIMM-temperature data when
the motherboard exposes those values through hwmon or NVIDIA exposes them
through nvidia-smi.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any


IMPORTANT_KERNEL_PATTERN = re.compile(
    r"invalid opcode|general protection fault|BUG:|soft lockup|hard lockup|"
    r"machine check|hardware error|\bmce:|NVRM: Xid|GPU has fallen|"
    r"out of memory|oom-kill|Killed process|segfault|I/O error|Buffer I/O error|"
    r"EXT4-fs error|XFS.*error|nvme.*(reset|timeout|abort)|thermal.*critical|"
    r"exited with irqs disabled",
    re.IGNORECASE,
)

GPU_BASE_FIELDS = [
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "pstate",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "fan.speed",
    "pcie.link.gen.current",
    "pcie.link.width.current",
]

PROCESS_FIELDNAMES = [
    "timestamp_local",
    "timestamp_utc",
    "sample",
    "root_pid",
    "pid",
    "ppid",
    "process_group_id",
    "session_id",
    "start_ticks",
    "state",
    "comm",
    "executable",
    "cmdline",
    "cpu_pct",
    "cpu_last",
    "rss_gib",
    "vm_size_gib",
    "threads",
    "fd_count",
    "read_bytes",
    "write_bytes",
    "voluntary_context_switches",
    "nonvoluntary_context_switches",
    "explicitly_tracked",
]

GPU_PROCESS_FIELDNAMES = [
    "timestamp_local",
    "timestamp_utc",
    "sample",
    "pid",
    "process_name",
    "gpu_uuid",
    "used_gpu_memory_mib",
    "in_watched_process_tree",
]

GPU_OPTIONAL_FIELDS = [
    "voltage.graphics",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sync_boost",
]


def _now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "unknown"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return None


def _read_number(path: Path, divisor: float = 1.0) -> float | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return float(raw) / divisor
    except ValueError:
        return None


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    cleaned = re.sub(r"\s*(MiB|GiB|W|MHz|MHz|%|V)$", "", value).strip()
    try:
        number = float(cleaned)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def record_pipeline_event(event: str, **details: Any) -> None:
    """Append a crash-durable event when pipeline telemetry is active.

    This is deliberately a no-op outside a monitored pipeline invocation.
    It lets lower-level inference code mark the exact image whose generate()
    call was in flight without passing a monitor object through every layer.
    """

    path = os.environ.get("PIPELINE_TELEMETRY_EVENTS_FILE")
    if not path:
        return
    try:
        _append_jsonl(
            Path(path),
            {
                "timestamp_local": _now_local(),
                "timestamp_utc": _now_utc(),
                "pid": os.getpid(),
                "event": event,
                **details,
            },
        )
    except OSError:
        # Telemetry must never make inference fail.
        return


def _run_capture(args: list[str], timeout: float = 8.0) -> tuple[int | None, str, str]:
    """Run a read-only diagnostic command with a bounded wait.

    A driver call can become uninterruptible after a kernel/GPU failure.  In
    that case we stop issuing further GPU queries rather than accumulating
    stuck nvidia-smi processes.
    """

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return None, "", f"{type(exc).__name__}: {exc}"

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process remained uninterruptible after timeout"
        return None, stdout, f"timeout after {timeout:.1f}s; {stderr}".strip()


def _write_command_section(file_obj: Any, title: str, args: list[str], timeout: float = 8.0) -> None:
    file_obj.write(f"\n===== {title} =====\n$ {' '.join(args)}\n")
    code, stdout, stderr = _run_capture(args, timeout=timeout)
    file_obj.write(stdout)
    if stderr:
        file_obj.write(f"\n[stderr]\n{stderr}\n")
    file_obj.write(f"\n[exit_code={code}]\n")


@dataclass(frozen=True)
class Sensor:
    key: str
    path: Path
    divisor: float
    unit: str
    device: str
    label: str


@dataclass
class RunningStats:
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0
    last: float | None = None

    def add(self, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        number = float(value)
        if not math.isfinite(number):
            return
        self.count += 1
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        self.total += number
        self.last = number

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": round(self.minimum, 6) if self.count else None,
            "max": round(self.maximum, 6) if self.count else None,
            "mean": round(self.total / self.count, 6) if self.count else None,
            "last": self.last,
        }


def _discover_hwmon() -> list[Sensor]:
    sensors: list[Sensor] = []
    definitions = {
        "temp": (1000.0, "C"),
        "in": (1000.0, "V"),
        "curr": (1000.0, "A"),
        "power": (1_000_000.0, "W"),
        "fan": (1.0, "RPM"),
    }
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        device = _read_text(hwmon / "name") or hwmon.name
        for prefix, (divisor, unit) in definitions.items():
            for input_path in sorted(hwmon.glob(f"{prefix}*_input")):
                stem = input_path.name.removesuffix("_input")
                label = _read_text(hwmon / f"{stem}_label") or stem
                key = f"hwmon.{_safe_name(device)}.{_safe_name(label)}.{stem}_{unit.lower()}"
                sensors.append(Sensor(key, input_path, divisor, unit, device, label))
    return sensors


def _discover_cpu_ids() -> list[int]:
    ids: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        match = re.fullmatch(r"cpu(\d+)", path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _discover_rapl() -> list[dict[str, Any]]:
    domains: list[dict[str, Any]] = []
    for path in sorted(Path("/sys/class/powercap").glob("intel-rapl:*")):
        energy_path = path / "energy_uj"
        if not energy_path.exists():
            continue
        name = _read_text(path / "name") or path.name
        domains.append(
            {
                "name": name,
                "key": f"rapl.{_safe_name(name)}.power_w",
                "energy_path": energy_path,
                "max_energy_uj": _read_number(path / "max_energy_range_uj"),
                "readable": os.access(energy_path, os.R_OK),
                "last_energy_uj": None,
                "last_monotonic": None,
            }
        )
    return domains


def _read_proc_stat() -> dict[str, tuple[int, int]]:
    values: dict[str, tuple[int, int]] = {}
    raw = _read_text(Path("/proc/stat")) or ""
    for line in raw.splitlines():
        parts = line.split()
        if not parts or not re.fullmatch(r"cpu\d*", parts[0]):
            continue
        try:
            numbers = [int(value) for value in parts[1:]]
        except ValueError:
            continue
        total = sum(numbers)
        idle = (numbers[3] if len(numbers) > 3 else 0) + (numbers[4] if len(numbers) > 4 else 0)
        values[parts[0]] = (total, idle)
    return values


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    raw = _read_text(Path("/proc/meminfo")) or ""
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        match = re.search(r"\d+", value)
        if match:
            result[key] = int(match.group(0))
    return result


def _read_pressure(resource: str) -> dict[str, float]:
    result: dict[str, float] = {}
    raw = _read_text(Path("/proc/pressure") / resource) or ""
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        category = parts[0]
        for item in parts[1:]:
            if "=" not in item:
                continue
            name, value = item.split("=", 1)
            if name.startswith("avg"):
                try:
                    result[f"psi.{resource}.{category}.{name}"] = float(value)
                except ValueError:
                    pass
    return result


def _read_diskstats() -> tuple[int, int]:
    block_devices = {path.name for path in Path("/sys/block").iterdir()} if Path("/sys/block").exists() else set()
    read_sectors = 0
    write_sectors = 0
    raw = _read_text(Path("/proc/diskstats")) or ""
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 14 or parts[2] not in block_devices:
            continue
        try:
            read_sectors += int(parts[5])
            write_sectors += int(parts[9])
        except ValueError:
            continue
    return read_sectors, write_sectors


def _read_process(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stat_raw = _read_text(Path(f"/proc/{pid}/stat"))
    if stat_raw and ") " in stat_raw:
        rest = stat_raw.rsplit(") ", 1)[1].split()
        try:
            result["pipeline.process_state"] = rest[0]
            result["pipeline.cpu_last"] = int(rest[36])
            result["_pipeline_ticks"] = int(rest[11]) + int(rest[12])
        except (IndexError, ValueError):
            pass
    status_raw = _read_text(Path(f"/proc/{pid}/status")) or ""
    for line in status_raw.splitlines():
        if line.startswith("VmRSS:"):
            match = re.search(r"\d+", line)
            if match:
                result["pipeline.rss_gib"] = int(match.group(0)) / 1024 / 1024
        elif line.startswith("Threads:"):
            match = re.search(r"\d+", line)
            if match:
                result["pipeline.threads"] = int(match.group(0))
    return result


def _read_process_stat_record(pid: int) -> dict[str, Any] | None:
    """Read the lightweight /proc stat identity needed to build a process tree."""
    stat_raw = _read_text(Path(f"/proc/{pid}/stat"))
    if not stat_raw or ") " not in stat_raw:
        return None
    prefix, suffix = stat_raw.rsplit(") ", 1)
    rest = suffix.split()
    try:
        comm = prefix.split("(", 1)[1]
        record: dict[str, Any] = {
            "pid": pid,
            "comm": comm,
            "state": rest[0],
            "ppid": int(rest[1]),
            "process_group_id": int(rest[2]),
            "session_id": int(rest[3]),
            "_ticks": int(rest[11]) + int(rest[12]),
            "threads": int(rest[17]),
            "start_ticks": int(rest[19]),
            "_rss_pages": int(rest[21]),
            "cpu_last": int(rest[36]),
        }
    except (IndexError, ValueError):
        return None
    return record


def _read_process_record(pid: int) -> dict[str, Any] | None:
    """Read one selected Linux process's detailed resources without signalling it."""

    record = _read_process_stat_record(pid)
    if record is None:
        return None

    status_raw = _read_text(Path(f"/proc/{pid}/status")) or ""
    for line in status_raw.splitlines():
        key, _, value = line.partition(":")
        match = re.search(r"\d+", value)
        if not match:
            continue
        number = int(match.group(0))
        if key == "VmRSS":
            record["rss_gib"] = number / 1024 / 1024
        elif key == "VmSize":
            record["vm_size_gib"] = number / 1024 / 1024
        elif key == "voluntary_ctxt_switches":
            record["voluntary_context_switches"] = number
        elif key == "nonvoluntary_ctxt_switches":
            record["nonvoluntary_context_switches"] = number

    io_raw = _read_text(Path(f"/proc/{pid}/io")) or ""
    for line in io_raw.splitlines():
        key, _, value = line.partition(":")
        if key in {"read_bytes", "write_bytes"}:
            try:
                record[key] = int(value.strip())
            except ValueError:
                pass
    try:
        record["fd_count"] = sum(1 for _ in Path(f"/proc/{pid}/fd").iterdir())
    except OSError:
        pass
    try:
        record["executable"] = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        pass
    try:
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        record["cmdline"] = raw_cmdline.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        pass
    return record


def _read_all_process_records() -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(path.name)
        except ValueError:
            continue
        record = _read_process_stat_record(pid)
        if record is not None:
            records[pid] = record
    return records


def _descendants(
    records: dict[int, dict[str, Any]],
    roots: set[int],
) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, record in records.items():
        children.setdefault(int(record.get("ppid", -1)), []).append(pid)
    found: set[int] = set()
    pending = list(roots)
    while pending:
        pid = pending.pop()
        if pid in found or pid not in records:
            continue
        found.add(pid)
        pending.extend(children.get(pid, ()))
    return found


def _read_vmstat_counter(name: str) -> int | None:
    raw = _read_text(Path("/proc/vmstat")) or ""
    match = re.search(rf"^{re.escape(name)}\s+(\d+)$", raw, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _cgroup_v2_path(pid: int) -> Path | None:
    raw = _read_text(Path(f"/proc/{pid}/cgroup")) or ""
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            candidate = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
            return candidate if candidate.is_dir() else None
    return None


class TelemetryWorker:
    def __init__(
        self,
        output_dir: Path,
        parent_pid: int,
        interval: float,
        label: str,
        duration: float | None,
        *,
        context: dict[str, Any] | None = None,
        archive_enabled: bool = True,
        exit_when_process_tree_disappears: bool = True,
    ):
        self.output_dir = output_dir
        self.parent_pid = parent_pid
        self.interval = max(0.2, interval)
        self.label = label
        self.duration = duration
        self.context = context or {}
        self.archive_enabled = archive_enabled
        self.exit_when_process_tree_disappears = exit_when_process_tree_disappears
        self.stop_event = Event()
        self.cpu_ids = _discover_cpu_ids()
        self.hwmon_sensors = _discover_hwmon()
        self.rapl_domains = _discover_rapl()
        self.previous_cpu = _read_proc_stat()
        self.previous_disk = _read_diskstats()
        self.previous_disk_time = time.monotonic()
        self.previous_process_ticks: int | None = None
        self.previous_process_time: float | None = None
        self.stats: dict[str, RunningStats] = {}
        self.errors: dict[str, int] = {}
        self.gpu_disabled_after_timeout = False
        self.gpu_optional_fields: list[str] = []
        self.journal_process: subprocess.Popen[str] | None = None
        self.system_journal_process: subprocess.Popen[str] | None = None
        self.started_monotonic = time.monotonic()
        self.started_local = _now_local()
        self.sample_count = 0
        self.process_previous: dict[tuple[int, int], tuple[int, float]] = {}
        self.tracked_identities: dict[int, int] = {}
        self.explicit_tracking_offset = 0
        self.explicit_tracked: dict[int, dict[str, Any]] = {}
        self.last_workload_pids: set[int] = set()
        self.cgroup_path = _cgroup_v2_path(self.parent_pid)

    def log(self, message: str) -> None:
        line = f"{_now_local()} {message}\n"
        with (self.output_dir / "monitor.log").open("a", encoding="utf-8") as file_obj:
            file_obj.write(line)
            file_obj.flush()
            os.fsync(file_obj.fileno())

    def _parent_alive(self) -> bool:
        if self.parent_pid <= 1:
            return True
        try:
            os.kill(self.parent_pid, 0)
            stat_raw = _read_text(Path(f"/proc/{self.parent_pid}/stat"))
            if not stat_raw:
                return False
            # A native-crashed parent can briefly remain as a zombie until
            # its shell/supervisor reaps it. Treat that as gone immediately.
            if ") " in stat_raw and stat_raw.rsplit(") ", 1)[1].startswith("Z "):
                return False
            return True
        except OSError:
            return False

    def _probe_gpu_optional_fields(self) -> None:
        if not Path("/usr/bin/nvidia-smi").exists():
            self.errors["nvidia_smi_missing"] = 1
            return
        for field in GPU_OPTIONAL_FIELDS:
            code, stdout, stderr = _run_capture(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                timeout=1.5,
            )
            if code is None and "timeout" in stderr:
                self.gpu_disabled_after_timeout = True
                self.errors["nvidia_smi_timeout"] = self.errors.get("nvidia_smi_timeout", 0) + 1
                return
            if code == 0 and stdout.strip():
                self.gpu_optional_fields.append(field)

    def _gpu_sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {"gpu.query_ok": 0}
        if self.gpu_disabled_after_timeout:
            result["gpu.error"] = "disabled after nvidia-smi timeout"
            return result

        fields = GPU_BASE_FIELDS + self.gpu_optional_fields
        code, stdout, stderr = _run_capture(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            timeout=min(max(self.interval * 0.8, 1.0), 3.0),
        )
        if code is None and "timeout" in stderr:
            self.gpu_disabled_after_timeout = True
            self.errors["nvidia_smi_timeout"] = self.errors.get("nvidia_smi_timeout", 0) + 1
        if code != 0:
            self.errors["nvidia_smi_query_failed"] = self.errors.get("nvidia_smi_query_failed", 0) + 1
            result["gpu.error"] = (stderr or stdout or f"exit code {code}").strip().replace("\n", " | ")[:500]
            return result

        rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
        if not rows:
            result["gpu.error"] = "nvidia-smi returned no GPU rows"
            return result
        result["gpu.query_ok"] = 1
        for ordinal, row in enumerate(rows):
            for field, raw in zip(fields, row):
                result[f"gpu{ordinal}.{field}"] = _coerce_value(raw)
        return result

    def _cpu_sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        current = _read_proc_stat()
        for name, (total, idle) in current.items():
            previous = self.previous_cpu.get(name)
            if previous:
                total_delta = total - previous[0]
                idle_delta = idle - previous[1]
                if total_delta > 0:
                    key = "cpu.total.util_pct" if name == "cpu" else f"cpu.{name}.util_pct"
                    result[key] = 100.0 * (total_delta - idle_delta) / total_delta
        self.previous_cpu = current

        for cpu_id in self.cpu_ids:
            base = Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
            freq = _read_number(base / "cpufreq/scaling_cur_freq", 1000.0)
            result[f"cpu.cpu{cpu_id}.freq_mhz"] = freq
            throttle = _read_number(base / "thermal_throttle/core_throttle_count")
            result[f"cpu.cpu{cpu_id}.core_throttle_count"] = throttle
        if self.cpu_ids:
            base = Path(f"/sys/devices/system/cpu/cpu{self.cpu_ids[0]}/thermal_throttle")
            result["cpu.package_throttle_count"] = _read_number(base / "package_throttle_count")
            result["cpu.package_throttle_total_ms"] = _read_number(base / "package_throttle_total_time_ms")
        return result

    def _memory_and_pressure_sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        mem = _read_meminfo()
        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        if total:
            result["memory.total_gib"] = total / 1024 / 1024
            result["memory.available_gib"] = available / 1024 / 1024
            result["memory.used_pct"] = 100.0 * (total - available) / total
        swap_total = mem.get("SwapTotal", 0)
        swap_free = mem.get("SwapFree", 0)
        result["memory.swap_used_gib"] = (swap_total - swap_free) / 1024 / 1024
        result["memory.dirty_mib"] = mem.get("Dirty", 0) / 1024
        result["memory.writeback_mib"] = mem.get("Writeback", 0) / 1024
        for resource in ("cpu", "memory", "io"):
            result.update(_read_pressure(resource))
        try:
            load1, load5, load15 = os.getloadavg()
            result.update({"system.load1": load1, "system.load5": load5, "system.load15": load15})
        except OSError:
            pass
        return result

    def _disk_sample(self) -> dict[str, Any]:
        now = time.monotonic()
        current = _read_diskstats()
        elapsed = now - self.previous_disk_time
        result: dict[str, Any] = {}
        if elapsed > 0:
            result["disk.read_mib_s"] = max(0, current[0] - self.previous_disk[0]) * 512 / 1024 / 1024 / elapsed
            result["disk.write_mib_s"] = max(0, current[1] - self.previous_disk[1]) * 512 / 1024 / 1024 / elapsed
        self.previous_disk = current
        self.previous_disk_time = now
        return result

    def _hwmon_sample(self) -> dict[str, Any]:
        return {sensor.key: _read_number(sensor.path, sensor.divisor) for sensor in self.hwmon_sensors}

    def _rapl_sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        now = time.monotonic()
        for domain in self.rapl_domains:
            energy = _read_number(domain["energy_path"])
            last_energy = domain["last_energy_uj"]
            last_time = domain["last_monotonic"]
            if energy is not None and last_energy is not None and last_time is not None and now > last_time:
                delta = energy - last_energy
                if delta < 0 and domain["max_energy_uj"]:
                    delta += domain["max_energy_uj"]
                result[domain["key"]] = delta / 1_000_000.0 / (now - last_time)
            else:
                result[domain["key"]] = None
            if energy is not None:
                domain["last_energy_uj"] = energy
                domain["last_monotonic"] = now
        return result

    def _process_sample(self) -> dict[str, Any]:
        now = time.monotonic()
        result = _read_process(self.parent_pid)
        ticks = result.pop("_pipeline_ticks", None)
        if ticks is not None and self.previous_process_ticks is not None and self.previous_process_time is not None:
            elapsed = now - self.previous_process_time
            if elapsed > 0:
                result["pipeline.cpu_pct"] = (
                    (ticks - self.previous_process_ticks) / os.sysconf("SC_CLK_TCK") / elapsed * 100.0
                )
        if ticks is not None:
            self.previous_process_ticks = ticks
            self.previous_process_time = now
        return result

    def _load_explicit_tracking(self) -> None:
        path = self.output_dir / "tracked_processes.jsonl"
        if not path.is_file():
            return
        try:
            with path.open(encoding="utf-8") as file_obj:
                file_obj.seek(self.explicit_tracking_offset)
                for line in file_obj:
                    try:
                        payload = json.loads(line)
                        pid = int(payload.get("pid"))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    self.explicit_tracked[pid] = payload
                self.explicit_tracking_offset = file_obj.tell()
        except OSError:
            self.errors["tracked_processes_read_failed"] = (
                self.errors.get("tracked_processes_read_failed", 0) + 1
            )

    def _process_tree_rows(self, timestamp_local: str, timestamp_utc: str) -> list[dict[str, Any]]:
        self._load_explicit_tracking()
        records = _read_all_process_records()

        # The recorder and its journal followers are children of the watched
        # orchestrator, but they are observers rather than pipeline workload.
        # Excluding the recorder subtree also makes its own overhead visible
        # only through host-wide counters instead of contaminating process data.
        excluded = _descendants(records, {os.getpid()})
        seeds = {self.parent_pid}
        seeds.update(self.explicit_tracked)
        seeds.update(
            pid
            for pid, start_ticks in self.tracked_identities.items()
            if pid in records and int(records[pid].get("start_ticks", -1)) == start_ticks
        )
        selected = _descendants(records, seeds) - excluded

        now = time.monotonic()
        rows: list[dict[str, Any]] = []
        active_previous: dict[tuple[int, int], tuple[int, float]] = {}
        for pid in sorted(selected):
            record = _read_process_record(pid) or dict(records[pid])
            start_ticks = int(record.get("start_ticks", -1))
            identity = (pid, start_ticks)
            ticks = int(record.pop("_ticks", 0))
            previous = self.process_previous.get(identity)
            if previous is not None and now > previous[1]:
                record["cpu_pct"] = (
                    (ticks - previous[0])
                    / os.sysconf("SC_CLK_TCK")
                    / (now - previous[1])
                    * 100.0
                )
            active_previous[identity] = (ticks, now)
            record.pop("_rss_pages", None)
            record.update(
                timestamp_local=timestamp_local,
                timestamp_utc=timestamp_utc,
                sample=self.sample_count,
                root_pid=self.parent_pid,
                explicitly_tracked=int(pid in self.explicit_tracked),
            )
            rows.append(record)
            self.tracked_identities[pid] = start_ticks
        self.process_previous = active_previous
        self.last_workload_pids = selected
        return rows

    def _gpu_process_rows(
        self,
        timestamp_local: str,
        timestamp_utc: str,
    ) -> list[dict[str, Any]]:
        if self.gpu_disabled_after_timeout or not Path("/usr/bin/nvidia-smi").exists():
            return []
        fields = "pid,process_name,gpu_uuid,used_gpu_memory"
        code, stdout, stderr = _run_capture(
            ["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"],
            timeout=min(max(self.interval * 0.8, 1.0), 3.0),
        )
        if code is None and "timeout" in stderr:
            self.gpu_disabled_after_timeout = True
            self.errors["nvidia_smi_process_timeout"] = (
                self.errors.get("nvidia_smi_process_timeout", 0) + 1
            )
            return []
        if code != 0:
            self.errors["nvidia_smi_process_query_failed"] = (
                self.errors.get("nvidia_smi_process_query_failed", 0) + 1
            )
            return []
        rows: list[dict[str, Any]] = []
        for raw in csv.reader(stdout.splitlines(), skipinitialspace=True):
            if len(raw) < 4:
                continue
            try:
                pid = int(raw[0].strip())
            except ValueError:
                continue
            rows.append(
                {
                    "timestamp_local": timestamp_local,
                    "timestamp_utc": timestamp_utc,
                    "sample": self.sample_count,
                    "pid": pid,
                    "process_name": raw[1].strip(),
                    "gpu_uuid": raw[2].strip(),
                    "used_gpu_memory_mib": _coerce_value(raw[3]),
                    "in_watched_process_tree": int(pid in self.last_workload_pids),
                }
            )
        return rows

    def _cgroup_sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        path = self.cgroup_path
        if path is None:
            return result
        for filename, key, divisor in (
            ("memory.current", "cgroup.memory_current_gib", 1024**3),
            ("memory.peak", "cgroup.memory_peak_gib", 1024**3),
            ("memory.swap.current", "cgroup.memory_swap_current_gib", 1024**3),
            ("pids.current", "cgroup.pids_current", 1),
        ):
            value = _read_number(path / filename, divisor)
            if value is not None:
                result[key] = value
        for filename, prefix in (("memory.events", "cgroup.memory_events"), ("cpu.stat", "cgroup.cpu")):
            raw = _read_text(path / filename) or ""
            for line in raw.splitlines():
                name, _, value = line.partition(" ")
                try:
                    result[f"{prefix}.{_safe_name(name)}"] = int(value.strip())
                except ValueError:
                    continue
        return result

    def _filesystem_sample(self) -> dict[str, Any]:
        try:
            values = os.statvfs(self.output_dir)
        except OSError:
            return {}
        return {
            "filesystem.free_gib": values.f_bavail * values.f_frsize / 1024**3,
            "filesystem.free_inodes": values.f_favail,
        }

    def sample(self) -> dict[str, Any]:
        started = time.perf_counter()
        result: dict[str, Any] = {
            "timestamp_local": _now_local(),
            "timestamp_utc": _now_utc(),
            "elapsed_s": time.monotonic() - self.started_monotonic,
            "sample": self.sample_count,
            "parent_alive": int(self._parent_alive()),
        }
        result.update(self._cpu_sample())
        result.update(self._memory_and_pressure_sample())
        result.update(self._disk_sample())
        result.update(self._hwmon_sample())
        result.update(self._rapl_sample())
        result.update(self._process_sample())
        result.update(self._cgroup_sample())
        result.update(self._filesystem_sample())
        result["system.oom_kill_count"] = _read_vmstat_counter("oom_kill")
        result.update(self._gpu_sample())
        result["monitor.sample_duration_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def _snapshot(self, filename: str, at_end: bool = False) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as file_obj:
            file_obj.write(f"Telemetry label: {self.label}\n")
            file_obj.write(f"Captured: {_now_local()}\n")
            commands = [
                ("date", ["date", "-Is"]),
                ("kernel", ["uname", "-a"]),
                ("kernel command line", ["cat", "/proc/cmdline"]),
                ("CPU summary", ["lscpu"]),
                ("CPU topology", ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ"]),
                ("memory", ["free", "-h"]),
                ("loaded modules", ["lsmod"]),
                ("NVIDIA module", ["modinfo", "nvidia"]),
                ("Intel microcode package", ["dpkg-query", "-W", "intel-microcode"]),
                ("kernel packages", ["dpkg-query", "-W", "linux-image*", "linux-modules*"]),
            ]
            for title, args in commands:
                _write_command_section(file_obj, title, args)

            file_obj.write("\n===== firmware/DMI =====\n")
            for name in ("sys_vendor", "product_name", "board_vendor", "board_name", "bios_vendor", "bios_version", "bios_date"):
                file_obj.write(f"{name}: {_read_text(Path('/sys/class/dmi/id') / name)}\n")
            file_obj.write("\n===== CPU microcode =====\n")
            cpuinfo = _read_text(Path("/proc/cpuinfo")) or ""
            microcodes = sorted(set(re.findall(r"^microcode\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)))
            file_obj.write("\n".join(microcodes) + "\n")

            if at_end:
                file_obj.write("\n===== monitor errors =====\n")
                file_obj.write(json.dumps(self.errors, indent=2) + "\n")

    def _capture_kernel_log(self, filename: str) -> None:
        code, stdout, stderr = _run_capture(
            ["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso-precise"], timeout=15.0
        )
        (self.output_dir / filename).write_text(
            stdout + (f"\n[stderr]\n{stderr}\n" if stderr else "") + f"\n[exit_code={code}]\n",
            encoding="utf-8",
        )

    def _capture_nvidia(self, filename: str) -> None:
        code, stdout, stderr = _run_capture(["nvidia-smi", "-q"], timeout=4.0)
        (self.output_dir / filename).write_text(
            stdout + (f"\n[stderr]\n{stderr}\n" if stderr else "") + f"\n[exit_code={code}]\n",
            encoding="utf-8",
        )

    def _start_kernel_follower(self) -> None:
        output = (self.output_dir / "kernel_during_pipeline.log").open("w", encoding="utf-8")
        try:
            self.journal_process = subprocess.Popen(
                ["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso-precise"],
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            output.write(f"Could not start journal follower: {exc}\n")
            output.close()
            self.errors["journal_follower_start_failed"] = 1
            return
        output.close()

        system_output = (self.output_dir / "system_warnings_during_pipeline.log").open(
            "w", encoding="utf-8"
        )
        try:
            self.system_journal_process = subprocess.Popen(
                [
                    "journalctl", "-f", "-n", "0", "-p", "warning..alert",
                    "-o", "short-iso-precise",
                ],
                stdout=system_output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            system_output.write(f"Could not start system journal follower: {exc}\n")
            system_output.close()
            self.errors["system_journal_follower_start_failed"] = 1
            return
        system_output.close()

    def _stop_kernel_follower(self) -> None:
        for process in (self.journal_process, self.system_journal_process):
            if process is None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass

    def _metadata(self) -> dict[str, Any]:
        temperature_sensors = [sensor for sensor in self.hwmon_sensors if sensor.unit == "C"]
        voltage_sensors = [sensor for sensor in self.hwmon_sensors if sensor.unit == "V"]
        dimm_sensors = [
            sensor for sensor in temperature_sensors
            if re.search(r"dimm|dram|memory|spd", f"{sensor.device} {sensor.label}", re.IGNORECASE)
        ]
        return {
            "schema_version": 2,
            "label": self.label,
            "monitor_pid": os.getpid(),
            "pipeline_pid": self.parent_pid,
            "watched_root_pid": self.parent_pid,
            "hostname": socket.gethostname(),
            "boot_id": _read_text(Path("/proc/sys/kernel/random/boot_id")),
            "started_local": self.started_local,
            "sample_interval_seconds": self.interval,
            "archive_enabled": self.archive_enabled,
            "context": self.context,
            "python": sys.version,
            "cpu_ids": self.cpu_ids,
            "suspect_core_note": "Historical invalid-opcode faults were on sibling CPUs 8-9 (physical core 4).",
            "hwmon_sensors": [
                {
                    "column": sensor.key,
                    "device": sensor.device,
                    "label": sensor.label,
                    "unit": sensor.unit,
                    "path": str(sensor.path),
                }
                for sensor in self.hwmon_sensors
            ],
            "rapl_domains": [
                {
                    "name": domain["name"],
                    "column": domain["key"],
                    "readable_without_root": domain["readable"],
                }
                for domain in self.rapl_domains
            ],
            "availability": {
                "cpu_temperature": bool(temperature_sensors),
                "motherboard_or_cpu_voltage": bool(voltage_sensors),
                "dimm_temperature": bool(dimm_sensors),
                "cpu_package_power_without_root": any(domain["readable"] for domain in self.rapl_domains),
                "gpu_voltage": "voltage.graphics" in self.gpu_optional_fields,
            },
            "limitations": [
                "Software telemetry cannot capture sub-millisecond electrical transients.",
                "Missing sensor values mean the firmware/driver did not expose them; they are not inferred.",
                "CPU package power requires readable Intel RAPL energy counters, which are commonly root-only.",
                "Consumer DDR5 DIMM temperature is unavailable unless an SPD/hwmon driver exposes it.",
                "A kernel hard-lockup can prevent the final sample, but every completed CSV row is fsync'd.",
                "Sampling is observational but not literally free; sample duration is recorded in monitor.sample_duration_ms.",
            ],
        }

    def _fieldnames(self) -> list[str]:
        base = ["timestamp_local", "timestamp_utc", "elapsed_s", "sample", "parent_alive"]
        dynamic: list[str] = []
        dynamic += ["cpu.total.util_pct"]
        for cpu_id in self.cpu_ids:
            dynamic += [
                f"cpu.cpu{cpu_id}.util_pct",
                f"cpu.cpu{cpu_id}.freq_mhz",
                f"cpu.cpu{cpu_id}.core_throttle_count",
            ]
        dynamic += ["cpu.package_throttle_count", "cpu.package_throttle_total_ms"]
        dynamic += [
            "system.load1", "system.load5", "system.load15",
            "memory.total_gib", "memory.available_gib", "memory.used_pct",
            "memory.swap_used_gib", "memory.dirty_mib", "memory.writeback_mib",
            "disk.read_mib_s", "disk.write_mib_s",
            "filesystem.free_gib", "filesystem.free_inodes",
            "system.oom_kill_count",
            "pipeline.process_state", "pipeline.cpu_last", "pipeline.cpu_pct",
            "pipeline.rss_gib", "pipeline.threads",
            "cgroup.memory_current_gib", "cgroup.memory_peak_gib",
            "cgroup.memory_swap_current_gib", "cgroup.pids_current",
            "cgroup.memory_events.low", "cgroup.memory_events.high",
            "cgroup.memory_events.max", "cgroup.memory_events.oom",
            "cgroup.memory_events.oom_kill", "cgroup.memory_events.oom_group_kill",
            "cgroup.cpu.usage_usec", "cgroup.cpu.user_usec", "cgroup.cpu.system_usec",
            "cgroup.cpu.nr_periods", "cgroup.cpu.nr_throttled",
            "cgroup.cpu.throttled_usec", "monitor.sample_duration_ms",
        ]
        for resource in ("cpu", "memory", "io"):
            for category in ("some", "full"):
                for avg in ("avg10", "avg60", "avg300"):
                    dynamic.append(f"psi.{resource}.{category}.{avg}")
        dynamic += [sensor.key for sensor in self.hwmon_sensors]
        dynamic += [domain["key"] for domain in self.rapl_domains]
        dynamic += ["gpu.query_ok", "gpu.error"]
        for field in GPU_BASE_FIELDS + self.gpu_optional_fields:
            dynamic.append(f"gpu0.{field}")
        return base + list(dict.fromkeys(dynamic))

    def _write_readme(self) -> None:
        text = f"""Hardware telemetry for anomaly-detection run
================================================

Label: {self.label}
Started: {self.started_local}
Sampling interval: {self.interval:.3f} seconds
Pipeline PID: {self.parent_pid}

Start with summary.json and pipeline_events.jsonl, then correlate timestamps
against telemetry.csv and kernel_during_pipeline.log.  The full kernel boot
logs before/after the pipeline are retained separately.

process_tree.csv records the watched root PID and all observed descendants,
including stable Linux start ticks so PID reuse is detectable. gpu_processes.csv
maps active CUDA process PIDs to immutable GPU UUIDs. tracked_processes.jsonl
contains child PIDs explicitly registered by an external orchestrator.

Important limitations
---------------------
* This recorder is read-only and changes no clock, voltage, fan or power setting.
* Software polling cannot capture sub-millisecond voltage/thermal transients.
* Empty/unsupported CPU voltage, DIMM temperature or RAPL power fields mean the
  firmware/kernel does not expose them to this unprivileged process.
* The separate recorder survives an ordinary pipeline SIGSEGV and stops after
  noticing the pipeline PID is gone.  A whole-kernel freeze can stop both.
* telemetry.csv is flushed and fsync'd after every sample to preserve the last
  completed readings before a native failure.
* Monitoring runs at a lower CPU scheduling priority when the OS permits it.
  monitor.sample_duration_ms quantifies the observer's polling cost.
"""
        (self.output_dir / "README.txt").write_text(text, encoding="utf-8")

    def _read_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        path = self.output_dir / "pipeline_events.jsonl"
        if not path.is_file():
            return events
        try:
            with path.open(encoding="utf-8") as file_obj:
                for line in file_obj:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        events.append(payload)
        except OSError:
            pass
        return events

    def _build_diagnosis(
        self,
        *,
        reason: str,
        important_kernel_lines: list[str],
        system_warning_lines: list[str],
    ) -> dict[str, Any]:
        events = self._read_events()
        active_stages: dict[int, dict[str, Any]] = {}
        failed_stage_exit: dict[str, Any] | None = None
        keyboard_interrupt_event: dict[str, Any] | None = None
        for event in events:
            name = event.get("event")
            if name == "stage_process_keyboard_interrupt":
                keyboard_interrupt_event = event
            try:
                child_pid = int(event.get("child_pid"))
            except (TypeError, ValueError):
                continue
            if name == "stage_process_started":
                active_stages[child_pid] = event
            elif name == "stage_process_exited":
                active_stages.pop(child_pid, None)
                try:
                    return_code = int(event.get("return_code", 0))
                except (TypeError, ValueError):
                    return_code = 0
                if return_code != 0:
                    failed_stage_exit = event

        evidence: list[dict[str, Any]] = []
        likely_causes: list[dict[str, Any]] = []
        kernel_text = "\n".join(important_kernel_lines)
        warning_text = "\n".join(system_warning_lines)
        if keyboard_interrupt_event is not None:
            likely_causes.append(
                {
                    "cause": "keyboard_interrupt",
                    "confidence": "high",
                    "basis": "The orchestrator recorded a KeyboardInterrupt while the stage child was active.",
                }
            )
        if failed_stage_exit is not None:
            return_code = int(failed_stage_exit.get("return_code", 0))
            likely_causes.append(
                {
                    "cause": "stage_process_signal" if return_code < 0 else "stage_process_nonzero_exit",
                    "confidence": "high",
                    "basis": (
                        f"The child exited with return code {return_code}"
                        + (f" (signal {-return_code})." if return_code < 0 else ".")
                    ),
                    "stage_exit": failed_stage_exit,
                }
            )
        if re.search(r"out of memory|oom-kill", kernel_text, re.IGNORECASE):
            likely_causes.append(
                {
                    "cause": "kernel_oom_kill",
                    "confidence": "high",
                    "basis": "Kernel messages reported an OOM kill during the monitored interval.",
                }
            )
        if re.search(r"NVRM: Xid|GPU has fallen", kernel_text, re.IGNORECASE):
            likely_causes.append(
                {
                    "cause": "nvidia_driver_or_gpu_fault",
                    "confidence": "high",
                    "basis": "Kernel messages reported an NVIDIA Xid/GPU failure.",
                }
            )
        if re.search(r"I/O error|EXT4-fs error|No space left", warning_text, re.IGNORECASE):
            likely_causes.append(
                {
                    "cause": "storage_or_filesystem_failure",
                    "confidence": "high",
                    "basis": "System warnings reported storage/filesystem errors.",
                }
            )

        for key in ("system.oom_kill_count", "cgroup.memory_events.oom_kill"):
            stats = self.stats.get(key)
            if stats is not None and stats.count and stats.maximum > stats.minimum:
                evidence.append(
                    {
                        "type": "counter_increase",
                        "metric": key,
                        "start": stats.minimum,
                        "end": stats.last,
                    }
                )
                if not any(item["cause"] == "kernel_oom_kill" for item in likely_causes):
                    likely_causes.append(
                        {
                            "cause": "memory_cgroup_or_host_oom_kill",
                            "confidence": "high",
                            "basis": f"{key} increased during the monitored interval.",
                        }
                    )

        if active_stages:
            evidence.append(
                {
                    "type": "stage_without_exit_record",
                    "stages": list(active_stages.values()),
                }
            )
            if reason == "watched_process_tree_disappeared" and not likely_causes:
                likely_causes.append(
                    {
                        "cause": "abrupt_external_or_native_termination",
                        "confidence": "medium",
                        "basis": "The watched process tree disappeared while a stage had no exit record.",
                    }
                )

        active_stage = list(active_stages.values())[-1] if active_stages else None
        stage_for_tail = active_stage or failed_stage_exit
        if stage_for_tail:
            stage_log = stage_for_tail.get("stage_log")
            if isinstance(stage_log, str) and Path(stage_log).is_file():
                try:
                    lines = Path(stage_log).read_text(encoding="utf-8", errors="replace").splitlines()
                    tail_path = self.output_dir / "active_stage_log_tail.txt"
                    tail_path.write_text("\n".join(lines[-250:]) + "\n", encoding="utf-8")
                    evidence.append(
                        {
                            "type": "active_stage_log_tail",
                            "source": stage_log,
                            "captured_to": str(tail_path),
                        }
                    )
                except OSError:
                    pass

        if not likely_causes and reason != "ablation_finished":
            likely_causes.append(
                {
                    "cause": "undetermined",
                    "confidence": "low",
                    "basis": "No definitive kernel, cgroup, process-exit, or system-journal signature was recorded.",
                }
            )
        return {
            "schema_version": 1,
            "generated_local": _now_local(),
            "monitor_stop_reason": reason,
            "last_event": events[-1] if events else None,
            "active_stage_at_stop": active_stage,
            "failed_stage_exit": failed_stage_exit,
            "likely_causes": likely_causes,
            "evidence": evidence,
            "important_kernel_events": important_kernel_lines,
            "system_warnings_tail": system_warning_lines[-500:],
            "interpretation_note": (
                "A high-confidence cause requires an explicit kernel/cgroup/exit signature. "
                "Absence of such a signature is reported as undetermined rather than guessed."
            ),
        }

    def _finish(self, reason: str) -> None:
        self._stop_kernel_follower()
        self._capture_kernel_log("kernel_after_pipeline.log")
        self._capture_nvidia("nvidia_smi_end.txt")
        self._snapshot("system_snapshot_end.txt", at_end=True)

        kernel_during = _read_text(self.output_dir / "kernel_during_pipeline.log") or ""
        important_lines = [line for line in kernel_during.splitlines() if IMPORTANT_KERNEL_PATTERN.search(line)]
        system_warnings = _read_text(self.output_dir / "system_warnings_during_pipeline.log") or ""
        diagnosis = self._build_diagnosis(
            reason=reason,
            important_kernel_lines=important_lines,
            system_warning_lines=system_warnings.splitlines(),
        )
        _atomic_json(self.output_dir / "diagnosis.json", diagnosis)
        summary = {
            "schema_version": 2,
            "label": self.label,
            "started_local": self.started_local,
            "ended_local": _now_local(),
            "duration_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "stop_reason": reason,
            "samples": self.sample_count,
            "pipeline_pid": self.parent_pid,
            "pipeline_alive_at_end": self._parent_alive(),
            "last_observed_workload_pids": sorted(self.last_workload_pids),
            "explicitly_tracked_processes": list(self.explicit_tracked.values()),
            "monitor_errors": self.errors,
            "important_kernel_events_during_pipeline": important_lines,
            "system_warning_lines": system_warnings.splitlines()[-500:],
            "metrics": {key: value.as_dict() for key, value in sorted(self.stats.items())},
        }
        archive_path = self.output_dir.parent / f"{self.output_dir.name}.tar.gz"
        summary["archive_path"] = str(archive_path)
        summary["archive_enabled"] = self.archive_enabled
        summary["archive_created"] = False
        _atomic_json(self.output_dir / "summary.json", summary)
        self.log(f"monitor stopped: reason={reason} samples={self.sample_count}")
        if not self.archive_enabled:
            return
        try:
            with tarfile.open(archive_path, mode="w:gz") as archive:
                archive.add(self.output_dir, arcname=self.output_dir.name)
            summary["archive_created"] = True
            _atomic_json(self.output_dir / "summary.json", summary)
        except (OSError, tarfile.TarError) as exc:
            summary["archive_error"] = f"{type(exc).__name__}: {exc}"
            _atomic_json(self.output_dir / "summary.json", summary)
            self.log(f"could not create IT archive: {type(exc).__name__}: {exc}")

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.nice(10)
        except OSError:
            self.errors["lower_priority_failed"] = 1
        self.log(f"monitor starting for pipeline pid {self.parent_pid}")
        self._probe_gpu_optional_fields()
        _atomic_json(self.output_dir / "metadata.json", self._metadata())
        self._write_readme()
        self._snapshot("system_snapshot.txt")
        self._capture_kernel_log("kernel_before_pipeline.log")
        self._capture_nvidia("nvidia_smi_start.txt")
        self._start_kernel_follower()

        fieldnames = self._fieldnames()
        (self.output_dir / ".ready").write_text(_now_local(), encoding="utf-8")
        reason = "normal_stop"
        next_sample = time.monotonic()
        try:
            with (
                (self.output_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as file_obj,
                (self.output_dir / "process_tree.csv").open("w", newline="", encoding="utf-8") as process_file,
                (self.output_dir / "gpu_processes.csv").open("w", newline="", encoding="utf-8") as gpu_process_file,
            ):
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
                process_writer = csv.DictWriter(
                    process_file, fieldnames=PROCESS_FIELDNAMES, extrasaction="ignore"
                )
                gpu_process_writer = csv.DictWriter(
                    gpu_process_file, fieldnames=GPU_PROCESS_FIELDNAMES, extrasaction="ignore"
                )
                writer.writeheader()
                process_writer.writeheader()
                gpu_process_writer.writeheader()
                for output_file in (file_obj, process_file, gpu_process_file):
                    output_file.flush()
                    os.fsync(output_file.fileno())

                while not self.stop_event.is_set():
                    if self.duration is not None and time.monotonic() - self.started_monotonic >= self.duration:
                        reason = "duration_elapsed"
                        break

                    cycle_started = time.perf_counter()
                    sample = self.sample()
                    timestamp_local = str(sample["timestamp_local"])
                    timestamp_utc = str(sample["timestamp_utc"])
                    process_rows = self._process_tree_rows(timestamp_local, timestamp_utc)
                    gpu_process_rows = self._gpu_process_rows(timestamp_local, timestamp_utc)
                    sample["monitor.sample_duration_ms"] = (
                        time.perf_counter() - cycle_started
                    ) * 1000.0
                    writer.writerow(sample)
                    process_writer.writerows(process_rows)
                    gpu_process_writer.writerows(gpu_process_rows)
                    for output_file in (file_obj, process_file, gpu_process_file):
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    self.sample_count += 1
                    for key, value in sample.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            self.stats.setdefault(key, RunningStats()).add(value)

                    if not process_rows and self.exit_when_process_tree_disappears:
                        reason = "watched_process_tree_disappeared"
                        break

                    next_sample += self.interval
                    wait_for = max(0.0, next_sample - time.monotonic())
                    self.stop_event.wait(wait_for)
                if self.stop_event.is_set():
                    reason = _read_text(self.output_dir / ".stop_reason") or "signal_stop"
        except Exception as exc:  # telemetry failures must still produce a summary
            reason = f"monitor_exception:{type(exc).__name__}"
            self.errors[reason] = self.errors.get(reason, 0) + 1
            self.log(f"monitor exception: {type(exc).__name__}: {exc}")
        finally:
            self._finish(reason)
        return 0


class HardwareMonitor:
    """Controller used by run_anomaly_detection.py."""

    def __init__(
        self,
        output_dir: Path,
        interval: float = 1.0,
        label: str = "anomaly_detection",
        *,
        watched_pid: int | None = None,
        context: dict[str, Any] | None = None,
        archive_enabled: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.label = label
        self.watched_pid = watched_pid if watched_pid is not None else os.getpid()
        self.context = context or {}
        self.archive_enabled = archive_enabled
        self.process: subprocess.Popen[Any] | None = None
        self.events_path = self.output_dir / "pipeline_events.jsonl"
        self.tracked_processes_path = self.output_dir / "tracked_processes.jsonl"

    def start(self, *, wait_for_ready: bool = True) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        worker_log = (self.output_dir / "worker_stdio.log").open("a", encoding="utf-8")
        args = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(self.output_dir),
            "--parent-pid",
            str(self.watched_pid),
            "--interval",
            str(self.interval),
            "--label",
            self.label,
            "--context-json",
            json.dumps(self.context, ensure_ascii=False, default=str),
        ]
        if not self.archive_enabled:
            args.append("--no-archive")
        try:
            self.process = subprocess.Popen(
                args,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            worker_log.close()
            return False
        worker_log.close()
        os.environ["PIPELINE_TELEMETRY_EVENTS_FILE"] = str(self.events_path)
        self.mark_event(
            "telemetry_controller_started",
            monitor_pid=self.process.pid,
            watched_root_pid=self.watched_pid,
            interval_s=self.interval,
        )

        if not wait_for_ready:
            return self.process.poll() is None

        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if (self.output_dir / ".ready").exists():
                return True
            if self.process.poll() is not None:
                return False
            time.sleep(0.1)
        # It may merely be waiting on a slow system snapshot. Keep it running.
        return self.process.poll() is None

    def mark_event(self, event: str, **details: Any) -> None:
        try:
            _append_jsonl(
                self.events_path,
                {
                    "timestamp_local": _now_local(),
                    "timestamp_utc": _now_utc(),
                    "pid": os.getpid(),
                    "controller_pid": os.getpid(),
                    "watched_root_pid": self.watched_pid,
                    "event": event,
                    **details,
                },
            )
        except OSError:
            return

    def track_process(self, pid: int, role: str, **details: Any) -> None:
        """Register a child PID durably so it remains tracked after reparenting."""

        try:
            _append_jsonl(
                self.tracked_processes_path,
                {
                    "timestamp_local": _now_local(),
                    "timestamp_utc": _now_utc(),
                    "controller_pid": os.getpid(),
                    "watched_root_pid": self.watched_pid,
                    "pid": int(pid),
                    "role": role,
                    **details,
                },
            )
        except (OSError, TypeError, ValueError):
            return

    def stop(self, reason: str = "pipeline_finished") -> None:
        if self.process is None:
            return
        self.mark_event("telemetry_controller_stopping", reason=reason)
        if self.process.poll() is None:
            try:
                (self.output_dir / ".stop_reason").write_text(reason, encoding="utf-8")
                self.process.send_signal(signal.SIGTERM)
                self.process.wait(timeout=12.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    pass
        if os.environ.get("PIPELINE_TELEMETRY_EVENTS_FILE") == str(self.events_path):
            os.environ.pop("PIPELINE_TELEMETRY_EVENTS_FILE", None)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Linux/NVIDIA hardware telemetry recorder")
    parser.add_argument("--output", required=True, type=Path, help="Telemetry output directory")
    parser.add_argument("--parent-pid", type=int, default=os.getppid(), help="Process PID to watch")
    parser.add_argument("--interval", type=float, default=1.0, help="Sample interval in seconds")
    parser.add_argument("--label", default="standalone", help="Human-readable run label")
    parser.add_argument("--duration-seconds", type=float, default=None, help="Optional standalone time limit")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help=(
            "Continue until SIGTERM/duration even without a watched process tree. "
            "Used by remote cluster recorders; host and all GPU-process data are still captured."
        ),
    )
    parser.add_argument("--context-json", default="{}", help="JSON metadata supplied by the controller")
    parser.add_argument("--no-archive", action="store_true", help="Keep the telemetry directory without creating a tar.gz copy")
    args = parser.parse_args()

    try:
        context = json.loads(args.context_json)
    except json.JSONDecodeError:
        context = {"unparsed_context": args.context_json}
    if not isinstance(context, dict):
        context = {"context": context}

    worker = TelemetryWorker(
        args.output.resolve(),
        args.parent_pid,
        args.interval,
        args.label,
        args.duration_seconds,
        context=context,
        archive_enabled=not args.no_archive,
        exit_when_process_tree_disappears=not args.standalone,
    )

    def request_stop(signum: int, _frame: Any) -> None:
        worker.log(f"received signal {signum}")
        worker.stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(_main())
