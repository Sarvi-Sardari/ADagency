"""Logging facade for anomaly detection and standalone post-run stages.

Logs remain inside each run's ``logs`` directory. Terminal capture and handler
construction live in :mod:`pipeline_core.logging` so planner and detection
stages use the same implementation without changing their public imports.
"""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console

from pipeline_core.logging import (
    capture_terminal_output,
    configure_logging,
    open_terminal_stream,
)

__all__ = ["capture_terminal_output", "open_terminal_stream", "setup_logging"]


def setup_logging(
    run_name: str,
    log_dir: Path,
    console: Console | None = None,
    file_level: int = logging.DEBUG,
    console_level: int | None = logging.WARNING,
) -> Path:
    """Configure one timestamped stage log and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{run_name}_{timestamp}.log"
    return configure_logging(
        log_file,
        console=console,
        file_level=file_level,
        console_level=console_level,
    )
