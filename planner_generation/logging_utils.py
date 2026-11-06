"""Configure planner-generation logs while sharing core handler behavior."""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console

from pipeline_core.logging import configure_logging


def setup_logging(
    dataset_name: str,
    log_dir: Path,
    console: Console | None = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.WARNING,
) -> Path:
    """Configure one dated planner log and return its path."""
    now = datetime.now()
    log_file = Path(log_dir) / now.strftime("%Y-%m-%d") / (
        f"{dataset_name}_{now.strftime('%H%M%S')}_log.txt"
    )
    return configure_logging(
        log_file,
        console=console,
        file_level=file_level,
        console_level=console_level,
    )
