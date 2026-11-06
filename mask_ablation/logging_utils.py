"""
logging_utils.py (mask_ablation)
==================================
Central logging configuration for the mask-generation-method ablation study.

Copied from anomaly_detection/logging_utils.py (same convention: every
pipeline_project sub-module is self-contained and does not import from its
siblings). See that file for the full rationale; summarized here:

Every module in mask_ablation should do::

    import logging
    logger = logging.getLogger(__name__)

and log through it instead of calling ``print()``. Nothing is configured
until :func:`setup_logging` is called (once, from run_mask_ablation.py).

One log file per CLI invocation, written to ``<output_root>/logs/`` so a
study's full history (config, per-combo progress, warnings, stack traces)
travels with its outputs, exactly like mask_generation.py's convention of
keeping logs inside the run/output directory instead of a separate
top-level pipeline_project/logs/ tree.
"""

import contextlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def open_terminal_stream():
    """Return an independent handle to the real terminal for Rich only.

    ``capture_terminal_output`` redirects OS file descriptors 1 and 2, not
    merely ``sys.stdout``/``sys.stderr``.  Duplicating stdout first means the
    Rich progress renderer keeps writing to the terminal while every ordinary
    Python, tqdm, C-extension, or third-party write goes to the invocation log.
    """
    return os.fdopen(
        os.dup(sys.__stdout__.fileno()),
        "w",
        encoding=sys.__stdout__.encoding or "utf-8",
        buffering=1,
    )


def setup_logging(
    run_name: str,
    log_dir: Path,
    console: Console | None = None,
    file_level: int = logging.DEBUG,
    console_level: int | None = None,
) -> Path:
    """
    Configure the root logger for one mask_ablation CLI invocation.

    Input: run_name (used to name the log file, e.g. "mask_ablation");
    log_dir (directory the log file is written into -- callers pass
    ``<output_root>/logs``); console (shared Rich console so log lines
    interleave correctly with an active `rich.progress.Progress` bar);
    file_level/console_level (minimum levels written to the log file /
    printed to the terminal respectively).  The ablation runner deliberately
    leaves ``console_level`` as ``None``: its terminal is reserved for the
    compact Rich progress display and final summary, while every diagnostic is
    retained in the invocation log.

    Output: path to the created log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(file_level if console_level is None else min(file_level, console_level))
    root_logger.handlers.clear()  # avoid duplicate handlers if run twice in one process

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    root_logger.addHandler(file_handler)

    if console_level is not None:
        console_handler = RichHandler(
            console=console or Console(),
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(console_level)
        root_logger.addHandler(console_handler)

    return log_file


@contextlib.contextmanager
def capture_terminal_output(log_file: Path):
    """Append ordinary stdout/stderr (prints, warnings and tqdm) to ``log_file``.

    The XAI core originated in a notebook and intentionally keeps detailed
    per-token diagnostics.  Redirecting at the CLI boundary also captures
    Transformers checkpoint bars and warnings without having to silence useful
    diagnostics or patch third-party libraries.  The Rich console is created
    with ``sys.__stdout__`` before entering this context, so live progress is
    still the only thing rendered in the user's terminal.
    """
    with Path(log_file).open("a", encoding="utf-8", buffering=1) as stream:
        stream.write("\n--- captured stdout/stderr begins ---\n")
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)
        try:
            # Some progress libraries retain sys.__stderr__ or write directly
            # through a file descriptor.  dup2 catches those paths as well.
            sys.__stdout__.flush()
            sys.__stderr__.flush()
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            # ``sys.__stdout__`` / ``sys.__stderr__`` can retain a buffered
            # write even though ordinary print() was redirected. Flush them
            # before restoring fd 1/2 so they cannot leak to the terminal.
            sys.__stdout__.flush()
            sys.__stderr__.flush()
            stream.flush()
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            stream.write("--- captured stdout/stderr ends ---\n")
