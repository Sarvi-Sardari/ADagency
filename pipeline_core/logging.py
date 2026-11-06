"""Common logging and terminal-capture primitives."""

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(
    log_file: Path,
    *,
    console: Console | None,
    file_level: int,
    console_level: int | None,
) -> Path:
    """Configure the root logger for one invocation and return its log path."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(file_level if console_level is None else min(file_level, console_level))
    root.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)
    if console_level is not None:
        rich_handler = RichHandler(
            console=console or Console(),
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(console_level)
        root.addHandler(rich_handler)
    return log_file


def open_terminal_stream() -> TextIO:
    """Open a terminal stream independent of stdout/stderr redirection."""
    return os.fdopen(
        os.dup(sys.__stdout__.fileno()),
        "w",
        encoding=sys.__stdout__.encoding or "utf-8",
        buffering=1,
    )


@contextlib.contextmanager
def capture_terminal_output(log_file: Path) -> Iterator[None]:
    """Redirect Python and OS stdout/stderr to an append-only log stream."""
    with Path(log_file).open("a", encoding="utf-8", buffering=1) as stream:
        stream.write("\n--- captured stdout/stderr begins ---\n")
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            sys.__stdout__.flush()
            sys.__stderr__.flush()
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            sys.__stdout__.flush()
            sys.__stderr__.flush()
            stream.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            stream.write("--- captured stdout/stderr ends ---\n")
