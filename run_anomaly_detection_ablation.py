#!/usr/bin/env python3
"""CLI facade for the full-pipeline ablation orchestrator."""

from typing import Any

from anomaly_detection.ablation import orchestration as _implementation
from anomaly_detection.ablation.orchestration import *  # noqa: F401,F403


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
