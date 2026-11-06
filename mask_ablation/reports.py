"""Compatibility facade for :mod:`mask_ablation.xai.reporting`."""

from typing import Any

from .xai import reporting as _implementation
from .xai.reporting import *  # noqa: F401,F403


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)
