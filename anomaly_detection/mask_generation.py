"""Compatibility facade for the structured mask-generation implementation.

Existing entry points and extensions may continue importing from this module;
the implementation now lives in :mod:`anomaly_detection.masking.orchestration`.
"""

from typing import Any

from .masking import orchestration as _implementation
from .masking.orchestration import *  # noqa: F401,F403
from .masking.orchestration import (
    _boxes_to_mask,
    _build_prompt,
    _parse_boxes,
    _secondary_reasoning,
)


def __getattr__(name: str) -> Any:
    """Preserve access to legacy private helpers during the transition."""
    return getattr(_implementation, name)
