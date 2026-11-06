"""Lazy bridge from the study orchestrator to the notebook-derived XAI core.

Keeping this bridge small has two benefits: config/evidence validation never
imports CUDA libraries, and the outer pipeline can unit-test resumability
without loading a multi-billion-parameter checkpoint.
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)


def _load_engine():
    try:
        return importlib.import_module(".xai_engine", package=__package__)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"The active Python environment is missing the XAI dependency "
            f"{error.name!r}. Activate/install the environment described by "
            "pipeline_project/requirements.txt before running an attribution arm."
        ) from error


def run_single_combo(
    *,
    model_path: str,
    adapter_name: str,
    output_dir: str,
    param_overrides: dict[str, Any],
    samples: list[dict[str, Any]],
    final_mask_callback: Callable[[str, dict[str, Any], dict[str, Any]], None],
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Generate attribution maps for the supplied pending anomaly samples.

    ``final_mask_callback`` is called immediately after each image-level map
    is durably saved by the core.  The outer pipeline uses it to atomically
    publish the metrics-compatible PNG, so completed images survive a later
    CUDA crash in the same dataset/method run.
    """

    engine = _load_engine()
    start = time.perf_counter()
    result = engine.run_pipeline(
        param_overrides=param_overrides,
        model_path=model_path,
        model_adapter=adapter_name,
        output_dir=output_dir,
        samples_override=samples,
        final_mask_callback=final_mask_callback,
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - start
    return {
        **result,
        "elapsed_sec": elapsed,
        "avg_sec_per_image": (
            elapsed / result["num_images"] if result.get("num_images") else None
        ),
    }
