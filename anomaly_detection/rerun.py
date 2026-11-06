"""Threshold selection + low-confidence VLM rerun stage.

This module is the ported/consolidated version of the two rerun-related
sections in the old notebook (online_anomaly_detector_thresholding_and_rerun.py):

1. "Thresholding" section — its `lowest_log_prob_getter` only ever needed a
   SINGLE per-image anomaly score (yes-step -> confidence/100, no-step ->
   (100-confidence)/100) to build the y_true/y_score arrays fed into
   roc_curve()/roc_auc_score() and sweep every threshold candidate. Whether a
   step's confidence is < 99 or < 100 never mattered there because the
   "bottom N steps" it also returned were discarded — only the single
   per-image score was used. That logic lives here as `_score_details()` /
   `lowest_log_prob_score()`.

2. "Rerun" section — a SECOND function with the same name (it shadowed the
   first one in the notebook) added `manual_threshold`/`no_steps` parameters
   because the bottom-N low-confidence steps now actually get used: they are
   the specific planner questions that get re-asked to the VLM one at a time.
   Called with `manual_threshold=False` (so the "< 100" filter keeps every
   step that wasn't scored at exactly 100% confidence) inside
   `rerun_based_on_thresholding`. That logic is `lowest_confidence_steps()`
   below, feeding `rerun_low_confidence_predictions()`.

Pipeline behavior (mirrors the notebook's `rerun_based_on_thresholding`):
  - Only images the decision maker currently predicts as "good" are eligible
    for rerun. Images already predicted "anomaly" are left untouched (assumed
    already correct).
  - Of those, only images whose per-image score (see #1 above) is >= the
    optimum threshold are rerun — i.e. borderline "good" predictions that
    could be false negatives.
  - For each eligible image, its bottom-N lowest-confidence planner steps
    (see #2 above) are re-asked to the VLM one at a time (in ascending
    confidence order). The first step answered "Yes" (anomaly) flips the
    image's prediction to anomaly and stops early; if none are "Yes" the
    original "good" prediction is kept.

For runs with a saved secondary-classification profile, the optional
``global-secondary`` policy instead:
  - takes the globally lowest-confidence N steps referenced by any configured
    secondary-class rule;
  - converts their mean confidence into one image-level uncertainty score;
  - calibrates an optimum threshold against whether the initial secondary
    prediction is correct;
  - reruns those exact N steps for every image crossing the threshold and
    reevaluates all configured class conditions using the complete updated
    answer set; and
  - confirms any cross-binary class change against a separately calibrated
    binary threshold. Same-binary secondary changes need no binary approval.

All rerun artifacts (threshold ROC plot, per-image rerun step JSON files, and
the final rerun_result.jsonl) are written under
<outdir>/<dataset_name>/rerun_results/ instead of directly inside the dataset
folder.

rerun_result.jsonl entries are keyed the same way all_data_summary.jsonl is:
(label_folder, source_label_folder, img_name). The source component prevents
same-name images in merged anomaly folders from overwriting or resuming one
another; non-merged datasets retain their flat folder structure.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import json_repair
import numpy as np
import yaml
from PIL import Image
from rich.console import Console
from rich.table import Column
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from .config import (
    ModelsConfig,
    artifact_group_dir,
    load_models_config,
    resolve_portable_file,
)
from .hardware_monitor import HardwareMonitor, record_pipeline_event
from .logging_utils import capture_terminal_output, open_terminal_stream, setup_logging
from .models import vlm
from .scoring import read_confidence_scores, score_details, lowest_log_prob_score
from .secondary_classification import (
    SecondaryClassificationProfile,
    binary_anomaly_score,
    classify_yes_steps,
    load_profile_snapshot,
    profile_fingerprint,
    score_classes,
)
from .utils import build_prompt

from cluster_deployment.config import load_deployment_config
from cluster_deployment.remote_monitor import build_monitor, monitored
from cluster_deployment.server_manager import managed_server_lifecycle

logger = logging.getLogger(__name__)

RERUN_RESULTS_SUBDIR = "rerun_results"
GLOBAL_SECONDARY_SCORE_VERSION = "mean_uncertainty_of_global_bottom_n_v1"


@dataclass(frozen=True)
class ThresholdCandidate:
    threshold: float
    false_positive_false_negative: int
    f1: float
    accuracy: float
    false_positive: int
    false_negative: int


from .rerun_stage.artifacts import (
    _entry_key,
    _img_key,
    _read_jsonl,
    _resolve_dataset_artifact,
    _resolve_probs_path,
)
from pipeline_core.artifacts import atomic_write_text


# ---------------------------------------------------------------------
# Resume support: rerun_result.jsonl is written incrementally (one line per
# processed image, flushed immediately) so a crash/segfault mid-run loses at
# most the single image being processed when it happened, never the whole
# run. On the NEXT invocation against the same outdir, every previously
# written entry is re-validated (not just "does the file exist") before
# being reused -- see `_validate_resumed_entry()`.
# ---------------------------------------------------------------------

_TERMINAL_RERUN_STATUSES = {
    "no_confidence_data",
    "not_rerun_below_threshold",
    "not_rerun_already_anomaly",
    "not_rerun_no_low_confidence_steps",
    "flipped_to_anomaly",
    "secondary_class_changed",
    "rerun_confirmed_good",
    "rerun_confirmed_anomaly",
    "rerun_rejected_binary_change",
    "not_rerun_no_class_uncertainty",
    "not_rerun_below_secondary_threshold",
    "flipped_to_good",
}


def _current_rerun_meta(
    threshold: float,
    num_steps: int,
    modelcfg: ModelsConfig,
    prompts_dir: Path,
    secondary_profile: SecondaryClassificationProfile | None = None,
    secondary_policy: str = "binary-gate",
    secondary_confidence_threshold: float = 90.0,
    binary_threshold: float | None = None,
) -> dict[str, Any]:
    """Fingerprint of everything that affects whether a PRIOR rerun_result.jsonl
    (from an earlier, possibly partial, attempt at this same outdir) is still
    valid to resume from. Mirrors mask_generation.py's per-method settings
    fingerprint -- if the threshold, step count, VLM, or prompts changed since
    the last attempt, silently mixing old and new results would be wrong."""
    rerun_prompt_path = prompts_dir / "rerun_step.j2"
    rerun_prompt_sha256 = (
        hashlib.sha256(rerun_prompt_path.read_bytes()).hexdigest()
        if rerun_prompt_path.is_file() else None
    )
    return {
        "threshold": round(float(threshold), 6),
        "num_steps": int(num_steps),
        "vlm_model": modelcfg.vlm.modelname,
        "vlm_source": modelcfg.vlm.source,
        "vlm_chat_template_kwargs": dict(modelcfg.vlm.chat_template_kwargs),
        "prompts_dir": str(prompts_dir),
        "rerun_prompt_sha256": rerun_prompt_sha256,
        "secondary_profile_fingerprint": (
            profile_fingerprint(secondary_profile) if secondary_profile else None
        ),
        "secondary_policy": secondary_policy,
        "secondary_confidence_threshold": round(float(secondary_confidence_threshold), 6),
        "binary_threshold": (
            round(float(binary_threshold), 6) if binary_threshold is not None else None
        ),
        "global_secondary_score_version": (
            GLOBAL_SECONDARY_SCORE_VERSION
            if secondary_policy == "global-secondary" else None
        ),
    }


def _meta_matches(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if previous.get("num_steps") != current.get("num_steps"):
        return False
    if previous.get("vlm_model") != current.get("vlm_model"):
        return False
    if previous.get("vlm_source") != current.get("vlm_source"):
        return False
    if previous.get("vlm_chat_template_kwargs", {}) != current.get("vlm_chat_template_kwargs", {}):
        return False
    # The prompt directory is provenance, not semantic input.  Cross-host
    # resumes legitimately rebase the same prompt tree to a different mount;
    # the content hash below is the portable compatibility contract.
    if previous.get("rerun_prompt_sha256") != current.get("rerun_prompt_sha256"):
        return False
    if previous.get("secondary_profile_fingerprint") != current.get("secondary_profile_fingerprint"):
        return False
    if previous.get("secondary_policy", "binary-gate") != current.get("secondary_policy", "binary-gate"):
        return False
    if previous.get("secondary_confidence_threshold", 90.0) != current.get("secondary_confidence_threshold", 90.0):
        return False
    if previous.get("binary_threshold") != current.get("binary_threshold"):
        return False
    if previous.get("global_secondary_score_version") != current.get("global_secondary_score_version"):
        return False
    try:
        return abs(float(previous.get("threshold")) - float(current.get("threshold"))) < 1e-6
    except (TypeError, ValueError):
        return False


def _validate_resumed_entry(entry: dict[str, Any], dataset_output_dir: Path) -> bool:
    """
    Decide whether a PREVIOUSLY written rerun_result.jsonl entry can be
    safely reused instead of recomputing it. Unlike a plain "does the file
    exist" check, this validates the actual CONTENT:
      - all required fields are present and internally consistent
        (label/binary_prediction agree with each other)
      - for statuses that imply the VLM was actually rerun
        (flipped_to_anomaly / rerun_confirmed_good), the per-image
        *_rerun_steps.json it points to actually exists, parses as a
        non-empty JSON list of step dicts, and its Yes/No outcomes are
        consistent with the recorded status (a crash can leave a
        stale/half-written steps file, or a status line flushed just before
        the steps file itself was written)
      - the recorded continuous "score" is present and finite wherever a
        score is expected for that status
    Returns False (meaning: re-run this image from scratch) for anything
    that doesn't fully check out.
    """
    required = {"label_folder", "img_name", "ground_truth", "binary_prediction", "label", "rerun_status"}
    if not required.issubset(entry.keys()):
        return False

    status = entry.get("rerun_status")
    if status not in _TERMINAL_RERUN_STATUSES:
        return False

    label = entry.get("label")
    prediction = entry.get("binary_prediction")
    if label not in ("good", "anomaly") or prediction not in (0, 1):
        return False
    if (label == "anomaly") != (prediction == 1):
        return False

    secondary_profile = load_profile_snapshot(dataset_output_dir)
    if secondary_profile is not None:
        secondary_required = {
            "yes_steps", "secondary_class", "secondary_class_scores",
            "secondary_binary_score",
        }
        if not secondary_required.issubset(entry):
            return False
        try:
            secondary_outcome = classify_yes_steps(secondary_profile, entry["yes_steps"])
        except (TypeError, ValueError):
            return False
        if (
            entry["secondary_class"] != secondary_outcome.secondary_class
            or entry["label"] != secondary_outcome.binary_label
            or entry["binary_prediction"] != secondary_outcome.binary_prediction
        ):
            return False
        class_scores = entry.get("secondary_class_scores")
        if not isinstance(class_scores, dict) or set(class_scores) != set(secondary_profile.class_names):
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in class_scores.values()
        ):
            return False
        saved_binary_score = entry.get("secondary_binary_score")
        if (
            isinstance(saved_binary_score, bool)
            or not isinstance(saved_binary_score, (int, float))
            or not math.isfinite(float(saved_binary_score))
            or not math.isclose(
                float(saved_binary_score),
                binary_anomaly_score(secondary_profile, class_scores),
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            return False

    if status in (
        "flipped_to_anomaly", "flipped_to_good", "secondary_class_changed",
        "rerun_confirmed_good", "rerun_confirmed_anomaly",
        "rerun_rejected_binary_change",
    ):
        steps_path_raw = entry.get("rerun_steps_filepath")
        if not steps_path_raw:
            return False
        # Rerun-step paths were historically stored as absolute paths. Runs
        # are frequently copied or renamed afterwards, so resolve them with
        # the same dataset-relative relocation fallback used during inference
        # instead of rejecting every otherwise-valid resumed entry.
        steps_path = _resolve_dataset_artifact(
            steps_path_raw,
            dataset_output_dir,
            Path(__file__).resolve().parent.parent,
        )
        if not steps_path.is_file() or steps_path.stat().st_size == 0:
            return False
        try:
            steps = json.loads(steps_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if not isinstance(steps, list) or not steps:
            return False
        if not all(isinstance(step, dict) and "is_anomaly" in step and "step_number" in step for step in steps):
            return False
        if secondary_profile is None:
            any_yes = any(bool(step.get("is_anomaly")) for step in steps)
            if status in ("flipped_to_anomaly", "rerun_confirmed_anomaly") and (
                prediction != 1 or not any_yes
            ):
                return False
            if status in ("flipped_to_good", "rerun_confirmed_good") and (
                prediction != 0 or any_yes
            ):
                return False
        elif status == "flipped_to_anomaly" and prediction != 1:
            return False
        elif status == "flipped_to_good" and prediction != 0:
            return False
        elif status == "rerun_confirmed_good" and prediction != 0:
            return False
        elif status == "rerun_confirmed_anomaly" and prediction != 1:
            return False

    if status != "no_confidence_data":
        score = entry.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False

    return True


def _load_resumable_rerun_state(
    results_dir: Path, dataset_output_dir: Path, current_meta: dict[str, Any], overwrite: bool,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    Loads and validates a previous (possibly partial) rerun_result.jsonl at
    this same outdir, returning a {entry_key: entry} map of entries safe to
    reuse. Returns an empty map (nothing to reuse) if overwrite is requested,
    no prior state exists, or the prior state doesn't validate. Raises if a
    prior fingerprint exists and its settings differ from the current run
    AND overwrite wasn't requested (mirrors mask_generation.py's fingerprint
    mismatch behavior) -- this prevents silently mixing rerun results
    computed under different thresholds/step-counts/models.
    """
    meta_path = results_dir / "rerun_meta.json"
    rerun_result_path = results_dir / "rerun_result.jsonl"

    if overwrite:
        if rerun_result_path.is_file() or meta_path.is_file():
            logger.info("overwrite requested: ignoring any previous rerun state at %s", results_dir)
        return {}

    if meta_path.is_file():
        try:
            previous_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            logger.warning(
                "Could not read previous rerun metadata (%s: %s); treating this as a fresh rerun.", meta_path, error
            )
            previous_meta = None
        if previous_meta is not None and not _meta_matches(previous_meta, current_meta):
            raise ValueError(
                f"Rerun settings changed since the last (possibly partial) rerun attempt at {results_dir} "
                f"(previous={previous_meta}, current={current_meta}). Pass --overwrite to discard the old "
                "rerun state and start fresh instead of silently mixing results computed under different settings."
            )

    if not rerun_result_path.is_file():
        return {}

    resumed: dict[tuple[str, str, str], dict[str, Any]] = {}
    discarded = 0
    for prior in _read_jsonl(rerun_result_path):
        if _validate_resumed_entry(prior, dataset_output_dir):
            resumed[_entry_key(prior)] = prior
        else:
            discarded += 1
            logger.warning(
                "Discarding unusable/incomplete prior rerun entry for %s/%s (status=%s) — will be reprocessed.",
                prior.get("label_folder"), prior.get("img_name"), prior.get("rerun_status"),
            )
    if resumed:
        logger.info(
            "Resuming rerun: %d previously-processed image(s) validated and will be reused (%d discarded).",
            len(resumed), discarded,
        )
    elif discarded:
        logger.info("Found a prior rerun_result.jsonl but none of its %d entries validated; reprocessing all images.", discarded)
    return resumed


# Public compatibility imports: callers can continue importing these names from
# anomaly_detection.rerun while their implementation lives in rerun_stage.
from .rerun_stage.thresholding import (
    class_aware_rerun_steps,
    global_secondary_rerun_score,
    lowest_confidence_steps,
)


def _collect_scores(outdir: Path, dataset_name: str) -> tuple[list[int], list[float]]:
    dataset_output_dir = outdir / dataset_name
    summary_path = dataset_output_dir / "all_data_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"First-run summary not found: {summary_path}")

    y_true: list[int] = []
    y_score: list[float] = []
    skipped_unusable = 0
    secondary_profile = load_profile_snapshot(dataset_output_dir)
    for result in _read_jsonl(summary_path):
        if secondary_profile is not None:
            score = result.get("secondary_binary_score")
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                y_true.append(int(result["gt_binary"]))
                y_score.append(float(score))
                continue
            skipped_unusable += 1
            logger.warning("Skipping secondary result without a usable binary score: %s", result)
            continue
        probs_path = result.get("probs_data_filepath")
        if not probs_path:
            logger.warning("Skipping result without probs_data_filepath: %s", result)
            continue

        _, details = lowest_log_prob_score(_resolve_probs_path(probs_path, dataset_output_dir))
        if details is None or not math.isfinite(float(details["score"])):
            skipped_unusable += 1
            logger.warning("Skipping result without usable probability data: %s", result)
            continue

        y_true.append(int(result["gt_binary"]))
        y_score.append(float(details["score"]))

    if not y_true:
        raise ValueError(
            f"No usable finite probability scores found in {summary_path}. "
            f"{skipped_unusable} image(s) have missing, NaN, or infinite confidence values. "
            "This run must be regenerated before the rerun threshold can be calculated."
        )
    if len(set(y_true)) < 2:
        raise ValueError("Threshold selection requires both good and anomaly ground-truth samples.")

    return y_true, y_score
def resolve_secondary_ground_truth(
    dataset_name: str,
    *,
    explicit_path: str | Path | None = None,
    metrics_config_path: str | Path | None = None,
) -> Path:
    """Resolve the class-label file used to calibrate secondary rerun need."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Secondary ground truth not found: {path}")
        return path
    if metrics_config_path is None:
        raise ValueError(
            "The global-secondary policy requires --secondary-ground-truth or "
            "--metrics-config so its threshold can be calibrated against secondary labels."
        )
    config_path = resolve_portable_file(metrics_config_path, PROJECT_ROOT, Path.cwd())
    if not config_path.is_file():
        raise FileNotFoundError(f"Metrics config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), list):
        raise ValueError(f"Metrics config has no datasets list: {config_path}")
    matches = [
        entry for entry in raw["datasets"]
        if isinstance(entry, dict) and entry.get("name") == dataset_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Metrics config must contain exactly one dataset named {dataset_name!r}."
        )
    dataset_entry = matches[0]
    settings = dataset_entry.get("secondary_metrics")
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        raise ValueError(
            f"Dataset {dataset_name!r} has no enabled secondary_metrics ground truth."
        )
    raw_ground_truth = settings.get("ground_truth_file")
    if not isinstance(raw_ground_truth, str) or not raw_ground_truth.strip():
        raise ValueError(
            f"Dataset {dataset_name!r} has no secondary_metrics.ground_truth_file."
        )
    ground_truth = Path(raw_ground_truth).expanduser()
    if not ground_truth.is_absolute():
        raw_root = dataset_entry.get("root_dir")
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ValueError(
                f"Dataset {dataset_name!r} needs root_dir for its relative secondary ground truth."
            )
        dataset_root = Path(raw_root).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
        ground_truth = dataset_root / ground_truth
    ground_truth = ground_truth.resolve()
    if not ground_truth.is_file():
        raise FileNotFoundError(f"Secondary ground truth not found: {ground_truth}")
    return ground_truth


def _collect_secondary_rerun_scores(
    outdir: Path,
    dataset_name: str,
    ground_truth_path: str | Path,
    *,
    num_steps: int,
) -> tuple[list[int], list[float]]:
    """Collect targets and scores for predicting an incorrect initial class.

    Target 1 means that the initial secondary prediction is wrong and therefore
    needs a rerun.  The score is calculated from the same global bottom-N steps
    that will actually be rerun later.
    """
    dataset_output_dir = outdir / dataset_name
    summary_path = dataset_output_dir / "all_data_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"First-run summary not found: {summary_path}")
    profile = load_profile_snapshot(dataset_output_dir)
    if profile is None:
        raise ValueError("Global secondary threshold selection requires a saved secondary profile.")

    raw_ground_truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    if not isinstance(raw_ground_truth, dict):
        raise ValueError("Secondary ground truth must be a JSON object keyed by image stem.")
    ground_truth = {str(key): str(value) for key, value in raw_ground_truth.items()}
    unknown = sorted(set(ground_truth.values()) - set(profile.class_names))
    if unknown:
        raise ValueError(f"Secondary ground truth contains unknown classes: {unknown}")

    y_true: list[int] = []
    y_score: list[float] = []
    seen_stems: set[str] = set()
    for result in _read_jsonl(summary_path):
        stem = Path(_img_key(result)).stem
        if stem in seen_stems:
            raise ValueError(
                f"Secondary ground truth is stem-keyed, but image stem {stem!r} is duplicated."
            )
        seen_stems.add(stem)
        if stem not in ground_truth:
            raise ValueError(f"Secondary ground truth lacks image id {stem!r}.")
        predicted_class = result.get("secondary_class")
        if predicted_class not in profile.class_names:
            raise ValueError(f"Summary image {stem!r} lacks a valid secondary prediction.")
        probs_path = result.get("probs_data_filepath")
        if not probs_path:
            raise ValueError(f"Summary image {stem!r} lacks probs_data_filepath.")
        confidence_scores, _ = read_confidence_scores(
            _resolve_probs_path(probs_path, dataset_output_dir)
        )
        score, _ = global_secondary_rerun_score(
            profile, confidence_scores, num_steps=num_steps
        )
        y_true.append(int(predicted_class != ground_truth[stem]))
        y_score.append(score)

    if not y_true:
        raise ValueError(f"No usable entries found in {summary_path}.")
    if len(set(y_true)) < 2:
        raise ValueError(
            "Secondary rerun threshold selection requires both correct and incorrect "
            "initial secondary predictions."
        )
    return y_true, y_score


def _candidate_for_threshold(threshold: float, y_true: list[int], y_score: list[float]) -> ThresholdCandidate:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    return ThresholdCandidate(
        threshold=float(threshold),
        false_positive_false_negative=false_positive + false_negative,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _select_candidate(candidates: list[ThresholdCandidate]) -> ThresholdCandidate:
    """Choose lowest FP+FN, then highest F1, then highest accuracy."""
    return min(
        candidates,
        key=lambda candidate: (
            candidate.false_positive_false_negative,
            -candidate.f1,
            -candidate.accuracy,
        ),
    )

def _save_threshold_plot(plot_path: str | Path, title: str, fpr: np.ndarray, tpr: np.ndarray, best: ThresholdCandidate, image_auroc: float,) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5.5), dpi=200)
    plt.plot(fpr, tpr, label=f"AUC={image_auroc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#ff7f0e", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(
        loc="lower right",
        title=(
            f"threshold={best.threshold:.4f}\n"
            f"FP+FN={best.false_positive_false_negative}, F1={best.f1:.4f}, acc={best.accuracy:.4f}"
        ),
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def calculate_optimum_threshold(
    outdir: str | Path,
    dataset_name: str,
    plot_path: str | Path | None = None,
    *,
    policy: str = "binary-gate",
    secondary_ground_truth_path: str | Path | None = None,
    num_steps: int = 3,
    results_subdir: str = RERUN_RESULTS_SUBDIR,
) -> float:
    """Calculate, print, plot, and return the best threshold for one run.

    ``outdir`` must contain ``<dataset_name>/all_data_summary.jsonl``. The plot
    is saved under ``<outdir>/<dataset_name>/rerun_results/`` by default, and
    the returned threshold can be passed straight into
    `rerun_low_confidence_predictions()`.
    """
    outdir = Path(outdir).expanduser().resolve() #Gets the full path of the output directory (the directory of the first run)
    if policy == "global-secondary":
        if secondary_ground_truth_path is None:
            raise ValueError("secondary_ground_truth_path is required for global-secondary.")
        y_true, y_score = _collect_secondary_rerun_scores(
            outdir, dataset_name, secondary_ground_truth_path, num_steps=num_steps
        )
        title = f"{dataset_name} | secondary rerun-need threshold"
        default_plot_name = f"{dataset_name}_secondary_rerun_threshold_roc.png"
    elif policy in {"binary-gate", "class-aware"}:
        y_true, y_score = _collect_scores(outdir, dataset_name)
        title = f"{dataset_name} | binary threshold selection"
        default_plot_name = f"{dataset_name}_threshold_selection_roc.png"
    else:
        raise ValueError(f"Unknown threshold policy: {policy!r}")
    image_auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    candidates = [_candidate_for_threshold(float(threshold), y_true, y_score) for threshold in thresholds]
    best = _select_candidate(candidates)

    logger.info("Threshold-selection AUROC (%s): %.6f", policy, image_auroc)
    logger.info(
        "Optimum threshold: %.6f | FP+FN=%d | F1=%.6f | accuracy=%.6f | FP=%d | FN=%d",
        best.threshold,
        best.false_positive_false_negative,
        best.f1,
        best.accuracy,
        best.false_positive,
        best.false_negative,
    )

    if plot_path is None:
        plot_path = outdir / dataset_name / results_subdir / default_plot_name
    _save_threshold_plot(plot_path, title, fpr, tpr, best, image_auroc)
    return best.threshold


# ---------------------------------------------------------------------
# VLM rerun of the low-confidence steps for borderline "good" predictions
# ---------------------------------------------------------------------

RERUN_FORMAT_MAX_ATTEMPTS = 2


from .rerun_stage.inference import (
    extract_rerun_confidence as _extract_rerun_confidence,
    load_planner_steps as _load_planner_steps,
    secondary_probabilities as _secondary_probabilities,
)


def _rerun_details_text(value: Any) -> str:
    """Convert structured model details into stable localization prose."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, nested in value.items():
            text = _rerun_details_text(nested)
            if text:
                label = str(key).replace("_", " ").strip()
                parts.append(f"{label}: {text}" if label else text)
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        return "; ".join(
            text for item in value if (text := _rerun_details_text(item))
        )
    return ""


def _parse_rerun_response(final_text: str, probs_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the strict rerun JSON and its decision-token confidence."""
    out = json_repair.loads(final_text)
    if not isinstance(out, dict):
        raise ValueError(f"Could not parse a JSON object from the rerun VLM output:\n{final_text}")
    decision = str(out.get("decision", "")).strip().lower()
    if decision not in {"yes", "no"}:
        raise ValueError("Rerun JSON decision must be Yes or No.")
    reason = out.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Rerun JSON reason must be a non-empty string.")
    out["reason"] = reason.strip()
    details = _rerun_details_text(out.get("details"))
    if not details:
        if decision == "no":
            # A negative step contributes no localization evidence.  Preserve
            # its explicit justification as details rather than failing an
            # hours-long resumable stage over an empty redundant field.
            details = out["reason"]
        else:
            raise ValueError(
                "Rerun JSON details must contain visible evidence for a Yes decision."
            )
    out["details"] = details
    expected_token = decision.upper()
    matching = [
        row for row in probs_data
        if str(row.get("token", "")).strip().upper() == expected_token
    ]
    if len(probs_data) != 1 or len(matching) != 1:
        raise ValueError(
            "Rerun confidence must contain exactly the JSON decision token; "
            f"decision={expected_token}, rows={probs_data!r}."
        )
    out["decision"] = "Yes" if decision == "yes" else "No"
    return out


def individual_step_rerun(
    modelcfg: ModelsConfig,
    prompts_dir: Path,
    vlmModel: type[vlm],
    ref_img: Image.Image,
    cand_img: Image.Image,
    step_text: str,
    *,
    telemetry_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-asks a single planner step to the VLM and returns its decision/reason/details."""
    prompt = build_prompt(prompts_dir, "rerun_step.j2", question=step_text)
    # require_step_context=False: rerun_step.j2's strict-JSON prompt never emits the
    # word "STEP" near its "Yes"/"No" token (unlike decision_maker.j2), so the
    # default STEP-context filter would always find zero probs_data here.
    event_details = dict(telemetry_context or {})
    event_details.update(image_count=2, model=modelcfg.vlm.modelname)
    record_pipeline_event("rerun_vlm_generate_start", **event_details)
    ablation_profile = os.environ.get("ANOMALY_ABLATION_PROFILE") == "1"
    vlm_call_time_sec = 0.0
    inference_time_sec = 0.0
    generated_token_count = 0
    timing_basis = "model_generate"
    attempt_prompt = prompt
    for attempt in range(1, RERUN_FORMAT_MAX_ATTEMPTS + 1):
        call_start = time.perf_counter()
        try:
            final_text, probs_data, generation_stats = vlmModel.run_vlm(
                modelcfg, [ref_img, cand_img], attempt_prompt,
                require_step_context=False,
            )
        except BaseException as exc:
            record_pipeline_event(
                "rerun_vlm_generate_python_failure",
                **event_details,
                exception_type=type(exc).__name__,
                elapsed_sec=round(time.perf_counter() - call_start, 6),
                attempt=attempt,
            )
            raise
        attempt_call_time = time.perf_counter() - call_start
        vlm_call_time_sec += attempt_call_time
        if "generation_time_sec" in generation_stats:
            inference_time_sec += float(generation_stats["generation_time_sec"])
        else:
            inference_time_sec += attempt_call_time
            timing_basis = "vlm_call_fallback"
        if ablation_profile:
            generated_token_count += int(generation_stats["generated_token_count"])
        try:
            out = _parse_rerun_response(final_text, probs_data)
            break
        except ValueError as exc:
            if attempt == RERUN_FORMAT_MAX_ATTEMPTS:
                logger.error("Invalid rerun response after %d attempts:\n%s", attempt, final_text)
                raise
            logger.warning(
                "Rerun response failed validation on attempt %d/%d: %s; retrying once.",
                attempt,
                RERUN_FORMAT_MAX_ATTEMPTS,
                exc,
            )
            attempt_prompt = (
                prompt
                + "\n\nCORRECTION REQUIRED: Return exactly one JSON object with "
                + 'decision set to "Yes" or "No" and non-empty string values '
                + "for reason and details. Return no markdown or commentary.\n"
                + f"Validation error: {exc}\nPrevious invalid response:\n"
                + final_text[-1500:]
            )
    record_pipeline_event(
        "rerun_vlm_generate_end",
        **event_details,
        elapsed_sec=round(inference_time_sec, 6),
        vlm_call_time_sec=round(vlm_call_time_sec, 6),
        timing_basis=timing_basis,
        generation_attempts=attempt,
    )

    out["step"] = step_text
    out["is_anomaly"] = str(out.get("decision", "")).strip().lower() == "yes"
    out["probs_data"] = probs_data
    out["inference_time_sec"] = round(inference_time_sec, 4)
    out["vlm_call_time_sec"] = round(vlm_call_time_sec, 4)
    out["timing_basis"] = timing_basis
    if ablation_profile:
        out["generated_token_count"] = generated_token_count
        out["generated_tokens_per_second"] = round(
            out["generated_token_count"] / inference_time_sec, 4
        ) if inference_time_sec > 0 else None
        logger.info(
            "Rerun VLM step: inference_time=%.3fs generated_tokens=%d "
            "tokens_per_second=%.2f",
            inference_time_sec,
            out["generated_token_count"],
            out["generated_tokens_per_second"] or 0.0,
        )
    return out


def _rerun_image(
    modelcfg: ModelsConfig, prompts_dir: Path, vlmModel: type[vlm], ref_img: Image.Image, cand_img: Image.Image,
    bottom_n_scores: list[tuple[int, float]], planner_steps: list[str], confidence_scores: dict[int, float],
    telemetry_context: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """
    Re-asks each of `bottom_n_scores`' steps in ascending-confidence order; stops at the first "Yes".

    Also recomputes the image's continuous per-image score (see scoring.py) using
    `confidence_scores` with every reran step's confidence overwritten by its rerun
    result, so the returned score reflects the rerun exactly like `lowest_log_prob_score()`
    would from a fresh *_prob_data.txt.
    """
    step_results: list[dict[str, Any]] = []
    outcome = "good"
    updated_scores = dict(confidence_scores)
    yes_steps: list[int] = []

    for step_number, confidence in bottom_n_scores:
        step_text = planner_steps[step_number - 1]
        result = individual_step_rerun(
            modelcfg,
            prompts_dir,
            vlmModel,
            ref_img,
            cand_img,
            step_text,
            telemetry_context={
                **(telemetry_context or {}),
                "step_number": step_number,
            },
        )
        result["step_number"] = step_number
        result["original_confidence"] = confidence
        step_results.append(result)

        updated_scores[step_number] = _extract_rerun_confidence(result, fallback=confidence)

        if result["is_anomaly"]:
            outcome = "anomaly"
            yes_steps.append(step_number)
            break

    score_info = score_details(updated_scores, yes_steps)
    return outcome, step_results, score_info


def _apply_binary_rerun_outcome(
    result: dict[str, Any], outcome: str, score: float,
) -> None:
    """Persist the binary rerun decision before status classification.

    Keeping this conversion in one small helper prevents the VLM outcome from
    being calculated successfully and then silently lost when the surrounding
    result dictionary still contains only the original prediction.
    """
    if outcome not in {"good", "anomaly"}:
        raise ValueError(f"Unexpected binary rerun outcome: {outcome!r}")
    result.update(
        binary_prediction=int(outcome == "anomaly"),
        label=outcome,
        score=float(score),
    )


def _rerun_image_secondary(
    modelcfg: ModelsConfig,
    prompts_dir: Path,
    vlmModel: type[vlm],
    ref_img: Image.Image,
    cand_img: Image.Image,
    bottom_n_scores: list[tuple[int, float]],
    planner_steps: list[str],
    confidence_scores: dict[int, float],
    original_yes_steps: list[int],
    profile: SecondaryClassificationProfile,
    telemetry_context: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, float], float]:
    """Rerun every selected step, then derive both classes from the final step set."""
    step_results: list[dict[str, Any]] = []
    updated_scores = dict(confidence_scores)
    effective_yes = set(original_yes_steps)

    for step_number, confidence in bottom_n_scores:
        if step_number < 1 or step_number > len(planner_steps):
            raise ValueError(f"Planner does not contain configured step {step_number}.")
        result = individual_step_rerun(
            modelcfg,
            prompts_dir,
            vlmModel,
            ref_img,
            cand_img,
            planner_steps[step_number - 1],
            telemetry_context={**(telemetry_context or {}), "step_number": step_number},
        )
        result["step_number"] = step_number
        result["original_confidence"] = confidence
        step_results.append(result)
        updated_scores[step_number] = _extract_rerun_confidence(result, fallback=confidence)
        if result["is_anomaly"]:
            effective_yes.add(step_number)
        else:
            effective_yes.discard(step_number)

    outcome = classify_yes_steps(profile, effective_yes)
    probabilities = _secondary_probabilities(updated_scores, effective_yes)
    class_scores = score_classes(profile, probabilities)
    return outcome, step_results, class_scores, binary_anomaly_score(profile, class_scores)


def rerun_low_confidence_predictions(
    outdir: str | Path,
    dataset_name: str,
    threshold: float,
    model_config_path: str | Path | None = None,
    prompts_dir: str | Path | None = None,
    num_steps: int = 3,
    console: Console | None = None,
    overwrite: bool = False,
    secondary_policy: str = "binary-gate",
    secondary_confidence_threshold: float = 90.0,
    binary_threshold: float | None = None,
    secondary_ground_truth_path: str | Path | None = None,
    results_subdir: str = RERUN_RESULTS_SUBDIR,
) -> Path:
    """
    Reruns bottom-N low-confidence planner steps according to
    ``secondary_policy`` (see the module docstring for the full decision
    rules), and writes
    <outdir>/<dataset_name>/rerun_results/rerun_result.jsonl.

    Per-image rerun step details are saved to
    <outdir>/<dataset_name>/rerun_results/texts/<label_folder>/[<source_label_folder>/]<img_name>_rerun_steps.json

    Resumability
    ------------
    rerun_result.jsonl is written INCREMENTALLY -- one line per processed
    image, flushed to disk immediately -- instead of only at the very end.
    This means a crash/segfault mid-run (e.g. the known sdpa VLM crash) loses
    at most the single image that was in flight, never the whole rerun.

    On the next invocation against the same outdir, any previously written
    rerun_result.jsonl is re-validated entry-by-entry (see
    `_validate_resumed_entry()` -- checks required fields, cross-checks
    label/binary_prediction consistency, and for reran images verifies the
    referenced *_rerun_steps.json actually exists/parses/agrees with the
    recorded outcome) before being reused; anything that doesn't fully check
    out is silently reprocessed instead of trusting a possibly
    half-written/stale file. A settings fingerprint (`rerun_meta.json`:
    threshold, num_steps, VLM model/source, prompts_dir) is also compared --
    if it changed since the prior attempt, this raises instead of silently
    mixing rerun results computed under different settings; pass
    ``overwrite=True`` to discard prior rerun state and start fresh.
    """
    outdir = Path(outdir).expanduser().resolve()
    dataset_output_dir = outdir / dataset_name
    summary_path = dataset_output_dir / "all_data_summary.jsonl"
    entries = _read_jsonl(summary_path)
    if not entries:
        raise ValueError(f"No usable entries found in {summary_path}")

    stage_start = time.perf_counter()
    project_root = Path(__file__).resolve().parent.parent
    model_config_path = Path(model_config_path) if model_config_path else project_root / "configs" / "model_configs.yaml"
    prompts_dir = Path(prompts_dir) if prompts_dir else project_root / "prompts" / "anomaly_detection_pipeline"

    modelcfg = load_models_config(model_config_path)
    secondary_profile = load_profile_snapshot(dataset_output_dir)

    if secondary_policy not in {"binary-gate", "class-aware", "global-secondary"}:
        raise ValueError(
            "secondary_policy must be 'binary-gate', 'class-aware', or 'global-secondary'."
        )
    if secondary_policy == "global-secondary" and secondary_profile is None:
        raise ValueError("global-secondary requires a saved secondary-classification profile.")
    if secondary_policy == "global-secondary" and binary_threshold is None:
        raise ValueError("global-secondary requires an independently calibrated binary_threshold.")
    if not 0.0 <= secondary_confidence_threshold <= 100.0:
        raise ValueError("secondary_confidence_threshold must be between 0 and 100.")
    results_subdir_path = Path(results_subdir)
    if results_subdir_path.is_absolute() or ".." in results_subdir_path.parts:
        raise ValueError("results_subdir must stay inside the dataset output directory.")

    results_dir = dataset_output_dir / results_subdir_path
    texts_dir = results_dir / "texts"
    meta_path = results_dir / "rerun_meta.json"
    rerun_result_path = results_dir / "rerun_result.jsonl"

    current_meta = _current_rerun_meta(
        threshold, num_steps, modelcfg, prompts_dir, secondary_profile,
        secondary_policy, secondary_confidence_threshold, binary_threshold,
    )
    resumed_map = _load_resumable_rerun_state(results_dir, dataset_output_dir, current_meta, overwrite)

    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(meta_path, json.dumps(current_meta, indent=2))

    # Lazily loaded on first ACTUAL rerun below -- a fully-resumed dataset
    # (every image already validated from a prior attempt) never needs the
    # VLM at all.
    vlmModel: type[vlm] | None = None

    summary: list[dict[str, Any]] = []
    rerun_step_times: list[float] = []
    rerun_step_tokens: list[int] = []
    num_images_rerun = 0
    num_images_resumed = 0
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", table_column=Column(ratio=1)),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console or Console(),
        expand=True,
    )
    progress.start()
    task = progress.add_task(f"[dim]{dataset_name}: starting rerun…", total=len(entries))
    try:
        with rerun_result_path.open("w", encoding="utf-8") as result_file:

            def _emit(result: dict[str, Any]) -> None:
                summary.append(result)
                result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                result_file.flush()
                # A rerun can take hours and may write through SSHFS.  Make
                # each completed record durable before proceeding so a
                # writeback failure is raised by the child instead of being
                # reported as a successful, resumable stage.
                os.fsync(result_file.fileno())

            for i, entry in enumerate(entries, start=1):
                img_name = _img_key(entry)
                label_folder = entry["label_folder"]
                source_label_folder = str(entry.get("source_label_folder") or label_folder)
                entry_key = (label_folder, source_label_folder, img_name)

                resumed_entry = resumed_map.get(entry_key)
                if resumed_entry is not None:
                    num_images_resumed += 1
                    logger.info(
                        "[%d/%d] %s/%s: resuming previously-computed rerun result (status=%s)",
                        i, len(entries), label_folder, img_name, resumed_entry.get("rerun_status"),
                    )
                    progress.update(
                        task, advance=1,
                        description=f"[dim]{dataset_name}: {label_folder}/{img_name} → resumed ({resumed_entry.get('rerun_status')})",
                    )
                    _emit(resumed_entry)
                    continue

                result: dict[str, Any] = {
                    "dataset": entry.get("dataset", dataset_name),
                    "label_folder": label_folder,
                    "source_label_folder": source_label_folder,
                    "img_name": img_name,
                    "ground_truth": entry["gt_binary"],
                }
                if secondary_profile is not None:
                    for field in (
                        "yes_steps", "secondary_class", "secondary_class_scores",
                        "secondary_binary_score",
                    ):
                        if field not in entry:
                            raise ValueError(
                                f"Secondary summary entry {entry_key!r} lacks {field!r}."
                            )
                        result[field] = entry[field]
                progress.update(task, description=f"[dim]{dataset_name}: {label_folder}/{img_name} — checking eligibility…")

                probs_path = _resolve_probs_path(entry["probs_data_filepath"], dataset_output_dir)
                #result["probs_data_filepath"] = str(probs_path)
                bottom_n_scores, details, confidence_scores = lowest_confidence_steps(probs_path, manual_threshold=False, num_steps=num_steps)

                if details is None:
                    result.update(binary_prediction=entry["binary_prediction"], label=entry["label"], rerun_status="no_confidence_data", score=None)
                    logger.info("[%d/%d] %s/%s: no confidence data — keeping original prediction", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[dim]{dataset_name}: {label_folder}/{img_name} → no confidence data")
                    _emit(result)
                    continue

                eligibility_score = (
                    float(entry["secondary_binary_score"])
                    if secondary_profile is not None
                    else float(details["score"])
                )
                rerun_lanes = {"secondary_lane": False, "binary_lane": False}
                if secondary_profile is not None and secondary_policy == "class-aware":
                    bottom_n_scores, rerun_lanes = class_aware_rerun_steps(
                        secondary_profile,
                        entry["yes_steps"],
                        confidence_scores,
                        binary_score=eligibility_score,
                        binary_threshold=threshold,
                        secondary_confidence_threshold=secondary_confidence_threshold,
                        num_steps=num_steps,
                    )
                    result["rerun_lanes"] = rerun_lanes
                elif secondary_profile is not None and secondary_policy == "global-secondary":
                    eligibility_score, bottom_n_scores = global_secondary_rerun_score(
                        secondary_profile, confidence_scores, num_steps=num_steps
                    )
                    result["rerun_eligibility_score"] = round(eligibility_score, 8)

                if secondary_policy == "global-secondary" and eligibility_score < threshold:
                    result.update(
                        binary_prediction=entry["binary_prediction"],
                        label=entry["label"],
                        rerun_status="not_rerun_below_secondary_threshold",
                        score=float(entry["secondary_binary_score"]),
                    )
                    logger.info(
                        "[%d/%d] %s/%s: secondary uncertainty %.4f < threshold %.4f — no rerun",
                        i, len(entries), label_folder, img_name, eligibility_score, threshold,
                    )
                    progress.update(
                        task, advance=1,
                        description=f"[dim]{dataset_name}: {label_folder}/{img_name} → below secondary threshold",
                    )
                    _emit(result)
                    continue

                if secondary_policy == "binary-gate" and eligibility_score < threshold:
                    result.update(binary_prediction=entry["binary_prediction"], label=entry["label"], rerun_status="not_rerun_below_threshold", score=eligibility_score)
                    logger.info("[%d/%d] %s/%s: score %.4f < threshold %.4f — no rerun", i, len(entries), label_folder, img_name, eligibility_score, threshold)
                    progress.update(task, advance=1, description=f"[dim]{dataset_name}: {label_folder}/{img_name} → below threshold, skipped")
                    _emit(result)
                    continue

                if secondary_policy == "binary-gate" and entry["label"] != "good":
                    result.update(binary_prediction=entry["binary_prediction"], label=entry["label"], rerun_status="not_rerun_already_anomaly", score=eligibility_score)
                    logger.info("[%d/%d] %s/%s: already predicted anomaly — no rerun needed", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[dim]{dataset_name}: {label_folder}/{img_name} → already anomaly, skipped")
                    _emit(result)
                    continue

                if (
                    secondary_profile is not None
                    and secondary_policy == "class-aware"
                    and not any(rerun_lanes.values())
                ):
                    result.update(
                        binary_prediction=entry["binary_prediction"],
                        label=entry["label"],
                        rerun_status="not_rerun_no_class_uncertainty",
                        score=eligibility_score,
                    )
                    logger.info(
                        "[%d/%d] %s/%s: no class-aware rerun lane triggered — keeping original prediction",
                        i, len(entries), label_folder, img_name,
                    )
                    progress.update(
                        task, advance=1,
                        description=f"[dim]{dataset_name}: {label_folder}/{img_name} → class-confident, skipped",
                    )
                    _emit(result)
                    continue

                if not bottom_n_scores:
                    result.update(binary_prediction=entry["binary_prediction"], label=entry["label"], rerun_status="not_rerun_no_low_confidence_steps", score=eligibility_score)
                    logger.info("[%d/%d] %s/%s: score >= threshold but no low-confidence steps to rerun", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[dim]{dataset_name}: {label_folder}/{img_name} → no low-confidence steps")
                    _emit(result)
                    continue

                logger.info("[%d/%d] %s/%s: rerunning steps %s", i, len(entries), label_folder, img_name, [s for s, _ in bottom_n_scores])
                progress.update(task, description=f"[bold yellow]{dataset_name}: {label_folder}/{img_name} — rerunning steps {[s for s, _ in bottom_n_scores]}…")
                ref_img = Image.open(
                    _resolve_dataset_artifact(
                        entry["preprocess_ref_img_path"], dataset_output_dir, project_root
                    )
                ).convert("RGB")
                cand_img = Image.open(
                    _resolve_dataset_artifact(
                        entry["preprocess_cand_img_path"], dataset_output_dir, project_root
                    )
                ).convert("RGB")
                planner_steps = _load_planner_steps(entry["planner_filepath"])

                if vlmModel is None:
                    vlmModel = vlm.load(modelcfg)
                telemetry_context = {
                        "phase": "rerun",
                        "dataset": dataset_name,
                        "label_folder": label_folder,
                        "source_label_folder": source_label_folder,
                        "img_name": img_name,
                        "image_identity": "/".join(
                            (dataset_name, label_folder, source_label_folder, img_name)
                        ),
                    }
                if secondary_profile is not None:
                    outcome, step_results, class_scores, secondary_binary_score = _rerun_image_secondary(
                        modelcfg, prompts_dir, vlmModel, ref_img, cand_img,
                        bottom_n_scores, planner_steps, confidence_scores,
                        entry["yes_steps"], secondary_profile,
                        telemetry_context=telemetry_context,
                    )
                    result.update(
                        yes_steps=outcome.yes_steps,
                        secondary_class=outcome.secondary_class,
                        secondary_class_scores={
                            name: round(score, 8) for name, score in class_scores.items()
                        },
                        secondary_binary_score=round(secondary_binary_score, 8),
                        binary_prediction=outcome.binary_prediction,
                        label=outcome.binary_label,
                        score=round(secondary_binary_score, 8),
                    )
                    rejected_binary_change = (
                        (
                            secondary_policy == "class-aware"
                            and rerun_lanes["secondary_lane"]
                            and not rerun_lanes["binary_lane"]
                            and outcome.binary_label != entry["label"]
                        )
                        or (
                            secondary_policy == "global-secondary"
                            and outcome.binary_label != entry["label"]
                            and not (
                                (
                                    outcome.binary_prediction == 1
                                    and secondary_binary_score >= float(binary_threshold)
                                )
                                or (
                                    outcome.binary_prediction == 0
                                    and secondary_binary_score < float(binary_threshold)
                                )
                            )
                        )
                    )
                    if rejected_binary_change:
                        for field in (
                            "yes_steps", "secondary_class", "secondary_class_scores",
                            "secondary_binary_score", "binary_prediction", "label",
                        ):
                            result[field] = entry[field]
                        result["score"] = float(entry["secondary_binary_score"])
                else:
                    rejected_binary_change = False
                    outcome, step_results, score_info = _rerun_image(
                        modelcfg, prompts_dir, vlmModel, ref_img, cand_img,
                        bottom_n_scores, planner_steps, confidence_scores,
                        telemetry_context=telemetry_context,
                    )
                    _apply_binary_rerun_outcome(
                        result,
                        outcome,
                        score_info["score"] if score_info is not None else details["score"],
                    )
                num_images_rerun += 1
                rerun_step_times.extend(step["inference_time_sec"] for step in step_results if "inference_time_sec" in step)
                rerun_step_tokens.extend(
                    int(step["generated_token_count"])
                    for step in step_results
                    if step.get("generated_token_count") is not None
                )

                # Flat for non-merged datasets; source-nested only for merged
                # groups. This is the same canonical rule as inference/masks.
                step_results_path = artifact_group_dir(
                    texts_dir, label_folder, source_label_folder
                ) / f"{img_name}_rerun_steps.json"
                step_results_path.parent.mkdir(parents=True, exist_ok=True)
                with step_results_path.open("w", encoding="utf-8") as f:
                    json.dump(step_results, f, indent=2, ensure_ascii=False)
                result["rerun_steps_filepath"] = str(step_results_path)

                original_prediction = int(entry["binary_prediction"])
                final_prediction = int(result.get("binary_prediction", original_prediction))
                if rejected_binary_change:
                    result["rerun_status"] = "rerun_rejected_binary_change"
                    logger.info("[%d/%d] %s/%s: rejected unapproved binary-label change", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[yellow]{dataset_name}: {label_folder}/{img_name} → binary change rejected")
                elif original_prediction == 0 and final_prediction == 1:
                    result.update(binary_prediction=1, label="anomaly", rerun_status="flipped_to_anomaly")
                    logger.info("[%d/%d] %s/%s: flipped to anomaly after rerun", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[bold red]{dataset_name}: {label_folder}/{img_name} → flipped to anomaly")
                elif original_prediction == 1 and final_prediction == 0:
                    result.update(binary_prediction=0, label="good", rerun_status="flipped_to_good")
                    logger.info("[%d/%d] %s/%s: flipped to good after rerun", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[bold cyan]{dataset_name}: {label_folder}/{img_name} → flipped to good")
                else:
                    if secondary_profile is None:
                        result.update(binary_prediction=entry["binary_prediction"], label=entry["label"])
                    result["rerun_status"] = (
                        "secondary_class_changed"
                        if secondary_profile is not None
                        and result["secondary_class"] != entry["secondary_class"]
                        else (
                            "rerun_confirmed_anomaly"
                            if final_prediction == 1
                            else "rerun_confirmed_good"
                        )
                    )
                    logger.info("[%d/%d] %s/%s: confirmed/reclassified after rerun", i, len(entries), label_folder, img_name)
                    progress.update(task, advance=1, description=f"[bold green]{dataset_name}: {label_folder}/{img_name} → rerun complete")
                _emit(result)
    finally:
        progress.stop()
        if vlmModel is not None:
            vlm.release()

    if num_images_resumed:
        logger.info("Rerun resume: reused %d/%d already-processed image(s); freshly processed %d.", num_images_resumed, len(entries), len(entries) - num_images_resumed)

    total_wall_time_sec = time.perf_counter() - stage_start
    total_step_inference_time_sec = sum(rerun_step_times)
    total_generated_tokens = sum(rerun_step_tokens)
    performance: dict[str, Any] = {
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"vlm": asdict(modelcfg.vlm)},
        "threshold": threshold,
        "num_steps": num_steps,
        "secondary_policy": secondary_policy,
        "secondary_confidence_threshold": secondary_confidence_threshold,
        "binary_threshold": binary_threshold,
        "global_secondary_score_version": (
            GLOBAL_SECONDARY_SCORE_VERSION
            if secondary_policy == "global-secondary" else None
        ),
        "num_entries_checked": len(entries),
        "num_images_rerun": num_images_rerun,
        "num_rerun_steps": len(rerun_step_times),
        "total_wall_time_sec": round(total_wall_time_sec, 4),
        "total_rerun_step_inference_time_sec": round(total_step_inference_time_sec, 4),
        "avg_rerun_step_inference_time_sec": (
            round(total_step_inference_time_sec / len(rerun_step_times), 4) if rerun_step_times else None
        ),
        "timing_basis": "model_generate",
    }
    if secondary_policy == "global-secondary" and secondary_ground_truth_path is not None:
        raw_ground_truth = json.loads(
            Path(secondary_ground_truth_path).read_text(encoding="utf-8")
        )
        ground_truth = {str(key): str(value) for key, value in raw_ground_truth.items()}
        coverage = {
            "initial_errors_total": 0,
            "initial_errors_selected": 0,
            "initial_errors_not_selected": 0,
            "initial_correct_selected": 0,
            "final_errors_after_selected_rerun": 0,
            "final_errors_not_selected": 0,
            "selected_errors_corrected": 0,
            "selected_correct_predictions_harmed": 0,
        }
        for entry, result in zip(entries, summary):
            actual = ground_truth[Path(_img_key(entry)).stem]
            initial_wrong = entry["secondary_class"] != actual
            final_wrong = result["secondary_class"] != actual
            selected = bool(result.get("rerun_steps_filepath"))
            coverage["initial_errors_total"] += int(initial_wrong)
            coverage["initial_errors_selected"] += int(initial_wrong and selected)
            coverage["initial_errors_not_selected"] += int(initial_wrong and not selected)
            coverage["initial_correct_selected"] += int(not initial_wrong and selected)
            coverage["final_errors_after_selected_rerun"] += int(final_wrong and selected)
            coverage["final_errors_not_selected"] += int(final_wrong and not selected)
            coverage["selected_errors_corrected"] += int(
                selected and initial_wrong and not final_wrong
            )
            coverage["selected_correct_predictions_harmed"] += int(
                selected and not initial_wrong and final_wrong
            )
        performance["secondary_rerun_coverage"] = coverage
    if os.environ.get("ANOMALY_ABLATION_PROFILE") == "1":
        performance.update(
            total_generated_tokens=total_generated_tokens,
            avg_generated_tokens_per_rerun_step=(
                round(total_generated_tokens / len(rerun_step_tokens), 4)
                if rerun_step_tokens else None
            ),
            generated_tokens_per_second=(
                round(total_generated_tokens / total_step_inference_time_sec, 4)
                if total_step_inference_time_sec > 0 else None
            ),
        )
    performance_path = results_dir / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    atomic_write_text(performance_path, json.dumps(performance, indent=2, default=str))
    logger.info("[%s] Rerun performance summary written to %s", dataset_name, performance_path)

    logger.info("Rerun complete: %d images processed, results saved to %s", len(summary), rerun_result_path)
    return rerun_result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the rerun threshold for one completed run, then rerun its low-confidence 'good' predictions."
    )
    parser.add_argument("outdir", nargs="?", help="First-run output directory containing the dataset folder")
    parser.add_argument("dataset_name", nargs="?", help="Dataset folder name inside outdir")
    parser.add_argument(
        "--outdir",
        dest="outdir_option",
        help="First-run output directory containing the dataset folder (alternative to the positional argument)",
    )
    parser.add_argument(
        "--dataset_name",
        "--dataset-name",
        dest="dataset_name_option",
        help="Dataset folder name inside outdir (alternative to the positional argument)",
    )
    parser.add_argument("--plot-path", default=None, help="Optional output path for the threshold ROC plot")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Skip automatic eligibility-threshold selection and use this value instead.",
    )
    parser.add_argument(
        "--binary-threshold", type=float, default=None,
        help=(
            "Binary confirmation threshold for global-secondary label flips. "
            "By default it is calibrated independently from the base binary scores."
        ),
    )
    parser.add_argument("--threshold-only", action="store_true", help="Only compute/print the threshold; skip the VLM rerun stage")
    parser.add_argument("--num-steps", type=int, default=3, help="Number of lowest-confidence planner steps to rerun per image")
    parser.add_argument(
        "--secondary-policy",
        choices=("binary-gate", "class-aware", "global-secondary"),
        default="binary-gate",
        help=(
            "Rerun eligibility policy. global-secondary calibrates one uncertainty "
            "threshold and reruns the globally lowest-confidence configured steps."
        ),
    )
    parser.add_argument(
        "--secondary-confidence-threshold",
        type=float,
        default=90.0,
        metavar="PERCENT",
        help="Class-aware lane reruns same-binary decision steps below this confidence.",
    )
    parser.add_argument(
        "--results-subdir",
        default=RERUN_RESULTS_SUBDIR,
        help="Dataset-relative result directory, useful for isolated rerun experiments.",
    )
    parser.add_argument(
        "--secondary-ground-truth",
        default=None,
        help="Secondary class-label JSON used to calibrate global-secondary rerun need.",
    )
    parser.add_argument(
        "--metrics-config",
        default=None,
        help=(
            "Metrics YAML from which global-secondary can resolve this dataset's "
            "secondary ground-truth JSON."
        ),
    )
    parser.add_argument("--model-config", default=None, help="Path to model_configs.yaml (default: configs/model_configs.yaml)")
    parser.add_argument("--prompts-dir", default=None, help="Path to the Jinja prompts directory (default: prompts/anomaly_detection_pipeline)")
    parser.add_argument(
        "--no-hardware-monitor",
        action="store_true",
        help="Disable local and cluster-node hardware telemetry for this rerun.",
    )
    parser.add_argument(
        "--hardware-monitor-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Hardware telemetry sampling interval.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Discard any previous (possibly partial/crashed) rerun_result.jsonl for this dataset and "
            "start the VLM rerun stage from scratch, instead of resuming/validating prior results."
        ),
    )
    args = parser.parse_args()

    args.outdir = args.outdir_option or args.outdir
    args.dataset_name = args.dataset_name_option or args.dataset_name
    if not args.outdir or not args.dataset_name:
        parser.error("outdir and dataset_name are required (use positional arguments or --outdir/--dataset_name)")

    resolved_outdir = Path(args.outdir).expanduser().resolve()
    interactive_terminal = sys.stdout.isatty()
    terminal_stream = open_terminal_stream() if interactive_terminal else None
    console = Console(file=terminal_stream) if terminal_stream is not None else Console()
    # Logs for this stage live inside the run directory being operated on
    # (<outdir>/logs/), timestamped and named after the dataset, since this
    # CLI is invoked once per dataset and may be re-run multiple times.
    # In an interactive terminal, reserve the console exclusively for Rich's
    # progress display. In non-interactive mode the ablation orchestrator
    # consumes the newline-based per-image INFO records from child stdout, so
    # retain the existing machine-readable stream there.
    log_file = setup_logging(
        f"rerun_{args.dataset_name}",
        log_dir=resolved_outdir / "logs",
        console=console,
        console_level=None if interactive_terminal else logging.INFO,
    )
    logger.info("Rerun stage: outdir=%s dataset=%s log_file=%s", resolved_outdir, args.dataset_name, log_file)

    output_capture = (
        capture_terminal_output(log_file)
        if interactive_terminal
        else contextlib.nullcontext()
    )
    try:
        with output_capture:
            model_config_path = Path(args.model_config).expanduser().resolve() if args.model_config else PROJECT_ROOT / "configs" / "model_configs.yaml"
            monitor = None
            if not args.no_hardware_monitor:
                telemetry_dir = resolved_outdir / "logs" / f'hardware_telemetry_rerun_{args.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                deployment = load_deployment_config(model_config_path)
                monitor = build_monitor(
                    HardwareMonitor(
                        output_dir=telemetry_dir,
                        interval=args.hardware_monitor_interval,
                        label=f"rerun_{args.dataset_name}",
                    ),
                    deployment,
                    interval=args.hardware_monitor_interval,
                    label=f"rerun_{args.dataset_name}",
                    context={"pipeline": "rerun", "dataset": args.dataset_name, "run_dir": str(resolved_outdir)},
                )
                if not monitor.start():
                    raise RuntimeError(f"Hardware telemetry failed to start: {telemetry_dir}")
                monitor.mark_event("pipeline_start", pipeline="rerun", dataset=args.dataset_name)

            with monitored(monitor), managed_server_lifecycle(f"rerun_{args.dataset_name}"):
                secondary_ground_truth = None
                if args.secondary_policy == "global-secondary":
                    secondary_ground_truth = resolve_secondary_ground_truth(
                        args.dataset_name,
                        explicit_path=args.secondary_ground_truth,
                        metrics_config_path=args.metrics_config,
                    )
                selected_threshold = (
                    args.threshold
                    if args.threshold is not None
                    else calculate_optimum_threshold(
                        args.outdir,
                        args.dataset_name,
                        args.plot_path,
                        policy=args.secondary_policy,
                        secondary_ground_truth_path=secondary_ground_truth,
                        num_steps=args.num_steps,
                        results_subdir=args.results_subdir,
                    )
                )
                selected_binary_threshold = args.binary_threshold
                if (
                    args.secondary_policy == "global-secondary"
                    and selected_binary_threshold is None
                ):
                    selected_binary_threshold = calculate_optimum_threshold(
                        args.outdir,
                        args.dataset_name,
                        policy="binary-gate",
                        num_steps=args.num_steps,
                        results_subdir=args.results_subdir,
                    )
                logger.info("Selected threshold for rerun for %s: %.6f", args.dataset_name, selected_threshold)
                if selected_binary_threshold is not None:
                    logger.info(
                        "Selected binary confirmation threshold for %s: %.6f",
                        args.dataset_name, selected_binary_threshold,
                    )
                if monitor:
                    monitor.mark_event("threshold_selected", dataset=args.dataset_name, threshold=selected_threshold)

                if not args.threshold_only:
                    rerun_low_confidence_predictions(
                        outdir=args.outdir,
                        dataset_name=args.dataset_name,
                        threshold=selected_threshold,
                        model_config_path=model_config_path,
                        prompts_dir=args.prompts_dir,
                        num_steps=args.num_steps,
                        console=console,
                        overwrite=args.overwrite,
                        secondary_policy=args.secondary_policy,
                        secondary_confidence_threshold=args.secondary_confidence_threshold,
                        binary_threshold=selected_binary_threshold,
                        secondary_ground_truth_path=secondary_ground_truth,
                        results_subdir=args.results_subdir,
                    )
    finally:
        if terminal_stream is not None:
            terminal_stream.close()
