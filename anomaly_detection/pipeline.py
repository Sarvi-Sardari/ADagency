"""
pipeline.py (anomaly_detection)
=================================
Per-dataset inference orchestration for the online anomaly-detection
pipeline: embeds reference images, finds each candidate's nearest reference
image, resolves the correct planner (single-class -> "good"; multi-class ->
CLIP text-matched class), runs the VLM decision maker, and writes
all_data_summary.jsonl.

Behavior change (2026-08): resume no longer trusts
``all_data_summary.jsonl`` merely because it exists. Before skipping a
completed dataset it validates exact expected image identities plus every
decision, probability, planner, and preprocessed-image artifact. Invalid
completed state raises without rewriting it. A genuine mid-dataset run has no
atomic final summary, so its valid per-image pairs are reused independently.

Metrics/ROC plots are intentionally NOT computed here — see metrics.py and
run_anomaly_detection.py for why that stage is separate and flag-gated.

Behavior change (2026-09): optional object cropping (``object_crop.py``) is
frozen per dataset run in ``object_crop_profile.json``. Resume rejects a run
whose crop settings changed, and completed-summary validation requires every
row to carry a valid ``object_crop`` record exactly when cropping is enabled.
"""

from __future__ import annotations
import os
import json
import logging
import math
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress

from .config import (
    AnomalyDetectionConfig,
    DatasetEntry,
    ModelsConfig,
    artifact_group_dir,
    resolve_portable_file,
)
from .models import vlm, Encoder
from .class_labeler import load_or_generate_class_labels
from .decision import parse_final_decision, resolve_decision_prompt_template, validate_yes_step_evidence
from .reference_selector import compute_ref_embeddings
from .scoring import validate_probability_file
from .utils import list_images, preprocess_image_for_embedding
from .inference_agent import inference_agent
from .object_crop import (
    ensure_object_crop_profile,
    load_object_crop_profile,
    object_crop_timing_summary,
    save_cropped_candidate_png,
    train_crop_manifest_warning,
    validate_summary_crop_record,
)
from .secondary_classification import (
    PROFILE_FILENAME,
    binary_anomaly_score,
    classify_yes_steps,
    load_profile_snapshot,
    profile_fingerprint,
    profile_snapshot,
)

logger = logging.getLogger(__name__)


def _ensure_secondary_profile(
    output_dataset_dir: Path,
    configured_profile: Any,
) -> None:
    """Freeze the profile for every run and reject unsafe resume changes."""
    snapshot_path = output_dataset_dir / PROFILE_FILENAME
    if configured_profile is None:
        if snapshot_path.exists():
            raise RuntimeError(
                f"Secondary classification is disabled, but this run already has "
                f"a saved profile at {snapshot_path}. Use a fresh run directory."
            )
        return

    if snapshot_path.exists():
        saved_profile = load_profile_snapshot(output_dataset_dir)
        if saved_profile is None or profile_fingerprint(saved_profile) != profile_fingerprint(configured_profile):
            raise RuntimeError(
                f"The configured secondary-classification profile differs from "
                f"the profile saved in {snapshot_path}. Use a fresh run directory."
            )
        return

    temporary_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(profile_snapshot(configured_profile), file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, snapshot_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()




def _discover_test_groups(dataset: DatasetEntry) -> dict[str, list[str]]:
    """
    Discover test folders from the dataset itself. The "good" folder is
    always kept separate; all other folders are either kept by name or
    combined into "anomalies" according to merge_anomalies flag.
    """
    test_dir = dataset.root_dir / "test"
    groups: dict[str, list[str]] = {}

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found for dataset '{dataset.name}': {test_dir}")

    anomaly_paths: list[str] = []
    for entry in sorted(test_dir.iterdir()):
        if not entry.is_dir():
            continue

        paths, _ = list_images(entry)
        if entry.name == "good":
            groups["good"] = paths          #Add the good images paths separately as "good" group
        elif dataset.merge_anomalies:
            anomaly_paths.extend(paths)     #Add all the anomaly images paths regardless of group or folder name
        else:
            groups[entry.name] = paths      #Add the images paths as per the group (folder name)

    if dataset.merge_anomalies:
        groups["anomalies"] = anomaly_paths    #Create a separate group called anomalies with all the non-good images

    return groups


def _load_completed_dataset_result(dataset: DatasetEntry, output_dataset_directory: Path, summary_filepath: Path) -> dict[str, Any]:
    """
    Rebuild `entire_dataset_run()`'s return dict for a dataset that was
    already fully processed (its all_data_summary.jsonl exists) in an
    earlier attempt at this same run_dir, without re-running any inference.
    Reuses the most recent existing performance_*.json if one was written;
    otherwise reconstructs a minimal one from the summary alone (no per-image
    timing is available since resumed images don't record it).
    """
    full_summary: list[dict[str, Any]] = []
    with summary_filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                full_summary.append(json.loads(line))

    results: dict[str, dict[str, list[int]]] = {}
    for entry in full_summary:
        group = results.setdefault(entry["label_folder"], {"y_true": [], "y_pred": []})
        group["y_true"].append(entry["gt_binary"])
        group["y_pred"].append(entry["binary_prediction"])

    performance_dir = output_dataset_directory / "performance"
    existing_performance_files = sorted(performance_dir.glob("performance_*.json")) if performance_dir.is_dir() else []
    if existing_performance_files:
        performance = json.loads(existing_performance_files[-1].read_text(encoding="utf-8"))
        if os.environ.get("ANOMALY_ABLATION_PROFILE") != "1":
            for field in (
                "num_images_with_token_stats", "total_generated_tokens",
                "avg_generated_tokens_per_image", "generated_tokens_per_second",
            ):
                performance.pop(field, None)
    else:
        timed_entries = [entry for entry in full_summary if entry.get("inference_time_sec")]
        inference_times = [float(entry["inference_time_sec"]) for entry in timed_entries]
        timing_bases = {
            str(entry.get("timing_basis", "legacy_vlm_call")) for entry in timed_entries
        }
        token_entries = [
            entry for entry in timed_entries
            if entry.get("generated_token_count") is not None
        ]
        total_generated_tokens = sum(
            int(entry["generated_token_count"]) for entry in token_entries
        )
        token_timing_sec = sum(
            float(entry["inference_time_sec"]) for entry in token_entries
        )
        performance: dict[str, Any] = {
            "dataset": dataset.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_images": len(full_summary),
            "num_images_with_vlm_timing": len(inference_times),
            "total_wall_time_sec": 0.0,
            "total_vlm_inference_time_sec": round(sum(inference_times), 4),
            "avg_vlm_inference_time_per_image_sec": (
                round(sum(inference_times) / len(inference_times), 4) if inference_times else None
            ),
            "timing_basis": (
                next(iter(timing_bases)) if len(timing_bases) == 1 else "mixed"
            ),
        }
        if os.environ.get("ANOMALY_ABLATION_PROFILE") == "1":
            performance.update(
                num_images_with_token_stats=len(token_entries),
                total_generated_tokens=total_generated_tokens,
                avg_generated_tokens_per_image=(
                    round(total_generated_tokens / len(token_entries), 4)
                    if token_entries else None
                ),
                generated_tokens_per_second=(
                    round(total_generated_tokens / token_timing_sec, 4)
                    if token_timing_sec > 0 else None
                ),
            )
        performance.update(object_crop_timing_summary(full_summary))

    return {
        "dataset_name": dataset.name,
        "output_dir": output_dataset_directory,
        "summary_filepath": summary_filepath,
        "performance": performance,
        "groups": results,
    }


def _write_summary_atomically(
    dataset: DatasetEntry,
    output_dataset_dir: Path,
    summary_filepath: Path,
    entries: list[dict[str, Any]],
    groups: dict[str, list[str]],
    label_policy: str = "model",
) -> None:
    """Validate a temporary summary and only then publish it as complete."""
    temporary_path = summary_filepath.with_suffix(summary_filepath.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for entry in entries:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        summary_valid, validation_error = _validate_completed_summary(
            dataset,
            output_dataset_dir,
            temporary_path,
            groups,
            label_policy=label_policy,
        )
        if not summary_valid:
            raise RuntimeError(
                f"[{dataset.name}] refusing to publish an incomplete summary: "
                f"{validation_error}"
            )
        os.replace(temporary_path, summary_filepath)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _resolve_summary_artifact(raw_path: Any, dataset_output_dir: Path) -> Path:
    """Resolve a summary artifact, including pre-migration timestamp names."""
    path = Path(str(raw_path)).expanduser()
    project_root = Path(__file__).resolve().parent.parent
    return resolve_portable_file(
        path,
        dataset_output_dir,
        project_root,
        Path.cwd(),
    )


def _valid_image_file(path: Path) -> bool:
    """Return whether an image artifact exists, is non-empty, and decodes."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _validate_completed_summary(
    dataset: DatasetEntry,
    output_dataset_dir: Path,
    summary_path: Path,
    groups: dict[str, list[str]],
    label_policy: str = "model",
) -> tuple[bool, str]:
    """Validate the completed-dataset contract without changing any files."""
    try:
        secondary_profile = load_profile_snapshot(output_dataset_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return False, f"saved secondary-classification profile is invalid: {error}"
    try:
        cropping_enabled = load_object_crop_profile(output_dataset_dir) is not None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return False, f"saved object-crop profile is invalid: {error}"
    expected = {
        (group_name, Path(image_path).parent.name, Path(image_path).name)
        for group_name, image_paths in groups.items()
        for image_path in image_paths
    }
    entries: list[dict[str, Any]] = []
    try:
        for line_number, raw_line in enumerate(
            summary_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            entry = json.loads(raw_line)
            if not isinstance(entry, dict):
                return False, f"summary line {line_number} is not a JSON object"
            entries.append(entry)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return False, f"summary is unreadable or malformed: {error}"
    if not entries:
        return False, "summary contains no image records"

    required = {
        "dataset", "label_folder", "source_label_folder", "img_name",
        "gt_binary", "binary_prediction", "label", "decision_filepath",
        "probs_data_filepath", "planner_filepath",
        "preprocess_ref_img_path", "preprocess_cand_img_path",
    }
    actual: set[tuple[str, str, str]] = set()
    candidate_owners: dict[Path, tuple[str, str, str]] = {}
    for position, entry in enumerate(entries, start=1):
        missing = required - entry.keys()
        if missing:
            return False, f"summary record {position} lacks {sorted(missing)}"
        identity = (
            str(entry["label_folder"]),
            str(entry["source_label_folder"]),
            str(entry["img_name"]),
        )
        if identity in actual:
            return False, f"summary contains duplicate image identity {identity!r}"
        actual.add(identity)
        if identity not in expected:
            return False, f"summary contains unexpected image identity {identity!r}"
        if entry["dataset"] != dataset.name:
            return False, f"{identity!r} records the wrong dataset name"

        expected_gt = 0 if identity[0].lower() == "good" else 1
        if entry["gt_binary"] != expected_gt:
            return False, f"{identity!r} has inconsistent ground truth"
        label = entry["label"]
        prediction = entry["binary_prediction"]
        if label not in {"good", "anomaly"} or prediction not in {0, 1}:
            return False, f"{identity!r} has an invalid prediction"
        if (label == "anomaly") != (prediction == 1):
            return False, f"{identity!r} has inconsistent label/prediction"
        crop_error = validate_summary_crop_record(
            entry.get("object_crop"), cropping_enabled=cropping_enabled
        )
        if crop_error is not None:
            return False, f"{identity!r} {crop_error}"

        if secondary_profile is not None:
            secondary_required = {
                "yes_steps", "secondary_class", "secondary_class_scores",
                "secondary_binary_score",
            }
            secondary_missing = secondary_required - entry.keys()
            if secondary_missing:
                return False, f"{identity!r} lacks {sorted(secondary_missing)}"
            yes_steps = entry["yes_steps"]
            if (
                not isinstance(yes_steps, list)
                or any(isinstance(step, bool) or not isinstance(step, int) for step in yes_steps)
                or yes_steps != sorted(set(yes_steps))
            ):
                return False, f"{identity!r} has invalid yes_steps"
            try:
                outcome = classify_yes_steps(secondary_profile, yes_steps)
            except ValueError as error:
                return False, f"{identity!r} has invalid secondary result: {error}"
            if entry["secondary_class"] != outcome.secondary_class:
                return False, f"{identity!r} has an inconsistent secondary class"
            if label != outcome.binary_label or prediction != outcome.binary_prediction:
                return False, f"{identity!r} has secondary/binary disagreement"
            scores = entry["secondary_class_scores"]
            if not isinstance(scores, dict) or set(scores) != set(secondary_profile.class_names):
                return False, f"{identity!r} has invalid secondary class-score keys"
            if any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                for score in scores.values()
            ):
                return False, f"{identity!r} has invalid secondary class scores"
            binary_score = entry["secondary_binary_score"]
            if (
                isinstance(binary_score, bool)
                or not isinstance(binary_score, (int, float))
                or not math.isfinite(float(binary_score))
                or not math.isclose(
                    float(binary_score),
                    binary_anomaly_score(secondary_profile, scores),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
            ):
                return False, f"{identity!r} has an inconsistent secondary binary score"

        decision_path = _resolve_summary_artifact(
            entry["decision_filepath"], output_dataset_dir
        )
        probability_path = _resolve_summary_artifact(
            entry["probs_data_filepath"], output_dataset_dir
        )
        expected_text_dir = artifact_group_dir(
            output_dataset_dir / "texts", identity[0], identity[1]
        ).resolve()
        if decision_path != expected_text_dir / f"{identity[2]}.txt":
            return False, f"{identity!r} points at a non-canonical decision path"
        if probability_path != expected_text_dir / f"{identity[2]}_prob_data.txt":
            return False, f"{identity!r} points at a non-canonical probability path"
        try:
            decision_text = decision_path.read_text(encoding="utf-8")
            if secondary_profile is None:
                validate_yes_step_evidence(decision_text)
            parsed_label, _, _ = parse_final_decision(
                decision_text, label_policy=label_policy
            )
        except (OSError, UnicodeError, ValueError) as error:
            return False, f"{identity!r} has invalid decision data: {error}"
        if parsed_label != label:
            return False, f"{identity!r} summary and decision labels disagree"
        probabilities_valid, probability_error = validate_probability_file(
            probability_path
        )
        if not probabilities_valid:
            return False, f"{identity!r} has invalid probability data: {probability_error}"

        candidate_path = _resolve_summary_artifact(
            entry["preprocess_cand_img_path"], output_dataset_dir
        )
        expected_candidate = (
            artifact_group_dir(
                output_dataset_dir / "preproc",
                identity[0],
                identity[1],
                prefix="cand_",
            )
            / f"processed_{identity[2]}"
        ).with_suffix(".png").resolve()
        if candidate_path != expected_candidate or not _valid_image_file(candidate_path):
            return False, f"{identity!r} has a missing/corrupt candidate image"
        previous_owner = candidate_owners.setdefault(candidate_path, identity)
        if previous_owner != identity:
            return False, f"{previous_owner!r} and {identity!r} share one candidate image"

        reference_path = _resolve_summary_artifact(
            entry["preprocess_ref_img_path"], output_dataset_dir
        )
        if not _valid_image_file(reference_path):
            return False, f"{identity!r} has a missing/corrupt reference image"
        planner_path = _resolve_summary_artifact(
            entry["planner_filepath"], output_dataset_dir
        )
        if not planner_path.is_file() or planner_path.stat().st_size == 0:
            return False, f"{identity!r} has a missing/empty planner"

    missing_identities = expected - actual
    if missing_identities:
        example = sorted(missing_identities)[0]
        return False, f"summary is missing {len(missing_identities)} image(s), e.g. {example!r}"
    return True, ""


def entire_dataset_run(
    datacfg: AnomalyDetectionConfig,
    modelcfg: ModelsConfig,
    vlmModel: type[vlm],
    encoder: type[Encoder],
    dataset: DatasetEntry,
    run_dir: Path,
    progress: Progress | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    """
    Description - Runs inference for one dataset end-to-end (embeds
    train/good reference images once, runs inference_agent per test group)
    and writes all_data_summary.jsonl.

    ``progress``, if given, is the SAME `rich.progress.Progress` instance the
    caller (run_anomaly_detection.py) already uses for the outer per-dataset
    bar. A nested per-image task is added here (and removed once this
    dataset finishes) so the live display shows both "which dataset" and
    "which image, currently doing what" at once.

    Output - dict with the dataset name, output directory, summary filepath,
    and per-group {"y_true": [...], "y_pred": [...]} — handed to metrics.py
    by the caller if metrics are enabled.

    Also writes a `performance_<timestamp>.json` next to the summary (model
    used + its full config, total wall time for this dataset, and the total/
    average per-image synchronized model-generation time derived from each
    summary row's `inference_time_sec` — see decision.py).
    """
    dataset_start = time.perf_counter()
    ref_dir = dataset.root_dir / "train" / "good"
    project_root = Path(__file__).resolve().parent.parent
    output_dataset_directory = run_dir / dataset.name
    preproc_base = output_dataset_directory / "preproc"
    text_base = output_dataset_directory / "texts"
    preproc_base.mkdir(parents=True, exist_ok=True)
    text_base.mkdir(parents=True, exist_ok=True)
    _ensure_secondary_profile(
        output_dataset_directory, datacfg.secondary_classification
    )
    ensure_object_crop_profile(
        output_dataset_directory, dataset.object_crop, datacfg.prompts_dir
    )
    crop_warning = train_crop_manifest_warning(
        dataset.name, dataset.root_dir, dataset.object_crop, datacfg.prompts_dir
    )
    if crop_warning is not None:
        logger.warning(crop_warning)

    groups = _discover_test_groups(dataset)
    summary_filepath = output_dataset_directory / "all_data_summary.jsonl"
    if summary_filepath.is_file():
        summary_valid, validation_error = _validate_completed_summary(
            dataset,
            output_dataset_directory,
            summary_filepath,
            groups,
            label_policy=datacfg.decision_label_policy,
        )
        if not summary_valid:
            if "invalid decision data: YES steps lack concrete inline visual evidence" in validation_error:
                archived_summary = summary_filepath.with_name(
                    f"{summary_filepath.name}.invalidated_bare_yes_{uuid.uuid4().hex}"
                )
                os.replace(summary_filepath, archived_summary)
                logger.warning(
                    "[%s] Archived legacy summary with bare YES evidence at %s; "
                    "rechecking its per-image artifacts before rebuilding.",
                    dataset.name,
                    archived_summary,
                )
            else:
                raise RuntimeError(
                    f"[{dataset.name}] all_data_summary.jsonl exists but the dataset "
                    f"is not safely complete: {validation_error}. Resume did not "
                    "modify it. Use a fresh run directory for legacy/corrupt "
                    "completed data."
                )
        else:
            if dataset.object_crop is not None and dataset.object_crop.save_cropped_png:
                for raw_line in summary_filepath.read_text(encoding="utf-8").splitlines():
                    if not raw_line.strip():
                        continue
                    entry = json.loads(raw_line)
                    source_image = (
                        dataset.root_dir / "test" / entry["source_label_folder"] / entry["img_name"]
                    )
                    save_cropped_candidate_png(
                        source_image, tuple(entry["object_crop"]["box_pixels"]),
                        output_dataset_directory, entry["label_folder"],
                        entry["source_label_folder"],
                    )
            logger.info("[%s] all_data_summary.jsonl already exists — dataset already complete, skipping.", dataset.name)
            return _load_completed_dataset_result(dataset, output_dataset_directory, summary_filepath)

    ref_img_paths, _ = list_images(ref_dir)
    ref_img_tensors = preprocess_image_for_embedding(ref_img_paths)
    ref_emb_np = compute_ref_embeddings(encoder=encoder, ref_img_tensors=ref_img_tensors) #Computes embeddings for all the reference images in the reference directory (train/good) for the dataset 
    del ref_img_tensors
    torch.cuda.empty_cache()

    # Pause the outer progress display while the interactive labeler owns the
    # terminal. This lets it redraw one clean screen per proposal.
    if progress is not None:
        progress.stop()
    try:
        resolved_class_labels = load_or_generate_class_labels(
            dataset=dataset,
            modelcfg=modelcfg,
            datacfg=datacfg,
            vlmModel=vlmModel,
            project_root=project_root,
            console=console,
        )
    finally:
        if progress is not None:
            progress.start()
            progress.refresh()

    logger.info("[%s] ref=%d  counts=%s", dataset.name, len(ref_img_paths), {k: len(v) for k, v in groups.items()})
    if dataset.classes == "multi":
        logger.info("[%s] class labels loaded: %s", dataset.name, list((resolved_class_labels or {}).keys()))

    full_summary: list[dict[str, Any]] = []
    results: dict[str, dict[str, list[int]]] = {}

    total_images = sum(len(paths) for paths in groups.values())
    image_task = (
        progress.add_task(
            f"  [dim]{dataset.name}: Starting…",
            total=total_images,
            avg_inference_sec="--",
            avg_crop_sec="--",
            avg_with_crop_sec="--",
            avg_output_tokens="--",
            generated_tokens_per_second="--",
            inference_count=0,
            inference_total_sec=0.0,
            crop_count=0,
            crop_total_sec=0.0,
            with_crop_total_sec=0.0,
            generated_token_count=0,
            token_timing_sec=0.0,
            token_sample_count=0,
        )
        if progress
        else None
    )
    if progress is not None:
        progress.refresh()

    for group_name, img_paths in groups.items():
        summary, y_actual, y_pred = inference_agent(
            dataset=dataset,
            datacfg=datacfg,
            modelcfg=modelcfg,
            vlmModel=vlmModel,
            encoder=encoder,
            cand_img_paths=img_paths,
            group_name=group_name,
            ref_emb_np=ref_emb_np,
            ref_img_paths=ref_img_paths,
            preproc_base=preproc_base,
            text_base=text_base,
            resolved_class_labels=resolved_class_labels,
            progress=progress,
            image_task=image_task,
        )
        full_summary.extend(summary)
        results[group_name] = {"y_true": y_actual, "y_pred": y_pred}

    if progress is not None and image_task is not None:
        progress.remove_task(image_task)
        progress.refresh()

    _write_summary_atomically(
        dataset,
        output_dataset_directory,
        summary_filepath,
        full_summary,
        groups,
        label_policy=datacfg.decision_label_policy,
    )

    logger.info("[%s] Summary written to %s", dataset.name, summary_filepath)

    total_wall_time_sec = time.perf_counter() - dataset_start
    inference_times = [
        float(entry["inference_time_sec"])
        for entry in full_summary
        if entry.get("inference_time_sec") is not None
        and float(entry["inference_time_sec"]) > 0
    ]
    total_inference_time_sec = sum(inference_times)
    timing_bases = {
        str(entry.get("timing_basis", "legacy_vlm_call"))
        for entry in full_summary
        if entry.get("inference_time_sec") is not None
    }
    token_entries = [
        entry for entry in full_summary
        if entry.get("generated_token_count") is not None
        and entry.get("inference_time_sec") is not None
        and float(entry["inference_time_sec"]) > 0
    ]
    total_generated_tokens = sum(int(entry["generated_token_count"]) for entry in token_entries)
    token_timing_sec = sum(float(entry["inference_time_sec"]) for entry in token_entries)
    performance: dict[str, Any] = {
        "dataset": dataset.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"vlm": asdict(modelcfg.vlm)},
        "decision_prompt_template": (
            datacfg.secondary_classification.prompt_template
            if datacfg.secondary_classification
            else resolve_decision_prompt_template(datacfg.decision_prompt_template)
        ),
        "secondary_profile_fingerprint": (
            profile_fingerprint(datacfg.secondary_classification)
            if datacfg.secondary_classification else None
        ),
        "num_images": len(full_summary),
        "num_images_with_vlm_timing": len(inference_times),
        "total_wall_time_sec": round(total_wall_time_sec, 4),
        "total_vlm_inference_time_sec": round(total_inference_time_sec, 4),
        "avg_vlm_inference_time_per_image_sec": (
            round(total_inference_time_sec / len(inference_times), 4) if inference_times else None
        ),
        "timing_basis": (
            next(iter(timing_bases)) if len(timing_bases) == 1 else "mixed"
        ),
    }
    if os.environ.get("ANOMALY_ABLATION_PROFILE") == "1":
        performance.update(
            num_images_with_token_stats=len(token_entries),
            total_generated_tokens=total_generated_tokens,
            avg_generated_tokens_per_image=(
                round(total_generated_tokens / len(token_entries), 4)
                if token_entries else None
            ),
            generated_tokens_per_second=(
                round(total_generated_tokens / token_timing_sec, 4)
                if token_timing_sec > 0 else None
            ),
        )
    # Object-crop cost is reported separately; avg_vlm_inference_time_per_
    # image_sec above stays decision-only for comparison with uncropped runs.
    performance.update(object_crop_timing_summary(full_summary))
    performance_dir = output_dataset_directory / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    performance_path = performance_dir / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    performance_path.write_text(json.dumps(performance, indent=2, default=str), encoding="utf-8")
    logger.info("[%s] Performance summary written to %s", dataset.name, performance_path)

    return {
        "dataset_name": dataset.name,
        "output_dir": output_dataset_directory,
        "summary_filepath": summary_filepath,
        "performance": performance,
        "groups": results,
    }
