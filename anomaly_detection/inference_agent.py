"""Per-image anomaly inference and artifact persistence.

Behavior change (2026-08): artifacts for a merged anomaly group now preserve
the candidate's real ``source_label_folder`` in their paths.  Previously,
different defect classes such as ``broken_large/000.png`` and
``contamination/000.png`` were both written as
``preproc/cand_anomalies/processed_000.png`` and silently overwrote each
other.  When the source and display labels differ, the new layout is
``preproc/cand_<label>/<source_label>/processed_<image>`` (and the equivalent
reference-image and text layouts). Non-merged groups keep their previous flat
paths. Multi-class names such as Orange/Banana/Cherry remain planner selectors
only; they never become artifact folders.

Behavior change (2026-09): when ``datasets[].object_crop.enabled`` is true,
each candidate is first cropped to the VLM-located object (see
``object_crop.py``). The crop box is cached under
``<dataset>/object_crops/<label>[/<source>]/<image>.json`` and recorded in the
summary row as ``object_crop``; CLIP reference selection and the letterboxed
``preproc/cand_*`` image both use the crop. A "not found" answer raises
``ObjectNotFoundError`` and stops the run with its evidence path.
"""

from __future__ import annotations
import os
import shutil
import uuid
import gc
import json
import logging
import numpy as np
import clip
from typing import Any
from pathlib import Path
from PIL import Image
import torch
from rich.progress import Progress, TaskID
from rich.progress import Task as ProgressTask
from .models import Encoder, vlm
from sklearn.metrics import pairwise_distances
from .utils import preprocess_img
from .decision import (
    decision_maker,
    parse_final_decision,
    missing_yes_evidence_steps,
    validate_yes_step_evidence,
    resolve_decision_prompt_template,
    _planner_step_count,
)
from .reference_selector import inference_ref_img_selector
from .config import AnomalyDetectionConfig, DatasetEntry, ModelsConfig, artifact_group_dir
from .object_crop import load_or_locate_object_crop, save_cropped_candidate_png
from .scoring import validate_probability_file
from .secondary_classification import (
    SecondaryClassificationProfile,
    binary_anomaly_score,
    classify_yes_steps,
    parse_step_log,
    score_classes,
    yes_probabilities,
)

logger = logging.getLogger(__name__)


def _archive_bare_yes_pair(decision_path: Path, probability_path: Path) -> None:
    """Preserve a formerly accepted bare-YES pair before recomputing it."""
    try:
        text = decision_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return
    if not missing_yes_evidence_steps(text):
        return
    archive_dir = decision_path.parent / ".invalidated_decisions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    for source in (decision_path, probability_path):
        if source.is_file():
            shutil.copy2(source, archive_dir / f"{source.name}.{suffix}")
    logger.warning("Archived bare-YES decision pair for %s in %s", decision_path, archive_dir)


def _reconstruct_decision_output(
    decision_result_filepath: Path,
    probsdata_result_filepath: Path,
    text_base: Path,
    label_policy: str = "model",
    secondary_profile: SecondaryClassificationProfile | None = None,
    planner_filepath: Path | None = None,
) -> dict[str, Any] | None:
    """
    Rebuild a decision_maker()-shaped summary dict from a PREVIOUSLY written
    decision text file, without calling the VLM again. Used to resume a run
    that crashed or was interrupted mid-dataset: every image whose decision
    text file already exists on disk was already fully decided in an earlier
    (partial) attempt at this same run_dir, so re-running the VLM on it would
    just waste time (and GPU risk) for a result we already have.

    Validates that probsdata_result_filepath actually exists and is readable
    (rerun.py will need it later to calculate threshold). If the file is
    missing or empty, that's a sign the earlier run crashed mid-VLM-call
    (before writing prob data), so we skip this reconstructed entry and let
    the caller re-run the VLM on it instead.

    Stores probs_data_filepath as RELATIVE to the dataset directory (not
    project root) so that rerun.py's _resolve_probs_path() can find it correctly.

    New runs persist timing/token statistics beside each decision so resumed
    images retain their performance measurements. Legacy decisions without a
    sidecar fall back to zero/unknown values without affecting correctness.
    """
    try:
        final_text = Path(decision_result_filepath).read_text(encoding="utf-8")
        if not final_text.strip():
            raise ValueError("decision file is empty")
        if secondary_profile is None:
            validate_yes_step_evidence(final_text)
        label, reason, details = parse_final_decision(
            final_text,
            label_policy=label_policy,
        )
    except (OSError, UnicodeError, ValueError) as error:
        logger.warning(
            "Decision file is incomplete or invalid for resume (%s: %s). "
            "This image will be re-run.",
            decision_result_filepath,
            error,
        )
        return None
    pred = 1 if label == "anomaly" else 0

    # A file can be non-empty but still contain truncated JSON or no usable
    # confidence rows. Resume therefore validates its complete content.
    probs_path = Path(probsdata_result_filepath)
    probabilities_valid, validation_error = validate_probability_file(probs_path)
    if not probabilities_valid:
        logger.warning(
            "Probability data is incomplete or invalid for %s (%s). "
            "This image will be re-run instead of resumed.",
            decision_result_filepath.name,
            validation_error,
        )
        return None

    secondary_fields: dict[str, Any] = {}
    if secondary_profile is not None:
        try:
            if planner_filepath is None:
                raise ValueError("planner path is required")
            planner_text = Path(planner_filepath).read_text(encoding="utf-8")
            expected_count = _planner_step_count(planner_text)
            if expected_count is None:
                raise ValueError("planner steps are not consecutively numbered")
            evidence_steps = {
                step for rule in secondary_profile.classes for step in rule.localization_steps
            }
            step_log = parse_step_log(
                final_text,
                list(range(1, expected_count + 1)),
                evidence_required_steps=evidence_steps,
            )
            probability_rows = [
                json.loads(line)
                for line in probs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            class_scores = score_classes(
                secondary_profile, yes_probabilities(probability_rows)
            )
            outcome = classify_yes_steps(secondary_profile, step_log.yes_steps)
            label = outcome.binary_label
            pred = outcome.binary_prediction
            secondary_fields = {
                "yes_steps": outcome.yes_steps,
                "secondary_class": outcome.secondary_class,
                "secondary_class_scores": {
                    name: round(score, 8) for name, score in class_scores.items()
                },
                "secondary_binary_score": round(
                    binary_anomaly_score(secondary_profile, class_scores), 8
                ),
            }
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            logger.warning(
                "Secondary decision artifacts are invalid for resume (%s: %s). This image will be re-run.",
                decision_result_filepath,
                error,
            )
            return None

    # Store paths relative to text_base.parent (dataset-directory-relative) so that
    # rerun.py's _resolve_probs_path() can find them. This avoids the
    # project-root-relative path issue where paths don't resolve correctly.
    try:
        probs_path_relative = Path(probsdata_result_filepath).relative_to(text_base.parent)
        decision_path_relative = Path(decision_result_filepath).relative_to(text_base.parent)
    except ValueError:
        # Fallback if relative_to fails (shouldn't happen in normal flow)
        probs_path_relative = Path(probsdata_result_filepath)
        decision_path_relative = Path(decision_result_filepath)

    ablation_profile = os.environ.get("ANOMALY_ABLATION_PROFILE") == "1"
    performance: dict[str, Any] = {}
    if ablation_profile:
        performance_path = Path(decision_result_filepath).with_suffix(".performance.json")
        try:
            loaded_performance = json.loads(performance_path.read_text(encoding="utf-8"))
            if isinstance(loaded_performance, dict):
                performance = loaded_performance
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    try:
        persisted_inference_time = float(performance.get("inference_time_sec", 0.0))
    except (TypeError, ValueError):
        persisted_inference_time = 0.0
    try:
        persisted_vlm_call_time = float(performance.get("vlm_call_time_sec", 0.0))
    except (TypeError, ValueError):
        persisted_vlm_call_time = 0.0
    try:
        persisted_token_count = int(performance["generated_token_count"])
    except (KeyError, TypeError, ValueError):
        persisted_token_count = None
    try:
        persisted_tokens_per_second = float(performance["generated_tokens_per_second"])
    except (KeyError, TypeError, ValueError):
        persisted_tokens_per_second = None

    result = {
        "label": label,
        "binary_prediction": pred,
        "decision_filepath": str(decision_path_relative),
        "probs_data_filepath": str(probs_path_relative),
        "inference_time_sec": persisted_inference_time,
        "vlm_call_time_sec": persisted_vlm_call_time,
        "timing_basis": performance.get("timing_basis", "legacy_vlm_call"),
        "reason": reason,
        "details": details,
        **secondary_fields,
    }
    if ablation_profile:
        result.update(
            generated_token_count=persisted_token_count,
            generated_tokens_per_second=persisted_tokens_per_second,
        )
    return result

def planner_selector(planners_root_dir: Path, dataset_name: str, label: str) -> Path:
    """
    Description - Returns the most recently generated planner file (by the
    timestamp embedded in its filename) for a given dataset + label, as
    written by planner_generation/planner.py.

    Expected layout: <planners_root_dir>/<dataset_name>/<label>/*_planner.txt
    """
    planner_folder = Path(planners_root_dir) / dataset_name / label
    if not planner_folder.exists():
        raise FileNotFoundError(
            f"No planner folder found for dataset='{dataset_name}' label='{label}': {planner_folder}\n"
            f"Run run_planner_generation.py for this dataset first."
        )

    filenames = os.listdir(planner_folder)
    if not filenames:
        raise FileNotFoundError(f"No planner files found in {planner_folder}")

    timestamps = sorted((item.split('_')[-2] for item in filenames), reverse=True)
    latest_timestamp = timestamps[0]
    for item in filenames:
        if item.split('_')[-2] == latest_timestamp:
            return planner_folder / item

    raise FileNotFoundError(f"Could not resolve the latest planner file in {planner_folder}")


def _get_task(progress: Progress, task_id: TaskID) -> ProgressTask:
    """
    Look up a live Task by its TaskID.

    ``Progress.tasks`` is a plain list of currently active tasks, NOT a
    mapping keyed by TaskID -- and TaskIDs are a monotonically increasing
    counter per Progress instance. Once any earlier task has been added and
    removed (e.g. ``entire_dataset_run()`` adds a nested per-image task per
    dataset and removes it when that dataset finishes), a later task's ID no
    longer matches its position in the (now shorter) list, so
    ``progress.tasks[task_id]`` can raise IndexError or silently return the
    wrong task. Looking up by ``Task.id`` is always correct regardless of
    prior add/remove history.
    """
    for task in progress.tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"No active Progress task with id={task_id}")


def label_selector(encoder: type[Encoder], class_labels: dict[str, str], ref_emb_row: np.ndarray) -> tuple[int, str]:
    """
    Description - Given a dataset's {label: text description} mapping and a
    single reference-image embedding row, returns the (index, label) of the
    class whose text description is closest to that embedding via CLIP
    text-image cosine similarity.
    """
    labels = list(class_labels.keys())
    texts = list(class_labels.values())

    text_features_list = []
    with torch.no_grad():
        for text in texts:
            tokens = clip.tokenize(text).to(encoder.device)
            embeddings = encoder.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            text_features_list.append(embeddings.squeeze(0))
        ref_emb_text = torch.stack(text_features_list).cpu().numpy()

    distances = pairwise_distances(ref_emb_row, ref_emb_text, metric="cosine")
    idx = int(np.argmin(distances, axis=1)[0])
    logger.debug("Closest class label: index=%d label=%s", idx, labels[idx])
    return idx, labels[idx]


def _select_planner(encoder: type[Encoder], dataset: DatasetEntry, datacfg: AnomalyDetectionConfig, ref_emb_row: np.ndarray, resolved_class_labels: dict[str, str] | None) -> Path:
    """Resolve which planner file to use for a candidate image, given the CLIP embedding row of its nearest reference image."""
    if dataset.classes == "multi":
        if not resolved_class_labels:
            raise ValueError(
                f"{dataset.name}: multi-class dataset requires class labels. "
                "Provide cached labels or class_examples for generation."
            )
        _, label = label_selector(encoder=encoder, class_labels=resolved_class_labels, ref_emb_row=ref_emb_row)
    else:
        label = "good"
    return planner_selector(datacfg.planners_dir, dataset.name, label)


def inference_agent(dataset: DatasetEntry, datacfg: AnomalyDetectionConfig, modelcfg: ModelsConfig, vlmModel: type[vlm], encoder: type[Encoder], cand_img_paths: list[str], 
                    group_name: str, ref_emb_np, ref_img_paths: list[str], preproc_base: Path, text_base: Path, resolved_class_labels: dict[str, str] | None,
                    progress: Progress | None = None, image_task: TaskID | None = None,
                    )-> tuple[list[dict[str, Any]], list[int], list[int]]:
    """
    Description - Runs inference for one group of candidate images (e.g. all "good" or all "logical_anomalies" images of a dataset) and returns the per-image result summaries 
    plus parallel ground-truth/prediction lists.

    ``group_name`` is the group key returned by ``_discover_test_groups()``
    (e.g. "good"/"anomalies" when ``merge_anomalies: true", or the real test
    subfolder name otherwise). ``label_folder`` remains the configured group,
    while ``source_label_folder`` preserves the image's real test subfolder.
    Whenever those names differ, persisted artifacts include both components;
    this is required because different source folders commonly reuse filenames
    such as ``000.png``.

    ``progress``/``image_task``, if given, are used to live-update a nested
    rich progress bar (shared with the caller) with the image currently being
    processed and, once decided, its predicted label.
    """
    summary: list[dict[str, Any]] = []
    y_actual: list[int] = [] #Ground Truth of inference (candidate) image 
    y_pred: list[int] = []

    for img_path in cand_img_paths:
        if torch.cuda.is_available():
            logger.debug("GPU mem: %.2fGB allocated, %.2fGB reserved", torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

        cand_img_path = Path(img_path)
        label_folder_for_display = cand_img_path.parent.name
        image_number = 1
        total_images = len(cand_img_paths)
        if progress is not None and image_task is not None:
            image_task_state = _get_task(progress, image_task)
            image_number = int(image_task_state.completed) + 1
            total_images = int(image_task_state.total)

        if progress is not None and image_task is not None:
            progress.update(image_task, description=f"  [dim]{dataset.name}: {label_folder_for_display}/{cand_img_path.name} [{image_number}/{total_images}] — selecting reference…")
            progress.refresh()

        label_folder = group_name
        source_label_folder = cand_img_path.parent.name

        # Optional object-centred crop. It must precede reference selection so
        # CLIP compares the located object, and it is cached so a resumed run
        # reuses the exact same box instead of calling the VLM again.
        crop_result = None
        crop_config = dataset.object_crop
        if crop_config is not None and crop_config.enabled:
            if progress is not None and image_task is not None:
                progress.update(image_task, description=f"  [dim]{dataset.name}: {label_folder_for_display}/{cand_img_path.name} [{image_number}/{total_images}] — locating object for crop…")
                progress.refresh()
            crop_cache_path = artifact_group_dir(
                text_base.parent / "object_crops", label_folder, source_label_folder
            ) / f"{cand_img_path.name}.json"
            crop_result = load_or_locate_object_crop(
                vlmModel, modelcfg, cand_img_path, crop_config,
                datacfg.prompts_dir, crop_cache_path,
            )
        crop_box = crop_result.box_pixels if crop_result is not None else None
        if crop_result is not None and crop_config.save_cropped_png:
            save_cropped_candidate_png(
                cand_img_path, crop_result.box_pixels, text_base.parent,
                label_folder, source_label_folder,
            )

        min_indices = inference_ref_img_selector(encoder=encoder, ref_emb_np=ref_emb_np, cand_img_path=cand_img_path, crop_box=crop_box) #Selects the closest reference image to candidate image based on cosine similarity of encodings
        ref_img_path = Path(ref_img_paths[min_indices[0]])
        
        logger.debug("Minimum reference indices: %s", min_indices)
        logger.debug("Selected reference image index: %d", min_indices[0])

        planner_filepath = _select_planner(encoder=encoder,
                           dataset=dataset, 
                                           datacfg=datacfg,
                                           ref_emb_row=ref_emb_np[min_indices[0]].reshape(1, -1),
                                           resolved_class_labels=resolved_class_labels,
                                           )

        # Folder grouping depends only on merge_anomalies (represented here by
        # label_folder versus source_label_folder). The semantic multi-class
        # label chosen above affects planner_filepath only and is intentionally
        # absent from every output path.
        ref_tag_name = f'{cand_img_path.name}_ref_{ref_img_path.name}'
        ref_img_out_dir = artifact_group_dir(
            preproc_base, label_folder, source_label_folder, prefix="ref_"
        )
        preprocess_ref_img_path = preprocess_img(modelcfg, ref_img_path, ref_img_out_dir, ref_tag_name)

        gt = 0 if label_folder.lower() == 'good' else 1
        cand_tag_name = cand_img_path.name
        cand_img_out_dir = artifact_group_dir(
            preproc_base, label_folder, source_label_folder, prefix="cand_"
        )
        preprocess_cand_img_path = preprocess_img(modelcfg, cand_img_path, cand_img_out_dir, cand_tag_name, crop_box=crop_box)

        ref_img = Image.open(preprocess_ref_img_path).convert("RGB")
        cand_img = Image.open(preprocess_cand_img_path).convert("RGB")

        # Use exactly the same conditional grouping for decisions. In
        # particular, non-merged LOCO stays texts/logical_anomalies (not a
        # duplicated texts/logical_anomalies/logical_anomalies directory).
        decision_output_folder = artifact_group_dir(
            text_base, label_folder, source_label_folder
        )
        decision_output_folder.mkdir(parents=True, exist_ok=True)
        decision_result_filepath = decision_output_folder / f'{cand_tag_name}.txt'
        probsdata_result_filepath = decision_output_folder / f'{cand_tag_name}_prob_data.txt'

        # A short-lived earlier implementation always appended the source
        # folder, even when source == label. Reuse those safe-but-duplicated
        # decision files only while continuing an interrupted dataset. All
        # newly written files use the canonical flat non-merged layout above;
        # completed datasets are skipped before this function is entered.
        legacy_decision_folder = text_base / label_folder / source_label_folder
        resume_pairs = [
            (decision_result_filepath, probsdata_result_filepath),
        ]
        if legacy_decision_folder != decision_output_folder:
            resume_pairs.append((
                legacy_decision_folder / f'{cand_tag_name}.txt',
                legacy_decision_folder / f'{cand_tag_name}_prob_data.txt',
            ))
        completed_pair = next((
            pair for pair in resume_pairs
            if pair[0].is_file() and pair[1].is_file()
        ), None)

        if progress is not None and image_task is not None:
            progress.update(image_task, description=f"  [dim]{dataset.name}: {label_folder}/{cand_tag_name} [{image_number}/{total_images}] — running VLM decision maker…")
            progress.refresh()

        if completed_pair is not None:
            # Resuming a previous partial attempt at this same run_dir: this
            # image's decision was already produced (and written to disk)
            # before the earlier process crashed/was interrupted -- reuse it
            # instead of calling the VLM again. UNLESS the prob_data file is
            # missing/empty (sign of a crash mid-VLM-call before prob data was
            # fully written), in which case re-run the VLM.
            resumed_decision_path, resumed_probs_path = completed_pair
            output = _reconstruct_decision_output(
                resumed_decision_path,
                resumed_probs_path,
                text_base,
                label_policy=datacfg.decision_label_policy,
                secondary_profile=datacfg.secondary_classification,
                planner_filepath=planner_filepath,
            )
            if output is None:
                _archive_bare_yes_pair(resumed_decision_path, resumed_probs_path)
                logger.info("Decision or probability data invalid for %s; re-running VLM instead of resuming.", cand_tag_name)
                output = decision_maker(
                    modelcfg=modelcfg,
                    prompts_dir=datacfg.prompts_dir,
                    vlmModel=vlmModel,
                    ref_img=ref_img,
                    cand_img=cand_img,
                    planner_filepath=planner_filepath,
                    decision_result_filepath=decision_result_filepath,
                    probsdata_result_filepath=probsdata_result_filepath,
                    text_base=text_base,
                    label_policy=datacfg.decision_label_policy,
                    decision_prompt_template=datacfg.decision_prompt_template,
                    secondary_profile=datacfg.secondary_classification,
                )
            else:
                logger.info("Resuming: decision already exists for %s — skipping VLM call.", cand_tag_name)
        else:
            output = decision_maker(
                modelcfg=modelcfg,
                prompts_dir=datacfg.prompts_dir,
                vlmModel=vlmModel,
                ref_img=ref_img,
                cand_img=cand_img,
                planner_filepath=planner_filepath,
                decision_result_filepath=decision_result_filepath,
                probsdata_result_filepath=probsdata_result_filepath,
                text_base=text_base,
                label_policy=datacfg.decision_label_policy,
                decision_prompt_template=datacfg.decision_prompt_template,
                secondary_profile=datacfg.secondary_classification,
            )
        output.update({
            "dataset": dataset.name,
            "label_folder": label_folder,
            "source_label_folder": source_label_folder,
            "gt_binary": gt,
            "planner_filepath": str(planner_filepath),
            "img_name": cand_tag_name,
            "preprocess_ref_img_path": str(preprocess_ref_img_path),
            "preprocess_cand_img_path": str(preprocess_cand_img_path),
            "decision_prompt_template": (
                datacfg.secondary_classification.prompt_template
                if datacfg.secondary_classification
                else resolve_decision_prompt_template(datacfg.decision_prompt_template)
            ),
        })
        if crop_result is not None:
            try:
                crop_metadata_path = str(crop_cache_path.relative_to(text_base.parent))
            except ValueError:
                crop_metadata_path = str(crop_cache_path)
            output["object_crop"] = crop_result.summary_record(crop_metadata_path)
        summary.append(output)

        if progress is not None and image_task is not None:
            task_fields = _get_task(progress, image_task).fields
            inference_count = int(task_fields["inference_count"]) + 1
            inference_total_sec = float(task_fields["inference_total_sec"]) + output["inference_time_sec"]
            generated_token_count = int(task_fields["generated_token_count"])
            token_timing_sec = float(task_fields["token_timing_sec"])
            token_sample_count = int(task_fields["token_sample_count"])
            output_tokens = output.get("generated_token_count")
            if output_tokens is not None and output["inference_time_sec"] > 0:
                generated_token_count += int(output_tokens)
                token_timing_sec += float(output["inference_time_sec"])
                token_sample_count += 1
            # Running crop averages (same basis as the dataset performance
            # JSON): only images whose crop record carries a measured time.
            crop_count = int(task_fields.get("crop_count", 0))
            crop_total_sec = float(task_fields.get("crop_total_sec", 0.0))
            with_crop_total_sec = float(task_fields.get("with_crop_total_sec", 0.0))
            crop_time = (output.get("object_crop") or {}).get("inference_time_sec")
            if crop_time is not None:
                crop_count += 1
                crop_total_sec += float(crop_time)
                with_crop_total_sec += float(crop_time) + float(output["inference_time_sec"])
            progress.update(
                image_task,
                advance=1,
                description=f"  [dim]{dataset.name}: {label_folder}/{cand_tag_name} [{image_number}/{total_images}] → {output['label']}[/dim]",
                avg_inference_sec=f"{inference_total_sec / inference_count:.2f}s",
                avg_crop_sec=f"{crop_total_sec / crop_count:.2f}s" if crop_count else "--",
                avg_with_crop_sec=(
                    f"{with_crop_total_sec / crop_count:.2f}s" if crop_count else "--"
                ),
                crop_count=crop_count,
                crop_total_sec=crop_total_sec,
                with_crop_total_sec=with_crop_total_sec,
                avg_output_tokens=(
                    f"{generated_token_count / token_sample_count:.1f}"
                    if token_sample_count else "--"
                ),
                generated_tokens_per_second=(
                    f"{generated_token_count / token_timing_sec:.1f}"
                    if token_timing_sec > 0 else "--"
                ),
                inference_count=inference_count,
                inference_total_sec=inference_total_sec,
                generated_token_count=generated_token_count,
                token_timing_sec=token_timing_sec,
                token_sample_count=token_sample_count,
            )
            progress.refresh()

        torch.cuda.empty_cache()
        gc.collect()

        y_actual.append(gt)
        y_pred.append(output["binary_prediction"])

    return summary, y_actual, y_pred
