"""Generate inspection planners from normal references and anomaly rules."""

import os
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from .config import DatasetConfig, ModelsConfig
from .models import vlm
from .utils import build_prompt
from pipeline_core.model_output import normalize_model_output

logger = logging.getLogger(__name__)

# Generate one rule at a time. Besides fitting smaller VLM context windows,
# isolation prevents a model from reordering, merging, or borrowing anomaly
# details from adjacent rules. Results are renumbered and joined afterward.
PLANNER_RULES_PER_BATCH = 1
PLANNER_TOKENS_PER_STEP = 160


def _clean_model_output(text: str) -> str:
    """Remove reasoning and structural wrappers before strict validation."""
    return normalize_model_output(text)


def _planner_step_count(text: str) -> int:
    """Count only the literal ``Step N`` format required downstream."""
    cleaned = _clean_model_output(text)
    return len(re.findall(r"(?im)^\s*Step\s+\d+\b", cleaned))


def _planner_rule_ids(text: str) -> list[str]:
    """Extract rule IDs only from the required ``Step N [Rule ID]:`` form."""
    cleaned = _clean_model_output(text)
    return re.findall(
        r"(?im)^\s*Step\s+\d+\s+\[Rule\s+([^\]]+)\]\s*:", cleaned
    )


def _strip_planner_rule_ids(text: str) -> str:
    """Remove validation-only rule IDs before publishing the planner."""
    return re.sub(
        r"(?im)^(\s*Step\s+\d+)\s+\[Rule\s+[^\]]+\]\s*:", r"\1:", text
    )


def _renumber_planner_steps(text: str, offset: int) -> str:
    """Renumber a validated batch while preserving its generated wording."""
    next_number = offset

    def replacement(match: re.Match[str]) -> str:
        nonlocal next_number
        next_number += 1
        return f"{match.group(1)}Step {next_number}"

    return re.sub(r"(?im)^(\s*)Step\s+\d+\b", replacement, text).strip()


def _rule_batches(
    anomaly_rules: str, expected_rule_count: int | None
) -> list[tuple[str, int, list[str]]]:
    """Split structured anomaly rules into context-safe, ordered JSON batches."""
    if expected_rule_count is None:
        return [(anomaly_rules, 0, [])]
    try:
        payload = json.loads(anomaly_rules)
    except json.JSONDecodeError:
        logger.warning("Anomaly rules are not JSON; planner batching is unavailable.")
        return [(anomaly_rules, expected_rule_count, [])]
    candidates = payload.get("candidate_anomalies") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or len(candidates) != expected_rule_count:
        logger.warning(
            "Structured anomaly-rule count does not match expected count; planner batching is unavailable."
        )
        return [(anomaly_rules, expected_rule_count, [])]
    return [
        (
            json.dumps({**payload, "candidate_anomalies": candidates[start:start + PLANNER_RULES_PER_BATCH]}, indent=2),
            len(candidates[start:start + PLANNER_RULES_PER_BATCH]),
            [str(rule.get("id", "")).strip() for rule in candidates[start:start + PLANNER_RULES_PER_BATCH]],
        )
        for start in range(0, len(candidates), PLANNER_RULES_PER_BATCH)
    ]


def _rule_based_fallback(batch_rules: str) -> str:
    """Build faithful yes/no questions from structured rules without class-specific logic.

    This is used only after the VLM fails the step/rule-ID contract.  Keeping the
    rule's own object, region, anomaly name, and visual description makes the
    fallback deterministic while preserving the approved semantics.
    """
    payload = json.loads(batch_rules)
    candidates = payload.get("candidate_anomalies")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Cannot build planner fallback from unstructured anomaly rules.")

    steps: list[str] = []
    for index, rule in enumerate(candidates, start=1):
        if not isinstance(rule, dict):
            raise ValueError("Cannot build planner fallback from a non-object anomaly rule.")
        rule_id = str(rule.get("id", "")).strip()
        object_name = str(rule.get("object_name", "the inspected object")).strip()
        region = str(rule.get("part_or_region", object_name)).strip()
        anomaly_name = str(rule.get("anomaly_name", "the specified anomaly")).strip()
        visual_description = str(rule.get("visual_description", "")).strip().rstrip(".?")
        if not rule_id or not visual_description:
            raise ValueError(
                "Cannot build planner fallback when a rule ID or visual description is missing."
            )
        location = region
        if object_name.casefold() not in region.casefold():
            location = f"{region} of the {object_name}"
        steps.append(
            f"Step {index} [Rule {rule_id}]: Is there evidence of {anomaly_name} "
            f"in the {location}, specifically {visual_description}?"
        )
    return "\n".join(steps)


def agent_planner(datacfg:DatasetConfig, modelcfg: ModelsConfig, vlmModel: vlm, img: Image.Image, ANOMALY_RULES: str, baseline_description: str, expected_rule_count: int | None = None)->str:

    """description: This function takes the anomaly free image and master description as input to give planner steps as question to find anomaly as output.\n
    Input: cfg: dataclass,img:anomaly free image; ANOMALY_RULES:human accepted anomaly rules; baseline_description: master description of the dataset.\n
    Output: gives the planner steps in form of question where 'yes' means anomaly and 'no' means non anomaly."""
    
    batches = _rule_batches(ANOMALY_RULES, expected_rule_count)
    generated_batches: list[str] = []
    step_offset = 0
    logger.info(
        "Planner rule batching: %d batch(es), sizes=%s",
        len(batches), [count for _, count, _ in batches],
    )

    for batch_index, (batch_rules, batch_count, expected_ids) in enumerate(batches, start=1):
        prompt = build_prompt(
            datacfg.prompts_dir,
            "planner.j2",
            baseline_description=baseline_description,
            ANOMALY_RULES=batch_rules,
            expected_rule_count=batch_count or None,
            expected_rule_ids=expected_ids,
        )
        max_attempts = 2 if batch_count else 1
        output_limit = min(
            modelcfg.vlm.generation.max_new_tokens,
            max(256, batch_count * PLANNER_TOKENS_PER_STEP) if batch_count else modelcfg.vlm.generation.max_new_tokens,
        )
        for attempt in range(1, max_attempts + 1):
            out, vlm_time = vlmModel.run_vlm(
                modelcfg.vlm, [img], prompt, max_new_tokens=output_limit
            )
            result = _clean_model_output(out[1]['content'])
            actual_step_count = _planner_step_count(result)
            actual_ids = _planner_rule_ids(result)
            ids_match = not expected_ids or actual_ids == expected_ids
            logger.info(
                "Planner batch %d/%d generated in %.3fs (attempt=%d, steps=%d, approved_rules=%s)",
                batch_index, len(batches), vlm_time, attempt, actual_step_count,
                batch_count if batch_count else "legacy-untracked",
            )
            if not batch_count or (actual_step_count == batch_count and ids_match):
                publishable = _strip_planner_rule_ids(result) if expected_ids else result
                generated_batches.append(_renumber_planner_steps(publishable, step_offset))
                step_offset += actual_step_count
                break
            logger.debug(
                "Planner raw response on failed validation (batch=%d, attempt=%d, "
                "expected_ids=%s, actual_ids=%s): %r",
                batch_index, attempt, expected_ids, actual_ids, result,
            )
            if attempt < max_attempts:
                logger.warning(
                    "Planner batch %d failed validation (steps=%d/%d, rule_ids=%s/%s); retrying once.",
                    batch_index, actual_step_count, batch_count, actual_ids, expected_ids,
                )
            else:
                fallback = _rule_based_fallback(batch_rules)
                fallback_step_count = _planner_step_count(fallback)
                fallback_ids = _planner_rule_ids(fallback)
                if fallback_step_count != batch_count or fallback_ids != expected_ids:
                    raise ValueError(
                        f"Planner batch {batch_index}/{len(batches)} and its rule-based fallback "
                        f"failed validation: steps={fallback_step_count}/{batch_count}, "
                        f"rule_ids={fallback_ids!r}/{expected_ids!r}. No planner file was written."
                    )
                logger.warning(
                    "Planner batch %d/%d failed VLM validation after two attempts; "
                    "using a deterministic question derived from the approved rule.",
                    batch_index, len(batches),
                )
                publishable = _strip_planner_rule_ids(fallback)
                generated_batches.append(_renumber_planner_steps(publishable, step_offset))
                step_offset += fallback_step_count
                break

    result = "\n".join(generated_batches).strip()
    actual_step_count = _planner_step_count(result)
    if expected_rule_count is not None and actual_step_count != expected_rule_count:
        raise ValueError(
            f"Batched planner produced {actual_step_count} steps for {expected_rule_count} approved rules."
        )
    return result

def planner_generator(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlmModel: vlm,
    dataset_name: str,
    anomaly_rules_text: str,
    master_description_paths_by_class: dict[str, Path],
    preprocessed_image_paths_by_class: dict[str, list[Path]],
    expected_rule_count: int | None = None,
) -> None:

    """Generate planners using only master descriptions and images from this run."""
    missing_master_classes = [label for label in datacfg.class_labels if label not in master_description_paths_by_class]
    missing_image_classes = [label for label in datacfg.class_labels if not preprocessed_image_paths_by_class.get(label)]
    if missing_master_classes or missing_image_classes:
        raise ValueError(
            "Missing current-run planner artifacts: "
            f"master descriptions for {missing_master_classes}, "
            f"preprocessed images for {missing_image_classes}"
        )
    os.makedirs(datacfg.output.planners_dir,exist_ok=True)
    planner_dataset_directory = os.path.join(datacfg.output.planners_dir,dataset_name)
    os.makedirs(planner_dataset_directory,exist_ok= True)
    rng = np.random.default_rng(datacfg.reference_selection.random_seed)
    for class_directory in datacfg.class_labels:
        planner_class_label_directory = os.path.join(planner_dataset_directory,class_directory)
        os.makedirs(planner_class_label_directory,exist_ok= True)
        image_description_path = Path(master_description_paths_by_class[class_directory])
        preprocessed_images = [Path(path) for path in preprocessed_image_paths_by_class[class_directory]]
        missing_files = [
            path for path in [image_description_path, *preprocessed_images]
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Current-run planner artifacts do not exist for class {class_directory!r}: {missing_files}"
            )
        pre_processed_image_img_path = rng.choice(preprocessed_images)
        img = Image.open(pre_processed_image_img_path).convert("RGB")
        with image_description_path.open('r', encoding='utf-8') as file:
            data = json.load(file)
            logger.debug("Generating planner steps from: %s", image_description_path.name)
        result = agent_planner(datacfg=datacfg,
                               modelcfg=modelcfg,
                               vlmModel=vlmModel,
                               img=img,
                               ANOMALY_RULES=anomaly_rules_text,
                               baseline_description=data,
                               expected_rule_count=expected_rule_count)
        # Keep generated filenames portable across Linux and Windows/RDP.
        # A colon is valid in Linux filenames but invalid in Windows filenames.
        planner_file_path = os.path.join(
            planner_class_label_directory,
            f'{class_directory}_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_planner.txt',
        )
        with open(planner_file_path, 'w', encoding='utf-8') as f:
            f.write(result)
        logger.info("Planner steps saved for class '%s' to %s", class_directory, planner_file_path)
        logger.debug("Planner steps content:\n%s", result)
