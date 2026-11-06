"""Discover or load the object names used to construct inspection planners."""

from PIL import Image
import re
import json
import ast
import logging
from pathlib import Path
from rich.console import Console
from .config import DatasetConfig, ModelsConfig
from .models import vlm
from .utils import build_prompt
from pipeline_core.model_output import normalize_model_output

logger = logging.getLogger(__name__)

def object_namer(datacfg:DatasetConfig, modelcfg: ModelsConfig, vlmModel: vlm, img_path:Path)->list[str]:
   
   """description: This function takes the image path and understand the scene of the image and return the list of objects present in a image.\n
   Input: cfg:dataclass ,img_path: filepath of the image (usually a good or anomaly free image is given)\n
   Output: list of string ie object names present in the images."""
   
   img = Image.open(img_path).convert("RGB")
   prompt = build_prompt(datacfg.prompts_dir, "object_namer.j2")
   out, vlm_time = vlmModel.run_vlm(modelcfg.vlm,[img],prompt)
   logger.info("Object names generated for %s in %.3fs", img_path, vlm_time)
   object_names = out[1]['content'] #(Hari's old code)
   # raw_output = out[1]['content']
   # object_names = _parse_object_name_list(raw_output)
   return object_names

def _parse_object_name_list(raw: str) -> list[str]:
    """
    Robustly parse a VLM response that should be a Python list of strings.
    Strips markdown code fences and surrounding commentary before evaluating.
    """
    # 1. If it's already a Python list, validate elements and return directly
    if isinstance(raw, list):
        if all(isinstance(x, str) for x in raw):
            return raw
        raise ValueError(f"List contains non-string elements: {raw!r}")
    # 2. Otherwise, treat as string and parse
    text = normalize_model_output(raw)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = json.loads(text)

    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"VLM object-name output is not a list[str]: {parsed!r}")
    
    return parsed


def _prompt_user_to_select_labels(candidate_labels: list[str], console: Console | None = None) -> list[str] | None:
    """
    Ask the user to choose which labels to keep from a VLM-generated list.

    Input format
    ------------
    - Enter: keep all labels
    - all: keep all labels
    - 1,3,5: keep specific indices
    - 2-4,7: keep index ranges + singles

    Returns
    -------
    list[str] | None
        Selected labels in original order with duplicates removed. ``None`` means
        the user chose to quit label generation.
    """
    if not candidate_labels:
        raise ValueError("No candidate labels were provided for selection.")

    prompt_console = console or Console()
    prompt_console.print("\n  [Object Namer] Candidate labels from VLM:")
    for idx, label in enumerate(candidate_labels, start=1):
        prompt_console.print(f"    {idx:>2}. {label}")

    prompt_text = (
        "\n  Select labels to keep "
        "(all (or) 1,3,5 (or) 2-4,7; q/quit to stop) "
        "(Note: If no inputs are given, all labels are selected by default): "
    )

    while True:
        user_input = prompt_console.input(prompt_text).strip().lower()

        if user_input in {"q", "quit"}:
            return None

        # Fast path: default or explicit full selection.
        if user_input in {"", "all"}:
            return list(dict.fromkeys(candidate_labels))

        chosen_indices: list[int] = []
        valid = True

        for token in [tok.strip() for tok in user_input.split(",") if tok.strip()]:
            if "-" in token:
                parts = [p.strip() for p in token.split("-", 1)]
                if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                    valid = False
                    break
                start_i, end_i = int(parts[0]), int(parts[1])
                if start_i > end_i:
                    valid = False
                    break
                chosen_indices.extend(range(start_i, end_i + 1))
            else:
                if not token.isdigit():
                    valid = False
                    break
                chosen_indices.append(int(token))

        # Reject malformed or out-of-bound selections and reprompt.
        if not valid or not chosen_indices:
            prompt_console.print("  Invalid selection format. Example: 1,3,5 or 2-4,7")
            continue

        max_idx = len(candidate_labels)
        if any(i < 1 or i > max_idx for i in chosen_indices):
            prompt_console.print(f"  Invalid index. Choose numbers between 1 and {max_idx}.")
            continue

        seen: set[int] = set()
        ordered_unique = [i for i in chosen_indices if not (i in seen or seen.add(i))]
        selected = [candidate_labels[i - 1] for i in ordered_unique]

        if not selected:
            prompt_console.print("  No labels selected. Please choose at least one label.")
            continue

        logger.info("[Object Namer] Selected %d labels: %s", len(selected), selected)
        return selected


def load_or_generate_object_names(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlmModel: vlm,
    label_example_img_path: Path,
    PROJECT_ROOT: Path,
    console: Console | None = None,
) -> list[str]:
    """
    Return the list of visible object names for this dataset.

    Behaviour
    ---------
        - If ``use_previous_labels`` is True and a cached labels file exists at
            ``prompts/<dataset_name>_object_labels.txt``, the cached names are returned
            immediately with no VLM call.
        - Otherwise, the VLM repeatedly identifies object names from
            ``label_example_img_path``. After each generation, selected labels are
            appended to the labels file. Generation continues until the user enters
            ``q`` or ``quit``.

    The labels file format is one object name per line.
    """
    labels_file = PROJECT_ROOT / "prompts" / "object_labels" / f"{datacfg.name}_object_labels.txt"

    if datacfg.object_labels.use_previous_labels and labels_file.exists():
        names = [ln.strip() for ln in labels_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        logger.info("[Object Namer] Loaded %d object labels from cache: %s", len(names), labels_file)
        return names

    labels_file.parent.mkdir(parents=True, exist_ok=True)
    accumulated_names = (
        [ln.strip() for ln in labels_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if labels_file.exists()
        else []
    )
    accumulated_names = list(dict.fromkeys(accumulated_names))

    while True:
        logger.info("[Object Namer] Generating another object-label candidate batch …")
        raw = object_namer(datacfg, modelcfg, vlmModel, label_example_img_path)

        # object_namer() returns the raw VLM content string; parse into a Python list
        candidate_names = raw if isinstance(raw, list) else _parse_object_name_list(raw)

        # Let the user decide which labels are relevant before appending to cache.
        if console is not None and console.is_terminal:
            # The alternate screen is restored on exit, removing the prompt and
            # selection history before the pipeline progress bar resumes.
            with console.screen():
                selected_names = _prompt_user_to_select_labels(candidate_names, console=console)
        else:
            selected_names = _prompt_user_to_select_labels(candidate_names, console=console)

        if selected_names is None:
            break

        new_names = [name for name in selected_names if name not in accumulated_names]
        if new_names:
            with labels_file.open("a", encoding="utf-8") as labels_handle:
                if labels_file.stat().st_size:
                    labels_handle.write("\n")
                labels_handle.write("\n".join(new_names))
            accumulated_names.extend(new_names)
            logger.info("[Object Namer] Appended %d labels to %s", len(new_names), labels_file)
        else:
            logger.info("[Object Namer] Selection contained no new labels.")

        continue_prompt = (console or Console()).input(
            "\n  Press Enter to generate another label batch, or enter q/quit to finish: "
        ).strip().lower()
        if continue_prompt in {"q", "quit"}:
            break

    if not accumulated_names:
        raise ValueError("No object labels were selected before quitting.")
    return accumulated_names
