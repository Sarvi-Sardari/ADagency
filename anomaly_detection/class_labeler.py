"""
class_labeler.py (anomaly_detection)
====================================
Load cached class-label descriptions for multi-class datasets, or generate
new ones from the user's ``train/good`` ``class_examples`` with VLM + user
approval and persist them for future runs. These semantic labels select
planners only; they do not define anomaly output folders.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from PIL import Image
from rich.console import Console
from .config import AnomalyDetectionConfig, DatasetEntry, ModelsConfig
from .models import vlm
from .utils import build_prompt

logger = logging.getLogger(__name__)


def _labels_file_path(project_root: Path, dataset_name: str) -> Path:
    return project_root / "prompts" / "class_labels" / f"{dataset_name}_data_labels.txt"


def _parse_cached_class_labels(path: Path) -> dict[str, str]:
    """Parse cached labels file lines in `label: description` format."""
    labels: dict[str, str] = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid line {idx} in {path}: expected 'label: description'.")
        label, desc = line.split(":", 1)
        label = label.strip()
        desc = desc.strip()
        if not label or not desc:
            raise ValueError(f"Invalid line {idx} in {path}: empty label/description.")
        labels[label] = desc

    if not labels:
        raise ValueError(f"No class labels found in {path}")

    return labels


def _parse_generated_label_list(raw: str | list[str]) -> list[str]:
    """Parse a VLM response expected to be a list of strings."""
    if isinstance(raw, list):
        if all(isinstance(x, str) and x.strip() for x in raw):
            return [x.strip() for x in raw]
        raise ValueError(f"Generated labels contain invalid entries: {raw!r}")

    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\\n?", "", text)
    text = re.sub(r"\\n?```$", "", text).strip()

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = json.loads(text)

    if not isinstance(parsed, list) or not all(isinstance(x, str) and x.strip() for x in parsed):
        raise ValueError(f"Generated labels are not list[str]: {parsed!r}")

    return [x.strip() for x in parsed]


def _generate_class_label_descr_from_examples(
    dataset: DatasetEntry,
    modelcfg: ModelsConfig,
    datacfg: AnomalyDetectionConfig,
    vlmModel: "type[vlm]",
    regeneration_count: int = 0,
) -> dict[str, str]:
    """
    Generate one free-text class description per class label using example
    images from dataset.class_examples.
    """
    if not dataset.class_examples:
        raise ValueError(
            f"{dataset.name}: class_examples are required to auto-generate class labels. "
            "Either add datasets[].class_examples or provide cached labels file."
        )

    label_names = list(dataset.class_examples.keys())
    example_paths = list(dataset.class_examples.values())
    logger.info(
        "[%s] Class-label examples (train/good): %s",
        dataset.name,
        [str(path) for path in example_paths],
    )
    images = [Image.open(path).convert("RGB") for path in example_paths]
    # The configured class names are used only by the caller to map output
    # positions back to classes. Do not put them in the VLM prompt: doing so
    # encourages the model to echo names instead of describing the images.
    prompt = build_prompt(datacfg.prompts_dir, "label_generator.j2")

    # Keep the first proposal reproducible. Regeneration explicitly requests
    # sampling so pressing r can produce a genuinely different proposal.
    output_text, _, _ = vlmModel.run_vlm(
        modelcfg,
        images,
        prompt,
        do_sample=regeneration_count > 0,
        temperature=0.7,
        require_confidence=False,
    )
    generated = _parse_generated_label_list(output_text)

    if len(generated) != len(label_names):
        raise ValueError(
            f"{dataset.name}: generated {len(generated)} labels for {len(label_names)} classes. "
            "Regenerate labels or check prompt/model output."
        )

    return {label: desc for label, desc in zip(label_names, generated)}


def _ask_user_accept_or_regenerate(
    labels_map: dict[str, str],
    example_paths: dict[str, Path],
    regeneration_count: int,
    console: Console | None = None,
) -> str:
    """Return one of: 'accept', 'regenerate', 'quit'."""
    if console is not None:
        console.clear()
        console.print(f"[bold]Class Labeler — regeneration count: {regeneration_count}[/bold]")
        console.print("[dim]Source images (resolved from train/good):[/dim]")
        for label, path in example_paths.items():
            console.print(f"  {label}: {path}")
        console.print("\n[bold]Proposed class labels:[/bold]")
    else:
        print(f"\nClass Labeler — regeneration count: {regeneration_count}")
        print("Source images (resolved from train/good):")
        for label, path in example_paths.items():
            print(f"  {label}: {path}")
        print("\nProposed class labels:")

    for idx, (label, desc) in enumerate(labels_map.items(), start=1):
        line = f"  {idx:>2}. {label}: {desc}"
        if console is not None:
            console.print(line)
        else:
            print(line)

    # Rich treats square brackets as markup tags. Escape them only when the
    # shared Rich console renders the prompt; plain input() needs the normal
    # spelling without the escape character.
    prompt_text = (
        "\nChoose: \\[a]ccept, \\[r]egenerate, \\[q]uit: "
        if console is not None
        else "\nChoose: [a]ccept, [r]egenerate, [q]uit: "
    )
    while True:
        choice = (console.input(prompt_text) if console is not None else input(prompt_text)).strip().lower()
        if choice in {"a", "accept", ""}:
            return "accept"
        if choice in {"r", "regen", "regenerate"}:
            return "regenerate"
        if choice in {"q", "quit", "exit"}:
            return "quit"
        if console is not None:
            console.print("Invalid choice. Enter a, r, or q.")
        else:
            print("Invalid choice. Enter a, r, or q.")


def _save_labels_file(path: Path, labels_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{label}: {desc}" for label, desc in labels_map.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cached_class_labels(
    dataset: DatasetEntry,
    project_root: Path,
) -> dict[str, str] | None:
    """Load an existing class-label cache without generation or user input.

    Persistent services must fail during preflight when a multi-class recipe
    is incomplete; they cannot safely open the batch runner's interactive
    accept/regenerate prompt after the web server starts.
    """
    if dataset.classes == "single":
        return None
    labels_file = _labels_file_path(Path(project_root), dataset.name)
    if not labels_file.is_file():
        raise FileNotFoundError(
            f"{dataset.name}: live inference requires the existing class-label "
            f"cache {labels_file}. Run the batch anomaly pipeline's label "
            "approval workflow before starting the web application."
        )
    return _parse_cached_class_labels(labels_file)


def load_or_generate_class_labels(
    dataset: DatasetEntry,
    modelcfg: ModelsConfig,
    vlmModel: type[vlm],
    datacfg: AnomalyDetectionConfig,
    project_root: Path,
    console: Console | None = None,
) -> dict[str, str] | None:
    """
    Return class-label descriptions for a dataset.

    Behavior for classes == 'multi':
    - If cached labels file exists, load and return it.
    - Else generate labels from class_examples and ask user to accept/regenerate/quit.
      Accepted labels are saved to prompts/class_labels/<dataset>_data_labels.txt.

    Raises when labels are required but cannot be obtained.
    """
    if dataset.classes == "single":
        return None

    labels_file = _labels_file_path(project_root, dataset.name)

    if labels_file.exists():
        loaded = load_cached_class_labels(dataset, project_root)
        assert loaded is not None
        logger.info("[%s] Loaded class labels from cache: %s", dataset.name, labels_file)
        return loaded

    regeneration_count = 0
    while True:
        generated = _generate_class_label_descr_from_examples(
            dataset=dataset,
            modelcfg=modelcfg,
            datacfg=datacfg,
            vlmModel=vlmModel,
            regeneration_count=regeneration_count,
        )
        action = _ask_user_accept_or_regenerate(
            generated,
            dataset.class_examples or {},
            regeneration_count,
            console=console,
        )

        if action == "accept":
            _save_labels_file(labels_file, generated)
            logger.info("[%s] Saved class labels to %s", dataset.name, labels_file)
            return generated

        if action == "quit":
            if labels_file.exists():
                loaded = _parse_cached_class_labels(labels_file)
                logger.info(
                    "[%s] User quit generation, but labels file now exists; loading cache: %s",
                    dataset.name,
                    labels_file,
                )
                return loaded
            raise RuntimeError(
                f"{dataset.name}: class labels are required for multi-class inference. "
                f"No cache file found at {labels_file}; user quit generation."
            )

        regeneration_count += 1
