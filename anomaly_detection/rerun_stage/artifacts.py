"""Artifact reading, path resolution, and collision-safe identities."""

import logging
from pathlib import Path
from typing import Any

from pipeline_core.artifacts import read_jsonl
from ..config import resolve_portable_file

logger = logging.getLogger(__name__)

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries. Malformed lines are skipped with a warning."""
    return read_jsonl(path, logger=logger)


def _resolve_probs_path(raw_path: str | Path, dataset_output_dir: Path) -> Path:
    """
    Resolve a raw path to a probability data file. The path may be:
      - Absolute: returned as-is
      - Relative to dataset output dir (e.g., \"texts/good/000.png_prob_data.txt\"): preferred format
      - Legacy project-root-relative (e.g., \"outputs/runs/.../capsule/texts/...\"): try both dataset_output_dir and parent
    """
    return resolve_portable_file(
        raw_path,
        dataset_output_dir,
        dataset_output_dir.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path.cwd(),
    )


def _resolve_dataset_artifact(
    raw_path: str | Path, dataset_output_dir: Path, project_root: Path
) -> Path:
    """Resolve an artifact whose saved absolute run root may have been moved.

    Completed runs are sometimes renamed for experiment bookkeeping. Their
    summaries can consequently contain an obsolete absolute prefix even though
    the dataset-relative suffix remains valid in the current run directory.
    """
    resolved = resolve_portable_file(raw_path, dataset_output_dir, project_root)
    if resolved.is_file():
        return resolved
    recorded = Path(raw_path).expanduser()
    parts = recorded.parts
    matching = [index for index, part in enumerate(parts) if part == dataset_output_dir.name]
    if matching:
        relocated = dataset_output_dir.joinpath(*parts[matching[-1] + 1 :])
        if relocated.is_file():
            return relocated.resolve()
    return resolved


def _img_key(entry: dict[str, Any]) -> str:
    """Same img_name identity used by all_data_summary.jsonl (see metrics.py's _img_key)."""
    return entry.get("img_name") or Path(entry["decision_filepath"]).name.removesuffix(".txt")


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Composite identity used to match a resumed rerun_result.jsonl entry back
    to its all_data_summary.jsonl entry. Plain img_name alone is NOT unique
    when merge_anomalies groups several real test subfolders under one
    label_folder (e.g. two different defect subfolders can each contain a
    '000.png') -- source_label_folder disambiguates that case."""
    label_folder = str(entry.get("label_folder", ""))
    # Old non-merged summaries did not store source_label_folder. Falling back
    # to label_folder preserves their resume identity without introducing a
    # duplicate label/label output directory.
    source_label_folder = str(entry.get("source_label_folder") or label_folder)
    return (label_folder, source_label_folder, _img_key(entry))

