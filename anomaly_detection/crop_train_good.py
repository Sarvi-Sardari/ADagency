"""Ad-hoc, optional: crop each dataset's train/good references to its object.

Why this exists
---------------
When ``datasets[].object_crop.enabled`` is true, anomaly detection crops every
TEST image on the fly to the VLM-located object (see
``anomaly_detection/object_crop.py``). Normal references in ``train/good`` are
not touched by detection. Running this script once makes the references
object-centred too, so CLIP retrieval and the reference/candidate comparison
see comparable crops. It is NOT a pipeline stage and is never run
automatically; run it only when you want cropped references.

What it does per dataset
------------------------
1. First run only: copies ``train/good`` to ``train/good_uncropped`` and
   verifies the copy byte-for-byte. Later runs always crop FROM
   ``good_uncropped``, so an already-cropped image is never cropped again.
2. Locates the object in every original with the same prompt
   (``image_crop.j2``), object labels, and padding as test-time cropping,
   writing crops to a hidden staging folder (``train/.good_object_crop_staging``)
   that doubles as a resume cache.
3. Publishes only if EVERY image succeeded: ``train/good`` is swapped for the
   cropped set in one rename and ``train/good_object_crop_manifest.json``
   records boxes and settings. If any object is not found or any response is
   malformed, the failures are reported and ``train/good`` is left untouched.
4. ``--restore`` swaps the originals back from ``good_uncropped`` (the cropped
   set is kept as ``train/.good_cropped_<timestamp>``).

Datasets are selected from the anomaly-detection config: by default those with
``object_crop.enabled: true``; ``--datasets`` selects explicitly. Nothing is
deleted except this script's own staging data and a replaced ``train/good``
that is verified identical to ``good_uncropped``.

Examples (from the project root, like ``python -m anomaly_detection.rerun``)::

    python -m anomaly_detection.crop_train_good --dry-run
    python -m anomaly_detection.crop_train_good \
        --dataset-config configs/anomaly_detection_configs.yaml --datasets single_box
    python -m anomaly_detection.crop_train_good --datasets single_box --restore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import (
    DatasetEntry,
    ModelsConfig,
    ObjectCropConfig,
    load_anomaly_detection_config,
    load_models_config,
)
from .object_crop import (
    TRAIN_CROP_MANIFEST,
    ObjectNotFoundError,
    atomic_save_image,
    locate_object,
    profile_snapshot,
    result_from_payload,
)
from pipeline_core.artifacts import atomic_write_text
from pipeline_core.imaging import IMAGE_EXTENSIONS

logger = logging.getLogger("crop_train_good")

BACKUP_NAME = "good_uncropped"
STAGING_NAME = ".good_object_crop_staging"


@dataclass(frozen=True)
class TrainPaths:
    train: Path
    good: Path
    backup: Path
    staging: Path
    manifest: Path

    @classmethod
    def for_dataset(cls, dataset: DatasetEntry) -> "TrainPaths":
        train = dataset.root_dir / "train"
        return cls(
            train=train,
            good=train / "good",
            backup=train / BACKUP_NAME,
            staging=train / STAGING_NAME,
            manifest=train / TRAIN_CROP_MANIFEST,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _dirs_identical(first: Path, second: Path) -> bool:
    """True when both folders hold the same file names with identical bytes."""
    first_files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    second_files = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    if first_files != second_files:
        return False
    return all(_sha256(first / name) == _sha256(second / name) for name in first_files)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_backup(paths: TrainPaths) -> None:
    """Create train/good_uncropped once, verified against train/good."""
    if paths.good.is_symlink():
        raise RuntimeError(
            f"{paths.good} is a symbolic link; replace it with a real directory "
            "before cropping so the linked data is never renamed or deleted."
        )
    if paths.backup.is_dir():
        if not _images(paths.backup):
            raise RuntimeError(f"{paths.backup} exists but contains no images.")
        return
    if paths.manifest.exists():
        raise RuntimeError(
            f"{paths.manifest} exists but {paths.backup} is missing, so the original "
            "uncropped references cannot be proven. Restore them manually first."
        )
    if not paths.good.is_dir() or not _images(paths.good):
        raise FileNotFoundError(f"No train/good images found in {paths.good}")
    temporary = paths.train / f".{BACKUP_NAME}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)  # our own interrupted copy, never user data
    shutil.copytree(paths.good, temporary, copy_function=shutil.copy2)
    if not _dirs_identical(paths.good, temporary):
        shutil.rmtree(temporary)
        raise RuntimeError(f"Backup verification failed for {paths.good}; nothing was changed.")
    temporary.rename(paths.backup)
    logger.info("Backed up %s -> %s", paths.good, paths.backup)


def _settings(crop_cfg: ObjectCropConfig, prompts_dir: Path, modelcfg: ModelsConfig) -> dict:
    return {**profile_snapshot(crop_cfg, prompts_dir), "model": modelcfg.vlm.modelname}


def _already_cropped(paths: TrainPaths, settings: dict) -> bool:
    if not paths.manifest.is_file() or not paths.good.is_dir():
        return False
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("settings") == settings
        and sorted(manifest.get("images", {})) == sorted(p.name for p in _images(paths.good))
    )


def _crop_one(
    model, modelcfg: ModelsConfig, source: Path, crop_cfg: ObjectCropConfig,
    prompts_dir: Path, paths: TrainPaths, settings: dict,
) -> dict:
    """Crop one original into staging, reusing a matching staged result."""
    staged_image = paths.staging / source.name
    staged_record = paths.staging / f"{source.name}.json"
    source_hash = _sha256(source)
    if staged_image.is_file() and staged_record.is_file():
        try:
            record = json.loads(staged_record.read_text(encoding="utf-8"))
            if record.get("source_sha256") == source_hash and record.get("settings") == settings:
                result_from_payload(record)
                with Image.open(staged_image) as opened:
                    opened.verify()
                return record
        except (OSError, ValueError, KeyError, TypeError):
            pass  # stale/corrupt staging entry: recompute below

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    result = locate_object(
        model, modelcfg, image, crop_cfg, prompts_dir,
        diagnostic_dir=paths.staging / f"{source.name}.diagnostics",
        image_label=str(source),
    )
    atomic_save_image(staged_image, image.crop(result.box_pixels))
    record = {
        **result.summary_record(),
        "source_image": source.name,
        "source_sha256": source_hash,
        "attempts": result.attempts,
        "response": result.response,
        "settings": settings,
    }
    atomic_write_text(staged_record, json.dumps(record, indent=2, ensure_ascii=False))
    return record


def _publish(paths: TrainPaths, records: dict[str, dict], settings: dict) -> None:
    """Swap train/good for the staged crops in a single directory rename."""
    new_dir = paths.train / ".good_object_crop_new"
    if new_dir.exists():
        shutil.rmtree(new_dir)  # our own interrupted publish directory
    new_dir.mkdir()
    image_names = set(records)
    for item in paths.backup.iterdir():
        if item.name in image_names:
            shutil.copy2(paths.staging / item.name, new_dir / item.name)
        elif item.is_file():
            shutil.copy2(item, new_dir / item.name)  # keep non-image side files
        else:
            shutil.copytree(item, new_dir / item.name, copy_function=shutil.copy2)

    previous = paths.train / f".good_replaced_{_timestamp()}"
    paths.good.rename(previous)
    new_dir.rename(paths.good)
    crop_times = [
        float(record["inference_time_sec"]) for record in records.values()
        if record.get("inference_time_sec") is not None
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dir": BACKUP_NAME,
        "settings": settings,
        "num_images": len(records),
        "total_object_crop_time_sec": round(sum(crop_times), 4),
        "avg_object_crop_time_per_image_sec": (
            round(sum(crop_times) / len(crop_times), 4) if crop_times else None
        ),
        "images": {
            name: {
                key: record.get(key)
                for key in ("box_1000", "box_pixels", "source_size", "label",
                            "attempts", "inference_time_sec")
            }
            for name, record in sorted(records.items())
        },
    }
    atomic_write_text(paths.manifest, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # The replaced folder is deleted only when it is provably the untouched
    # original (identical to the verified backup); otherwise it is kept.
    if _dirs_identical(previous, paths.backup):
        shutil.rmtree(previous)
    else:
        logger.warning("Kept the replaced train/good contents at %s", previous)
    shutil.rmtree(paths.staging)


def crop_dataset(
    dataset: DatasetEntry, model, modelcfg: ModelsConfig, prompts_dir: Path, *, force: bool,
) -> bool:
    """Return True when train/good is (now) cropped with the current settings."""
    crop_cfg = dataset.object_crop
    assert crop_cfg is not None and crop_cfg.object_labels
    paths = TrainPaths.for_dataset(dataset)
    settings = _settings(crop_cfg, prompts_dir, modelcfg)
    if not force and _already_cropped(paths, settings):
        logger.info("[%s] train/good is already cropped with identical settings; skipping.", dataset.name)
        return True

    _ensure_backup(paths)
    sources = _images(paths.backup)
    records: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for index, source in enumerate(sources, start=1):
        print(f"[{dataset.name}] {index}/{len(sources)} {source.name}", flush=True)
        try:
            records[source.name] = _crop_one(
                model, modelcfg, source, crop_cfg, prompts_dir, paths, settings
            )
        except ObjectNotFoundError as error:
            failures.append((source.name, f"OBJECT NOT FOUND - {error}"))
        except ValueError as error:
            failures.append((source.name, f"INVALID MODEL OUTPUT - {error}"))

    if failures:
        print(f"\n[{dataset.name}] {len(failures)} of {len(sources)} image(s) failed; "
              f"train/good was NOT changed. Completed crops are cached in {paths.staging}.")
        for name, message in failures:
            print(f"  - {name}: {message}")
        return False

    _publish(paths, records, settings)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    average = manifest.get("avg_object_crop_time_per_image_sec")
    print(f"[{dataset.name}] Cropped {len(records)} references into {paths.good} "
          f"(originals: {paths.backup}; manifest: {paths.manifest}; avg crop time "
          f"{f'{average:.2f}s' if average is not None else 'n/a'}/image).")
    return True


def restore_dataset(dataset: DatasetEntry) -> None:
    """Put the originals back into train/good, keeping the cropped set aside."""
    paths = TrainPaths.for_dataset(dataset)
    if not paths.backup.is_dir():
        raise FileNotFoundError(f"[{dataset.name}] nothing to restore: {paths.backup} is missing.")
    stamp = _timestamp()
    temporary = paths.train / ".good_restore.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(paths.backup, temporary, copy_function=shutil.copy2)
    kept = None
    if paths.good.exists():
        kept = paths.train / f".good_cropped_{stamp}"
        paths.good.rename(kept)
    temporary.rename(paths.good)
    if paths.manifest.exists():
        paths.manifest.rename(paths.train / f"good_object_crop_manifest.restored_{stamp}.json")
    print(f"[{dataset.name}] Restored originals into {paths.good}"
          + (f"; cropped set kept at {kept}" if kept else "") + ".")


def _select_datasets(datasets: list[DatasetEntry], names: list[str] | None) -> list[DatasetEntry]:
    if names:
        by_name = {dataset.name: dataset for dataset in datasets}
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise ValueError(f"Datasets not in the config: {missing}")
        return [by_name[name] for name in names]
    return [d for d in datasets if d.object_crop is not None and d.object_crop.enabled]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ad-hoc: back up train/good to train/good_uncropped and replace it "
                    "with VLM object crops (same settings as test-time object_crop).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-config", default="configs/anomaly_detection_configs.yaml",
                        help="Anomaly-detection config providing datasets and object_crop settings")
    parser.add_argument("--model-config", default="configs/model_configs.yaml",
                        help="Model config for the VLM that locates the object")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="Datasets to process (default: those with object_crop.enabled)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without loading a model or changing files")
    parser.add_argument("--restore", action="store_true",
                        help="Restore train/good from train/good_uncropped")
    parser.add_argument("--force", action="store_true",
                        help="Re-crop even if the manifest already matches the current settings")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    datacfg = load_anomaly_detection_config(args.dataset_config)
    selected = _select_datasets(datacfg.datasets, args.datasets)
    if not selected:
        print("No datasets selected: enable object_crop for a dataset or pass --datasets.")
        return 1

    print("Datasets:")
    for dataset in selected:
        paths = TrainPaths.for_dataset(dataset)
        crop_cfg = dataset.object_crop
        state = "cropped (manifest present)" if paths.manifest.is_file() else "uncropped"
        good_count = len(_images(paths.good)) if paths.good.is_dir() else 0
        print(f"  - {dataset.name}: {paths.good} [{good_count} images, {state}]"
              f"; backup {'present' if paths.backup.is_dir() else 'will be created'}")
        if not args.restore:
            if crop_cfg is None or not crop_cfg.object_labels:
                raise ValueError(f"{dataset.name}: no object labels ({crop_cfg.labels_file if crop_cfg else '?'})")
            if not crop_cfg.enabled:
                print("    WARNING: object_crop is disabled for this dataset, so its test "
                      "images will NOT be cropped during detection.")
            print(f"    labels={list(crop_cfg.object_labels)} padding_ratio={crop_cfg.padding_ratio}")
    if args.dry_run:
        return 0
    action = "restore the originals of" if args.restore else "replace train/good with object crops for"
    if not args.yes and input(f"\nThis will {action} the datasets above. Type 'yes' to continue: ").strip().lower() != "yes":
        print("Cancelled; nothing was changed.")
        return 1

    if args.restore:
        for dataset in selected:
            restore_dataset(dataset)
        return 0

    from anomaly_detection.models import vlm
    from cluster_deployment.server_manager import managed_server_lifecycle

    modelcfg = load_models_config(args.model_config)
    all_ok = True
    with managed_server_lifecycle("train_good_object_crop"):
        vlm.load(modelcfg)
        try:
            for dataset in selected:
                all_ok = crop_dataset(dataset, vlm, modelcfg, datacfg.prompts_dir, force=args.force) and all_ok
        finally:
            vlm.release()
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
