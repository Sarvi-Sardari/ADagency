"""
utils.py (anomaly_detection)
==============================
Shared helper functions for the online anomaly-detection pipeline: image
listing/preprocessing, Jinja prompt building, and VRAM cleanup. Mirrors
planner_generation/utils.py's conventions; self-contained on purpose (see
config.py's module docstring).
"""

import gc
import logging
from pathlib import Path
from PIL import Image
import torch
from .config import ModelsConfig
from .models import Encoder, vlm
from pipeline_core.prompting import build_prompt, build_prompt_blocks, get_jinja_env
from pipeline_core.imaging import (
    letterbox as shared_letterbox,
    list_images as list_image_files,
    preprocess_image_paths,
)

logger = logging.getLogger(__name__)

def list_images(folder: Path) -> tuple[list[str], list[str]]:
    return list_image_files(folder)


def preprocess_image_for_embedding(image_paths: list[str]) -> torch.Tensor:
    return preprocess_image_paths(image_paths, Encoder.preprocess)


def letterbox(
    image: Image.Image,
    target_w: tuple[int, int],
    target_h: tuple[int, int],
    resample: str,
) -> Image.Image:
    return shared_letterbox(
        image,
        target_w,
        target_h,
        resample,
        truncate_fixed_size=True,
    )



def preprocess_img(
    modelcfg: ModelsConfig,
    src: Path,
    out_dir: Path,
    tag: str,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Path:
    """Description - Preprocessing of inference/reference images before they
    are handed to the VLM (letterbox to the configured target size).

    ``crop_box`` (left, top, right, bottom in original pixels) is applied
    BEFORE letterboxing when object cropping is enabled, so the VLM, rerun,
    and mask stages all see the same object-centred geometry. The original
    dataset image is never modified."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"processed_{tag}").with_suffix(".png")
    with Image.open(src) as im:
        img = im.convert("RGB")
        if crop_box is not None:
            img = img.crop(tuple(crop_box))
        lb = letterbox(img, modelcfg.vlm.image_processing.target_w, modelcfg.vlm.image_processing.target_h, modelcfg.vlm.image_processing.resample)
        lb.save(out)
    return out


def clear_vram() -> None:
    """
    Release GPU memory. Unlike planner_generation (which swaps between a VLM
    and an LLM), anomaly detection only ever loads one VLM, so this simply
    clears the CUDA cache/garbage between dataset runs.
    """
    if vlm.vlm_pipe_line is not None:
        vlm.vlm_pipe_line = None
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(torch.cuda.current_device()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
