"""Prompt rendering, image preprocessing, and model-lifecycle helpers."""

import gc
import logging
from pathlib import Path
from PIL import Image
import torch
from .config import ModelsConfig
from .models import Encoder, llm, vlm
from pipeline_core.prompting import build_prompt, build_prompt_blocks, get_jinja_env
from pipeline_core.imaging import (
    letterbox as shared_letterbox,
    list_images as list_image_files,
    preprocess_image_paths,
)

logger = logging.getLogger(__name__)

def list_images(folder: Path) -> tuple[list[str], list[str]]:
    return list_image_files(folder)


def preprocess_image_for_embedding(
    encodercfg: Encoder, image_paths: list[str]
) -> torch.Tensor:
    return preprocess_image_paths(image_paths, encodercfg.preprocess)


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
        truncate_fixed_size=False,
    )



def preprocess_img(modelcfg: ModelsConfig, src: Path, out_dir: Path, tag: str) -> Path:

    """Description - Preporcessing of inference images\n
    Input: cfg: dataclass,src: filepath of image, out_dir: final output directory, tag: actual image filename with its file extension\n
    Output: out: processed image filepath"""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"processed_{tag}").with_suffix(".png")
    with Image.open(src) as im:
        img = im.convert("RGB")
        lb = letterbox(img, modelcfg.vlm.image_processing.target_w, modelcfg.vlm.image_processing.target_h, modelcfg.vlm.image_processing.resample)
        lb.save(out)
    return out

def clear_vram() -> None:
    """
    Release GPU memory. Used between model swaps (VLM → LLM → VLM).
    Uses the class-level pipeline handles on the vlm / llm singletons.
    """
    if vlm.vlm_pipe_line is not None:
        vlm.vlm_pipe_line = None
    if llm.llm_pipe_line is not None:
        llm.llm_pipe_line = None
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(torch.cuda.current_device()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
