"""Shared image discovery, CLIP preprocessing, and letterboxing."""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def list_images(folder: Path) -> tuple[list[str], list[str]]:
    """Return sorted image paths and filenames from one directory."""
    names = sorted(
        name for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    return [str(folder) + "/" + name for name in names], names


def preprocess_image_paths(
    paths: Sequence[str], preprocess: Callable[[Image.Image], torch.Tensor]
) -> torch.Tensor:
    """Open RGB images, apply an encoder transform, and stack the tensors."""
    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        tensors.append(preprocess(image))
    return torch.stack(tensors)


def letterbox(
    image: Image.Image,
    target_w: tuple[int, int],
    target_h: tuple[int, int],
    resample: str,
    *,
    truncate_fixed_size: bool = False,
) -> Image.Image:
    """Resize within configured bounds and pad while preserving aspect ratio."""
    try:
        resample_value = _RESAMPLE[resample.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported resample mode: {resample!r}. Expected one of {sorted(_RESAMPLE)}."
        )
    min_w, max_w = target_w
    min_h, max_h = target_h
    if min_w <= 0 or min_h <= 0:
        raise ValueError("Minimum image dimensions must be positive.")
    if min_w > max_w:
        raise ValueError("target_w minimum cannot exceed maximum.")
    if min_h > max_h:
        raise ValueError("target_h minimum cannot exceed maximum.")

    width, height = image.size
    if min_w == max_w and min_h == max_h:
        scale = min(max_w / width, max_h / height)
        size_fn = int if truncate_fixed_size else round
        resized_w = max(1, size_fn(width * scale))
        resized_h = max(1, size_fn(height * scale))
        resized = image.resize((resized_w, resized_h), resample=resample_value)
        canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
        canvas.paste(resized, ((max_w - resized_w) // 2, (max_h - resized_h) // 2))
        return canvas

    max_scale = min(max_w / width, max_h / height)
    min_scale = max(min_w / width, min_h / height)
    if min_w <= width <= max_w and min_h <= height <= max_h:
        scale = 1.0
    elif width > max_w or height > max_h:
        scale = max_scale
    else:
        scale = min_scale if min_scale <= max_scale else max_scale
    resized_w = min(max(1, round(width * scale)), max_w)
    resized_h = min(max(1, round(height * scale)), max_h)
    resized = (
        image.copy()
        if (resized_w, resized_h) == image.size
        else image.resize((resized_w, resized_h), resample=resample_value)
    )
    canvas_w, canvas_h = max(resized_w, min_w), max(resized_h, min_h)
    if (canvas_w, canvas_h) == resized.size:
        return resized
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    canvas.paste(resized, ((canvas_w - resized_w) // 2, (canvas_h - resized_h) // 2))
    return canvas
