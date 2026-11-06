"""Model-independent conversion helpers used by mask backends."""

import json
import re
from typing import Any

import numpy as np

from pipeline_core.model_output import normalize_model_output

def _normalize_box(raw: Any) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    # Some otherwise compatible VLMs use the common 0..1 convention despite
    # the prompt requesting 0..1000. Detect that unambiguously before clamping
    # so their boxes do not collapse into a one-pixel corner mask.
    if values and min(values) >= 0.0 and max(values) <= 1.0:
        values = [value * 1000.0 for value in values]
    x0, y0, x1, y1 = (min(1000.0, max(0.0, value)) for value in values)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _parse_boxes(text: str) -> tuple[list[list[float]], list[str]]:
    def decode(value_text: str) -> Any:
        cleaned = normalize_model_output(value_text)
        cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.I).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                import json_repair
                return json_repair.loads(cleaned)
            except Exception as error:
                raise ValueError(f"Could not parse bounding-box JSON: {text}") from error

    value = decode(text)
    # Some VLMs occasionally JSON-encode the requested JSON a second time,
    # yielding a top-level string such as
    # "[{\"label\": \"scratch\", \"box\": [1, 2, 3, 4]}]".  Unwrap a
    # small, bounded number of string layers, while retaining the strict type
    # check below for ordinary prose or other invalid responses.
    for _ in range(2):
        if not isinstance(value, str):
            break
        decoded = decode(value)
        if decoded == value:
            break
        value = decoded
    if isinstance(value, dict):
        value = value.get("boxes", [value])
    if not isinstance(value, list):
        raise ValueError(f"Bounding-box response must be a JSON list, got {type(value).__name__}")

    boxes: list[list[float]] = []
    labels: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        box = _normalize_box(item.get("box"))
        if box is not None:
            boxes.append(box)
            labels.append(str(item.get("label") or "anomaly"))
    return boxes, labels


def _boxes_to_mask(boxes: list[list[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        left = max(0, min(width - 1, int(np.floor(x0 * width / 1000.0))))
        top = max(0, min(height - 1, int(np.floor(y0 * height / 1000.0))))
        right = max(left + 1, min(width, int(np.ceil(x1 * width / 1000.0))))
        bottom = max(top + 1, min(height, int(np.ceil(y1 * height / 1000.0))))
        mask[top:bottom, left:right] = 1.0
    return mask

def _segmentation_output_to_map(output: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Converts a SAM3-style output dict (masks_logits + scores) into a single
    [H, W] anomaly score map. A brand-new backend whose output doesn't match
    this shape should register its own generator instead of reusing this."""
    import torch
    import torch.nn.functional as functional

    masks = output.get("masks_logits")
    scores = output.get("scores")
    if masks is None or not isinstance(masks, torch.Tensor) or masks.numel() == 0:
        return np.zeros((height, width), dtype=np.float32)
    masks = masks.detach().float()
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.shape[-2:] != (height, width):
        masks = functional.interpolate(
            masks.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
    masks = masks.clamp(0, 1)
    if isinstance(scores, torch.Tensor) and scores.numel() == masks.shape[0]:
        masks = masks * scores.detach().float().reshape(-1, 1, 1).clamp(0, 1)
    return masks.max(dim=0).values.cpu().numpy().astype(np.float32)
