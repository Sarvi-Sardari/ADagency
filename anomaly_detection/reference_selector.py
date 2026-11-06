"""
reference.py (anomaly_detection)
==================================
CLIP-based reference-image selection and class matching for the online
anomaly-detection pipeline.

Unlike planner_generation/reference.py, there is no reference_image_creator()
equivalent here: the online pipeline doesn't split train/good into per-class
folders — it embeds the whole train/good pool once and, for every candidate
image, finds the nearest reference image by cosine distance. For multi-class
datasets, that nearest reference embedding is then matched against the
dataset's `class_labels` text descriptions (via CLIP text-image similarity)
to decide which class's planner to use.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import pairwise_distances

from .models import Encoder
from .utils import preprocess_image_for_embedding

logger = logging.getLogger(__name__)


def compute_ref_embeddings(encoder: Encoder, ref_img_tensors: torch.Tensor) -> np.ndarray:
    """Description - Encodes the preprocessed reference-image tensors into
    L2-normalised CLIP embeddings.\n
    Output - numpy array of shape (N, 512)."""
    with torch.no_grad():
        ref_emb = Encoder.model.encode_image(ref_img_tensors.to(encoder.device))
        ref_emb = ref_emb / ref_emb.norm(dim=1, keepdim=True)
    return ref_emb.cpu().numpy()

def inference_ref_img_selector(
    encoder: type[Encoder],
    ref_emb_np: np.ndarray,
    cand_img_path: Path,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Description - Embeds the candidate image and returns the index of the
    closest reference image (by cosine distance).

    With object cropping enabled, ``crop_box`` restricts the CLIP embedding to
    the located object so retrieval compares object with (optionally
    anomaly_detection/crop_train_good.py-cropped) object rather than background with background."""
    if crop_box is None:
        cand_tensor = preprocess_image_for_embedding([cand_img_path]).to(encoder.device)
    else:
        with Image.open(cand_img_path) as opened:
            cropped = opened.convert("RGB").crop(tuple(crop_box))
        cand_tensor = Encoder.preprocess(cropped).unsqueeze(0).to(encoder.device)
    with torch.no_grad():
        cand_emb = encoder.model.encode_image(cand_tensor)
        cand_emb = cand_emb / cand_emb.norm(dim=1, keepdim=True)
    cand_emb_np = cand_emb.cpu().numpy() 
    del cand_tensor, cand_emb
    torch.cuda.empty_cache()
    distances = pairwise_distances(cand_emb_np, ref_emb_np, metric="cosine")
    return np.argmin(distances, axis=1)
