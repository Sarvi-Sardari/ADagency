"""Mask ensembling, expansion, and display normalization."""

import time

import torch
import torch.nn.functional as F

from ..igos_core_methods_helper import upscale_superpixels

def run_igos_ensemble(method, args, seeds, **kwargs):
    """
    Runs iGOS once per seed and averages the mask in its NATIVE optimization
    representation:

      - segmented mode: compact [1, 1, 1, K] region mask
      - dense mode:     dense [1, 1, H_p, W_p] patch-grid mask

    This allows the same ensemble and phrase-union logic to work with and
    without SLIC/SAM3 segmentation.
    """
    native_masks = []
    per_seed_results = []
    last_result = None

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        t0_seed = time.time()
        result = method(args, **kwargs)
        t1_seed = time.time()

        # iGOS_p returns:
        # final_mask, losses..., loss_comb_del, compact_mask
        final_mask = result[0]
        compact_mask = result[-1]

        # Segmented runs should be averaged before region expansion.
        # Dense runs already return their native patch-grid mask.
        native_mask = (
            compact_mask
            if compact_mask is not None
            else final_mask
        )

        if native_mask is not None:
            native_masks.append(native_mask.detach())

        per_seed_results.append(
            (seed, result, round(t1_seed - t0_seed, 2))
        )
        last_result = result

    avg_native_mask = (
        torch.stack(native_masks, dim=0).mean(dim=0)
        if native_masks
        else None
    )

    return avg_native_mask, last_result, per_seed_results

def expand_native_visibility_mask(
    native_mask,
    use_segmentation,
    sp_map_patch,
    sp_map_rgb,
    target_height,
    target_width,
    patch_height=None,
    patch_width=None,
):
    """
    Converts the optimizer's native visibility mask into:

      1. Qwen patch-grid resolution for deletion/insertion metrics.
      2. RGB-image resolution for heatmap visualization.

    Mask convention:
        1 = candidate remains visible
        0 = candidate is replaced by the blur baseline
    """
    if use_segmentation:
        # Compact region mask -> Qwen patch grid.
        patch_visibility = upscale_superpixels(
            native_mask,
            sp_map_patch,
        ).detach()

        # Compact region mask -> full RGB resolution.
        rgb_visibility = upscale_superpixels(
            native_mask,
            sp_map_rgb,
        )
    else:
        # Normal dense ablations optimize directly on the Qwen patch grid.
        # The paper profile instead optimizes the repository's 32x32 mask and
        # bilinearly expands it to both Qwen's grid and the rendered image.
        if patch_height is None or patch_width is None:
            patch_visibility = native_mask.detach()
        else:
            patch_visibility = F.interpolate(
                native_mask,
                size=(int(patch_height), int(patch_width)),
                mode="bilinear",
                align_corners=False,
            ).detach()

        # Bilinear interpolation recreates the original non-superpixel
        # smooth patch-mask visualization.
        rgb_visibility = F.interpolate(
            native_mask,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    return patch_visibility, rgb_visibility


def normalize_display_score(score: torch.Tensor) -> torch.Tensor:
    """Match save_heatmaps()'s per-map 5th/99th-percentile normalization.

    Individual keyword and phrase maps can have very different raw scales
    (especially with SAM3's compact region masks). Fusing those raw values
    with ``max`` silently lets the larger-scale map erase the other one even
    when its separately rendered heatmap is clearly hot. Normalize before
    fusion so the visual and final-map semantics agree.
    """
    flat = score.detach().reshape(-1)
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.99)
    spread = torch.clamp(hi - lo, min=0.05)
    return ((score - lo) / spread).clamp(0.0, 1.0)

