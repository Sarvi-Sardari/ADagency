"""Paper-faithful single-mask iGOS optimizer for the XAI ablation stage.

This module intentionally contains only the method described by equations
(6)--(8) of *Where do Large Vision-Language Models Look at when Answering
Questions?*.  It is isolated from the later SLIC/SAM3/phrase experiments so a
config-selected paper run cannot inherit their optimization shortcuts.

The model adapter is the only architecture-specific boundary.  The objective,
mask resolution, interpolation, noise, regularizers, GNC decay, and NAG update
below follow the paper and the canonical ``LVLM_Interpretation_Orig/methods.py``
implementation.  In particular, insertion swaps image and blurred baseline;
the later ``Advanced_IGOS_PP`` Qwen wrapper accidentally repeats the deletion
direction in those calls and is therefore not used as the mathematical source.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def _phi(image: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Equation (2): preserve ``image`` where mask=1, use baseline where mask=0."""
    return image.mul(mask) + baseline.mul(1.0 - mask)


def _paper_interval_score(
    *,
    model: Any,
    adapter: Any,
    image: torch.Tensor,
    baseline: torch.Tensor,
    visibility: torch.Tensor,
    output_ids: torch.Tensor,
    positions: list[int],
    num_iter: int,
    noise_std: float,
) -> float:
    """Integrated score using the cumulative log-likelihood from paper Eq. (7).

    Qwen's packed visual tokens are detached for each LVLM forward to avoid
    retaining the full model graph.  Their gradient is immediately pushed back
    through the differentiable packing and mask graph; this is algebraically
    equivalent to a single end-to-end backward pass.
    """
    if not positions:
        raise ValueError("paper_original requires at least one crucial token")

    device = image.device
    positions_tensor = torch.tensor(positions, device=device, dtype=torch.long)
    intervals = torch.linspace(
        1.0 / num_iter, 1.0, num_iter, device=device, dtype=visibility.dtype
    )
    total = 0.0

    for alpha in intervals:
        interval_mask = visibility * alpha.view(1, 1, 1, 1)
        perturbed = _phi(image, baseline, interval_mask)
        if noise_std:
            # Canonical repository behavior: one independent N(0, 0.2) draw.
            perturbed = perturbed + torch.randn_like(perturbed) * noise_std

        patch_grid = perturbed[0].permute(1, 2, 0)
        # Packing belongs to the registered model adapter. Qwen uses merged
        # block order; another LVLM may use a different differentiable layout.
        packed_pixels = adapter.pack_visual_grid(patch_grid)
        detached_pixels = packed_pixels.detach().float().requires_grad_(True)

        model_inputs = adapter.current_base_inputs.copy()
        model_inputs["pixel_values"] = adapter.inject_candidate_pixels(
            detached_pixels.to(model.dtype)
        )
        probabilities = adapter.token_probs_from_dict(model_inputs, output_ids)

        # No mean, LLR weighting, percentile weighting, or probability clipping:
        # this is precisely the cumulative log-likelihood in Eq. (7).
        step_score = torch.log(probabilities)[positions_tensor].sum() / num_iter
        step_score.backward()
        packed_pixels.backward(detached_pixels.grad, retain_graph=True)
        total += float(step_score.detach().item())

        del probabilities, step_score, detached_pixels, model_inputs, perturbed

    return total


def _bilateral_tv(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    tv_beta: int = 2,
    sigma: float = 0.01,
) -> torch.Tensor:
    """Canonical iGOS bilateral-TV term, evaluated in RGB image space."""
    mask_2d = mask[:, 0]
    mask_vertical = torch.mean(
        torch.abs(mask_2d[:, :-1] - mask_2d[:, 1:])
        .reshape(mask.shape[0], -1)
        .pow(tv_beta),
        dim=1,
    )
    mask_horizontal = torch.mean(
        torch.abs(mask_2d[:, :, :-1] - mask_2d[:, :, 1:])
        .reshape(mask.shape[0], -1)
        .pow(tv_beta),
        dim=1,
    )

    up_mask = F.interpolate(
        mask, size=rgb_image.shape[-2:], mode="bilinear", align_corners=False
    )
    image_vertical = (
        rgb_image[:, :, :-1, :] - rgb_image[:, :, 1:, :]
    ).mean(dim=1)
    bilateral_vertical = torch.mean(
        torch.exp(-(image_vertical**2) / sigma).reshape(mask.shape[0], -1)
        * torch.abs(up_mask[:, :, :-1, :] - up_mask[:, :, 1:, :])
        .reshape(mask.shape[0], -1)
        .pow(tv_beta),
        dim=1,
    )

    image_horizontal = rgb_image[:, :, :, :-1] - rgb_image[:, :, :, 1:]
    bilateral_horizontal = torch.mean(
        torch.exp(-(image_horizontal**2) / sigma)
        .mean(dim=1)
        .reshape(mask.shape[0], -1)
        * torch.abs(up_mask[:, :, :, :-1] - up_mask[:, :, :, 1:])
        .reshape(mask.shape[0], -1)
        .pow(tv_beta),
        dim=1,
    )
    return 0.5 * (
        mask_vertical
        + mask_horizontal
        + bilateral_vertical
        + bilateral_horizontal
    )


def paper_igos_single_mask(
    args: Any,
    *,
    model: Any,
    adapter: Any,
    init_mask: torch.Tensor,
    image: torch.Tensor,
    baseline: torch.Tensor,
    label: torch.Tensor,
    size: int = 32,
    iterations: int = 5,
    ig_iter: int = 5,
    L1: float = 1.0,
    L2: float = 0.1,
    L3: float = 10.0,
    lr: float = 10.0,
    opt: str = "NAG",
    positions: list[int] | None = None,
    rgb_image: torch.Tensor | None = None,
    progress_callback: Callable[..., None] | None = None,
    iteration_callback: Callable[..., None] | None = None,
    log: Callable[[str], None] | None = None,
    **_: Any,
):
    """Optimize the paper's one 32x32 visibility mask with GNC and NAG."""
    if opt != "NAG":
        raise ValueError("paper_original currently permits only the paper NAG optimizer")
    if rgb_image is None:
        raise ValueError("paper_original requires the RGB image for bilateral TV")
    if tuple(init_mask.shape[-2:]) != (size, size):
        raise ValueError(
            f"paper_original expected a {size}x{size} initial mask, got "
            f"{tuple(init_mask.shape[-2:])}"
        )

    log = print if log is None else log
    positions = list(positions or [])
    masks = Variable(init_mask.detach().clone().float().to(image.device), requires_grad=True)
    momentum_state = torch.zeros(1, device=image.device)
    loss_del_history: list[float] = []
    loss_ins_history: list[float] = []
    loss_l1_history: list[float] = []
    loss_tv_history: list[float] = []
    loss_l2_history: list[float] = []

    for iteration in range(iterations):
        if progress_callback is not None:
            progress_callback(
                iteration=iteration + 1, iterations=iterations, ig_iter=ig_iter
            )

        visibility = F.interpolate(
            masks, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        total_gradient = torch.zeros_like(masks)

        loss_del = _paper_interval_score(
            model=model,
            adapter=adapter,
            image=image,
            baseline=baseline,
            visibility=visibility,
            output_ids=label,
            positions=positions,
            num_iter=ig_iter,
            noise_std=float(args.ig_noise_std),
        )
        total_gradient += masks.grad.clone()
        masks.grad.zero_()

        # Equation (8) insertion: swap original and blurred baseline.
        loss_ins = _paper_interval_score(
            model=model,
            adapter=adapter,
            image=baseline,
            baseline=image,
            visibility=visibility,
            output_ids=label,
            positions=positions,
            num_iter=ig_iter,
            noise_std=float(args.ig_noise_std),
        )
        total_gradient -= masks.grad.clone()
        masks.grad.zero_()

        l2_coefficient = L2 * math.exp(-float(args.gamma) * iteration)
        loss_l1 = L1 * torch.mean(
            torch.abs(1.0 - masks).reshape(masks.shape[0], -1), dim=1
        )
        loss_tv = L3 * _bilateral_tv(rgb_image, masks, tv_beta=2, sigma=0.01)
        loss_l2 = l2_coefficient * torch.sum(
            (1.0 - masks) ** 2, dim=(1, 2, 3)
        )
        (loss_l1 + loss_tv + loss_l2).sum().backward()
        total_gradient += masks.grad.clone()
        masks.grad.zero_()

        acceleration = iteration / (iteration + int(args.momentum))
        previous_state = momentum_state
        momentum_state = masks.data - lr * total_gradient
        masks.data = momentum_state + acceleration * (
            momentum_state - previous_state
        )
        masks.data.clamp_(0.0, 1.0)

        if iteration_callback is not None:
            iteration_callback(masks.detach(), iteration=iteration + 1,
                               iterations=iterations, ig_iter=ig_iter)

        loss_del_history.append(loss_del)
        loss_ins_history.append(loss_ins)
        loss_l1_history.append(float(loss_l1.item()))
        loss_tv_history.append(float(loss_tv.item()))
        loss_l2_history.append(float(loss_l2.item()))
        log(
            "paper_original iteration "
            f"{iteration + 1}/{iterations}: deletion={loss_del:.4f}, "
            f"insertion={loss_ins:.4f}, L1={loss_l1.item():.4f}, "
            f"BTV={loss_tv.item():.4f}, L2={loss_l2.item():.4f}"
        )

    # Match iGOS_p's return contract.  This is the native 32x32 visibility
    # mask; the caller bilinearly expands it separately for Qwen and rendering.
    return (
        masks.detach(),
        loss_del_history,
        loss_ins_history,
        loss_l1_history,
        loss_tv_history,
        loss_l2_history,
        None,
        None,
    )
