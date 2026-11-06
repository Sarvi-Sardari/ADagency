#igos_core_methods_helper.py

"""
Helper functions for the IGOS explanation methods.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS

This local adaptation keeps the optimization helpers self-contained inside
the pipeline and adds the superpixel and model-adapter operations required by
the mask-ablation engine.
"""

import torch
import torch.nn.functional as F
import math
from torch.nn import UpsamplingBilinear2d
from .igos_core_utils import pred_probs


# A dedicated RNG makes IG interval noise reproducible even if upstream call
# (not the global torch RNG) makes this specific noise draw reproducible
# regardless of what else runs before it in the pipeline (SLIC, SAM3, sample
# ordering) -- something torch.manual_seed(seed) alone can't guarantee once call
# order upstream changes. CPU generator + .to(device) below avoids any CUDA
# multi-device generator mismatch issues.
_NOISE_GENERATOR = torch.Generator(device='cpu').manual_seed(20260710)

def blocks_to_grid(flat_patches, H_p, W_p, merge_size=2):
    x = flat_patches.view(H_p // merge_size, W_p // merge_size, merge_size, merge_size, -1)
    x = x.permute(0, 2, 1, 3, 4)
    return x.reshape(H_p, W_p, -1)

def grid_to_blocks(grid, merge_size=2):
    H_p, W_p, dim = grid.shape
    x = grid.view(H_p // merge_size, merge_size, W_p // merge_size, merge_size, dim)
    x = x.permute(0, 2, 1, 3, 4).reshape(H_p * W_p, dim)
    return x
def upscale_superpixels(masks, sp_label_map):
    """
    Expands a compact per-superpixel mask into a dense grid by broadcasting each
    superpixel's scalar value to every cell that belongs to it. Differentiable:
    gradient w.r.t. `masks` accumulates correctly for repeated indices (same
    mechanism as an embedding-table lookup).

    masks: anything that flattens to [B, num_superpixels]
    sp_label_map: [H, W] long tensor, sp_label_map[i,j] = superpixel id of cell (i,j)
    returns: [B, 1, H, W]
    """
    B = masks.shape[0]
    flat = masks.reshape(B, -1)
    H, W = sp_label_map.shape
    idx = sp_label_map.reshape(-1).to(flat.device)
    dense = flat[:, idx]
    return dense.view(B, 1, H, W)


def pool_labels_to_grid(label_map_full, H_p, W_p):
    """
    Downsamples a pixel-resolution SLIC label map to the model's patch grid
    (H_p x W_p) via majority vote per patch. Majority vote, not average/nearest,
    because labels are categorical IDs -- averaging two superpixel IDs produces
    a meaningless number, not a valid third region.
    """
    H, W = label_map_full.shape
    ph, pw = H // H_p, W // W_p
    lm = label_map_full[: H_p * ph, : W_p * pw].reshape(H_p, ph, W_p, pw)
    lm = lm.permute(0, 2, 1, 3).reshape(H_p, W_p, ph * pw)
    mode_labels, _ = torch.mode(lm, dim=-1)
    return mode_labels  # [H_p, W_p]

def build_superpixel_adjacency(sp_map_patch, num_superpixels, rgb_image, sp_map_rgb):
    """
    Finds which superpixels are physical neighbors, and how similar their mean
    colors are. Needed because the per-pixel bilateral TV term is structurally
    a no-op for superpixel masks: it only fires at superpixel boundaries, and
    SLIC boundaries are placed exactly where real image edges are -- so the
    per-pixel bilateral weight collapses to ~0 at precisely the locations that
    matter. This computes the same "smooth where similar, sharp where different"
    idea one level up: between whole regions, using their mean colors, not
    between individual boundary pixels.
    """
    # Adjacent-cell pairs on the (cheap) patch grid -- good enough for adjacency,
    # and consistent with the grid the model itself masks over.
    left, right = sp_map_patch[:, :-1].reshape(-1), sp_map_patch[:, 1:].reshape(-1)
    top, bottom = sp_map_patch[:-1, :].reshape(-1), sp_map_patch[1:, :].reshape(-1)
    pairs = torch.cat([torch.stack([left, right], dim=1),
                        torch.stack([top, bottom], dim=1)], dim=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]          # keep only actual boundaries
    pairs, _ = torch.sort(pairs, dim=1)
    pairs = torch.unique(pairs, dim=0)                  # [E, 2], deduped, undirected

    # Mean color per superpixel, from the full-resolution RGB image + sp_map_rgb --
    # computed once here since neither the image nor the segmentation changes
    # during optimization.
    img = rgb_image[0].permute(1, 2, 0).reshape(-1, 3)      # [H*W, 3]
    flat_labels = sp_map_rgb.reshape(-1).to(img.device)
    sums = torch.zeros(num_superpixels, 3, device=img.device)
    sums.index_add_(0, flat_labels, img)
    counts = torch.bincount(flat_labels, minlength=num_superpixels).float().clamp(min=1).unsqueeze(1)
    mean_colors = sums / counts                              # [K, 3]

    color_sqdist = (mean_colors[pairs[:, 0]] - mean_colors[pairs[:, 1]]).pow(2).sum(dim=1)
    return pairs, color_sqdist   # edge_index [E,2], squared color distance [E]

# one-shot, noise-free superpixel relevance probe. A single backward pass
# with the mask fixed at "everything visible" (all ones) shows which superpixels
# the model is actually sensitive to, BEFORE the noisy iterative optimization
# below even starts -- one backward pass vs. the `iterations` it currently pays
# for to discover the same thing the hard way, and fully deterministic since
# noise=False.
def estimate_superpixel_importance(args, model, model_name, image, baseline, label,
                                    sp_map_patch, num_superpixels, prompt, image_size,
                                    positions, adapter, ig_iter=5, keyword_weights=None): #ig_iter=5  #CLADUE
    
    probe_mask = torch.ones((1, 1, 1, num_superpixels), dtype=torch.float32,
                             device='cuda', requires_grad=True)
    up_probe = upscale_superpixels(probe_mask, sp_map_patch)
    
    # integrated_gradient(args, model, model_name, image, baseline, label, up_probe,
    #                     ig_iter, noise=False, prompt=prompt, image_size=image_size,
    #                     positions=positions, adapter=adapter)
    # importance = probe_mask.grad.detach().abs().view(-1)   # [num_superpixels]
    # probe_mask.grad = None
    
    # probe both del and ins direction, summing gradients, since that's
    # what the actual loss_function optimizes over -- ranking by only one
    # direction doesn't match what the real run is sensitive to.
    integrated_gradient(args, model, model_name, image, baseline, label, up_probe,
                        ig_iter, noise=False, prompt=prompt, image_size=image_size,
                        positions=positions, adapter=adapter, keyword_weights=keyword_weights)
    grad_del = probe_mask.grad.detach().clone()
    probe_mask.grad = None

    integrated_gradient(args, model, model_name, baseline, image, label, up_probe,
                        ig_iter, noise=False, prompt=prompt, image_size=image_size,
                        positions=positions, adapter=adapter, keyword_weights=keyword_weights)
    grad_ins = probe_mask.grad.detach().clone()
    probe_mask.grad = None

    importance = (grad_del.abs() + grad_ins.abs()).view(-1)   # [num_superpixels]
    return importance
#######################################################################################################################################################

def cosine_decay(init, iter):
    return init * (1 + math.cos(math.pi * iter)) / 2

def exp_decay(init, iter, gamma=0.2):
    return init * math.exp(-gamma * iter)

def tv_norm(image, beta=2):
    """
    Calculates the total variation.
    :param image:
    :param beta:
    :return:
    """
    image = image[:, 0, :, :]
    a = torch.mean(torch.abs((image[:, :-1, :] - image[:, 1:, :]).view(image.shape[0], -1)).pow(beta), dim=1)
    b = torch.mean(torch.abs((image[:, :, :-1] - image[:, :, 1:]).view(image.shape[0], -1)).pow(beta), dim=1)
    return a + b


def bilateral_tv_norm(image, mask, resolution, tv_beta=2, sigma=1, expand_fn=None) :
    """
        Calculates the bilateral total variation.

    :param image:
    :param mask:
    :param tv_beta:
    :param sigma:
    :return:
    """

    #########################################################################################################################
    if expand_fn is None:
        mask_ = mask[:, 0, :]
        a = torch.mean(torch.abs((mask_[:, :-1, :] - mask_[:, 1:, :]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
        b = torch.mean(torch.abs((mask_[:, :, :-1] - mask_[:, :, 1:]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    else:
        # Superpixel index k and k+1 are NOT spatial neighbors -- a raw diff between
        # adjacent entries of the compact mask vector is meaningless here. Skip it;
        # bil_a/bil_b below (computed on the EXPANDED mask, which is spatially valid)
        # carry the entire smoothness penalty instead.
        a = torch.zeros(mask.shape[0], device=mask.device)
        b = torch.zeros(mask.shape[0], device=mask.device)
    #########################################################################################################################

    # tv term
    # mask_ = mask[:, 0, :]
    # a = torch.mean(torch.abs((mask_[:, :-1, :] - mask_[:, 1:, :]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    # b = torch.mean(torch.abs((mask_[:, :, :-1] - mask_[:, :, 1:]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    
    # bilateral tv in the image space
    if isinstance(image, list):
        image = image[0]
    elif image.shape[0] > 1:
        image = image[0].unsqueeze(0)

    #up_mask_ = upscale(mask, image, resolution)
    up_mask_ = expand_fn(mask) if expand_fn is not None else upscale(mask, image, resolution)
    bil_a = torch.mean(torch.exp(-(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1) ** 2 / sigma).view(mask.shape[0], -1)
                    * torch.abs((up_mask_[:, :, :-1, :] - up_mask_[:, :, 1:, :]).view(up_mask_.shape[0], -1)).pow(tv_beta), dim=1)
    bil_b = torch.mean(torch.exp(-(image[:, :, :, :-1] - image[:, :, :, 1:]) ** 2 / sigma).mean(dim=1).view(mask.shape[0], -1)
                    * torch.abs((up_mask_[:, :, :, :-1] - up_mask_[:, :, :, 1:]).view(up_mask_.shape[0], -1)).pow(tv_beta), dim=1)
    return 0.5 * (a + b + bil_a + bil_b)


def upscale(masks, images, resolution):
    if isinstance(images, list):
        output_masks = []
        for image in images:
            upscale_fn = UpsamplingBilinear2d(size=image.shape[-2:]).cuda()
            output_masks.append(upscale_fn(masks).expand((-1,1,)+image.shape[-2:]))
    elif images.shape[0] == 1:
        target_size = images.shape[-2:]
        if masks.shape[-1] > target_size[-1] or masks.shape[-2] > target_size[-2]:
            # Downsampling to patches: Use Average Pooling so gradients flow to ALL high-res pixels
            # If Mask is High-Res (672x448) and Image is Patches (e.g. 48x32) -> Pool it safely
            output_masks = F.adaptive_avg_pool2d(masks, target_size).expand((-1,1,)+target_size)
        else:
            # Upsampling to RGB image: Use Bilinear Interpolation
            # If Mask is Low-Res and Image is RGB -> Upsample it
            output_masks = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False).expand((-1,1,)+target_size)
        # upscale_fn = UpsamplingBilinear2d(size=images.shape[-2:]).cuda()
        # output_masks = upscale_fn(masks).expand((-1,1,)+images.shape[-2:])
    else:
        # divide mask to patches
        #patch_size = int(masks.shape[-1] / math.sqrt(images.shape[0] - 1))
        if resolution[0] != resolution[1]:
            ratio = resolution[0] / resolution[1]
            if ratio > 1:
                new_height = masks.shape[-1]
                new_width = int(new_height * ratio)
            else:
                new_width = masks.shape[-1]
                new_height = int(new_width / ratio)
            masks = F.interpolate(masks, size=(new_width, new_height), mode='bilinear', align_corners=False)
        patch_size = int(images.shape[-1] * masks.shape[-1] / resolution[-1])
        patches = masks.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, masks.shape[1], patch_size, patch_size)
        upscale_fn = UpsamplingBilinear2d(size=images.shape[-2:]).cuda()
        up_patches = upscale_fn(patches)
        up_mask_origin = upscale_fn(masks)
        output_masks = torch.cat((up_mask_origin, up_patches))
        assert output_masks.shape[0] == images.shape[0]

    return output_masks

def interval_score(args, model, model_name, images, baseline, label, up_masks, num_iter, noise=True, 
                   prompt=None, image_size=None, positions=None, adapter=None, keyword_weights=None):                         #Added adapter parameter for Qwen3-VL
    if model_name == 'llava' or model_name == 'llava_next' or model_name == 'mgm':
        # The intervals to approximate the integral over
        intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
        interval_masks = up_masks.unsqueeze(1) * intervals
        local_images = phi(images.unsqueeze(1), baseline.unsqueeze(1), interval_masks)
        if noise:
            local_images = local_images + torch.randn_like(local_images) * .2

        local_images = local_images.transpose(0, 1)
        input_ids = torch.cat((prompt, label), dim=1)
        positions = torch.tensor(positions).to(input_ids.device)

        losses = torch.tensor(0.).to(input_ids.device)
        for single_img in local_images:
            single_img = single_img.half()
            probs = pred_probs(args, model, input_ids, label, single_img, image_size)
            #losses += probs[positions].mean()
            losses += torch.log(probs)[positions].sum()
    
    elif model_name == 'cambrian':
        # The intervals to approximate the integral over
        intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
        interval_masks = [up_mask.unsqueeze(1) * intervals for up_mask in up_masks]
        local_images = [phi(image.unsqueeze(1), base.unsqueeze(1), interval_mask) for image, base, interval_mask in zip(images, baseline, interval_masks)]

        if noise:
            local_images = [local_image + torch.randn_like(local_image) * .2 for local_image in local_images]

        local_images = [local_image.squeeze(0) for local_image in local_images]
        local_images = [[local_images[i][j] for i in range(len(local_images))] for j in range(num_iter)]
        input_ids = torch.cat((prompt, label), dim=1)
        positions = torch.tensor(positions).to(input_ids.device)
        losses = torch.tensor(0.).to(input_ids.device)

        for single_img in local_images:
            single_img = [item.unsqueeze(0).half() for item in single_img]
            probs = pred_probs(args, model, input_ids, label, single_img, image_size)
            #losses += probs[positions].mean()
            losses += torch.log(probs)[positions].sum()

    ###################################Code added for Qwen3-VL #########################################
    # elif model_name == 'qwen3_vl' and adapter is not None:
    #     intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
    #     interval_masks = up_masks.unsqueeze(1) * intervals
    #     local_images = phi(images.unsqueeze(1), baseline.unsqueeze(1), interval_masks)

    #     # up_masks: [1, 1, H, W]
    #     # interval_masks: [1, num_iter, 1, H, W]
        
    #     # images/baseline: [1, C, H, W]
    #     # local_images: [1, num_iter, C, H, W]
        
    #     # if noise:
    #     #     local_images = local_images + torch.randn_like(local_images) * 0.05 #0.05

    #     if noise:
    #         # average several FIXED-generator noise draws per interval
    #         # sample instead of one torch.randn_like() draw off the global RNG.
    #         # This keeps the same landscape-smoothing benefit the single noisy
    #         # draw was for, but the variance-reduction now happens INSIDE one
    #         # deterministic run instead of across several independently-seeded
    #         # full optimization runs -- which is what let a bad seed's mask drag
    #         # a good seed's mask down when they got averaged together afterward.
    #         # num_noise_samples=2 costs roughly 2/3 of what your old 3-seed
    #         # ensemble cost, not 3x more.
    #         num_noise_samples = 2
    #         noisy_sum = torch.zeros_like(local_images)
    #         for _ in range(num_noise_samples):
    #             noisy_sum = noisy_sum + torch.randn(local_images.shape, generator=_NOISE_GENERATOR).to(local_images.device) * 0.05
    #         local_images = local_images + noisy_sum / num_noise_samples

    #     local_images = local_images.transpose(0, 1)
    #     target_device = model.device if hasattr(model, 'device') else 'cuda'
    #     positions_tensor = torch.tensor(positions).to(target_device)
    #     #losses = torch.tensor(0.).to(target_device)
    #     total_loss_val = 0.0

    #     merge_size = adapter.processor.image_processor.merge_size

    #     for single_img in local_images:

    #         # # Clamp the tensor to valid pixel values and convert to PIL Image
    #         # pil_img = F_t.to_pil_image(single_img[0].clamp(0, 1).cpu())
    #         # # Use the adapter instead of legacy pred_probs
    #         # probs = adapter.token_probs(pil_img, prompt, label)
    #         # losses += torch.log(probs)[positions_tensor].sum()

    #         # 1. Take the 3D tensor [C, H, W] and flatten it back to Qwen's 1D patch sequence
    #         # patch_dim = single_img[0].shape[0] 
    #         # masked_pixel_values = single_img[0].permute(1, 2, 0).reshape(-1, patch_dim).to(model.dtype)

    #         patch_dim = single_img[0].shape[0]
    #         masked_pixel_grid = single_img[0].permute(1, 2, 0) #.to(model.dtype)      # [H_p, W_p, patch_dim] raster order
    #         masked_pixel_values = grid_to_blocks(masked_pixel_grid, merge_size)     # back to Qwen's native block order

    #         ###############################################################################################

    #         # Isolate the mask graph from the LLM graph     #Implemeted to reduce VRAM usage
    #         detached_pixels = masked_pixel_values.detach().float().requires_grad_(True)

    #         # 2. Feed the PyTorch tensor directly into the dictionary, bypassing NumPy/PIL
    #         model_inputs = adapter.current_base_inputs.copy()
    #         model_inputs["pixel_values"] = detached_pixels.to(model.dtype) #Previously had masked_pixel_values instead of detached_pixels   #Implemeted to reduce VRAM usage 

    #         # 3. Get probabilities using the function that preserves the PyTorch math chain
    #         probs = adapter.token_probs_from_dict(model_inputs, label)

    #         ########################################################################
    #         #            Implemeted to reduce VRAM usage
    #         ########################################################################
    #         #step_loss = torch.log(probs)[positions_tensor].sum() / num_iter
    #         #step_loss = probs[positions_tensor].mean() / num_iter  #For superpixel tried again with lpog probs instead of mean
    #         #step_loss = torch.log(probs.clamp(min=1e-5))[positions_tensor].sum() / num_iter

    #         # Normalize by BOTH num_iter and the number of keyword positions -- the old
    #         # sum-over-positions version made loss magnitude (and therefore the gradient
    #         # magnitude the NAG step multiplies by lr) scale with how many keyword
    #         # tokens this particular sample happened to have, which is why lr=0.2
    #         # (tuned on a different scale) instantly saturated masks to 0/1.
    #         #step_loss = torch.log(probs.clamp(min=1e-5))[positions_tensor].sum() / (num_iter * len(positions_tensor))
    #         #step_loss = torch.log(probs.clamp(min=1e-5))[positions_tensor].sum() / (num_iter * len(positions_tensor)) (Without Keyword Weighting with importance)

    #         # weight each keyword's log-prob by its own importance score instead
    #         # of counting every word equally. keyword_weights now arrives as a real
    #         # parameter (see integrated_gradient/interval_score signatures above),
    #         # not pulled from **kwargs -- interval_score never had a **kwargs catch-all,
    #         # so kwargs.get() would have silently failed before.
    #         if keyword_weights is not None:
    #             kw_weights = torch.tensor(keyword_weights, device=probs.device, dtype=probs.dtype)
    #             kw_weights = kw_weights / kw_weights.sum()
    #             step_loss = (torch.log(probs.clamp(min=1e-5))[positions_tensor] * kw_weights).sum() / num_iter
    #         else:
    #             step_loss = torch.log(probs.clamp(min=1e-5))[positions_tensor].sum() / (num_iter * len(positions_tensor))
            

    #         # i. Backprop ONLY the LLM (This instantly frees the massive LLM memory)
    #         step_loss.backward()
            
    #         # ii. Push the gradients across the shield into the tiny mask graph
    #         masked_pixel_values.backward(detached_pixels.grad, retain_graph=True)
            
    #         total_loss_val += step_loss.item()

    #         # 3. Flush the VRAM before the next iteration
    #         del probs, step_loss, detached_pixels, model_inputs
    #         torch.cuda.empty_cache()

    #         # Because we already pushed gradients manually, we return a flat tensor
    #     return torch.tensor(total_loss_val, device=target_device, requires_grad=False)
    #         ##########################################################################

    #         # Accumulate the raw sum of losses
    #         #losses += torch.log(probs)[positions_tensor].sum()

    elif model_name == 'qwen3_vl' and adapter is not None:      #Model VRAM saving and faster calculation of masks (from Gemini)
        target_device = model.device if hasattr(model, 'device') else 'cuda'
        positions_tensor = torch.tensor(positions).to(target_device)
        
        intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda()
        total_loss_val = 0.0
        merge_size = adapter.processor.image_processor.merge_size

        for i in range(num_iter):
            # 1. Lazy evaluation: build only ONE interval image at a time
            interval_mask = up_masks * intervals[i].view(1, 1, 1, 1)
            single_img = phi(images, baseline, interval_mask)

            if noise:
                # Retained your specific reproducible noise generator
                num_noise_samples = int(getattr(args, "ig_noise_samples", 2))
                noise_std = float(getattr(args, "ig_noise_std", 0.05))
                noisy_sum = torch.zeros_like(single_img)
                for _ in range(num_noise_samples):
                    noisy_sum += (
                        torch.randn(single_img.shape, generator=_NOISE_GENERATOR)
                        .to(single_img.device)
                        * noise_std
                    )
                single_img = single_img + noisy_sum / num_noise_samples

            # 2. Extract patches
            masked_pixel_grid = single_img[0].permute(1, 2, 0) 
            masked_pixel_values = grid_to_blocks(masked_pixel_grid, merge_size)

            # 3. Isolate the mask graph from the 18GB LLM graph
            detached_pixels = masked_pixel_values.detach().float().requires_grad_(True)
            model_inputs = adapter.current_base_inputs.copy()
            #model_inputs["pixel_values"] = detached_pixels.to(model.dtype)
            # replace candidate patches only.
            # In reference-context mode, reference patches remain unchanged.
            model_inputs["pixel_values"] = adapter.inject_candidate_pixels(detached_pixels.to(model.dtype))

            # 4. Forward & Backward
            probs = adapter.token_probs_from_dict(model_inputs, label)
            
            if getattr(args, "token_weighting", "llr_sqrt") == "paper_sum":
                # Paper Eq. (7): cumulative (not averaged or LLR-weighted)
                # log-likelihood over the selected crucial tokens.
                step_loss = (
                    torch.log(probs.clamp(min=1e-5))[positions_tensor].sum()
                    / num_iter
                )
            elif keyword_weights is not None:
                kw_weights = torch.tensor(keyword_weights, device=probs.device, dtype=probs.dtype)
                kw_weights = kw_weights / kw_weights.sum()
                step_loss = (torch.log(probs.clamp(min=1e-5))[positions_tensor] * kw_weights).sum() / num_iter
            else:
                step_loss = torch.log(probs.clamp(min=1e-5))[positions_tensor].sum() / (num_iter * len(positions_tensor))

            step_loss.backward()
            masked_pixel_values.backward(detached_pixels.grad, retain_graph=True)
            total_loss_val += step_loss.item()

            # 5. Aggressive cleanup of local references (Replaces inner empty_cache)
            del probs, step_loss, detached_pixels, model_inputs, single_img, interval_mask

        return torch.tensor(total_loss_val, device=target_device, requires_grad=False)

    #####################################################################################################

    return losses / num_iter


def integrated_gradient(args, model, model_name, image, baseline, label, up_masks, num_iter,
                        noise=True, prompt=None, image_size=None, positions=None, adapter=None, keyword_weights=None): #Added adapter parameter for Qwen3-VL
    loss = interval_score(
                args,
                model,
                model_name, 
                image,
                baseline,
                label,
                up_masks,
                num_iter,
                noise,
                prompt,
                image_size,
                positions,
                adapter=adapter,     #Added adapter parameter for Qwen3-VL
                keyword_weights=keyword_weights
                )
    ###################### Implemented to reduce VRAM usage #################
    #loss.sum().backward(retain_graph=True)
    
    # Only run global backward if the graph wasn't already manually backpropped
    if loss.requires_grad:
        loss.sum().backward(retain_graph=True)
    #########################################################################
    
    return loss.sum().item()

def line_search(masks, total_grads, loss_func, alpha=8, beta=0.0001, decay=0.2,):
    # Speed up computations, reduce memory usage, and ensure no autograd
    # graphs are created
    with torch.no_grad():
        i = 0
        mod = len(masks.shape) - 3
        num_inputs = masks.shape[0]
        # The indices of masks that still need their alphas updated
        indices = torch.ones(num_inputs, dtype=torch.bool).cuda()
        # Create initial alpha values for each mask
        alphas = torch.ones(num_inputs).cuda() * alpha

        up_masks = upscale(masks.view(-1,*masks.shape[mod:])).view(-1, *masks.shape[1:mod], 1, upscale.out_size, upscale.out_size)

        # Compute the base loss used in the condition
        base_losses = loss_func(up_masks, masks, indices).view(-1)
        t = -beta * (total_grads ** 2).view(num_inputs, -1).sum(dim=1).view(num_inputs)

        while True:
            # Create a new mask with the updated alpha value to
            # see if it meets condition
            new_masks = torch.clamp(masks[indices] - alphas[indices].view(-1,*(1,) * mod,1,1) * total_grads[indices], 0, 1)
            up_masks = upscale(new_masks.view(-1,*masks.shape[mod:])).view(-1,*masks.shape[1:mod], 1, upscale.out_size, upscale.out_size)
            # Calculate new losses
            losses = loss_func(up_masks, new_masks, indices).view(-1)
            # Get indices for each alpha that meets the condition for
            # their corresponding mask
            indices[indices] = losses > base_losses[indices] + alphas[indices] * t[indices]
            # Same for this, but for if the alpha values are too low (\alpha_l)
            indices[indices] *= (alphas[indices] >= 0.00001)
            # Break out of the loop if all alpha values satisfy the condition
            # or are too low
            if not indices.sum():
                break
            # Otherwise update alphas
            alphas[indices] *= decay
            i += 1
    return alphas.view(-1,1,1,1)


def phi(img, baseline, mask):
    """
        Composes an image from img and baseline according to the mask values.

    :param img:
    :param baseline:
    :param mask:
    :return:
    """
    return img.mul(mask) + baseline.mul(1-mask)


def metric(args, image, baseline, mask, model, model_name, label, label_i, pred_data, size=28, prompt=None, image_size=None, positions=None, resolution=None, adapter=None): #Added adapter parameter for Qwen3-VL
    with torch.no_grad():
        # The dimensions for the image
        #img_size = image.shape[-1]
        # Compute the total number of pixels in a mask
        # mask_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        # num_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        H_m, W_m = mask.shape[-2], mask.shape[-1]
        num_pixels = H_m * W_m
        # The original helper sampled approximately 50 points on *each* curve,
        # which means roughly 100 additional LVLM forwards for every candidate
        # heatmap.  Keep 50 as the compatibility default, but allow ablation
        # presets to use a coarse curve when the metric is only ranking maps.
        metric_num_steps = max(1, int(getattr(args, "metric_num_steps", 50)))
        step = max(1, math.ceil(num_pixels / metric_num_steps))
        # Used for indexing with batch sizes
        #l = torch.arange(1)
        l = torch.arange(mask.shape[0], device=image.device)
        # The unmasked score
        og_scores = score_output(args, image, image_size, model, model_name, l, label, prompt, positions, adapter=adapter) #Added adapter parameter for Qwen3-VL
        # The baseline score
        blur_scores = score_output(args, baseline, image_size, model, model_name, l, label, prompt, positions, adapter=adapter) #Added adapter parameter for Qwen3-VL
        # Initial values for the curves
        del_curve = [og_scores]
        ins_curve = [blur_scores]
        index = [0.]

        # True_mask is used to hold 1 or 0. Either show that pixel or blur it.
        #true_mask = torch.ones((mask.shape[0], mask_pixels)).cuda()
        true_mask = torch.ones((mask.shape[0], num_pixels), device=image.device, dtype=mask.dtype)
        del_scores = torch.zeros(mask.shape[0])
        ins_scores = torch.zeros(mask.shape[0])
        # Sort each mask by values and store the indices.
        elements = torch.argsort(mask.view(mask.shape[0], -1), dim=1)
        for pixels in range(0, num_pixels, step):
            # Get the indices used in this iteration
            indices = elements[l,pixels:pixels+step].squeeze().view(1, -1)
            # Set those indices to 0
            true_mask[l, indices.permute(1,0)] = 0
            #up_mask = upscale(true_mask.view(-1, 1, size,size), image, resolution)
            up_mask = upscale(true_mask.view(-1, 1, H_m, W_m), image, resolution)
            # Mask the image for deletion
            if isinstance(image, list):
                del_image = [phi(x, y, z).half() for x, y, z in zip(image, baseline, up_mask)]
            else:
                del_image = phi(image, baseline, up_mask).half()
            # Calculate new scores
            outputs = score_output(args, del_image, image_size, model, model_name, l, label, prompt, positions, adapter=adapter) #Added adapter parameter for Qwen3-VL
            del_curve.append(outputs)
            index.append((pixels+step)/num_pixels)
            outputs = (outputs-blur_scores) / (og_scores-blur_scores)
            del_scores += outputs.cpu() * step if pixels + step < num_pixels else num_pixels - pixels

            # Mask the image for insertion
            if isinstance(image, list):
                ins_image = [phi(x, y, z).half() for x, y, z in zip(baseline, image, up_mask)]
            else:
                ins_image = phi(baseline, image, up_mask).half()

            # Calculate the new scores
            outputs = score_output(args, ins_image, image_size, model, model_name, l, label, prompt, positions, adapter=adapter) #Added adapter parameter for Qwen3-VL

            ins_curve.append(outputs)
            outputs = (outputs-blur_scores) / (og_scores-blur_scores)
            ins_scores += outputs.cpu() * step if pixels + step < num_pixels else num_pixels - pixels

        # Force scores between 0 and 1.
        del_scores /= num_pixels
        ins_scores /= num_pixels

        del_curve = list(map(lambda x: [y.item() for y in x], zip(*del_curve)))
        ins_curve = list(map(lambda x: [y.item() for y in x], zip(*ins_curve)))

    return del_scores, ins_scores, del_curve, ins_curve, index


def score_output(args, image, image_size, model, model_name, l, label, prompt, positions, adapter=None):  #Added adapter parameter for Qwen3-VL
    
    ######################### Code added for Qwen3-VL #################################    
    if model_name=='qwen3_vl' and adapter is not None:
        target_device = model.device if hasattr(model, 'device') else 'cuda'
        positions_tensor = torch.tensor(positions).to(target_device)

        # image is NOT RGB here.
        # Shape: [1, patch_dim, H_p, W_p], e.g. [1, 1536, H_p, W_p]
        merge_size = adapter.processor.image_processor.merge_size

        pixel_grid = image[0].permute(1, 2, 0).to(dtype=model.dtype, device=target_device)  # raster order
        pixel_values = grid_to_blocks(pixel_grid, merge_size)

        #pixel_values = (image[0].permute(1, 2, 0).reshape(-1, patch_dim).to(dtype=model.dtype, device=target_device))
        ###########################################################################

        model_inputs = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in adapter.current_base_inputs.items()}

        #model_inputs["pixel_values"] = pixel_values
        # deletion/insertion evaluation must perturb only
        # candidate patches, exactly like the optimization loop.
        model_inputs["pixel_values"] = adapter.inject_candidate_pixels(pixel_values)

        probs = adapter.token_probs_from_dict(model_inputs, label)

        scores = probs[positions_tensor].sum().unsqueeze(0)
        
        # #Clamp tensor to valid pixel values and convert to PIL Image
        # pil_img = F_t.to_pil_image(image[0].clamp(0, 1).cpu())

        # #Use the precomputed adapter
        # probs = adapter.token_probs(pil_img, prompt, label)

        # scores = probs[positions_tensor].sum(-1).unsqueeze(0)
        
        return scores/ len(positions)
    ################################################################################### 
    else:
        input_ids = torch.cat((prompt, label), dim=1)
        probs_pred = pred_probs(args, model, input_ids, label, image, image_size).unsqueeze(0)
        scores = probs_pred[:, torch.tensor(positions).to(probs_pred.device)].sum(-1) 
        
        return scores / len(positions)
