#igos_core_methods.py

"""
Different explanation methods from the Integrated-Gradient Optimized Saliency map methods.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS

This is a self-contained copy of LVLM_Interpretation/methods_GPT.py,
relocated into pipeline_project/mask_ablation/. Two kinds of changes were
made relative to methods_GPT.py:

  1. The cross-module import below (methods_helper_GPT -> igos_core_methods_helper).
  2. iGOS_p()'s DENSE (non-superpixel) regularization/clamp/gradient-clip path
     now honors four new optional kwargs -- dense_l2_reduction,
     mask_clamp_range, grad_clip_norm, dense_l2_floor_fraction -- so the
     "original" (segmentation_method='none') method can be run with EITHER:
       (a) the original dense regularization formulas only
           (LVLM_Interpretation_Orig/methods.py):
           L2 as a raw sum, mask clamped to [0, 1], no gradient-norm clipping,
           no L2 floor -- useful for component ablations, or
       (b) today's GPT-tuned dense variant (L2 as a mean, clamp [0.02, 0.98],
           grad_clip_norm=5.0, L2 floor at 1% of its initial value) for
           side-by-side comparison. The GPT-tuned values remain the defaults.
     The complete paper method (including its 32x32 native mask, token
     selection, score, optimizer, and renderer) now lives in
     ``paper_original.py``; these switches alone must not be labelled paper
     fidelity. The superpixel/SAM3/hybrid code paths are
     completely unchanged from methods_GPT.py.
"""

from torch.autograd import Variable
from .igos_core_methods_helper import *

def iGOS_pp(
        args,
        model,
        model_name, 
        init_mask,
        image,
        baseline,
        label,
        size=32,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='LS',
        softmax=True,
        **kwargs):

    def regularization_loss(image, masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1), \
               L3 * bilateral_tv_norm(image, masks, resolution, tv_beta=2, sigma=0.01), \
               L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])

    def ins_loss_function(up_masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    model_name, 
                    baseline[indices],
                    image[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise
                    )
        return losses.sum(dim=1).view(-1)

    def del_loss_function(up_masks, indices, noise=True):
        losses = interval_score(
                    model,
                    model_name, 
                    image[indices],
                    baseline[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise,
                    )
        return losses.sum(dim=1).view(-1)

    def loss_function(up_masks, masks, indices):
        loss = del_loss_function(up_masks[:, 0], indices)
        loss += ins_loss_function(up_masks[:, 1], indices)
        loss += del_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        loss += ins_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        return loss + regularization_loss(image[indices], masks[:, 0] * masks[:, 1])

    masks_del = torch.ones((1, 1, size, size), dtype=torch.float32, device='cuda')
    masks_del = masks_del * init_mask.cuda()
    masks_del = Variable(masks_del, requires_grad=True)
    masks_ins = torch.ones((image.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_ins = masks_ins * init_mask.cuda()
    masks_ins = Variable(masks_ins, requires_grad=True)
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    resolution = kwargs.get('resolution', None)

    if opt == 'NAG':
        cita_d=torch.zeros(1).cuda()
        cita_i=torch.zeros(1).cuda()
    
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)

    losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins = [], [], [], [], [], [], []
    for i in range(iterations):
        up_masks1 = upscale(masks_del, image, resolution)
        up_masks2 = upscale(masks_ins, image, resolution)

        # Compute the integrated gradient for the combined mask, optimized for deletion
        loss_comb_del = integrated_gradient(args, model, model_name, image, baseline, label, up_masks1 * up_masks2, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads1 = masks_del.grad.clone()
        total_grads2 = masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the combined mask, optimized for insertion
        loss_comb_ins = integrated_gradient(args, model, model_name, baseline, image, label, up_masks1 * up_masks2, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads1 -= masks_del.grad.clone()  # Negative because insertion loss is 1 - score.
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the deletion mask
        loss_del = integrated_gradient(args, model, model_name, image, baseline, label, up_masks1, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Compute the integrated graident for the insertion mask
        loss_ins = integrated_gradient(args, model, model_name, baseline, image, label, up_masks2, ig_iter, prompt=prompt, image_size=image_size, positions=positions)
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        # Average them to balance out the terms with the regularization terms
        total_grads1 /= 2
        total_grads2 /= 2

        # Computer regularization for combined masks
        L2 = exp_decay(args.L2, i, args.gamma)
        loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks_del * masks_ins)
        losses = loss_l1 + loss_tv + loss_l2
        losses.sum().backward()
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()

        if opt == 'LS':
            masks = torch.cat((masks_del.unsqueeze(1), masks_ins.unsqueeze(1)), 1)
            total_grads = torch.cat((total_grads1.unsqueeze(1), total_grads2.unsqueeze(1)), 1)
            lrs = line_search(masks, total_grads, loss_function, lr)
            masks_del.data -= total_grads1 * lrs
            masks_ins.data -= total_grads2 * lrs
        
        if opt == 'NAG':
            e = i / (i + args.momentum)
            cita_d_p = cita_d
            cita_i_p = cita_i
            cita_d = masks_del.data - lr * total_grads1
            cita_i = masks_ins.data - lr * total_grads2
            masks_del.data = cita_d + e * (cita_d - cita_d_p)
            masks_ins.data = cita_i + e * (cita_i - cita_i_p)

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0,1)
        masks_ins.data.clamp_(0,1)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_comb_del.append(loss_comb_del)
        losses_comb_ins.append(loss_comb_ins)
        losses_l1.append(loss_l1.item())
        losses_tv.append(loss_tv.item())
        losses_l2.append(loss_l2.item())
        print(f'iteration: {i} lr: {lr:.4f} loss_comb_del: {loss_comb_del:.4f}, loss_comb_ins: {loss_comb_ins:.4f}, loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')

    return masks_del * masks_ins, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins


def iGOS_p(
        args,
        model,
        model_name, 
        init_mask,
        image,
        baseline,
        label,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='LS',
        softmax=True,
        log=None,
        **kwargs):

    def regularization_loss(image, masks, resolution):
        # return L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1), \
        #        L3 * bilateral_tv_norm(image, masks, resolution, tv_beta=2, sigma=0.01), \
        #        L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])
        
        # We use rgb_image for bilateral_tv_norm so it calculates RGB edges correctly
        tv_source = rgb_image if rgb_image is not None else image
        #####################################################################################################################
        if use_superpixels:
            flat = masks.view(-1)
            l1_term = (L1 * (sp_counts.view(-1)* torch.abs(1.0 - flat)).sum() / total_patches).unsqueeze(0)
            l2_term = (L2 * (sp_counts.view(-1) * (1.0 - flat).pow(2)).sum() / total_patches).unsqueeze(0)
            edge_diffs = (flat[sp_edge_index[:, 0]] - flat[sp_edge_index[:, 1]]).pow(2)
            tv_term = (L3 * (sp_edge_weight * edge_diffs).sum() / (sp_edge_weight.sum() + 1e-8)).unsqueeze(0)
        else:
            # Dense patch-grid regularization -- this is the branch used by the
            # "original"/paper method (segmentation_method='none'). Whether it
            # reproduces LVLM_Interpretation_Orig/methods.py EXACTLY or the
            # GPT-tuned variant is controlled entirely by `dense_l2_reduction`
            # (see kwarg default below and its docstring note), not hardcoded here.
            l1_term = L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1)
            if dense_l2_reduction == "sum":
                # ORIGINAL paper formula (LVLM_Interpretation_Orig/methods.py):
                # L2 * torch.sum((1 - masks) ** 2, dim=[1, 2, 3])
                l2_term = L2 * torch.sum((1 - masks) ** 2, dim=[1, 2, 3])
            elif dense_l2_reduction == "mean":
                # use a mean rather than a raw sum. This keeps L2 on a comparable scale when the number of dense patch variables differs from the number of superpixels.
                l2_term = L2 * torch.mean((1 - masks) ** 2, dim=[1, 2, 3])
            else:
                raise ValueError(f"Unknown dense_l2_reduction={dense_l2_reduction!r}; expected 'sum' or 'mean'.")
            tv_term = L3 * bilateral_tv_norm(tv_source, masks, resolution, tv_beta=2, sigma=0.01)
        
        return l1_term, tv_term, l2_term

        # expand_fn_rgb = (lambda m: upscale_superpixels(m, sp_map_rgb)) if use_superpixels else None
        
        # if use_superpixels:
        #     # Exact identity, not an approximation: sum_k count_k * f(mask_k) equals
        #     # summing f(mask_dense) over every one of the H_p*W_p patches in the
        #     # expanded mask. This reproduces precisely what the ORIGINAL (non-superpixel)
        #     # L1/L2 formulas computed -- they just happened to have one parameter per
        #     # patch, so "mean/sum over parameters" and "mean/sum over patches" were the
        #     # same thing. Once a parameter can represent many patches, those two stop
        #     # being the same thing unless we weight by count explicitly, right here.
        #     flat = masks.view(-1)
        #     l1_term = (L1 * (sp_counts.view(-1) * torch.abs(1 - flat)).sum() / total_patches).unsqueeze(0)
        #     l2_term = (L2 * (sp_counts.view(-1) * (1 - flat) ** 2).sum()).unsqueeze(0)
        # else:
        #     l1_term = L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1)
        #     l2_term = L2 * torch.sum((1 - masks) ** 2, dim=[1, 2, 3])

        # return l1_term, \
        #        L3 * bilateral_tv_norm(tv_source, masks, resolution, tv_beta=2, sigma=0.01, expand_fn=expand_fn_rgb), \
        #        l2_term

        # return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1), \
        #        L3 * bilateral_tv_norm(tv_source, masks, resolution, tv_beta=2, sigma=0.01, expand_fn=expand_fn_rgb), \
        #        L2 * torch.sum((1 - masks) ** 2, dim=[1, 2, 3])

        # return L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1), \
        #        L3 * bilateral_tv_norm(tv_source, masks, resolution, tv_beta=2, sigma=0.01), \
        #        L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])
        #####################################################################################################################
        # return L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1), \
        #        L3 * tv_norm(masks, beta=2), \
        #        L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])
    
    def loss_function(up_masks, masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    model_name, 
                    baseline[indices],
                    image[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise,
                    prompt=prompt,              #Prompt and adapter parameters are added for adapting Qwen3-VL
                    positions=positions,
                    adapter=adapter,
                    keyword_weights=keyword_weights,
                    )
        losses += interval_score(
                    model,
                    model_name, 
                    image[indices],
                    baseline[indices],
                    label[indices],
                    up_masks,
                    ig_iter,
                    noise,
                    prompt=prompt,
                    positions=positions,
                    adapter=adapter,
                    keyword_weights=keyword_weights,
                    )
        loss_l1, loss_tv, loss_l2 = regularization_loss(
            image[indices], masks, resolution
        )
        return losses.sum(dim=1).view(-1) + loss_l1 + loss_tv + loss_l2
    
    #masks = torch.ones((1,1,size,size), dtype=torch.float32, device='cuda')
    #masks = masks * init_mask.cuda()
    #Match the mask strictly to the exact dimensions of init_mask          
    masks = torch.ones_like(init_mask).float().cuda()           #Optimized for rectangular masks

    masks = Variable(masks, requires_grad=True)
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    resolution = kwargs.get('resolution', None)
    adapter = kwargs.get('adapter', None) #Adapter added for Qwen3-VL
    rgb_image = kwargs.get('rgb_image', None) #Extract the rgb_image from visual_tensor
    # Per-keyword importance weights, same order as `positions` above.
    # Passed through unchanged to integrated_gradient -> interval_score, which is
    # where it's actually used to weight the loss.
    keyword_weights = kwargs.get('keyword_weights', None)
    progress_callback = kwargs.get('progress_callback')
    iteration_callback = kwargs.get('iteration_callback')
    if log is None: log = print

    # ------------------------------------------------------------------
    # DENSE-METHOD FIDELITY SWITCHES. All four default to exactly today's
    # GPT-tuned behavior (fully backward compatible with existing slic/sam3/
    # hybrid method configs). The "paper_igos_plus_sharp" method preset
    # (see configs/mask_ablation_configs.yaml) overrides all four so the
    # segmentation_method='none' path reproduces LVLM_Interpretation_Orig/
    # methods.py's dense regularization/update rules (mask_clamp_range=[0,1], no gradient-norm
    # clipping, no L2 floor, L2 as a raw sum). Any other combination is a
    # valid ablation arm in its own right -- these are plain config knobs,
    # not a hidden hardcoded choice.
    dense_l2_reduction = kwargs.get('dense_l2_reduction', 'mean')       # 'sum' (original paper) | 'mean' (GPT tuning, default)
    mask_clamp_range = kwargs.get('mask_clamp_range', (0.02, 0.98))      # original paper used (0.0, 1.0)
    grad_clip_norm = kwargs.get('grad_clip_norm', 5.0)                   # original paper had no gradient-norm clipping at all; set None to disable
    dense_l2_floor_fraction = kwargs.get('dense_l2_floor_fraction', 0.01)  # original paper had no L2 floor; set 0.0 to disable


    #####################################################################################################################
    sp_map_patch = kwargs.get('sp_map_patch', None)     # [H_p, W_p]
    sp_map_rgb   = kwargs.get('sp_map_rgb', None)        # [H_rgb, W_rgb]
    num_superpixels = kwargs.get('num_superpixels', None)
    use_superpixels = sp_map_patch is not None
    get_ig_iter_schedule = kwargs.get('get_ig_iter_schedule', None)
    if use_superpixels:
        masks = torch.ones((1, 1, 1, num_superpixels), dtype=torch.float32, device='cuda')  # one scalar per region
    else:
        masks = torch.ones_like(init_mask).float().cuda()
    masks = Variable(masks, requires_grad=True)

    # Precompute once, right after sp_map_patch/num_superpixels are known and masks is created —
    # this needs use_superpixels, sp_map_patch, and num_superpixels all in scope, which is only
    # true at this point in iGOS_p, not before.
    if use_superpixels:
        sp_counts = torch.bincount(sp_map_patch.reshape(-1), minlength=num_superpixels).float().clamp(min=1)
        sp_counts = sp_counts.view(1, 1, 1, num_superpixels).cuda()
        total_patches = sp_map_patch.numel()   # H_p * W_p — the denominator L1's mean() used to use implicitly

    # Precompute the superpixel adjacency graph ONCE. This is what actually
        # gives the optimizer a reason to keep neighboring superpixels coherent --
        # without it, L1/L2 can independently push any single superpixel to 0 or 1
        # with nothing pulling it back toward what its neighbors are doing, which
        # is exactly the scattered/checkerboard pattern you're seeing.
        sp_edge_index, sp_edge_sqdist = build_superpixel_adjacency(sp_map_patch, num_superpixels, rgb_image, sp_map_rgb)
        sp_tv_sigma = 0.05  # retune alongside L3 -- see note below
        sp_edge_weight = torch.exp(-sp_edge_sqdist / sp_tv_sigma)
        log(f'  sp graph: {sp_edge_index.shape[0]} edges, mean weight {sp_edge_weight.mean().item():.4f}, '
              f'weight sum {sp_edge_weight.sum().item():.4f}')  # sanity check for the tv_term normalization below
        
        # rank superpixels by the one-shot probe above, then freeze the
        # bottom (1 - sp_keep_fraction) at neutral for the whole run. Regions the
        # model shows near-zero sensitivity to can no longer pick up spurious
        # "hot" values from a noise draw or from TV/L1 pressure alone -- lower
        # sp_keep_fraction (e.g. 0.3) for faster/more conservative runs, raise it
        # (e.g. 0.7) if boundary_check overlays show real defects being pruned.
        sp_keep_fraction = kwargs.get('sp_keep_fraction', 0.5)
        k_keep = max(1, int(num_superpixels * sp_keep_fraction))
        importance = estimate_superpixel_importance(args, model, model_name, image, baseline, label,
                                                      sp_map_patch, num_superpixels, prompt, image_size,
                                                      positions, adapter, ig_iter=12, keyword_weights=keyword_weights)
        
        # turn the importance probe into a soft L1 anchor -- high-importance
        # superpixels anchor near 0 (encouraged to become salient), low-importance ones
        # anchor near 1 (encouraged to stay neutral). This is what regularization_loss's
        # `anchor` argument reads; see its docstring above for why.
        #_imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        #probe_anchor = (1.0 - _imp_norm).clamp(0.05, 0.98).detach()

        # compute reliability before it's needed for the keep-fraction decision below
        _sorted_imp, _ = torch.sort(importance.detach(), descending=True)
        _top1 = _sorted_imp[0].item()
        _top2 = _sorted_imp[1].item() if len(_sorted_imp) > 1 else 1e-8
        peak_ratio_top2 = _top1 / (_top2 + 1e-8)

        #active_idx = torch.topk(importance, k_keep).indices
        diff_prior = kwargs.get('reference_diff_prior', None)
        if diff_prior is not None:
            diff_t = torch.tensor(diff_prior, device=importance.device, dtype=importance.dtype)
            combined_score = 0.5 * (importance / (importance.max() + 1e-8)) + 0.5 * diff_t
        else:
            combined_score = importance

        # rescue an unreliable probe when there's no reference to fall back on --
        # but only for POSITIVE (anomaly) steps. For NO/good steps a flat/noisy probe is
        # the expected outcome (there's no single hidden region to rescue -- "everything
        # is empty" isn't localizable), so widening there just hands the optimizer more
        # freedom to paint arbitrary regions hot, which is what produced the checkerboard
        # overlays on the empty-box images.
        is_negative_label = kwargs.get('is_negative_label', False)
        # also widen when the keyword selection itself was thin (few positions,
        # or the single strongest keyword's importance is weak) -- a probe computed
        # from a single low-importance keyword is unreliable regardless of how peaked
        # its superpixel distribution looks, since the whole probe is downstream of
        # that one word's log-prob delta. keyword_weights is already threaded into
        # iGOS_p via kwargs (see keyword_weights = kwargs.get('keyword_weights', None)
        # near the top of this function).
        n_keywords = len(keyword_weights or [])
        # FIX: keyword_weights are normalized before reaching iGOS_p().
        # Therefore their maximum is normally below 1.0 and must not be used
        # as evidence that the keyword set is weak or unreliable.
        thin_keywords = n_keywords < 2
        probe_is_unreliable = (peak_ratio_top2 < 1.5 or thin_keywords)
        #max_kw_importance = max(keyword_weights or [0.0])
        #thin_keywords = (n_keywords < 2) or (max_kw_importance < 1.0)
        #if (peak_ratio_top2 < 1.5 or thin_keywords) and diff_prior is None and not is_negative_label:
        if (probe_is_unreliable and diff_prior is None and not is_negative_label):
            log(f'  [WARN] unreliable probe (top1/top2={peak_ratio_top2:.2f}, '
                  f'n_keywords={n_keywords} '
                  f'with no reference fallback -- widening keep_fraction')
            sp_keep_fraction = min(1.0, kwargs.get('sp_keep_fraction', 0.5) * 1.5)
            kwargs = dict(kwargs, sp_dampen_floor=max(kwargs.get('sp_dampen_floor', 0.15), 0.35))
        #elif (peak_ratio_top2 < 1.5 or thin_keywords) and diff_prior is None and is_negative_label:
        elif (probe_is_unreliable and diff_prior is None and is_negative_label):
            log(f'  [INFO] unreliable/thin probe on a NO/good step -- expected, keeping pre-filter tight')
        # if peak_ratio_top2 < 1.5 and diff_prior is None and not is_negative_label:
        #     log(f'  [WARN] unreliable probe (top1/top2={peak_ratio_top2:.2f}) with no reference '
        #           f'fallback -- widening keep_fraction and raising dampen floor')
        #     sp_keep_fraction = min(1.0, kwargs.get('sp_keep_fraction', 0.5) * 1.5)
        #     kwargs = dict(kwargs, sp_dampen_floor=max(kwargs.get('sp_dampen_floor', 0.15), 0.35))
        # elif peak_ratio_top2 < 1.5 and diff_prior is None and is_negative_label:
        #     log(f'  [INFO] unreliable probe on a NO/good step -- expected, keeping pre-filter tight')

        # rescue an unreliable probe when there's no reference to fall back on
        # if peak_ratio_top2 < 1.5 and diff_prior is None:
        #     log(f'  [WARN] unreliable probe (top1/top2={peak_ratio_top2:.2f}) with no reference '
        #           f'fallback -- widening keep_fraction and raising dampen floor')
        #     sp_keep_fraction = min(1.0, kwargs.get('sp_keep_fraction', 0.5) * 1.5)
        #     kwargs = dict(kwargs, sp_dampen_floor=max(kwargs.get('sp_dampen_floor', 0.15), 0.35))
        k_keep = max(1, int(num_superpixels * sp_keep_fraction))

        active_idx = torch.topk(combined_score, k_keep).indices

        # Hard-zero any superpixel majority-overlapping the static exclusion zone,
        # regardless of what the probe says about it -- see run_pipeline() for why.
        static_exclude_mask = kwargs.get('static_exclude_mask', None)
        if static_exclude_mask is not None:
            excl_t = torch.from_numpy(static_exclude_mask).to(sp_map_rgb.device)
            excluded_sp_ids = set()
            for sp_id in range(num_superpixels):
                region = (sp_map_rgb == sp_id)
                if region.sum() > 0 and (excl_t & region).float().mean() > 0.5:
                    excluded_sp_ids.add(sp_id)
            keep_mask = torch.tensor([idx.item() not in excluded_sp_ids for idx in active_idx])
            active_idx = active_idx[keep_mask]
            log(f'  excluded {len(excluded_sp_ids)} superpixels overlapping static rig zone')
            
        # soft dampening instead of a hard 0/1 mask. sp_dampen_floor is
        # the gradient fraction a "not selected" superpixel still receives --
        # a single noisy or texture-biased probe ranking should slow the optimizer
        # down on likely-irrelevant regions, not permanently forbid it from ever
        # discovering that one of them was actually the defect. 0.15 keeps most of
        # the compute saving while leaving a real path back for a wrongly-ranked
        # region across `iterations` steps of data gradient.
        sp_dampen_floor = kwargs.get('sp_dampen_floor', 0.35)
        active_sp_mask = torch.full((num_superpixels,), sp_dampen_floor, device='cuda')
        active_sp_mask[active_idx] = 1.0
        active_sp_mask = active_sp_mask.view(1, 1, 1, num_superpixels)
        log(f'  sp pre-filter: {k_keep}/{num_superpixels} at full weight, rest dampened to {sp_dampen_floor} '
              f'(importance range kept: {importance[active_idx].min().item():.4f} - {importance[active_idx].max().item():.4f})')
        # active_sp_mask = torch.zeros(num_superpixels, device='cuda')
        # active_sp_mask[active_idx] = 1.0
        # active_sp_mask = active_sp_mask.view(1, 1, 1, num_superpixels)
        # log(f'  sp pre-filter: keeping {k_keep}/{num_superpixels} superpixels '
        #       f'(importance range kept: {importance[active_idx].min().item():.4f} - {importance[active_idx].max().item():.4f})')

        # dump per-superpixel importance + selection as an overlay image, so you
        # can SEE whether the pre-filter kept the anomaly region or threw it away before
        # the optimizer ever ran. This is a static probe artifact -- if the anomaly region
        # is already grey/dropped here, iterations/L1/L2/L3 tuning downstream can't fix it.
        diag_dir = kwargs.get('diag_dir', None)
        diag_tag = kwargs.get('diag_tag', 'sample')
        if diag_dir is not None:
            import os as _os
            _os.makedirs(diag_dir, exist_ok=True)
            imp_full = importance.detach().cpu()
            log(f'  sp pre-filter FULL importance stats: min={imp_full.min().item():.4f}, '
                  f'max={imp_full.max().item():.4f}, mean={imp_full.mean().item():.4f}, '
                  f'median={imp_full.median().item():.4f}')
            # a simple "peakedness" score -- ratio of the top value to the
            # SECOND-highest value, and top value to the mean. A large ratio means one
            # region clearly stands out (trustworthy pre-filter). A ratio near 1.0
            # means the distribution is flat/noisy -- the top-K selection is close to
            # arbitrary, which is exactly what distinguished your two sample images.
            sorted_imp, _ = torch.sort(imp_full, descending=True)
            top1 = sorted_imp[0].item()
            top2 = sorted_imp[1].item() if len(sorted_imp) > 1 else 1e-8
            mean_val = imp_full.mean().item()
            peak_ratio_top2 = top1 / (top2 + 1e-8)
            peak_ratio_mean = top1 / (mean_val + 1e-8)
            reliability = "RELIABLE (clear standout)" if peak_ratio_top2 > 2.0 else "UNRELIABLE (flat/noisy distribution)"
            log(f'  [diag] importance peakedness: top1={top1:.4f}, top2={top2:.4f}, '
                  f'top1/top2 ratio={peak_ratio_top2:.2f}, top1/mean ratio={peak_ratio_mean:.2f} -> {reliability}')
            # per-superpixel dump: id, importance, kept?
            kept_set = set(active_idx.detach().cpu().tolist())
            for sp_id in range(num_superpixels):
                marker = "KEPT" if sp_id in kept_set else "dampened"
                log(f'    sp[{sp_id}] importance={imp_full[sp_id].item():.4f} -> {marker}')
            # visual overlay: expand active_sp_mask to RGB res and save as heatmap
            sp_map_rgb_local = kwargs.get('sp_map_rgb', None)
            rgb_image_local = kwargs.get('rgb_image', None)
            if sp_map_rgb_local is not None and rgb_image_local is not None:
                from .igos_core_methods_helper import upscale_superpixels as _ups
                from PIL import Image as _Image
                importance_norm = (imp_full - imp_full.min()) / (imp_full.max() - imp_full.min() + 1e-8)
                imp_dense = _ups(importance_norm.view(1, 1, 1, -1).cuda(), sp_map_rgb_local)  # [1,1,H,W]
                sel_dense = _ups(active_sp_mask, sp_map_rgb_local)  # [1,1,H,W], already 0.15/1.0
                base_img = rgb_image_local[0].detach().cpu().permute(1, 2, 0).numpy()
                base_img = (base_img * 255).clip(0, 255).astype('uint8')
                imp_vis = (imp_dense[0, 0].detach().cpu().numpy() * 255).astype('uint8')
                sel_vis = (sel_dense[0, 0].detach().cpu().numpy() * 255).astype('uint8')
                _Image.fromarray(base_img).save(_os.path.join(diag_dir, f'{diag_tag}_base.png'))
                _Image.fromarray(imp_vis).save(_os.path.join(diag_dir, f'{diag_tag}_importance_raw.png'))
                _Image.fromarray(sel_vis).save(_os.path.join(diag_dir, f'{diag_tag}_prefilter_selection.png'))
                log(f'  [diag] saved prefilter visualization to {diag_dir}/{diag_tag}_*.png')
    
    #######################################################################################################################

    if opt == 'NAG':
        cita=torch.zeros(1).cuda()

    losses_del, losses_ins, losses_l1, losses_tv, losses_l2 = [], [], [], [], []
    for i in range(iterations):
        if progress_callback is not None:
            progress_callback(
                iteration=i + 1,
                iterations=iterations,
                ig_iter=(get_ig_iter_schedule(i, iterations, ig_iter)
                         if get_ig_iter_schedule is not None else ig_iter),
            )
        total_grads = torch.zeros_like(masks).cuda()
        #up_masks = upscale(masks, image, resolution)

        # THE single call that decides what the model actually sees: a bilinear-blurred
        # patch-grid mask (old path) or a mask that steps cleanly at real object
        # boundaries (new path). Everything downstream (integrated_gradient,
        # interval_score) is unaware of the difference -- it just receives a dense
        # H_p x W_p tensor either way.
        up_masks = upscale_superpixels(masks, sp_map_patch) if use_superpixels else upscale(masks, image, resolution)
        #####################################################################################
        
        # reallocate the flat ig_iter budget across the run -- see
        # get_ig_iter_schedule() docstring in the main script. `ig_iter` (the
        # function argument) is now treated as the BASE value the schedule
        # scales around, not the literal value passed to integrated_gradient.
        if get_ig_iter_schedule is not None:
            ig_iter_i = get_ig_iter_schedule(i, iterations, ig_iter)
        else:
            ig_iter_i = ig_iter
            
        loss_del = integrated_gradient(args, model, model_name, image, baseline, label, up_masks, ig_iter_i, prompt=prompt, image_size=image_size, positions=positions, adapter=adapter, keyword_weights=keyword_weights) #Added adapter parameter for Qwen3-VL
        total_grads += masks.grad.clone()
        masks.grad.zero_()
        # grad_del = masks.grad.clone()
        # if use_superpixels:
        #     grad_del = grad_del / sp_counts        # <- undoes the "bigger region = bigger gradient" artifact
        # total_grads += grad_del
        # masks.grad.zero_()


        loss_ins = integrated_gradient(args, model, model_name, baseline, image, label, up_masks, ig_iter_i, prompt=prompt, image_size=image_size, positions=positions, adapter=adapter, keyword_weights=keyword_weights) #Added adapter parameter for Qwen3-VL
        total_grads += -masks.grad.clone()
        masks.grad.zero_()
        # grad_ins = masks.grad.clone()
        # if use_superpixels:
        #     grad_ins = grad_ins / sp_counts
        # total_grads += -grad_ins
        # masks.grad.zero_()

         # --- Diagnostic: snapshot the DATA gradient's norm before regularization
        # is added in. total_grads at this point holds ONLY the del+ins contribution.
        data_grad_norm = total_grads.norm().item()

        L2 = exp_decay(args.L2, i, args.gamma)
        # Floor the decay so the "pull toward 1" anchor never fully vanishes late in
        # training. By iteration 30 the old exp_decay had already dropped L2 to ~0.004
        # (from 0.1) -- almost no anchor left, which lines up with where std started
        # oscillating hardest (iterations 30-39 in the log you posted).
        # Set dense_l2_floor_fraction=0.0 (see kwarg near the top of this function) to
        # disable this floor entirely and match the original paper's plain exp_decay.
        L2 = max(L2, dense_l2_floor_fraction * args.L2)
        loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks, resolution)
        #log(f'  sp_tv edges: {sp_edge_index.shape[0]}, mean edge weight: {sp_edge_weight.mean().item():.4f}')
        if use_superpixels:
            log(f"  sp_tv edges: {sp_edge_index.shape[0]}, mean edge weight: {sp_edge_weight.mean().item():.4f}")
        else:
            log(f"  dense patch TV: mask_shape={tuple(masks.shape[-2:])}")
        losses = loss_l1 + loss_tv + loss_l2
        losses.sum().backward()
        # --- Diagnostic: masks.grad here is STILL just the regularization gradient
        # (total_grads hasn't absorbed it yet) -- so its norm is directly comparable
        # to data_grad_norm above. If reg consistently dominates data by a wide
        # margin, that's the smoking gun for "regularization swamping the signal."
        reg_grad_norm = masks.grad.norm().item()
        log(f'  grad norms -> data: {data_grad_norm:.4f}, reg: {reg_grad_norm:.4f}, '
              f'ratio(reg/data): {reg_grad_norm / (data_grad_norm + 1e-8):.4f}')
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        # zero the gradient for every superpixel the pre-filter didn't
        # select, so NAG/line-search can never move it off neutral this run,
        # no matter what L1/L2/TV or the data term compute for it. This is what
        # turns "ranked by importance" into an actual reduction of the search
        # space rather than just a soft prior.
        if use_superpixels:
            total_grads = total_grads * active_sp_mask

        # Safety net: cap the step size regardless of what scale the data-loss term
        # happens to be on. This is what should have prevented iteration-0 saturation
        # in the log-sum runs -- a scale mismatch in the loss shouldn't be able to
        # single-handedly wipe out the whole mask in one NAG step.
        # The original paper had NO such gradient-norm clipping at all -- set
        # grad_clip_norm=None (see kwarg near the top of this function) to disable.
        if grad_clip_norm is not None:
            grad_norm = total_grads.norm()
            if grad_norm > grad_clip_norm:
                total_grads = total_grads * (grad_clip_norm / (grad_norm + 1e-8))

        # how concentrated is the per-superpixel gradient this iteration?
        # A healthy run should show grad mass concentrating on a SHRINKING set of
        # superpixels over time. If the top-K stays flat/uniform across iterations,
        # the optimizer isn't discovering structure -- it's just applying uniform
        # pressure everywhere, which is consistent with your Del/Ins staying near 0.
        if use_superpixels and i % 5 == 0:
            grad_flat = total_grads.view(-1).detach().abs()
            top5 = torch.topk(grad_flat, min(5, grad_flat.numel()))
            log(f'  [diag it={i}] |grad| top5 sp_ids={top5.indices.tolist()} '
                  f'vals={[round(v,4) for v in top5.values.tolist()]}, '
                  f'grad_std={grad_flat.std().item():.4f}, grad_mean={grad_flat.mean().item():.4f}')

        if opt == 'LS':
            lrs = line_search(masks, total_grads, loss_function, lr)
            masks.data -= total_grads * lrs
        
        if opt == 'NAG':
            e = i / (i + args.momentum) 
            #e = min(i / (i + args.momentum), 0.5)
            cita_p = cita
            cita = masks.data - lr * total_grads
            masks.data = cita + e * (cita - cita_p)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_l1.append(loss_l1.item())
        losses_tv.append(loss_tv.item())
        losses_l2.append(loss_l2.item())

        log(f'iteration: {i}/{iterations} lr: {lr:.4f} loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')
        masks.grad.zero_()
        # The original paper clamps to the full [0, 1] range; mask_clamp_range
        # defaults to the GPT-tuned (0.02, 0.98) so the optimizer can never fully
        # saturate a patch -- override to (0.0, 1.0) for exact paper fidelity.
        masks.data.clamp_(mask_clamp_range[0], mask_clamp_range[1])

        if iteration_callback is not None:
            iteration_callback(masks.detach(), iteration=i + 1, iterations=iterations,
                               ig_iter=ig_iter_i)

        log(f'  mask stats -> min: {masks.min().item():.4f}, max: {masks.max().item():.4f}, std: {masks.std().item():.4f}')

    # Hand back the DENSE (H_p x W_p) mask, not the compact K-vector. This is the
    # load-bearing decision that keeps metric() and save_heatmaps() unmodified --
    # they already expect exactly this shape/contract from the non-superpixel path.
    final_mask = upscale_superpixels(masks, sp_map_patch).detach() if use_superpixels else masks

    # Also return the COMPACT per-superpixel vector. save_heatmaps() cannot recover
    # this from final_mask -- by the time it's on the patch grid, which pixels
    # belonged to which superpixel is already lost. The caller needs this raw
    # K-length vector to re-expand at full RGB resolution via sp_map_rgb, or the
    # saved heatmap will show a bilinear blur of the patch grid instead of the
    # real superpixel shapes (this is why the overlay and boundary_check don't match).
    compact_mask = masks.detach() if use_superpixels else None

    return final_mask, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, None, compact_mask #Removed 'masks' as the first parameter to be returned with final_mask
