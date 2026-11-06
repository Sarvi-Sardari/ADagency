"""Image baselines and SLIC/SAM3 segmentation preprocessing."""

import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic

def get_kernel_size(image_size, mode="strong"):
    w, h = image_size
    if mode == "paper_repo":
        if w >= 2000:
            return 501
        if w >= 800:
            return 301
        if w >= 500:
            return 201
        return 101
    if mode != "strong":
        raise ValueError(
            f"Unknown blur_kernel_mode={mode!r}; expected 'paper_repo' or 'strong'."
        )
    k = max(3, min(w, h) // 3)   #10 # Increased from // 10 to // 3 to guarantee the anomaly is destroyed in the baseline #This ensures that the blur window/filter will always be 10% of the shortest dimension of the image
    return k + 1 if k % 2 == 0 else k  #Gaussian Blur algo requires kernel size to be odd (so that pixel matrix has an exact single center pixel) 

def make_blur_baseline(image, log=print, kernel_mode="strong"):
    # CLEANUP: this function has exactly one responsibility:
    # create a blurred version of the candidate.
    kernel_size = get_kernel_size(image.size, mode=kernel_mode)

    # Blur the candidate while preserving its geometry and dimensions.
    blur = cv2.GaussianBlur(np.asarray(image), (kernel_size, kernel_size), sigmaX=kernel_size - 1,)

    # Diagnostic only: confirms that the blur removed enough information.
    orig_np = np.asarray(image).astype(np.float32)
    diff = np.abs(orig_np - blur.astype(np.float32))

    log(
        f"  [diag] blur baseline diff -> "
        f"mean={diff.mean():.2f}, max={diff.max():.2f}, "
        f"kernel_size={kernel_size}"
    )

    return Image.fromarray(blur.astype(np.uint8))

def get_iter_schedule(num_superpixels, args):
    """Return the config-selected optimization budget.

    The notebook prototype silently hard-coded ``20/4`` here, so changing
    ``Args.iterations`` or ``Args.ig_iter`` (including the original-repository
    and paper presets) had no effect.  Ablation arguments must be authoritative;
    any future dynamic schedule should be exposed as an explicit config mode.
    """
    del num_superpixels
    return int(args.iterations), int(args.ig_iter)

def get_ig_iter_schedule(iteration_idx, total_iterations, base_ig_iter):
    """
    Reallocates the ig_iter budget across the run instead of holding it flat.
    Early iterations see a near-uniform mask (init_mask=1s) -- the IG interval
    integral has little curvature to resolve yet, so 2 steps approximates it about
    as well as 4-6 would. Late iterations are where the mask has actually started
    discriminating between superpixels, which is exactly where a coarse IG estimate
    costs you gradient accuracy. Same total step count as a flat ig_iter=4 over most
    schedules, just spent where it matters.
    """
    progress = iteration_idx / max(1, total_iterations - 1)
    if progress < 0.34:
        return max(2, base_ig_iter - 2)
    elif progress < 0.67:
        return base_ig_iter
    else:
        return base_ig_iter + 2

# Functions to use sam3 instead of slic for segmentation masks
################################################################################################################
def load_and_resize_image(image_path, args):
    """Single source of truth for the resize every sample gets -- used by
    both the SAM3 preprocessing pass and the main loop, so their target_width/
    target_height can never drift apart from each other."""
    image = Image.open(image_path).convert("RGB")
    target_width = (args.input_size // 28) * 28
    target_height = int(target_width * image.height / image.width)
    target_height = max(28, (target_height // 28) * 28)
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image, target_width, target_height

def segment_with_sam3(img_np, sam3_generator, min_mask_area=64):
    """
    Runs SAM3's automatic (prompt-free) mask generator and flattens the
    resulting, possibly-overlapping instance masks into a single dense
    label map -- same [H, W] integer-array contract slic() returns, so
    every downstream function (pool_labels_to_grid, build_superpixel_adjacency,
    upscale_superpixels) needs zero changes.
    """
    pil_img = Image.fromarray(img_np)
    outputs = sam3_generator(pil_img, points_per_batch=64)
    masks = outputs["masks"]

    H, W = img_np.shape[:2]
    label_map = np.full((H, W), -1, dtype=np.int64)

    # Paint largest masks first -- SAM3's masks can overlap, SLIC's never do,
    # so bigger/more-confident regions claim pixels before slivers can.
    order = sorted(range(len(masks)), key=lambda i: np.asarray(masks[i]).sum(), reverse=True)
    next_id = 0
    for i in order:
        m = np.asarray(masks[i]).astype(bool)
        if m.sum() < min_mask_area:
            continue
        unclaimed = m & (label_map == -1)
        if unclaimed.sum() == 0:
            continue
        label_map[unclaimed] = next_id
        next_id += 1

    # Anything no mask claimed (background, gaps) becomes one shared region --
    # guarantees full coverage, which pool_labels_to_grid/adjacency assume.
    if (label_map == -1).any():
        label_map[label_map == -1] = next_id
        next_id += 1

    return label_map, next_id

def segment_with_slic_within_sam3(img_np, sam3_generator, args, total_n_seg, sam3_labels=None):
    """
    Runs SAM3 first to get semantically-coherent object masks (the "separation"
    SAM3 is actually good at), then subdivides EACH SAM3 mask independently with
    its own SLIC call restricted to that mask's pixels -- so no superpixel is
    ever allowed to straddle a real object boundary, while defects that live
    *inside* one object (the actual MVTec-LOCO anomaly scale) still get the
    fine intra-object subdivision that SAM3 alone never provides (SAM3 returns
    one blob per object, which is too coarse to localize anything smaller than
    the whole part).

    Requires skimage>=0.19 for slic()'s `mask=` parameter.
    """
    if sam3_labels is None:
        sam3_labels, n_sam3 = segment_with_sam3(img_np, sam3_generator, min_mask_area=args.sam3_min_mask_area)
    else:
        n_sam3 = int(sam3_labels.max()) + 1

    H, W = img_np.shape[:2]
    joint = np.zeros((H, W), dtype=np.int64)
    next_id = 0
    total_area = H * W

    # log SAM3's own region count and area distribution before subdivision --
    # if SAM3 returns only 3-8 regions on a compartmentalized object, one SAM3 region
    # is very likely straddling both the anomalous object and empty background, and no
    # amount of downstream SLIC subdivision or iGOS+ tuning can un-merge them once the
    # hybrid pipeline has committed to those outer boundaries. This is the first thing
    # to check when a specific sample won't converge to a clean positive Del regardless
    # of iterations/L1/dampening.
    print(f"  [hybrid diag] {n_sam3} SAM3 regions, areas: "
          f"{sorted([(sam3_labels == i).sum() for i in range(n_sam3)], reverse=True)}")

    # Flat, uniform regions (e.g. the plain background prop) shouldn't out-earn
    # textured regions just for being large -- score budget by structure instead.
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edge_map = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    total_edge = edge_map.sum() + 1e-6

    # Sort regions largest-first so the biggest (most important to subdivide
    # well) regions get first claim on the shared budget, and smaller regions
    # -- which is where SLIC overshoot is worst, per your log -- are naturally
    # squeezed toward n_seg_region=1 as the remaining budget shrinks.
    region_ids = sorted(range(n_sam3), key=lambda i: (sam3_labels == i).sum(), reverse=True)
    remaining_budget = total_n_seg

    for idx, sam3_id in enumerate(region_ids):
        region_mask = sam3_labels == sam3_id
        area = region_mask.sum()
        if area == 0:
            continue

        # CRITICAL FIX 1: Reserve Budget
        # Guarantee we save at least 1 segment for every SAM3 region we haven't processed yet
        regions_left = len(region_ids) - idx
        max_affordable = max(1, remaining_budget - regions_left + 1)

        # --- START OF LINEAR ALLOCATION PATCH ---
        # Replaced brittle exponential bias with a robust, non-starving hybrid metric.
        # 40% of the budget scales purely with area (guarantees flat walls don't collapse).
        # 60% of the budget scales with texture/edges (handles fine details on object bins).
        area_weight = area / total_area
        edge_weight = edge_map[region_mask].sum() / total_edge
        
        combined_weight = (0.40 * area_weight) + (0.60 * edge_weight)
        n_seg_region = max(1, min(max_affordable, round(total_n_seg * combined_weight)))
        # --- END OF LINEAR ALLOCATION PATCH ---

        if n_seg_region == 1:
            joint[region_mask] = next_id
            next_id += 1
            remaining_budget -= 1
            continue

        ys, xs = np.where(region_mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub_img = img_np[y0:y1, x0:x1]
        sub_mask = region_mask[y0:y1, x0:x1]

        # CRITICAL FIX 2: SLIC Dampening
        # SLIC naturally overshoots inside small/irregular masks, so we ask for slightly fewer
        ask_seg = max(1, int(n_seg_region * 0.85))

        sub_labels = slic(sub_img, n_segments=ask_seg, compactness=args.sp_compactness,
                           start_label=0, channel_axis=-1, mask=sub_mask)
        
        actual_sub_ids = np.unique(sub_labels[sub_mask])

        # CRITICAL FIX 3: Fast Area-Merge (Replaces buggy RAG graph merge)
        # If SLIC still overshot the maximum we can afford, force it back down
        if len(actual_sub_ids) > max_affordable:
            # Get areas of each sub-segment and sort from largest to smallest
            sizes = [(local_id, (sub_labels[sub_mask] == local_id).sum()) for local_id in actual_sub_ids]
            sizes.sort(key=lambda x: x[1], reverse=True)
            
            biggest_id = sizes[0][0]
            keep_ids = set([x[0] for x in sizes[:max_affordable]])
            
            # Instantly absorb all the tiny overflow slivers into the largest local segment
            for local_id in actual_sub_ids:
                if local_id not in keep_ids:
                    sub_labels[sub_mask & (sub_labels == local_id)] = biggest_id
                    
            # Re-evaluate surviving IDs (now mathematically guaranteed to be <= max_affordable)
            actual_sub_ids = np.unique(sub_labels[sub_mask])

        # Map the surviving local IDs into the global joint mask    
        for local_id in actual_sub_ids:
            joint[y0:y1, x0:x1][sub_mask & (sub_labels == local_id)] = next_id
            next_id += 1

        remaining_budget -= len(actual_sub_ids)


    return joint, next_id

def precompute_sam3_segmentations(samples, args):
    """
    Runs SAM3 once per UNIQUE image (steps share images) as a standalone
    pass, entirely before Qwen3-VL loads. This is what fixes the OOM --
    SAM3 and Qwen3-VL are never resident on the GPU together.
    Returns {image_path: (label_map, n_labels)}.
    """
    import gc
    import sys

    checkpoint = Path(args.sam3_model_id)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "sam3.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint file not found: {checkpoint}")

    # All SAM3/hybrid arms for a dataset use the same source image and prompt
    # settings. Persist the label maps next to the method diagnostics so later
    # arms can reuse them without loading or running SAM3 again.
    shared_cache_dir = Path(args.output_dir).parent / "_shared_sam3_cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    unique_paths = list(dict.fromkeys(s.image_path for s in samples))
    cache = {}
    missing = []
    for path in unique_paths:
        source = Path(path).expanduser().resolve()
        source_stat = source.stat()
        checkpoint_stat = checkpoint.stat()
        cache_signature = {
            "source": str(source),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "input_size": int(args.input_size),
            "prompts": list(args.sam3_text_prompts),
            "confidence_threshold": float(args.sam3_confidence_threshold),
            "min_mask_area": int(args.sam3_min_mask_area),
        }
        digest = hashlib.sha256(
            json.dumps(cache_signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        cache_path = shared_cache_dir / f"{source.stem}_{digest}.npz"
        try:
            with np.load(cache_path, allow_pickle=False) as stored:
                labels = stored["labels"].astype(np.int64, copy=False)
                n_labels = int(stored["n_labels"])
            if labels.ndim != 2 or n_labels < 1:
                raise ValueError("invalid cached SAM3 label map")
            cache[path] = (labels, n_labels)
            print(f"  [SAM3 cache] {source.name} -> {n_labels} segments")
        except (OSError, ValueError, KeyError):
            missing.append((path, cache_path))

    if not missing:
        return cache

    # This project's checkpoint declares ``sam3_video``. Transformers 4.57 in
    # the XAI environment cannot construct that architecture through its
    # generic mask-generation pipeline. Reuse the local SAM3 image loader that
    # already powers the standard mask-generation stage instead of upgrading
    # Transformers and risking a Qwen3-VL compatibility regression.
    repo_dir = str(Path(args.sam3_repo_dir).expanduser().resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
    )
    sam3_processor = Sam3Processor(
        sam3_model,
        device="cuda",
        confidence_threshold=float(args.sam3_confidence_threshold),
    )

    class LocalSam3MaskGenerator:
        """Adapt the local text-prompted SAM3 processor to the mask contract."""

        def __init__(self, processor):
            self.processor = processor

        def __call__(self, image, **_kwargs):
            state = self.processor.set_image(image)
            masks = []
            for prompt in args.sam3_text_prompts:
                self.processor.reset_all_prompts(state)
                output = self.processor.set_text_prompt(
                    prompt=str(prompt), state=state
                )
                output_masks = output.get("masks")
                if output_masks is None:
                    continue
                for mask in output_masks:
                    mask_np = mask.detach().squeeze().cpu().numpy().astype(bool)
                    if mask_np.ndim == 2:
                        masks.append(mask_np)
            return {"masks": masks}

    sam3_generator = LocalSam3MaskGenerator(sam3_processor)

    for path, cache_path in missing:
        image, _, _ = load_and_resize_image(path, args)
        img_np = np.asarray(image)
        cache[path] = segment_with_sam3(img_np, sam3_generator, min_mask_area=args.sam3_min_mask_area)
        temporary = cache_path.with_name(
            f".{cache_path.name}.tmp{os.getpid()}.npz"
        )
        np.savez_compressed(
            temporary,
            labels=cache[path][0].astype(np.int64, copy=False),
            n_labels=np.asarray(cache[path][1], dtype=np.int64),
        )
        os.replace(temporary, cache_path)
        print(f"  [SAM3 preprocessing] {os.path.basename(path)} -> {cache[path][1]} segments")

    del sam3_generator, sam3_processor, sam3_model
    gc.collect()
    torch.cuda.empty_cache()   # SAM3's VRAM is fully released before Qwen3-VL loads

    return cache

###############################################################################################################

