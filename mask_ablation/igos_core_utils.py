#igos_core_utils.py

"""
Utility functions for interpretation generation.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS

This local adaptation contains the probability, rendering, and persistence
utilities used by the mask-ablation engine.
"""

import torch
import torch.nn.functional as F
import os
import cv2
import sys

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from textwrap import fill
import yake
import torch.utils.checkpoint as checkpoint
from io import BytesIO

# mean and standard deviation for the imagenet dataset
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

special_ids = [12, 13, 29871, 29889, 29892, 1919, 869, 29899, 29897, 29898, 278, 322, 310, 263, 385, 445, 393, 338, 411]
CHOICES = ['A','B','C','D','E','F']

def eprint(*args, **kwargs):
    """
        Prints to the std.err

    :param args:
    :param kwargs:
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


def get_data(args, row):
    if args.dataset == 'MMVP':
        photo_id = row['lndex']
        image_path = os.path.join(args.image_folder, f"{photo_id}.jpg")
        image = Image.open(image_path).convert('RGB')

        if args.choices:
            opts = row['Options'].split(' ')
            cur_prompt = f"{row['Question']}\nA. {opts[1]}\nB. {opts[3]}\n"
            qs = cur_prompt + "Answer only with the option's letter A or B from the given choices directly."
        else:
            qs = row['Question']
            cur_prompt = qs
    elif args.dataset == 'cvbench':
        image_path = os.path.join(args.image_folder, row["filename"])
        image = Image.open(image_path).convert('RGB')
        if args.choices:
            choices = row['choices']
            choices_text = ''
            for j,item in enumerate(choices):
                choices_text += f'\n{CHOICES[j]}. {item}'

            cur_prompt = row['question'] + choices_text
            qs = cur_prompt + '\nAnswer with the option\'s letter from the given choices directly.'
        else:
            qs = row['question']
            cur_prompt = qs
    elif args.dataset == 'mmstar':
        image_stream = BytesIO(row["image"]['bytes'])
        image = Image.open(image_stream).convert('RGB')
        if args.choices:
            cur_prompt = row['question']
            qs = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        else:
            qs = row['question'].split('?')[0] + '?'
            cur_prompt = qs
            
    elif args.dataset == 'llava-bench':
        image_path = os.path.join(args.image_folder, row["image"])
        image = Image.open(image_path).convert('RGB')
        cur_prompt = row['text']
        qs = cur_prompt
    return image, qs, cur_prompt
    

def get_kernel_size(image_size):
    if image_size[0] >=2000:
        kernel_size = 501
    elif image_size[0] >= 800:
        kernel_size = 301
    elif image_size[0] >= 500:
        kernel_size = 201
    else:
        kernel_size = 101
    return kernel_size

# def save_heatmaps(masks, images, size, dataset_name, image_id, step_number, heatmap_dir, overlay_dir): #(index, index_o, outdir, model_name, box, classes, labels, out=224): #Added dataset_name, image_id, step_number, heatmap_dir and overlay_dir parameters for file name saving
#     if isinstance(images, list):
#         images = images[1]
#     elif images.shape[0] != 1:
#         images = images[0].unsqueeze(0)
#     #masks = masks.view(-1, 1, size, size) #Removed to optimize for rectangular masks

#     # #Normalize BEFORE upsampling to preserve high contrast 
#     # masks = (masks - torch.min(masks)) / (torch.max(masks) - torch.min(masks)) #Optimized for rectangular masks

#     up = torch.nn.UpsamplingBilinear2d(size=images.shape[-2:]).cuda()

#     u_mask = up(masks)
#     u_mask = u_mask.permute((0,2, 3, 1))

#     # Normalize the mask
#     #u_mask = (u_mask - torch.min(u_mask)) / (torch.max(u_mask) - torch.min(u_mask))
#     u_mask = u_mask.cpu().detach().numpy()
#     lo, hi = np.percentile(u_mask, 5), np.percentile(u_mask, 99) #Changed the lo percentile to 40 from 5

#     # If the mask barely varies (near-uniform, e.g. a NO/good step with nothing to
#     # highlight), lo/hi collapse toward each other and (u_mask - lo)/(hi - lo) blows up
#     # numerically -- guard with a minimum spread so flat masks stay flat/blue instead of
#     # amplifying noise into false-looking hot spots.
#     if hi - lo < 0.05:
#         hi = lo + 0.05

#     u_mask = np.clip((u_mask - lo) / (hi - lo), 0, 1)
#     #u_mask = u_mask.cpu().detach().numpy()

#     #Apply threshold to isolate the anomaly
#     #u_mask[u_mask < 0.6] = 0

#     # deprocess images
#     # images = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
#     # images = images.numpy()
    
#     #visual_tensor already contains raw RGB values in [0,1].
#     images = (images.cpu().detach().permute(0, 2, 3, 1).numpy())

#     for i, (image, u_mask) in enumerate(zip(images, u_mask)):

#         # OpenCV applyColorMap expects a 2-D intensity map.
#         #mask_2d = (u_mask[..., 0] if u_mask.ndim == 3 else u_mask)
        
#         # get the color map and normalize to 0-1
#         heatmap = cv2.applyColorMap(np.uint8(255 * u_mask), cv2.COLORMAP_JET)
#         #heatmap = cv2.applyColorMap(np.uint8(255 * u_mask), cv2.COLORMAP_TURBO)
#         #heatmap = np.float32(heatmap / 255)
#         heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
#         heatmap = np.float32(heatmap / 255.0)
#         # apply heatmap color only where the anomaly map is active.
#         # Background pixels remain close to the original image instead of
#         # receiving a permanent 50% blue tint.
#         #alpha = (0.65 * np.power(mask_2d, 0.8))[..., None]
#         #overlay = ((1.0 - alpha) * image + alpha * heatmap)
#         #overlay = np.clip(overlay, 0.0, 1.0)

#         # overlay the mask over the image
#         #overlay = (u_mask ** 0.8) *0.5* image + (1 - u_mask ** 0.8) * heatmap
#         overlay = 0.5 * heatmap + 0.5 * image
#         #cv2.normalize(overlay.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         overlay[overlay < 0] = 0
#         overlay[overlay > 1] = 1

#         #plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_heatmap.jpg'), heatmap)
#         #plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_overlay.jpg'), overlay)

#         # ==========================================================
#         #               NEW FILENAME FORMATTING
#         # 
#         # Format: <type>_<dataset_name>_<image_id>_step_<step_number>.png
#         # 
#         # Note: If batch size is > 1, we append '_i' to avoid overwriting, 
#         # though your pipeline currently processes batch size of 1.
#         # ==========================================================

#         suffix = f"_{i}" if i > 0 else ""

#         heatmap_name = f'heatmap_{dataset_name}_{image_id}_step_{step_number}{suffix}.png'
#         overlay_name = f'overlay_{dataset_name}_{image_id}_step_{step_number}{suffix}.png'

#         plt.imsave(os.path.join(heatmap_dir, heatmap_name), heatmap)
#         plt.imsave(os.path.join(overlay_dir, overlay_name), overlay)

def save_heatmaps(
    masks,
    images,
    size,
    dataset_name,
    image_id,
    step_number,
    heatmap_dir,
    overlay_dir,
    raw_mask_dir=None,
    write_files=True,
    display_masks=None,
    render_profile="ablation",
):
    """
    Prepare metadata for four representations of an anomaly-score map and,
    when ``write_files`` is true, save them:

    1. raw .npy:
       Exact float32 continuous anomaly scores. Use this for AUROC/AUPRO.

    2. raw preview PNG:
       Grayscale visualization using a fixed [0, 1] scale.
       Useful for inspecting the real score strength without percentile stretching.

    3. colored heatmap PNG:
       Percentile-normalized visualization for easier visual interpretation.

    4. overlay PNG:
       Colored heatmap blended with the RGB image.

    Expected input:
        masks contains anomaly scores where:
            0 = not anomalous
            1 = strongly anomalous
    """

    if isinstance(images, list):
        images = images[1]
    elif images.shape[0] != 1:
        images = images[0].unsqueeze(0)

    # Create output directories here so every call is safe.
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Keep backward compatibility. When no explicit raw directory is passed,
    # create a sibling folder next to heatmaps/ and overlays/.
    if raw_mask_dir is None:
        common_parent = os.path.dirname(heatmap_dir)
        raw_mask_dir = os.path.join(common_parent, "raw_masks")

    os.makedirs(raw_mask_dir, exist_ok=True)

    # Resize the continuous anomaly map to the same spatial size as visual_tensor.
    # This remains a continuous float map; no normalization has happened yet.
    upsampled_masks = F.interpolate(masks.float(), size=images.shape[-2:], mode="bilinear", align_corners=False)
    display_masks = masks if display_masks is None else display_masks
    upsampled_display_masks = F.interpolate(
        display_masks.float(), size=images.shape[-2:], mode="bilinear", align_corners=False
    )

    # [B, 1, H, W] -> [B, H, W]
    raw_masks = (upsampled_masks.detach().cpu().numpy())

    if raw_masks.ndim != 4:
        raise ValueError("save_heatmaps expected masks with shape [B, C, H, W], but received shape {raw_masks.shape}")

    if raw_masks.shape[1] != 1:
        raise ValueError("save_heatmaps expected a one-channel anomaly map, but received {raw_masks.shape[1]} channels")

    raw_masks = raw_masks[:, 0, :, :]
    display_masks_np = upsampled_display_masks.detach().cpu().numpy()[:, 0, :, :]

    # The caller normally already clamps the anomaly score, but do it again
    # before saving to prevent small numerical values outside [0, 1].
    raw_masks = np.clip(raw_masks, 0.0, 1.0).astype(np.float32)

    # visual_tensor already contains raw RGB values in [0, 1].
    images_np = (images.detach().cpu().permute(0, 2, 3, 1).numpy())

    saved_paths = []

    for i, (image, raw_mask_2d, display_mask_2d) in enumerate(
        zip(images_np, raw_masks, display_masks_np)
    ):
        suffix = f"_{i}" if i > 0 else ""

        base_name = (f"{dataset_name}_{image_id}_step_{step_number}{suffix}")

        raw_npy_name = f"raw_anomaly_{base_name}.npy"
        raw_preview_name = f"raw_preview_{base_name}.png"
        heatmap_name = f"heatmap_{base_name}.png"
        overlay_name = f"overlay_{base_name}.png"

        raw_npy_path = os.path.join(raw_mask_dir, raw_npy_name)

        raw_preview_path = os.path.join(raw_mask_dir, raw_preview_name)

        heatmap_path = os.path.join(heatmap_dir, heatmap_name)

        overlay_path = os.path.join(overlay_dir, overlay_name)

        # ==========================================================
        # 1. SAVE THE EXACT RAW CONTINUOUS ANOMALY SCORES
        # ==========================================================
        # This preserves every float32 value and is the file that
        # must later be loaded for AUROC and AUPRO.
        if write_files:
            np.save(raw_npy_path, raw_mask_2d.astype(np.float32))

        # ==========================================================
        # 2. SAVE A FIXED-SCALE RAW PREVIEW
        # ==========================================================
        # No percentile normalization is applied here.
        #
        # black = score 0
        # white = score 1
        #
        # Therefore, two different images use exactly the same scale.
        if write_files:
            plt.imsave(raw_preview_path, raw_mask_2d, cmap="gray", vmin=0.0, vmax=1.0)

        # ==========================================================
        # 3. CREATE THE DISPLAY-ENHANCED HEATMAP
        # ==========================================================
        # Work on a copy so raw_mask_2d remains unchanged.
        # Human-facing artifacts use the optional smoothed display map. The
        # exact raw mask above remains the lossless metric/publishing source.
        display_mask = np.clip(display_mask_2d.copy(), 0.0, 1.0)

        if render_profile == "paper_original":
            # Original Qwen repository visualization: global min/max
            # normalization happens on the native 32x32 mask before bilinear
            # resize; the caller supplies that normalized result here.
            # Rendering is JET followed by heatmap*0.4 + original image.
            lo = 0.0
            hi = 1.0
        elif render_profile == "ablation":
            lo = float(np.percentile(display_mask, 5))
            hi = float(np.percentile(display_mask, 99))
        else:
            raise ValueError(f"Unknown heatmap render_profile={render_profile!r}")

        # Avoid amplifying an almost-uniform map into a false-looking
        # high-contrast heatmap.
        minimum_spread = 1e-12 if render_profile == "paper_original" else 0.05
        if hi - lo < minimum_spread:
            hi = lo + minimum_spread

        display_mask = np.clip((display_mask - lo) / (hi - lo), 0.0, 1.0)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255.0 * display_mask), cv2.COLORMAP_JET)

        heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ==========================================================
        # 4. CREATE THE RGB OVERLAY
        # ==========================================================
        if render_profile == "paper_original":
            overlay = 0.5 * heatmap + 0.5 * image #overlay = 0.4 * heatmap + image 
        else:
            overlay = 0.5 * heatmap + 0.5 * image

        overlay = np.clip(overlay, 0.0, 1.0)

        if write_files:
            plt.imsave(heatmap_path, heatmap)
            plt.imsave(overlay_path, overlay)

        saved_paths.append(
            {
                "raw_mask_path": raw_npy_path,
                "raw_preview_path": raw_preview_path,
                "heatmap_path": heatmap_path,
                "overlay_path": overlay_path,
                "raw_shape": list(raw_mask_2d.shape),
                "raw_min": float(raw_mask_2d.min()),
                "raw_max": float(raw_mask_2d.max()),
                "raw_mean": float(raw_mask_2d.mean()),
                "raw_std": float(raw_mask_2d.std()),
                "display_percentile_low": lo,
                "display_percentile_high": hi,
            }
        )

    return saved_paths


def save_masks(masks, index, categories, mask_name, outdir):
    """
        Saves the generated masks as numpy.ndarrays.

    :param masks:
    :param index:
    :param categories:
    :param mask_name:
    :param outdir:
    :return:
    """
    masks = masks.cpu().detach().numpy()
    for i, (mask, category) in enumerate(zip(masks, categories), start=index):
        np.save(os.path.join(outdir, f'{mask_name}_{i+1}_mask_{category}.npy'), mask)

def save_loss(loss_del, loss_ins, loss_l1, loss_tv, loss_l2, index, index_o, outdir, loss_comb_del=None, loss_comb_ins=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    iterations = np.arange(len(loss_del))

    if loss_comb_del:
        ax1.plot(iterations, loss_comb_del, label='loss comb del', marker='^')
    if loss_comb_ins:
        ax1.plot(iterations, loss_comb_ins, label='loss comb ins', marker='<')
    ax1.plot(iterations, loss_del, label='loss del', marker='o')
    ax1.plot(iterations, loss_ins, label='loss ins', marker='s')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(iterations, loss_tv, label='loss TV', marker='o')
    ax2.plot(iterations, loss_l2, label='loss L2', marker='+')
    ax2.plot(iterations, loss_l1, label='loss L1', marker='x')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Value')
    ax2.grid(True)
    ax2.legend()

    plt.savefig(os.path.join(outdir, f'{index}_{index_o}_losses.jpg'), bbox_inches='tight', pad_inches = 0)
    plt.close()

def save_curves(del_curve, ins_curve, index_curve, index, index_o, outdir):
    """
        Save the deletion/insertion curves for the generated masks.

    :param del_curve:
    :param ins_curve:
    :param index_curve:
    :param index:
    :param index_o:
    :param outdir:
    :return:
    """
    for i in range(len(del_curve)):
        fig, (ax, ax1) = plt.subplots(2, 1)
        ax.plot(index_curve, del_curve[i], color='r', label='deletion')
        ax.fill_between(index_curve, del_curve[i], facecolor='maroon', alpha=0.4)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=14)
        ax.set_yticks(np.arange(0, 1.01, 1))
        ax.legend(['Deletion'], fontsize='x-large')
        ax.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(del_curve[i])),  fontsize=14, horizontalalignment='center', verticalalignment='center')

        ax1.plot(index_curve, ins_curve[i], color='b', label='Insertion')
        ax1.fill_between(index_curve, ins_curve[i], facecolor='darkblue', alpha=0.4)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=14)
        ax1.set_yticks(np.arange(0, 1.01, 1))
        ax1.legend(['Insertion'], fontsize='x-large')
        ax1.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(ins_curve[i])), fontsize=14, horizontalalignment='center', verticalalignment='center')

        # save the plot
        plt.savefig(os.path.join(outdir, f'{index}_{index_o}_curves.jpg'), bbox_inches='tight', pad_inches = 0)
        plt.close()


def save_images(images, index, index_o, outdir, classes, labels, pred_data, text=None):
    """
        saves original images into output directory

    :param images:
    :param index:
    :param index_o:
    :param outdir:
    :param classes:
    :param labels:
    :return:
    """
    if isinstance(images, list):
        images = images[1]
    elif images.shape[0] != 1:
        images = images[0].unsqueeze(0)
    images_ = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    for i, image in enumerate(images_):
        wrapped_text = fill(text, width=60)
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.subplots_adjust(top=0.8)
        ax.imshow(image.numpy())
        ax.axis('off')
        fig.text(0.5, 0.9, wrapped_text, ha='center', va='top', wrap=True, fontsize=10)
        plt.savefig(os.path.join(outdir, f'{index+i}_{index_o}_image.jpg'), bbox_inches='tight', pad_inches=0.2)


def auc(array):
    """
        calculates area under the curve (AUC)

    :param array:
    :return:
    """
    return (sum(array) - array[0]/2 - array[-1]/2)/len(array)


def get_initial(pred_data, k, init_posi, init_val, input_size, out_size):
    """
        filter the detection results by the threshold (predicted score)

    :param pred_data:
    :param k:
    :param initial_posi:
    :param init_val:
    :param input_size:
    :param out_size:
    :return:
    """
    interval_r = (pred_data['boxes'][:,2] - pred_data['boxes'][:,0]) / k
    interval_c = (pred_data['boxes'][:,3] - pred_data['boxes'][:,1]) / k
    num_row = init_posi // k
    num_col = init_posi - num_row * k
    init_boxes = np.concatenate([
                            [pred_data['boxes'][:,0] + interval_r * num_row], # x1
                            [pred_data['boxes'][:,1] + interval_c * num_col], # y1
                            [pred_data['boxes'][:,0] + interval_r * (num_row + 1)], # x2
                            [pred_data['boxes'][:,1] + interval_c * (num_col + 1)], # y2
                            ], axis=0).T 

    pred_data['init_masks'] = []
    down = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size))

    for ith, box in enumerate(init_boxes):
        init_mask = torch.zeros((input_size, input_size)).unsqueeze(0)
        init_mask[int(box[0]) : int(box[2]), int(box[1]) : int(box[3])] = 1

        if 'masks' in pred_data.keys():
            init_mask = init_mask * pred_data['masks'][ith]
        
        init_mask = down(init_mask.unsqueeze(0)) * init_val
        pred_data['init_masks'].append(1 - init_mask)
    return pred_data


def generate(args, model, input_ids, image, image_size):
    output_ids = model.generate(
            input_ids,
            images=image,
            #image_sizes=image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
    output_ids = output_ids[:,1:-1]
    return output_ids

def match_keywords(lst, sublist):
    lst_str = ','.join(map(str, lst))
    sublist_str = ','.join(map(str, sublist))
    start_idx = lst_str.find(sublist_str)

    if start_idx == -1:
        return None

    start = lst_str[:start_idx].count(',')
    end = start + len(sublist) - 1
    return [start, end]

def pred_probs(args, model, prompt, output_ids, image, image_size):
    if args.model == 'llava' or args.model == 'llava_next':
        def custom_forward(input_ids, images, image_size):
            (inputs, _, _, _, inputs_embeds, labels, _) = model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, images, image_sizes=image_size)
            return inputs, inputs_embeds, labels
        inputs, inputs_embeds, labels = checkpoint.checkpoint(custom_forward, prompt, image, image_size)
        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
        outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
    
    elif args.model == 'mgm':
        (inputs, _, _, _, inputs_embeds, labels) = model.prepare_inputs_labels_for_multimodal(prompt, None, None, None, None, image)
        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
        outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
    
    elif args.model == 'cambrian':
        (
            inputs,
            _,
            _,
            _,
            inputs_embeds,
            labels,
            vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list,
            final_vision_feature_size,
            global_context_feature,
        ) = model.prepare_inputs_labels_for_multimodal(
            prompt,
            None,
            None,
            None,
            None,
            image,
            image_sizes=image_size
        )
        model.vision_tower_aux_feature_list = vision_tower_aux_feature_list
        model.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
        model.final_vision_feature_size = final_vision_feature_size
        model.global_context_feature = global_context_feature

        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
        outputs = model(inputs, attention_mask, position_ids, None, inputs_embeds, labels, True, False, False, return_dict=True)

    else:
        raise ValueError('Model not supported: {}'.format(args.model))
    logits = outputs['logits']
    #logits.requires_grad_(True)
    logits_pred = logits[:,-output_ids.shape[-1]-1:-1,:]
    probs_pred = F.softmax(logits_pred, dim=-1) # (1, len, 32000)
    probs_pred = torch.gather(probs_pred, 2, output_ids.unsqueeze(-1)).squeeze()
    if probs_pred.dim() == 0:
        probs_pred = probs_pred.unsqueeze(0)
    return probs_pred

def find_keywords(args, model, input_ids, output_ids, image, blur_image, image_size, output_text, tokenizer=None, use_yake=False):
    if output_ids.shape[-1] == 1:
        full_prompt = torch.cat((input_ids, output_ids), dim=1)
        probs = pred_probs(args, model, full_prompt, output_ids, image, image_size)
        probs_blur = pred_probs(args, model, full_prompt, output_ids, blur_image, image_size)
        if probs_blur.squeeze().item() > 0.5*probs.squeeze().item():
            positions = []
            keywords = []
        else:
            positions = [0]
            keywords = [output_text]

    else:
        if use_yake:
            # select keywords using nlp tool
            num_words = len(output_text.split())
            if num_words <= 10:
                keywords_num = 3
            else:
                keywords_num = num_words // 4

            kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.2, top=keywords_num, features=None)
            keywords = kw_extractor.extract_keywords(output_text)
            keywords = [kw[0] for kw in keywords]
            positions = []
            for kw in keywords:
                kw_ids = tokenizer.encode(kw, add_special_tokens=False)
                position = match_keywords(output_ids[0].tolist(), kw_ids)
                if position:
                    positions.extend(position)
        else:
            # select keywords according to logits drop
            full_prompt = torch.cat((input_ids, output_ids), dim=1)
            probs = pred_probs(args, model, full_prompt, output_ids, image, image_size)
            probs_blur = pred_probs(args, model, full_prompt, output_ids, blur_image, image_size)
            # condition = (probs_blur <= 0.4*probs) & (~torch.isin(output_ids[0], torch.tensor(special_ids).to(probs.device)))
            condition = (torch.log(probs)-torch.log(probs_blur) > 1.0)& (probs>=0.0) & (~torch.isin(output_ids[0], torch.tensor(special_ids).to(probs.device)))
            positions = torch.where(condition)[0].tolist()
            keywords = [tokenizer.decode(output_ids[0][idx]).strip() for idx in positions]
    
    return positions, keywords
