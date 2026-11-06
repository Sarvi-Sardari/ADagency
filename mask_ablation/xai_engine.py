"""Run the config-driven XAI mask-ablation engine.

The engine builds attribution samples from anomaly-run artifacts, selects
model-output evidence, creates dense or segmented masks, runs the configured
iGOS optimization, and persists diagnostics and score maps. Model paths and
all experiment parameters come from the mask-ablation configuration.
"""

# %%
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Must precede CUDA initialization.
# Some managed/remote runs expose a read-only user config directory. Keep
# Matplotlib's cache in a writable temporary location unless the caller has
# already chosen one.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvtech_xai_matplotlib")
import json
import time
import re
import cv2
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List
from PIL import Image

import torch
import torchvision.transforms.functional as F_t
import torch.nn.functional as F
import torch._dynamo

from skimage.segmentation import slic
from itertools import groupby
import torchvision.transforms.functional as TF
import nltk
# Phrase ablations use NLTK's POS tagger. Do not perform a hidden network
# download at module import: offline/resumed studies must fail predictably.
# ``build_meaningful_visual_phrases`` emits an actionable error only when a
# phrase-based arm actually needs the missing resource.

# All model checkpoints for this engine live under pipeline_project/models/ --
# never under LVLM_Interpretation/ or the repo root. This resolves to
# .../pipeline_project/models regardless of the current working directory.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_VLM_MODEL_PATH = str(MODELS_DIR / "vlm_model")
DEFAULT_SAM3_MODEL_PATH = str(MODELS_DIR / "sam3_model")
DEFAULT_SAM3_REPO_PATH = str(Path(__file__).resolve().parent.parent.parent / "sam3")

# ByteDance XAI Imports -- relocated, self-contained copies (see igos_core_*.py
# in this same folder). No dependency on LVLM_Interpretation/ at runtime.
from .igos_core_utils import save_heatmaps
from .igos_core_methods import iGOS_p, iGOS_pp
from .igos_core_methods_helper import metric, blocks_to_grid, pool_labels_to_grid
from .paper_original import paper_igos_single_mask

from .xai.optimization import (
    expand_native_visibility_mask,
    normalize_display_score,
    run_igos_ensemble,
)
# %%
from .xai.evidence import (
    HeatmapSample,
    build_heatmap_samples,
    build_heatmap_samples_v2,
    build_rerun_heatmap_samples,
    extract_yes_steps_from_file,
    get_reference_path_for_category,
    parse_final_reason_details,
    parse_output_file,
    parse_planner_file,
    parse_prob_data_file,
    read_jsonl,
    resolve_dataset_jsonl_and_base_dir,
    resolve_image_path,
)
# %%
from .xai.segmentation import (
    get_ig_iter_schedule,
    get_iter_schedule,
    get_kernel_size,
    load_and_resize_image,
    make_blur_baseline,
    precompute_sam3_segmentations,
    segment_with_sam3,
    segment_with_slic_within_sam3,
)
from .xai.adapters import (
    MODEL_ADAPTER_REGISTRY,
    Qwen3VLPrecomputedAdapter,
    load_qwen3_vl_adapter,
    register_model_adapter,
)
def build_output_spans(adapter, output_text, output_ids):
    """
    Maps Qwen output-token positions onto dynamically detected textual
    word spans.

    No object names, anomaly terms, POS tags, or vocabulary lists are
    hardcoded. Word boundaries come directly from the output string,
    while token membership comes from tokenizer character offsets.
    """

    # use exactly the same stripped text used by encode_answer().
    # This keeps character offsets aligned with output_ids.
    text = output_text.strip()

    # tokenize while requesting each token's character interval.
    # Qwen offsets may include leading spaces, so we must not assume that
    # whitespace appears as a gap between consecutive token offsets.
    encoded = adapter.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    # convert offsets to ordinary Python tuples.
    offset_mapping = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]

    # verify that these offsets describe the same token sequence
    # used later for probability computation.
    retokenized_ids = encoded["input_ids"]
    target_ids = (output_ids[0].detach().cpu().tolist())

    if retokenized_ids != target_ids:
        raise ValueError(
            "Tokenizer offset mapping does not match output_ids. "
            "The exact same stripped output_text must be used by "
            "encode_answer() and build_output_spans()."
        )

    spans = []

    # detect word-like spans from the original text itself.
    # This is structural, not vocabulary-based:
    #   mechanical       -> one span
    #   discoloration    -> one span
    #   residue          -> one span
    #   high-quality     -> one span
    #   comma/parenthesis -> no span
    #
    # [^\W_] means a Unicode letter or digit but excludes underscore.
    word_pattern = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", flags=re.UNICODE)

    for match in word_pattern.finditer(text):
        char_start = match.start()
        char_end = match.end()

        token_positions = []

        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            # ignore empty/special offset intervals.
            if token_end <= token_start:
                continue

            # calculate actual character overlap.
            # This works even when a Qwen token includes a leading space.
            overlap_start = max(char_start, token_start)
            overlap_end = min(char_end, token_end)

            if overlap_end > overlap_start:
                token_positions.append(token_index)

        # retain only textual spans that map to at least one
        # actual Qwen token.
        if not token_positions:
            continue

        spans.append({
                        # Full textual word/span reconstructed from the original output.
                        "text": match.group(0),
                        # All Qwen token indices overlapping this word.
                        "positions": token_positions,
                        # Character coordinates retained for diagnostics.
                        "char_start": char_start,
                        "char_end": char_end,
                        # Every regex match contains letters/digits by construction.
                        "has_content": True,
                    })
    # diagnostic guard against accidental whole-sentence grouping.
    # A normal multiword sentence should produce multiple textual spans.
    whitespace_word_count = len(re.findall(r"[^\W_]+(?:[-'’][^\W_]+)*", text, flags=re.UNICODE,))

    if (whitespace_word_count > 1 and len(spans) == 1):
        raise RuntimeError("build_output_spans() collapsed a multiword output into one span. Check tokenizer offset handling.")

    return spans

    #return spans


# ============================================================
# build meaningful visual noun phrases from the complete
# generated output, not only from words above the keyword threshold.
# ============================================================
def build_meaningful_visual_phrases(adapter, output_text, output_ids, keyword_info, max_phrases=8):
    """
    Builds object/defect phrases such as:

        "red triangular item"
        "gray cube"
        "black mechanical component"
        "small metal plate"

    Important design:
      - High-importance keywords are used as visual anchors.
      - Nearby adjectives and the required head noun may be included even
        when those supporting words did not clear the keyword threshold.
      - A phrase is kept only when it contains at least one selected visual
        anchor and at least one noun.
      - Determiners and verbs are excluded from optimization positions.
    """

    # recover every complete textual word and its Qwen token positions.
    all_spans = build_output_spans(adapter=adapter, output_text=output_text, output_ids=output_ids)
    # Needed for sentence-boundary detection in the phrase collection loop below.
    # Uses the same strip() transformation as build_output_spans internally.
    text = output_text.strip()

    # POS-tag the complete sentence in context.
    # Tagging all words together is more reliable than tagging isolated tokens.
    span_texts = [span["text"] for span in all_spans]
    try:
        tagged_spans = nltk.pos_tag(span_texts)
    except LookupError as error:
        raise RuntimeError(
            "Phrase-based XAI requires the NLTK averaged_perceptron_tagger_eng "
            "resource. Install it once with: python -m nltk.downloader "
            "averaged_perceptron_tagger_eng"
        ) from error

    # attach each contextual POS tag to its aligned textual span.
    for span, (_, pos_tag) in zip(all_spans, tagged_spans):
        span["pos_tag"] = pos_tag

    # map every selected keyword token position back to its importance.
    selected_position_importance = {}

    for keyword in keyword_info:
        for position in keyword["positions"]:
            selected_position_importance[position] = max(
                selected_position_importance.get(position, 0.0),
                float(keyword["importance"]),
            )

    # mark complete spans that overlap at least one selected keyword token.
    for span in all_spans:
        span["is_visual_anchor"] = any(position in selected_position_importance for position in span["positions"])

        # supporting words below threshold receive zero anchor importance.
        span["anchor_importance"] = max([selected_position_importance.get(position, 0.0) for position in span["positions"]], default=0.0)

    # Noun phrases may contain adjectives and nouns.
    adjective_tags = {"JJ", "JJR", "JJS"}
    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
    allowed_phrase_tags = adjective_tags | noun_tags

    phrase_candidates = []
    index = 0

    while index < len(all_spans):
        current_span = all_spans[index]

        # only adjective/noun sequences can start an object phrase.
        if current_span["pos_tag"] not in allowed_phrase_tags:
            index += 1
            continue

        phrase_spans = []
        cursor = index

        # Collect a contiguous adjective/noun sequence, stopping at sentence
        # boundaries. The char positions stored in each span let us detect whether
        # a period appears in the gap between consecutive spans -- if so, the spans
        # belong to different sentences and must not be merged into one phrase.
        # Without this guard the phrase extractor incorrectly bridges sentences,
        # e.g. "inspection plan. Dirt" becomes phrase "inspection plan Dirt".
        while (cursor < len(all_spans) and all_spans[cursor]["pos_tag"] in allowed_phrase_tags):
            if cursor > index:
                # Stop if a sentence-ending period separates this span from the previous.
                prev_end = all_spans[cursor - 1]["char_end"]
                curr_start = all_spans[cursor]["char_start"]
                if '.' in text[prev_end:curr_start]:
                    break  # stop -- current span is in the next sentence
            phrase_spans.append(all_spans[cursor])
            cursor += 1

        # require a head noun so phrases such as "are dark"
        # or adjective-only fragments are rejected.
        contains_noun = any(span["pos_tag"] in noun_tags for span in phrase_spans)

        # require at least one model-selected visual anchor.
        # This prevents unrelated nouns in the sentence from becoming targets.
        contains_visual_anchor = any(span["is_visual_anchor"] for span in phrase_spans)

        if contains_noun and contains_visual_anchor:
            # optimize every adjective/noun position in the phrase,
            # including supporting nouns that were below the keyword threshold.
            phrase_positions = sorted({position for span in phrase_spans for position in span["positions"]})

            # readable phrase text uses complete reconstructed words.
            phrase_text = " ".join(span["text"] for span in phrase_spans)

            # phrase importance comes only from selected anchors.
            # Supporting nouns do not receive invented visual importance.
            phrase_importance = sum(span["anchor_importance"] for span in phrase_spans if span["is_visual_anchor"])

            phrase_candidates.append({
                "text": phrase_text,
                "positions": phrase_positions,
                "importance": float(phrase_importance),
                "spans": phrase_spans,
            })

        index = max(cursor, index + 1)

    # remove duplicate phrases using exact output-token positions.
    unique_phrases = []
    seen_position_sets = set()

    for phrase in phrase_candidates:
        position_key = tuple(phrase["positions"])

        if position_key in seen_position_sets:
            continue

        seen_position_sets.add(position_key)
        unique_phrases.append(phrase)

    # retain all meaningful phrases up to a configurable safety limit.
    # This replaces the previous hardcoded strongest-phrase-only behavior.
    unique_phrases.sort(key=lambda phrase: phrase["positions"][0])

    return unique_phrases[:max_phrases]

def flatten_balanced_phrase_targets(phrase_candidates):
    """
    Converts multiple phrases into one optimization target.

    Every phrase receives equal total weight, regardless of:
      - how many tokens it contains;
      - how high the strongest phrase importance is.

    This prevents "red triangular item" from dominating "gray cube".
    """

    if not phrase_candidates:
        return [], []

    # one equal budget per meaningful phrase.
    phrase_budget = 1.0 / len(phrase_candidates)

    position_to_weight = {}

    for phrase in phrase_candidates:
        phrase_positions = sorted(set(phrase["positions"]))

        if not phrase_positions:
            continue

        # divide this phrase's equal budget across its positions.
        per_position_weight = phrase_budget / len(phrase_positions)

        for position in phrase_positions:
            position_to_weight[position] = (position_to_weight.get(position, 0.0) + per_position_weight)

    # interval_score requires positions and weights in identical order.
    phrase_positions = sorted(position_to_weight.keys())

    phrase_weights = [position_to_weight[position] for position in phrase_positions]

    # Normalize again in case phrases overlap on one or more positions.
    total_weight = sum(phrase_weights) + 1e-8

    phrase_weights = [weight / total_weight for weight in phrase_weights]

    assert len(phrase_positions) == len(phrase_weights)

    return phrase_positions, phrase_weights

def find_boilerplate_clause_start(adapter, output_ids, output_text):
    """
    Finds the token index where the templated classification suffix begins,
    using the literal blank line template delimiter inserted by the pipeline.
    """
    import re as _re
    m = _re.search(r'\n\s*\n\s*\w+;\s*reason\s*=', output_text)
    if m is None:
        return None
    char_offset = m.start()
    
    # Map the exact string index cleanly back into matching token counts
    prefix_text = output_text[:char_offset]
    prefix_ids = adapter.tokenizer(prefix_text.strip(), add_special_tokens=False)["input_ids"]
    return len(prefix_ids)
    
def find_keywords_from_precomputed(
    adapter,
    image,
    baseline,
    question,
    output_ids,
    threshold=1.0,
    output_text="",
    reference=None,
    percentile=None,
    exclude_first_token=False,
    selection_profile="span",
    log=print,
):
    """
    Use a pre-computed adapter to identify keyword tokens and their importance scores.
    
    New Args:
        percentile: If not None, filter keywords to keep only those in the given 
                   percentile range (e.g., 50 for top 50% by importance).
                   Overrides the flat threshold if both are provided.
    """
    
    if reference is None:
        probs = adapter.token_probs(image, question, output_ids)
        probs_blur = adapter.token_probs(baseline, question, output_ids)
    else:
        probs = adapter.token_probs_with_reference(candidate=image,reference=reference,question=question,output_ids=output_ids)
        probs_blur = adapter.token_probs_with_reference(candidate=baseline,reference=reference,question=question,output_ids=output_ids)

    token_ids = output_ids[0].to(probs.device)

    # one-time diagnostic for probability precision. bfloat16 epsilon is unsuitable as an absolute probability cutoff.
    print(f"  [diag keyword selection] probability dtype={probs.dtype}, old dtype floor={float(torch.finfo(probs.dtype).eps * 32):.6f}, new floor=1e-8")

    # This delta IS the "how image-dependent is this word" score — keep it instead of discarding it
    importance = torch.log(probs + 1e-12) - torch.log(probs_blur + 1e-12)

    if selection_profile == "paper_tokenwise":
        # Paper Eq. (6) is token-level, not word/span-level: retain every
        # autoregressive answer token whose LLR exceeds alpha, except k=1.
        # No percentile pruning, phrase restoration, or boilerplate heuristic
        # is allowed in this profile.
        selected = []
        special_ids_set = set(adapter.tokenizer.all_special_ids)
        for position in range(output_ids.shape[-1]):
            if exclude_first_token and position == 0:
                continue
            token_id = int(output_ids[0, position].item())
            if token_id in special_ids_set:
                continue
            token_importance = float(importance[position].float().item())
            if token_importance <= float(threshold):
                continue
            selected.append(
                {
                    "token": adapter.tokenizer.decode([token_id]).strip(),
                    "positions": [position],
                    "position": position,
                    "importance": token_importance,
                    "all_span_positions": [position],
                }
            )
        return selected
    if selection_profile != "span":
        raise ValueError(f"Unknown token selection profile: {selection_profile!r}")

    # dynamically group tokenizer pieces into complete
    # word-like spans using character offsets.
    output_spans = build_output_spans(adapter=adapter, output_text=output_text, output_ids=output_ids)

    # Keep the existing classification-suffix exclusion.
    clause_start = find_boilerplate_clause_start(adapter, output_ids, output_text)

    keyword_info = []

    for span in output_spans:
        positions = [
            position
            for position in span["positions"]
            if not (exclude_first_token and position == 0)
        ]
        if not positions:
            continue

        # reject punctuation-only spans using a structural
        # character test, not a vocabulary list.
        if not span["has_content"]:
            continue

        # reject spans belonging to the excluded generated
        # boilerplate suffix, when such a suffix exists.
        if (clause_start is not None and positions[0] >= clause_start):
            continue

        # remove model special tokens dynamically.
        if any(token_ids[position].item() in adapter.tokenizer.all_special_ids for position in positions):
            continue

        position_tensor = torch.tensor(positions, device=probs.device, dtype=torch.long)

        span_probs = probs[position_tensor]

        span_probs_blur = probs_blur[position_tensor]

        # convert probabilities to float32 before logarithms and
        # numerical checks. Model probabilities may be bfloat16, whose
        # epsilon is too large for probability filtering.
        span_probs_fp32 = span_probs.float()

        # use the same float32 representation for the blurred
        # candidate probabilities.
        span_probs_blur_fp32 = span_probs_blur.float()

        # calculate visual dependence in float32.
        # This avoids low-probability anomaly tokens being distorted by
        # bfloat16 precision.
        subtoken_importance = (torch.log(span_probs_fp32.clamp(min=1e-12)) - torch.log(span_probs_blur_fp32.clamp(min=1e-12)))

        # use only a tiny underflow guard.
        # This is a numerical constant, not a word-, image-, or
        # dataset-specific threshold.
        numerical_floor = 1e-8

        # retain a token when either candidate or baseline has
        # a real non-underflow probability. A visually relevant token
        # does not need an absolute probability above 0.25.
        reliable_subtokens = (torch.maximum(span_probs_fp32, span_probs_blur_fp32) > numerical_floor)

        # the token is selected primarily using the paper's
        # candidate-versus-baseline log-likelihood-ratio criterion.
        selected_local_indices = torch.where(reliable_subtokens & (subtoken_importance > threshold))[0]
        
        # reject this word only when none of its constituent
        # Qwen tokens are visually dependent.
        if selected_local_indices.numel() == 0:
            continue

        # convert local indices within the span back to the
        # original output-token positions.
        selected_positions = [positions[local_index] for local_index in (selected_local_indices.detach().cpu().tolist())]

        # the span receives the strongest token-level visual
        # evidence belonging to that span.
        span_importance = subtoken_importance[selected_local_indices].max()

        keyword_info.append({
                                # Full reconstructed word, used only for readable metadata.
                                "token": span["text"].strip(),
                                # only visually relevant subtoken positions are optimized, not every autoregressive continuation piece.
                                "positions": selected_positions,
                                # Retained for compatibility with existing diagnostics.
                                "position": selected_positions[0],
                                # Model-derived visual relevance score.
                                "importance": float(span_importance.item()),
                                # optional diagnostic showing all tokenizer positions belonging to the complete word before LLR selection.
                                "all_span_positions": positions,
                            })
        
    # Apply percentile-based filtering if specified
    if percentile is not None and keyword_info:
        importances = [item["importance"] for item in keyword_info]
        percentile_threshold = np.percentile(importances, percentile)
        keyword_info = [item for item in keyword_info if item["importance"] >= percentile_threshold]
        # Re-sort by importance after filtering
        keyword_info = sorted(keyword_info, key=lambda x: x["importance"], reverse=True)
        log(f"  [diag] applied percentile-based filtering: {percentile}th percentile → threshold={percentile_threshold:.3f}, selected {len(keyword_info)} keywords")
    elif threshold is not None:
        # Original flat threshold logic (kept for backward compatibility)
        keyword_info = [item for item in keyword_info if item["importance"] >= threshold]

    # sort by model-derived visual importance only.
    #keyword_info.sort(key=lambda item: item["importance"], reverse=True)

    return keyword_info

# %%
class Args:
    # ``paper_original`` selects an isolated implementation of paper Eqs.
    # (6)--(8). ``ablation`` retains the notebook-derived experimental paths.
    implementation_profile = "ablation"
    method = "iGOS+"
    L1, L2, L3 =  1.0, 0.05, 15.0 #20.0 #1.3, 0.05, 15.0 #1.3, 0.05, 8.0 #1.3, 0.05, 25.0 #1.0, 0.05, 25.0 #8.0 #1.0, 0.05, 15 #3.0, 0.1, 35.0
    momentum, ig_iter, lr, iterations = 3, 4, 0.025, 20 #3, 10, 0.03, 40   # <- treated as the anchor values below
    opt = "NAG"
    gamma = 0.08
    # "fixed" reproduces the original repository's constant ig_iter.
    # "redistribute" is the later notebook experiment that spends more IG
    # samples near the end of optimization.
    ig_schedule = "redistribute"
    ig_noise_std = 0.05
    ig_noise_samples = 2
    input_size, size = 448, 448 #448, 224 #448, 448 #672, 448 #DEEPSEEK
    diverse_k, init_posi, init_val = 1, 0, 0.0
    # Use percentile-based filtering instead of flat threshold
    # Set to None to disable percentile filtering, or 50 for 50th percentile (median)
    keyword_percentile = 50  # Keep only top 50% by importance #Change 28.07.2026
    keyword_threshold = 0.5 #1.0 #0.5 #1.0 #Change 28.07.2026
    num_superpixels = [120] #[50, 100, 150] #[120] #[80] #[30, 56, 135]
    sp_compactness = 5.0 #10.0
    sigma = 4
    ensemble_seeds = [20]   # multiple seeds -> averaged for the final mask, but each is also saved individually

    # maximum number of meaningful noun phrases retained from one
    # reasoning output. This is a runtime guard, not an importance selection.
    max_phrases_per_sample = 8

    # Number of iGOS iterations for the single all-phrases joint run.
    phrase_joint_iterations = 15

    # contribution of the phrase-joint map during score-space fusion.
    # 1.0 means a full union; values below 1 reduce phrase-only false positives.
    phrase_union_weight = 0.85

    # require a minimally useful phrase mask before adding it.
    phrase_min_deletion_score = 0.0

    # Later anomaly tuning rejected a reasoning-step map when its deletion
    # score was non-positive. The paper/repository method does not apply that
    # post-hoc filter, so it can be disabled per ablation arm.
    step_map_filter = "positive_deletion"

    sp_keep_fraction = 1.0 #0.65 #0.45 #0.3 #0.5 #1.0 #0.5   # fraction of superpixels kept active after importance pre-filter
    mass_ratio_keep = 0.35 #0.35

    # Available modes:
#   "none"   = dense Qwen patch-grid mask; no SLIC, SAM3, or regions
#   "slic"   = SLIC superpixels
#   "sam3"   = SAM3 masks only
#   "hybrid" = SAM3 boundaries with SLIC subdivisions
    segmentation_method = "hybrid"  
    sam3_model_id = DEFAULT_SAM3_MODEL_PATH
    sam3_repo_dir = DEFAULT_SAM3_REPO_PATH
    # Generic industrial concepts give SAM3 semantic outer boundaries. Hybrid
    # SLIC then subdivides each returned region for fine anomaly localization.
    sam3_text_prompts = ["object", "part", "component"]
    sam3_confidence_threshold = 0.5
    sam3_min_mask_area = 50  #200      # discard SAM3 masks smaller than this many pixels (noise/slivers)
    sp_size_bias = 1.0 #0.2 #1.5 #0.2 #2.0 #1.0              # >1 = fewer/bigger sub-superpixels overall, <1 = more/smaller, 1.0 = unchanged

    # `iterations`/`ig_iter` above are tuned for iter_schedule_base_n_seg superpixels.
    # get_iter_schedule() scales them up/down for every other entry in num_superpixels.
    iter_schedule_base_n_seg = 40
    iter_schedule_min_iterations, iter_schedule_max_iterations = 15, 70
    iter_schedule_min_ig_iter, iter_schedule_max_ig_iter = 3, 6 #5, 20 #GEMINI CHANGE

    # New knobs
    save_joint_heatmap = True          # keep the "all keywords together" map as a summary view
    save_per_keyword_heatmaps = True   # the new per-keyword breakdown
    per_keyword_iterations = 25        # a single position tends to converge faster than the joint mask; lower this if runtime is a problem
    max_keywords_per_sample = None #Change 28.07.2026 

    # controls how the retrieved reference is used.
    # "none"    = candidate-only paper-style control.
    # "context" = reference stays fixed while candidate is perturbed.
    reference_mode = "none" #"context"

    # Pixel-replacement baseline used by Phi(I, I_tilde, M).
    # "blur" is the paper default; "reference" is the requested ablation and
    # falls back to blur with an explicit diagnostic if no reference artifact
    # exists for an image.
    baseline_mode = "blur"
    blur_kernel_mode = "strong"

    output_dir =  os.path.join("..","xai_test_sample",f"{datetime.now().strftime('%Y-%m-%d')}",f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_")

    #   Allowed Values:
    #   yes_steps -> generates one sample per YES step
    #   final_reason_details -> generates one synthetic sample per image using the consolidated final reason/details instead of individual YES steps
    mask_generation_mode = "final_reason_details"

    # controls which mask VARIANTS get computed/saved per step.
    #   "keyword_only"  -> only the keyword-joint mask (no phrase extraction at all)
    #   "phrase_only"   -> only the phrase-joint mask (built from noun phrases, no
    #                      standalone per-keyword mask contributes to the saved output)
    #   "both_combined" -> today's default: keyword_joint + phrase_joint +
    #                      keyword_phrase_combined (score-space union of the two)
    mask_content_mode = "both_combined"

    # Segmented masks are already piecewise constant. Gaussian smoothing makes
    # their boundaries bleed into neighboring normal pixels, so sharp study
    # arms set this to "none". "gaussian" preserves notebook compatibility.
    mask_smoothing = "gaussian"

    # Runtime/output controls for the final-only ablation.  A four-point curve
    # uses eight perturbed-image forwards rather than the original ~100.  The
    # exact 50-point behavior remains available by overriding metric_num_steps.
    metric_num_steps = 50
    save_intermediate_outputs = True
    save_diagnostics = True
    # Opt-in presentation capture.  The normal ablation path remains final-map
    # only; when enabled, all step-level heatmaps/overlays are written to the
    # dedicated presentation directory instead of the diagnostics directory.
    presentation_enabled = False
    presentation_output_dir = ""
    # "balanced_auc" selects one complete reasoning-step map by rewarding high
    # insertion AUC and LOW deletion AUC. "pixelwise_max" preserves the legacy
    # union of every accepted YES-step map.
    final_map_selection = "pixelwise_max"

    # Consumed by the outer publisher. They live here too so config overrides
    # pass the engine's strict unknown-argument check.
    published_mask_mode = "continuous"
    published_mask_threshold = 0.60

    # How selected LLR token positions contribute to f. ``paper_sum`` matches
    # Eq. (7), a plain sum of selected log probabilities. ``llr_sqrt`` keeps
    # the notebook's later importance-weighted experiment.
    token_weighting = "llr_sqrt"
    # Equation (6) excludes the first autoregressive answer token. The cloned
    # repository did not enforce this, so it is an explicit ablation switch.
    exclude_first_token = False

    # ------------------------------------------------------------------
    # DENSE REGULARIZATION ABLATION SWITCHES (only affect segmentation_method="none"
    # dense path inside igos_core_methods.iGOS_p; see that file's docstring).
    # Defaults below reproduce TODAY'S GPT-tuned dense behavior. The
    # These reproduce only the legacy dense regularization/update variants.
    # The complete locked method is implementation_profile="paper_original".
    # ------------------------------------------------------------------
    dense_l2_reduction = "mean"          # 'sum' (original paper) | 'mean' (GPT tuning)
    mask_clamp_range = (0.02, 0.98)       # original paper used (0.0, 1.0)
    grad_clip_norm = 5.0                  # original paper had no gradient-norm clipping; set None to disable
    dense_l2_floor_fraction = 0.01        # original paper had no L2 floor; set 0.0 to disable

    # when True, run_pipeline() also folds in rerun-derived samples --
    # see build_rerun_heatmap_samples() / build_heatmap_samples_v2() below.
    include_rerun = False


def _validate_paper_original_profile(args):
    """Prevent a nominal paper run from silently becoming a hybrid preset."""
    expected = {
        "method": "iGOS+",
        "segmentation_method": "none",
        "mask_content_mode": "keyword_only",
        "mask_generation_mode": "decision_output",
        "baseline_mode": "blur",
        "blur_kernel_mode": "paper_repo",
        "reference_mode": "none",
        "token_weighting": "paper_sum",
        "exclude_first_token": True,
        "keyword_percentile": None,
        "keyword_threshold": 1.0,
        "size": 32,
        "input_size": 336,
        "L1": 1.0,
        "L2": 0.1,
        "L3": 10.0,
        "gamma": 0.2,
        "opt": "NAG",
        "momentum": 3,
        "iterations": 5,
        "ig_iter": 5,
        "lr": 10.0,
        "ig_schedule": "fixed",
        "ig_noise_std": 0.2,
        "ig_noise_samples": 1,
        "ensemble_seeds": [0],
        "mask_smoothing": "none",
    }
    mismatches = []
    for field, required in expected.items():
        actual = getattr(args, field)
        if actual != required:
            mismatches.append(f"{field}={actual!r} (required {required!r})")
    if mismatches:
        raise ValueError(
            "implementation_profile='paper_original' is a locked fidelity "
            "profile; inconsistent config values: " + "; ".join(mismatches)
        )

def run_pipeline(
    *,
    param_overrides: dict | None = None,
    model_path: str | None = None,
    run_dir: str | None = None,
    dataset: str | None = None,
    dataset_jsonl: str | None = None,
    base_dir: str | None = None,
    output_dir: str | None = None,
    model_adapter: str = "qwen3_vl",
    samples_override: list[dict[str, Any]] | None = None,
    final_mask_callback: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> dict:
    """
    Runs the full per-image heatmap-generation pipeline (unchanged core logic).

    All parameters are OPTIONAL. They exist so an external orchestrator (e.g.
    pipeline_project/mask_ablation/pipeline.py, driven entirely by
    configs/mask_ablation_configs.yaml) can drive this engine across a grid of
    models/methods/datasets without editing this file per run:

      - param_overrides: dict of {Args field name: value} applied on top of
        the `Args` defaults above (e.g. {"segmentation_method": "slic",
        "num_superpixels": [80], "mask_content_mode": "phrase_only",
        "include_rerun": True}). Raises AttributeError on an unknown field
        name (fails fast instead of silently ignoring a typo). This is the
        ONLY mechanism needed to reach every documented Args field from a
        config file -- nothing behavior-relevant needs to be edited here.
      - model_path: overrides the default local Qwen3-VL checkpoint path
        (pipeline_project/models/vlm_model).
      - run_dir + dataset: PRIMARY dataset-resolution interface, matching the
        convention configs/mask_generation_configs.yaml / metrics_configs.yaml
        already use -- resolves to <run_dir>/<dataset>/all_data_summary.jsonl,
        and (when Args.include_rerun / param_overrides["include_rerun"] is
        True) also folds in <run_dir>/<dataset>/rerun_results/rerun_result.jsonl.
      - dataset_jsonl + base_dir: legacy override interface (bypasses run_dir/
        dataset resolution entirely and disables rerun folding, since there is
        no run_dir to locate rerun_results/ under). Kept for direct/adhoc use.
      - output_dir: overrides Args.output_dir (where heatmaps/overlays/
        raw_masks/heatmap_metadata.jsonl are written for this run).
      - model_adapter: registered differentiable model-family adapter.
      - samples_override: already-resolved, rerun-aware evidence records from
        mask_ablation/evidence.py. This is the pipeline interface; the legacy
        JSONL loaders remain available for direct notebook use.
      - final_mask_callback: invoked after every image-level final raw map is
        saved, allowing the outer pipeline to publish a resumable standard PNG.

    Returns a small summary dict (num_samples, output_dir) so callers can
    record it alongside their own wall-clock timing.
    """
    def notify(**fields: Any) -> None:
        """Forward lightweight status to the outer Rich progress renderer."""
        if progress_callback is None:
            return
        try:
            progress_callback(**fields)
        except Exception:
            # Status rendering must never interrupt a costly attribution run.
            pass

    args = Args()
    if param_overrides:
        for key, value in param_overrides.items():
            if not hasattr(args, key):
                raise AttributeError(f"Unknown Args field in param_overrides: {key!r}")
            setattr(args, key, value)
    if args.implementation_profile not in {"ablation", "paper_original"}:
        raise ValueError(
            "implementation_profile must be 'ablation' or 'paper_original'."
        )
    if args.implementation_profile == "paper_original":
        _validate_paper_original_profile(args)
    sam3_candidate = Path(str(args.sam3_model_id)).expanduser()
    if not sam3_candidate.is_absolute():
        project_relative_sam3 = Path(__file__).resolve().parent.parent / sam3_candidate
        if project_relative_sam3.exists():
            args.sam3_model_id = str(project_relative_sam3.resolve())
    if output_dir is not None:
        args.output_dir = output_dir
    if args.presentation_enabled and args.presentation_output_dir:
        args.output_dir = str(args.presentation_output_dir)

    valid_segmentation_methods = {"none", "slic", "sam3", "hybrid"}
    if args.segmentation_method not in valid_segmentation_methods:
        raise ValueError(
            f"Unknown segmentation_method={args.segmentation_method!r}. "
            f"Expected one of {sorted(valid_segmentation_methods)}."
        )
    valid_mask_content_modes = {"keyword_only", "phrase_only", "both_combined"}
    if args.mask_content_mode not in valid_mask_content_modes:
        raise ValueError(
            f"Unknown mask_content_mode={args.mask_content_mode!r}. "
            f"Expected one of {sorted(valid_mask_content_modes)}."
        )
    if args.baseline_mode not in {"blur", "reference"}:
        raise ValueError("baseline_mode must be 'blur' or 'reference'.")
    if args.reference_mode not in {"none", "context"}:
        raise ValueError("reference_mode must be 'none' or 'context'.")
    if args.blur_kernel_mode not in {"paper_repo", "strong"}:
        raise ValueError("blur_kernel_mode must be 'paper_repo' or 'strong'.")
    if args.token_weighting not in {"paper_sum", "llr_sqrt"}:
        raise ValueError("token_weighting must be 'paper_sum' or 'llr_sqrt'.")
    if args.ig_schedule not in {"fixed", "redistribute"}:
        raise ValueError("ig_schedule must be 'fixed' or 'redistribute'.")
    if args.step_map_filter not in {"none", "positive_deletion"}:
        raise ValueError("step_map_filter must be 'none' or 'positive_deletion'.")
    if args.mask_smoothing not in {"gaussian", "none"}:
        raise ValueError("mask_smoothing must be 'gaussian' or 'none'.")
    if args.published_mask_mode not in {"continuous", "heatmap_hot_regions"}:
        raise ValueError(
            "published_mask_mode must be 'continuous' or 'heatmap_hot_regions'."
        )
    if not 0.0 <= float(args.published_mask_threshold) <= 1.0:
        raise ValueError("published_mask_threshold must be in [0, 1].")
    if args.ig_noise_std < 0:
        raise ValueError("ig_noise_std cannot be negative.")
    if not isinstance(args.ig_noise_samples, int) or args.ig_noise_samples < 1:
        raise ValueError("ig_noise_samples must be a positive integer.")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initializing heatmap samples (which contain the questions, paths to files and outputs from the model)
    if samples_override is not None:
        samples = [HeatmapSample(**sample) for sample in samples_override]
        BASE_DIR = str(Path.cwd())
    elif dataset_jsonl is not None or base_dir is not None:
        # Legacy direct-path interface -- no run_dir means no rerun_results/ to find.
        BASE_DIR = base_dir if base_dir is not None else ".."
        summary_jsonl_path = dataset_jsonl if dataset_jsonl is not None else os.path.join(BASE_DIR, "mvtech_loco_runs", "xai_test_sample", "all_data_summary.jsonl")
        samples = build_heatmap_samples(summary_jsonl_path, BASE_DIR, mask_generation_mode=args.mask_generation_mode)
    else:
        if run_dir is None or dataset is None:
            raise ValueError("run_pipeline() requires either (run_dir AND dataset) or (dataset_jsonl / base_dir).")
        BASE_DIR = str(Path(run_dir))
        samples = build_heatmap_samples_v2(run_dir, dataset, mask_generation_mode=args.mask_generation_mode, include_rerun=args.include_rerun)

    print(f"Total anomaly reasoning steps identified: {len(samples)}")
    n_yes = sum(1 for s in samples if s.predicted_token == "YES")
    n_no = sum(1 for s in samples if s.predicted_token == "NO")
    print(f"  -> YES steps: {n_yes}, NO steps: {n_no}")
    if not samples:
        return {"num_samples": 0, "num_images": 0, "output_dir": args.output_dir}

    sam3_segmentation_cache = {}
    if args.segmentation_method in ("sam3", "hybrid"):
        sam3_segmentation_cache = precompute_sam3_segmentations(samples, args)

    # 2.1 Load Model (From HuggingFace)
    # model_name = "Qwen/Qwen3-VL-8B-Instruct"
    # adapter = Qwen3VLPrecomputedAdapter(model, processor)

    # 2.2 Load Model from Local Folder/Directory (pipeline_project/models/vlm_model by default)
    local_model_path = model_path if model_path is not None else DEFAULT_VLM_MODEL_PATH

    notify(phase="loading model", mask_type="--", step="--", iteration="--")
    adapter_loader = MODEL_ADAPTER_REGISTRY.get(model_adapter.strip().lower())
    if adapter_loader is None:
        raise ValueError(
            f"No differentiable XAI adapter is registered for {model_adapter!r}. "
            f"Available adapters: {sorted(MODEL_ADAPTER_REGISTRY)}."
        )
    model, processor, adapter, model_family = adapter_loader(local_model_path)
    notify(phase="model ready", mask_type="--", step="--", iteration="--")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 2048

    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    if args.implementation_profile == "paper_original":
        method = paper_igos_single_mask
    else:
        method = iGOS_p if args.method == "iGOS+" else iGOS_pp
    ig_schedule_fn = (
        None if args.ig_schedule == "fixed" else get_ig_iter_schedule
    )

    # 3. Generate Heatmaps, grouped per image so multi-step images can be
    # combined into one final_step_combined heatmap once all steps are done.
    for image_path, step_iter in groupby(samples, key=lambda s: s.image_path):
        step_group = list(step_iter)
        # Contains anomaly-score maps only from YES steps.
        # In these maps: 1 = anomalous/important, 0 = normal/not important.
        per_step_anomaly_scores = []
        # Same-order metric provenance for choosing one complete step map.
        # Keeping maps intact avoids the false-positive union created by a
        # pixelwise maximum over unrelated reasoning steps.
        per_step_metric_records = []
        #per_step_final_heatmaps = []   # collects final_step_heatmap from each step below
        step_cache = [] # Contains metric data only for the YES steps that were actually processed.

        # count the upstream reasoning steps whose predicted answer was YES, regardless of whether heatmap generation later succeeds.
        n_predicted_yes_steps = sum(step_sample.predicted_token.strip().upper() == "YES" for step_sample in step_group)

        # count only YES steps that successfully reach the end of token selection, segmentation, optimization, and map construction.
        n_successful_yes_heatmaps = 0

        # One log file per image, saved alongside the slic_check_*.png diagnostics
        # in args.output_dir. Filename derived from the image itself, not the step,
        # since it's shared across every step of this image.
        image_log_name = os.path.splitext(os.path.basename(image_path))[0]
        log_file_path = os.path.join(args.output_dir, f"log_{image_log_name}.txt")

        def log(msg):
            print(msg)
            with open(log_file_path, "a", encoding="utf-8") as lf:
                lf.write(str(msg) + "\n")

        sp_labels_cache_for_image = None   # (cache_key, label_array) -- filled on first use below, reused across all questions of this image

        # step_group is non-empty here (groupby guarantees at least one item per
        # group) and every sample in it shares the same image_path -- but NOT
        # necessarily the same decision_filepath/dataset_name/label, since two
        # different rows in the JSONL could theoretically point at the same
        # resolved image. In practice one candidate image belongs to exactly one
        # dataset/label, so reading it off step_group[0] is safe and avoids
        # referencing `sample` before the per-step loop defines it.
        first_sample = step_group[0]
        notify(
            phase="preparing image",
            image=os.path.basename(image_path),
            image_path=image_path,
            mask_type=args.segmentation_method,
            step=f"0/{len(step_group)}",
            iteration="--",
        )
        print(first_sample.decision_filepath)
        #ref_img_path = get_reference_path_for_category(first_sample.decision_filepath, first_sample.dataset_name, BASE_DIR, label=first_sample.label, log=log) #get_reference_path_for_category(image_path)   # your existing CLIP-retrieval lookup
        # Image-level information must be available even when every step is NO.
        dataset_name = first_sample.dataset_name
        processed_image_id = os.path.basename(first_sample.image_path)

        target_heatmap_dir = os.path.join(args.output_dir, "heatmaps")
        target_overlay_dir = os.path.join(args.output_dir, "overlays")
        target_raw_mask_dir = os.path.join(args.output_dir, "raw_masks")
        os.makedirs(target_heatmap_dir, exist_ok=True)
        os.makedirs(target_overlay_dir, exist_ok=True)
        os.makedirs(target_raw_mask_dir, exist_ok=True)

        jsonl_file_path = os.path.join(
            args.output_dir,
            "heatmap_metadata.jsonl",
        )

        reference_image = None

        # Load the exact upstream-selected reference when either reference
        # ablation needs it. Pipeline samples carry the explicit path; the
        # filename lookup remains only for legacy notebook calls.
        if args.reference_mode == "context" or args.baseline_mode == "reference":
            ref_img_path = first_sample.reference_image_path
            if ref_img_path is None:
                ref_img_path = get_reference_path_for_category(
                    first_sample.decision_filepath,
                    first_sample.dataset_name,
                    BASE_DIR,
                    label=first_sample.label,
                    log=log,
                )

            if ref_img_path is not None and Path(ref_img_path).is_file():
                reference_image, _, _ = load_and_resize_image(ref_img_path, args)
                log(f"  [reference] loaded: {ref_img_path}")
            else:
                log(
                    "  [reference] unavailable; fixed-context mode is disabled "
                    "and reference-baseline mode falls back to blur"
                )

        context_reference = (
            reference_image if args.reference_mode == "context" else None
        )

        for i, sample in enumerate(step_group):
            # log(f"\nProcessing Image {os.path.basename(sample.image_path)} | Step {sample.step} "
            #       f"({i+1}/{len(step_group)} steps)")
            log(f"\nProcessing Image {os.path.basename(sample.image_path)} | Step {sample.step} [{sample.predicted_token}] " #Include NO STEP
                  f"({i+1}/{len(step_group)} steps)")
            
            # NO means this step found no anomaly.
    # Its evidence must not be added to the final anomaly-localization map.
            if sample.predicted_token.strip().upper() != "YES":
                log(
                f"  [SKIP ANOMALY MAP] Step {sample.step} predicted "
                f"{sample.predicted_token}; excluded from anomaly-map generation."
                )
                continue
        
        
            source_image = Image.open(sample.image_path).convert("RGB")
            # target_width = args.input_size
            # target_height = int(target_width / (image.width / image.height))
            #Snap dimensions to perfect multiples of 28
            target_width = (args.input_size // 28) * 28
            target_height = int(target_width * source_image.height / source_image.width)
            target_height = max(28, (target_height // 28) * 28)
            resample = (
                Image.Resampling.BICUBIC
                if args.implementation_profile == "paper_original"
                else Image.Resampling.LANCZOS
            )
            if (
                args.implementation_profile == "paper_original"
                and args.baseline_mode == "blur"
            ):
                # The repository blurs the source-resolution PIL image first,
                # then applies the same differentiable-model resize to both.
                source_baseline = make_blur_baseline(
                    source_image, log=log, kernel_mode=args.blur_kernel_mode
                )
                image = source_image.resize((target_width, target_height), resample)
                baseline = source_baseline.resize(image.size, resample)
            else:
                image = source_image.resize((target_width, target_height), resample)
            #baseline = make_baseline_image(image, reference_np=reference_image_cache.get(image_path), log=log) #make_baseline_image(image)
            if (
                args.implementation_profile == "paper_original"
                and args.baseline_mode == "blur"
            ):
                pass
            elif args.baseline_mode == "reference" and reference_image is not None:
                baseline = reference_image.resize(image.size, Image.Resampling.LANCZOS)
                log("  [baseline] using retrieved reference image")
            else:
                baseline = make_blur_baseline(
                    image, log=log, kernel_mode=args.blur_kernel_mode
                )
            img_np = np.asarray(image)
            
            from skimage.segmentation import mark_boundaries

            # ============================================================
            # SEGMENTATION-INDEPENDENT SETUP (computed once per sample —
            # none of this depends on n_seg, so it doesn't belong inside
            # the per-segmentation loop below)
            # ============================================================
            output_ids = adapter.encode_answer(sample.output_text)
            # Pass the raw output text parameter downward so it can be handled by NLTK and Regex
            keyword_info = find_keywords_from_precomputed(adapter, image, baseline, sample.question, output_ids, 
                                                          threshold=args.keyword_threshold, #if args.keyword_percentile is None else None, 
                                                          output_text=sample.output_text, 
                                                          reference=context_reference,
                                                          percentile=getattr(args, 'keyword_percentile', None),  # Use percentile if configured
                                                          exclude_first_token=args.exclude_first_token,
                                                          selection_profile=(
                                                              "paper_tokenwise"
                                                              if args.implementation_profile == "paper_original"
                                                              else "span"
                                                          ),
                                                          log=log
                                                          )
            
            # show exactly what the generalized span selector found.
            # This diagnostic runs before deciding whether to skip iGOS.
            log("  [diag] selected visual spans: " 
                + json.dumps([{"text": item["token"], 
                               "positions": item["positions"], 
                               "all_span_positions": item.get("all_span_positions",item["positions"]), 
                               "importance": round(item["importance"],3)} for item in keyword_info], 
                               ensure_ascii=False,
                            )
                )
            
            if not keyword_info:
                log(f"  [SKIP] Step {sample.step} [{sample.predicted_token}] -- no keywords cleared threshold={args.keyword_threshold}")
                continue
            
            log(f"  [diag] full output text for step {sample.step}: \"{sample.output_text}\"")
            log(f"  [diag] per-token breakdown (ALL tokens, not just those above threshold={args.keyword_threshold}):")

            if context_reference is None:
                all_probs = adapter.token_probs(image, sample.question, output_ids)
                all_probs_blur = adapter.token_probs(baseline, sample.question, output_ids)
            else:
                all_probs = adapter.token_probs_with_reference(candidate=image, reference=context_reference, question=sample.question, output_ids=output_ids)
                all_probs_blur = adapter.token_probs_with_reference(candidate=baseline, reference=context_reference, question=sample.question, output_ids=output_ids)
            log(f"  [diag] probability dtype={all_probs.dtype}; selection numerical floor=1e-8")
            special_ids_set = set(adapter.tokenizer.all_special_ids)
            for idx in range(output_ids.shape[-1]):
                tid = output_ids[0, idx].item()
                if tid in special_ids_set:
                    continue
                tok_text = adapter.tokenizer.decode([tid]).strip()
                p_img = all_probs[idx].item()
                p_blur = all_probs_blur[idx].item()
                import math as _math
                imp = _math.log(p_img + 1e-12) - _math.log(p_blur + 1e-12)
                passed = "PASSED" if imp > args.keyword_threshold else "below threshold"
                log(f"    [{idx}] '{tok_text}' prob_image={p_img:.4f} prob_baseline={p_blur:.4f} "
                    f"importance={imp:.3f} -> {passed}")
                
            if args.max_keywords_per_sample:
                keyword_info = keyword_info[:args.max_keywords_per_sample]  # already sorted by importance, so this keeps the most relevant ones

            # positions = [k["position"] for k in keyword_info]
            # keywords  = [k["token"] for k in keyword_info]
            positions = []  #These lists now contain every token position belonging to every dynamically reconstructed text span.
            #Repeat the span-level weight for its subtoken positions.
            # This list must remain the same length and order as `positions`.
            keyword_weights_list = []

            # retain the complete reconstructed span text for diagnostics.
            keywords = []

            # compress the raw LLR range.
            # sqrt keeps stronger spans more important, but prevents one very large LLR from overwhelming all smaller anomaly descriptions.
            span_weights = [math.sqrt(max(keyword["importance"], 0.0)) for keyword in keyword_info]

            # Normalize all selected-span weights to sum to one.
            span_weight_sum = (sum(span_weights) + 1e-8)

            for keyword, span_weight in zip(keyword_info, span_weights):
                span_positions = keyword["positions"]
                span_text = keyword["token"]
                positions.extend(span_positions)
                keywords.append(span_text)
                # Normalized total weight assigned to this span.
                normalized_span_weight = (span_weight / span_weight_sum)
                # divide only among this span's subtokens.
                # Tokenizer fragmentation no longer changes a span's total weight.
                per_token_weight = (normalized_span_weight / max(1, len(span_positions)))
                keyword_weights_list.extend([per_token_weight] * len(span_positions))

            if args.token_weighting == "paper_sum":
                # Eq. (7): f is the unweighted cumulative log-likelihood of
                # every crucial token. A unit weight preserves the sum inside
                # interval_score; no word-level LLR weighting is introduced.
                keyword_weights_list = [1.0] * len(positions)
                log("  [targets] paper_sum: unweighted selected-token log-probability sum")

            # preserve repeated spans and their token positions.
            # A dictionary keyed only by text would overwrite duplicate words.
            keyword_scores = [
                                {"text": keyword["token"],
                                 "positions": keyword["positions"],
                                 "importance": round(keyword["importance"],3)
                                } for keyword in keyword_info
                             ]
            
            # diagnostic contract check.
            # interval_score() assumes one weight per selected token position.
            assert len(positions) == len(keyword_weights_list), (f"Position/weight mismatch: {len(positions)} positions versus {len(keyword_weights_list)} weights")

            if context_reference is None:
                # Candidate-only control.
                base_inputs = adapter.prepare_inputs(image=image, question=sample.question)
                blur_inputs = adapter.prepare_inputs(image=baseline, question=sample.question)

                candidate_start = 0
                candidate_end = (base_inputs["pixel_values"].shape[0])

            else:
                # candidate and fixed reference.
                base_inputs = adapter.prepare_reference_context_inputs(candidate=image, reference=context_reference, question=sample.question)
                blur_inputs = adapter.prepare_reference_context_inputs(candidate=baseline, reference=context_reference, question=sample.question)

                grids = base_inputs["image_grid_thw"]

                # Qwen concatenates visual patches in image order.
                patch_counts = [int(grid_item.prod().item()) for grid_item in grids]

                assert len(patch_counts) == 2, ("Reference-context mode expected exactly candidate and reference images.")

                assert (base_inputs["pixel_values"].shape[0] == sum(patch_counts)), ("Visual patch counts do not match the concatenated pixel_values tensor.")

                # Candidate was supplied first.
                candidate_start = 0
                candidate_end = patch_counts[0]

            # Save the complete input, including reference patches when present.
            adapter.current_base_inputs = base_inputs

            # only this range may be replaced by the optimized mask.
            adapter.candidate_patch_slice = slice(candidate_start, candidate_end)

            # Extract candidate patches for the differentiable mask grid.
            img_pv = base_inputs["pixel_values"][adapter.candidate_patch_slice]

            # Extract only blurred-candidate patches.
            blur_pv = blur_inputs["pixel_values"][adapter.candidate_patch_slice]

            # Candidate is image 0 in both modes.
            grid = base_inputs["image_grid_thw"][0]
            H_p, W_p = grid[1].item(), grid[2].item()

            merge_size = processor.image_processor.merge_size
            image_grid = blocks_to_grid(img_pv, H_p, W_p, merge_size)
            baseline_grid = blocks_to_grid(blur_pv, H_p, W_p, merge_size)

            image_tensor = image_grid.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            baseline_tensor = baseline_grid.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            visual_tensor = F_t.to_tensor(image).unsqueeze(0).cuda()
            if args.implementation_profile == "paper_original":
                # Match the repository's BTV calculation on normalized RGB
                # rather than on Qwen's packed patch-vector channels.
                image_mean = torch.tensor(
                    processor.image_processor.image_mean,
                    device=visual_tensor.device,
                    dtype=visual_tensor.dtype,
                ).view(1, 3, 1, 1)
                image_std = torch.tensor(
                    processor.image_processor.image_std,
                    device=visual_tensor.device,
                    dtype=visual_tensor.dtype,
                ).view(1, 3, 1, 1)
                regularization_tensor = (visual_tensor - image_mean) / image_std
            else:
                regularization_tensor = visual_tensor

            label = output_ids.detach()
            if label.dim() == 1:
                label = label.unsqueeze(0)
            all_positions = positions
            if args.implementation_profile == "paper_original":
                # Repository/paper optimization happens on one coarse square
                # mask, independently of the LVLM's visual-token grid.
                init_mask = torch.ones((1, 1, args.size, args.size)).cuda()
            else:
                init_mask = torch.ones((1, 1, H_p, W_p)).cuda()
            pred_data = {
                "labels": label, "keywords": positions,
                "boxes": np.array([[0, 0, target_width, target_height]]),
                "no_res": False, "pred_text": sample.output_text, "keywords_text": keywords, "init_masks": [init_mask]
            }

            step_cache.append({
                "image_tensor": image_tensor, "baseline_tensor": baseline_tensor,
                "label": label, "pred_data": pred_data, "all_positions": all_positions,
                "question": sample.question, "image_size": image.size,
            })

            step_num = sample.step

            # ============================================================
            # construct contiguous multiword visual phrases.
            #
            # A joint sentence-level loss may localize only a sufficient subset of a multiword object description. A contiguous phrase such as "red triangular item" provides a more object-specific refinement.
            # ============================================================
            phrase_candidates = []

            # ============================================================
            # construct meaningful object/defect phrases from the
            # complete generated output.
            #
            # Unlike the old implementation, this may restore a required
            # head noun such as "cube", "item", or "component" even when
            # that noun did not independently clear keyword_threshold.
            # ============================================================
            if args.mask_content_mode == "keyword_only":
                phrase_candidates = []
            else:
                phrase_candidates = build_meaningful_visual_phrases(
                    adapter=adapter,
                    output_text=sample.output_text,
                    output_ids=output_ids,
                    keyword_info=keyword_info,
                    # Keep multiple meaningful phrases so secondary anomaly
                    # objects are not deleted before optimization.
                    max_phrases=args.max_phrases_per_sample,
                )

            # log exactly which positions were selected keywords
            # and which positions were restored as phrase-support tokens.
            selected_keyword_position_set = {position for keyword in keyword_info for position in keyword["positions"]}

            phrase_log_data = []

            for phrase in phrase_candidates:
                phrase_log_data.append({
                    "text": phrase["text"],
                    "positions": phrase["positions"],
                    "anchor_positions": [position for position in phrase["positions"] if position in selected_keyword_position_set],
                    "support_positions": [position for position in phrase["positions"] if position not in selected_keyword_position_set],
                    "importance": round(phrase["importance"], 3),
                })

            log("  [diag] meaningful phrase candidates: " + json.dumps(phrase_log_data, ensure_ascii=False))

            # Resolve phrase targets before entering the mask loop. In
            # phrase-only mode these become the PRIMARY optimization target;
            # we must not spend a complete keyword optimization merely to
            # discard its mask afterward.
            phrase_positions, phrase_weights = flatten_balanced_phrase_targets(
                phrase_candidates
            )
            if args.mask_content_mode == "phrase_only" and not phrase_positions:
                log(
                    f"  [SKIP] Step {sample.step} [{sample.predicted_token}] -- "
                    "phrase_only found no meaningful phrase target"
                )
                continue

            per_seg_patch_masks = []   # patch-grid resolution -> feeds the combined metric() # visibility masks used by metric()
            per_seg_rgb_heatmaps = []  # RGB resolution        -> feeds the combined visual   # anomaly-score maps used for visualization/combination

            # retain the evaluation scores associated with each segmentation
            # level so the selected final step map can use its own Del/Ins values.
            per_seg_del_scores = []
            per_seg_ins_scores = []

            # ============================================================
            # Build the active mask configurations.
            #
            # None means a dense Qwen patch-grid mask with no segmentation.
            # Integer values mean segmented runs with that requested or
            # detected number of regions.
            # ============================================================
            if args.segmentation_method == "none":
                active_mask_configs = [None]

            elif args.segmentation_method == "sam3":
                sam3_labels_cache, sam3_n_labels = (sam3_segmentation_cache[sample.image_path])
                active_mask_configs = [sam3_n_labels]

            elif args.segmentation_method == "hybrid":
                sam3_labels_cache, _ = (sam3_segmentation_cache[sample.image_path])
                active_mask_configs = [max(args.num_superpixels)]

            elif args.segmentation_method == "slic":
                active_mask_configs = [max(args.num_superpixels)]

            else:
                raise ValueError(f"Unsupported segmentation method: {args.segmentation_method}")

            # ============================================================
            # SEGMENTATION-DEPENDENT LOOP: one full iGOS+ run per n_seg,
            # saved individually, then averaged into a combined mask below
            # ============================================================
            for n_seg in active_mask_configs:

                use_segmentation = n_seg is not None

                if not use_segmentation:
                    # ========================================================
                    # DENSE PATCH-GRID MODE
                    # ========================================================
                    # No label maps or region graph are created.
                    sp_map_patch = None
                    sp_map_rgb = None
                    num_superpixels = None

                    if args.implementation_profile == "paper_original":
                        num_mask_variables = int(args.size * args.size)
                        mode_tag = "paper_original_32x32"
                    else:
                        # One trainable scalar for every Qwen visual-grid location.
                        num_mask_variables = int(H_p * W_p)
                        mode_tag = "dense"

                    iterations_i, ig_iter_i = get_iter_schedule(num_mask_variables, args)

                    log(
                        f"  [{mode_tag}] native_shape={tuple(init_mask.shape[-2:])}, "
                        f"qwen_grid={H_p}x{W_p}, variables={num_mask_variables}, "
                        f"iterations={iterations_i}, ig_iter={ig_iter_i}"
                    )

                else:
                    # ========================================================
                    # EXISTING SEGMENTATION MODES
                    # ========================================================
                    cache_key = (n_seg, args.segmentation_method)

                    if (sp_labels_cache_for_image is not None and sp_labels_cache_for_image[0] == cache_key):
                        sp_labels_initial = sp_labels_cache_for_image[1]

                    elif args.segmentation_method == "sam3":
                        sp_labels_initial = sam3_labels_cache
                        sp_labels_cache_for_image = (cache_key, sp_labels_initial)

                    elif args.segmentation_method == "hybrid":
                        sp_labels_initial, _ = segment_with_slic_within_sam3(img_np, None, args, n_seg, sam3_labels=sam3_labels_cache)
                        sp_labels_cache_for_image = (cache_key, sp_labels_initial)

                    else:
                        sp_labels_initial = slic(img_np, n_segments=n_seg, compactness=args.sp_compactness, start_label=0, channel_axis=-1)
                        sp_labels_cache_for_image = (cache_key, sp_labels_initial)

                    sp_labels_full = torch.from_numpy(sp_labels_initial.astype(np.int64)).cuda()

                    if args.save_diagnostics:
                        boundary_vis = mark_boundaries(img_np, sp_labels_full.cpu().numpy(), color=(1, 0, 0))
                        Image.fromarray((boundary_vis * 255).astype(np.uint8)).save(os.path.join(args.output_dir,f"{args.segmentation_method}_check_{n_seg}sp_{os.path.basename(sample.image_path)}.png"))

                    # Renumber labels to a dense 0 ... K-1 range.
                    unique_ids, sp_map_rgb = torch.unique(sp_labels_full, return_inverse=True)
                    sp_map_rgb = sp_map_rgb.view(sp_labels_full.shape)

                    num_superpixels = int(unique_ids.numel())
                    sp_map_patch = pool_labels_to_grid(sp_map_rgb, H_p, W_p)

                    num_mask_variables = num_superpixels
                    mode_tag = (f"{args.segmentation_method}_{n_seg}sp")

                    iterations_i, ig_iter_i = get_iter_schedule(num_mask_variables, args)

                    log(f"  [{mode_tag} -> {num_superpixels} actual] iterations={iterations_i}, ig_iter={ig_iter_i}")

                phrase_only = args.mask_content_mode == "phrase_only"
                primary_positions = phrase_positions if phrase_only else all_positions
                primary_weights = phrase_weights if phrase_only else keyword_weights_list
                primary_iterations = (
                    args.phrase_joint_iterations if phrase_only else iterations_i
                )
                primary_diag_suffix = "_phrase_joint" if phrase_only else ""
                primary_mask_type = (
                    f"{mode_tag}:phrase" if phrase_only else mode_tag
                )

                t0 = time.time()            #Time whole run for each mask

                def presentation_iteration(native_mask, *, stage: str, iteration: int, iterations: int, ig_iter: int) -> None:
                    """Persist an ordered smooth frame for GIF/presentation use."""
                    if not args.presentation_enabled:
                        return
                    try:
                        frame_root = Path(args.output_dir) / "frames" / str(processed_image_id) / stage
                        frame_root.mkdir(parents=True, exist_ok=True)
                        _, rgb_visibility = expand_native_visibility_mask(
                            native_mask, use_segmentation, sp_map_patch, sp_map_rgb,
                            target_height, target_width,
                            patch_height=H_p if args.implementation_profile == "paper_original" else None,
                            patch_width=W_p if args.implementation_profile == "paper_original" else None,
                        )
                        score = (1.0 - rgb_visibility).clamp(0.0, 1.0)
                        score = TF.gaussian_blur(score, kernel_size=[5, 5], sigma=[1.0, 1.0])
                        arr = score[0, 0].detach().float().cpu().numpy()
                        # Match normal ablation heatmaps (dense/SAM3): use
                        # 5th/99th percentile display normalization while
                        # retaining the unnormalized continuous mask in .npy.
                        lo = float(np.percentile(arr, 5))
                        hi = float(np.percentile(arr, 99))
                        if hi - lo < 0.05:
                            hi = lo + 0.05
                        display_arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
                        stem = f"frame_{int(iteration):04d}"
                        np.save(frame_root / f"{stem}_mask.npy", arr.astype(np.float32))
                        Image.fromarray(np.rint(arr * 255).astype(np.uint8), mode="L").save(frame_root / f"{stem}_mask.png")
                        color = cv2.applyColorMap(np.rint(display_arr * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                        Image.fromarray(color).save(frame_root / f"{stem}_heatmap.png")
                        base = visual_tensor[0].detach().float().cpu().permute(1, 2, 0).numpy()
                        base = np.clip(base, 0.0, 1.0)
                        overlay = np.clip(0.5 * base + 0.5 * (color.astype(np.float32) / 255.0), 0.0, 1.0) #Changed the heatmap saving to 50:50 blend instead of 55:45 blend
                        Image.fromarray(np.rint(overlay * 255).astype(np.uint8)).save(frame_root / f"{stem}_overlay.png")
                        with (frame_root / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
                            manifest.write(json.dumps({"stage": stage, "iteration": int(iteration), "iterations": int(iterations), "ig_iter": int(ig_iter), "mask": f"{stem}_mask.png", "heatmap": f"{stem}_heatmap.png", "overlay": f"{stem}_overlay.png"}) + "\n")
                    except Exception as error:
                        log(f"presentation frame failed: {type(error).__name__}: {error}")

                if args.presentation_enabled:
                    initial_native = (
                        torch.ones((1, 1, 1, num_superpixels), device=init_mask.device)
                        if use_segmentation else torch.ones_like(init_mask)
                    )
                    presentation_iteration(
                        initial_native, stage="keywords", iteration=0,
                        iterations=primary_iterations, ig_iter=ig_iter_i,
                    )
                notify(
                    phase="optimising",
                    image=processed_image_id,
                    mask_type=primary_mask_type,
                    step=f"{step_num}/{len(step_group)}",
                    iteration=f"0/{primary_iterations} · IG {ig_iter_i}",
                    iteration_current=0,
                    iteration_total=primary_iterations,
                )

                def optimisation_progress(*, iteration: int, iterations: int, ig_iter: int) -> None:
                    notify(
                        phase="optimising",
                        image=processed_image_id,
                        mask_type=primary_mask_type,
                        step=f"{step_num}/{len(step_group)}",
                        iteration=f"{iteration}/{iterations} · IG {ig_iter}",
                        iteration_current=iteration,
                        iteration_total=iterations,
                    )

                avg_native_mask, last_result, per_seed_results = run_igos_ensemble(
                    method, args, seeds=args.ensemble_seeds,
                    adapter=adapter, model=model, model_name=model_family,
                    is_negative_label=(
                        False if phrase_only else sample.predicted_token == "NO"
                    ), # gate the unreliable-probe widen fallback below
                    init_mask=init_mask, image=image_tensor, baseline=baseline_tensor, label=label,
                    size=args.size, iterations=primary_iterations, ig_iter=ig_iter_i, L1=args.L1, L2=args.L2, L3=args.L3,
                    lr=args.lr, opt=args.opt, prompt=sample.question, image_size=[image.size], keyword_weights=primary_weights,
                    positions=primary_positions, processor=processor, resolution=None, rgb_image=regularization_tensor,
                    sp_map_patch=sp_map_patch, sp_map_rgb=sp_map_rgb, num_superpixels=num_superpixels,sp_keep_fraction=args.sp_keep_fraction, log=log, reference_diff_prior=None,
                    # routes the prefilter diagnostic dump in item 1 above to a
                    # per-sample folder, tagged so multiple steps/images don't collide.
                    diag_dir=(os.path.join(args.output_dir, "diagnostics") if args.save_diagnostics else None),
                    diag_tag=f"{dataset_name}_{processed_image_id}_step{step_num}_{mode_tag}{primary_diag_suffix}", #f"{dataset_name}_{processed_image_id}_step{step_num}_sp{n_seg}",
                    get_ig_iter_schedule=ig_schedule_fn, static_exclude_mask=None,
                    # Dense-method fidelity switches -- see Args class docstring
                    # and igos_core_methods.iGOS_p's regularization_loss.
                    dense_l2_reduction=args.dense_l2_reduction, mask_clamp_range=args.mask_clamp_range,
                    grad_clip_norm=args.grad_clip_norm, dense_l2_floor_fraction=args.dense_l2_floor_fraction,
                    progress_callback=optimisation_progress,
                    iteration_callback=lambda mask, iteration, iterations, ig_iter: presentation_iteration(mask, stage="keywords", iteration=iteration, iterations=iterations, ig_iter=ig_iter),
                )
                masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, compact_mask = last_result

                # ============================================================
                # refine the strongest contiguous visual phrase and
                # union it with the existing joint compact mask.
                # ============================================================

                if avg_native_mask is None:
                    log(f"  [SKIP] No valid optimized mask for step={step_num}, mode={mode_tag}")
                    continue
                
                # The first optimization result is the JOINT keyword mask.
                # Preserve it before phrase refinement so it can be saved and
                # evaluated independently.
                joint_native_mask = avg_native_mask.detach().clone()

                # ============================================================
                # SAVE THE JOINT KEYWORD MASK BEFORE PHRASE REFINEMENT
                # ============================================================
                joint_masks_patch, joint_visibility_mask = expand_native_visibility_mask(
                    native_mask=joint_native_mask,
                    use_segmentation=use_segmentation,
                    sp_map_patch=sp_map_patch,
                    sp_map_rgb=sp_map_rgb,
                    target_height=target_height,
                    target_width=target_width,
                    patch_height=H_p,
                    patch_width=W_p,
                )

                if args.implementation_profile == "paper_original":
                    # The repository normalizes the native optimized mask
                    # before cv2.resize. Doing it here (rather than stretching
                    # Qwen-patch values afterward) preserves that ordering and
                    # produces the intended continuous bilinear field.
                    native_min = joint_native_mask.amin(
                        dim=(-2, -1), keepdim=True
                    )
                    native_max = joint_native_mask.amax(
                        dim=(-2, -1), keepdim=True
                    )
                    native_spread = native_max - native_min
                    normalized_native_visibility = torch.where(
                        native_spread > 1e-12,
                        (
                            (joint_native_mask - native_min)
                            / native_spread.clamp_min(1e-12)
                        ).clamp(0.0, 1.0),
                        torch.ones_like(joint_native_mask),
                    )
                    joint_visibility_mask = F.interpolate(
                        normalized_native_visibility,
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                if use_segmentation and args.mask_smoothing == "gaussian":
                    joint_k = max(3,int((target_width * target_height / num_superpixels) ** 0.5) // 2| 1)
                    joint_visibility_smooth = TF.gaussian_blur(joint_visibility_mask, kernel_size=[joint_k, joint_k], sigma=[joint_k / 3, joint_k / 3])
                else:
                    joint_visibility_smooth = joint_visibility_mask

                joint_anomaly_score = (1.0 - joint_visibility_smooth).clamp(0.0, 1.0)

                if args.save_intermediate_outputs:
                    joint_del, joint_ins, _, _, _ = metric(
                        args, image_tensor, baseline_tensor, joint_masks_patch.detach(), model, model_family,
                        label, 0, pred_data, size=args.size, prompt=sample.question, image_size=[image.size],
                        positions=primary_positions, adapter=adapter, resolution=None,
                    )
                else:
                    # The combined per-step map is scored once below. Avoid an
                    # identical metric pass solely for an intermediate file.
                    joint_del = torch.full((1,), float("nan"))
                    joint_ins = torch.full((1,), float("nan"))

                # In phrase-only mode the primary mask already targets phrases,
                # so it must not be mislabeled/saved as a keyword-only output.
                if args.mask_content_mode != "phrase_only":
                    joint_image_id = f"{processed_image_id}_{mode_tag}_keyword_joint" #(f"{processed_image_id}_{mode_tag}_joint")

                    saved_outputs = save_heatmaps(
                        joint_anomaly_score,
                        visual_tensor,
                        args.size,
                        dataset_name=dataset_name,
                        image_id=joint_image_id,
                        step_number=step_num,
                        heatmap_dir=target_heatmap_dir,
                        overlay_dir=target_overlay_dir,
                        raw_mask_dir=target_raw_mask_dir,
                        write_files=args.save_intermediate_outputs,
                    )

                    saved_output = saved_outputs[0]

                    with open(jsonl_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"dataset": dataset_name,
                                            "image_id": processed_image_id,
                                            "step": step_num,
                                            "mask_type": "keyword_joint", #"joint",
                                            "mask_mode": ("segmented" if use_segmentation else "dense_patch"),
                                            "segmentation_method": args.segmentation_method,
                                            "n_seg_requested": (n_seg if use_segmentation else None),
                                            "n_seg_actual": (num_superpixels if use_segmentation else None),
                                            "n_mask_variables": num_mask_variables,
                                            "iterations": iterations_i,
                                            "ig_iter": ig_iter_i,
                                            "predicted_token": sample.predicted_token,
                                            "keywords": keyword_scores,
                                            "target_positions": all_positions,
                                            "phrase": None,
                                            "deletion_score": float(joint_del.sum().item()),
                                            "insertion_score": float(joint_ins.sum().item()),
                                            "raw_mask_path": saved_output["raw_mask_path"],
                                            "raw_preview_path": saved_output["raw_preview_path"],
                                            "heatmap_path": saved_output["heatmap_path"],
                                            "overlay_path": saved_output["overlay_path"],
                                            "raw_mask_shape": saved_output["raw_shape"],
                                            "raw_mask_min": saved_output["raw_min"],
                                            "raw_mask_max": saved_output["raw_max"],
                                            "raw_mask_mean": saved_output["raw_mean"],
                                            "raw_mask_std": saved_output["raw_std"],
                                            "display_percentile_low": (saved_output["display_percentile_low"]),
                                            "display_percentile_high": (saved_output["display_percentile_high"]),
                                            }) + "\n")

                primary_label = "phrase-primary" if phrase_only else "joint"
                log(f"   -> [{primary_label} mask, mode={mode_tag}] Del={joint_del.sum().item():.3f}, Ins={joint_ins.sum().item():.3f}")

                # ========================================================
                # optimize ALL meaningful phrases together in one
                # balanced phrase-joint run.
                # ========================================================
                phrase_joint_native_mask = None
                phrase_joint_anomaly_score = None
                phrase_joint_del_value = None
                phrase_joint_ins_value = None

                if phrase_only:
                    # The primary run above already targeted the balanced
                    # phrase positions. Reuse it directly instead of running
                    # and discarding a keyword mask first.
                    phrase_joint_native_mask = joint_native_mask
                    phrase_joint_last_result = last_result
                    phrase_joint_runtime_sec = round(time.time() - t0, 2)
                elif phrase_positions:
                    phrase_text_summary = [phrase["text"] for phrase in phrase_candidates]

                    log(f"  [phrase-joint refinement] phrases={phrase_text_summary!r}, positions={phrase_positions}, weights={[round(weight, 4) for weight in phrase_weights]}")

                    phrase_joint_t0 = time.time()
                    phrase_mask_type = f"{mode_tag}:phrase"
                    if args.presentation_enabled:
                        presentation_iteration(
                            initial_native, stage="phrases", iteration=0,
                            iterations=args.phrase_joint_iterations, ig_iter=ig_iter_i,
                        )
                    notify(
                        phase="optimising phrase",
                        image=processed_image_id,
                        mask_type=phrase_mask_type,
                        step=f"{step_num}/{len(step_group)}",
                        iteration=f"0/{args.phrase_joint_iterations} · IG {ig_iter_i}",
                        iteration_current=0,
                        iteration_total=args.phrase_joint_iterations,
                    )

                    def phrase_optimisation_progress(*, iteration: int, iterations: int, ig_iter: int) -> None:
                        notify(
                            phase="optimising phrase",
                            image=processed_image_id,
                            mask_type=phrase_mask_type,
                            step=f"{step_num}/{len(step_group)}",
                            iteration=f"{iteration}/{iterations} · IG {ig_iter}",
                            iteration_current=iteration,
                            iteration_total=iterations,
                        )

                    phrase_joint_native_mask, phrase_joint_last_result, _ = run_igos_ensemble(method, args, seeds=args.ensemble_seeds, adapter=adapter, model=model, model_name=model_family,
                            # the phrase-joint target belongs to a YES step.
                            is_negative_label=False, init_mask=init_mask, image=image_tensor, baseline=baseline_tensor, label=label, size=args.size,
                            # a single phrase-joint run replaces all separate phrase runs.
                            iterations=args.phrase_joint_iterations, ig_iter=ig_iter_i, L1=args.L1, L2=args.L2, L3=args.L3, lr=args.lr, opt=args.opt, prompt=sample.question, image_size=[image.size],
                            # every meaningful phrase contributes to this one balanced target.
                            positions=phrase_positions, keyword_weights=phrase_weights, processor=processor, resolution=None, rgb_image=regularization_tensor, sp_map_patch=sp_map_patch, sp_map_rgb=sp_map_rgb, num_superpixels=num_superpixels,
                            sp_keep_fraction=args.sp_keep_fraction, log=log, reference_diff_prior=None, diag_dir=(os.path.join(args.output_dir, "diagnostics") if args.save_diagnostics else None),
                            diag_tag=(f"{dataset_name}_{processed_image_id}_step{step_num}_{mode_tag}_phrase_joint"), get_ig_iter_schedule=ig_schedule_fn, static_exclude_mask=None,
                            # Dense-method fidelity switches -- see Args class docstring
                            # and igos_core_methods.iGOS_p's regularization_loss.
                            dense_l2_reduction=args.dense_l2_reduction, mask_clamp_range=args.mask_clamp_range,
                            grad_clip_norm=args.grad_clip_norm, dense_l2_floor_fraction=args.dense_l2_floor_fraction,
                            progress_callback=phrase_optimisation_progress,
                            iteration_callback=lambda mask, iteration, iterations, ig_iter: presentation_iteration(mask, stage="phrases", iteration=iteration, iterations=iterations, ig_iter=ig_iter),
                        )

                    phrase_joint_runtime_sec = round(time.time() - phrase_joint_t0, 2)

                if phrase_joint_native_mask is not None:
                    # expand phrase-joint mask in exactly the same
                    # representation as the keyword-joint mask.
                    phrase_joint_patch_visibility, phrase_joint_rgb_visibility = (
                        expand_native_visibility_mask(
                            native_mask=phrase_joint_native_mask.detach(),
                            use_segmentation=use_segmentation,
                            sp_map_patch=sp_map_patch,
                            sp_map_rgb=sp_map_rgb,
                            target_height=target_height,
                            target_width=target_width,
                            patch_height=H_p,
                            patch_width=W_p,
                        )
                    )

                    # use the same boundary softening as the keyword mask.
                    if use_segmentation and args.mask_smoothing == "gaussian":
                        phrase_joint_k = max(3, int((target_width * target_height / num_superpixels) ** 0.5) // 2 | 1)
                        phrase_joint_rgb_visibility_smooth = TF.gaussian_blur(phrase_joint_rgb_visibility, kernel_size=[phrase_joint_k, phrase_joint_k], sigma=[phrase_joint_k / 3, phrase_joint_k / 3])
                    else:
                        phrase_joint_rgb_visibility_smooth = (phrase_joint_rgb_visibility)

                    # convert visibility to anomaly scores.
                    phrase_joint_anomaly_score = (1.0 - phrase_joint_rgb_visibility_smooth).clamp(0.0, 1.0)

                    # evaluate against the phrase target itself.
                    if args.save_intermediate_outputs:
                        phrase_joint_del, phrase_joint_ins, _, _, _ = metric(args, image_tensor, baseline_tensor, phrase_joint_patch_visibility.detach(), model, model_family, label, 0,
                            pred_data, size=args.size, prompt=sample.question, image_size=[image.size], positions=phrase_positions, adapter=adapter, resolution=None,
                        )
                        phrase_joint_del_value = float(phrase_joint_del.sum().item())
                        phrase_joint_ins_value = float(phrase_joint_ins.sum().item())
                    else:
                        # Its contribution is assessed once as part of the final
                        # content map; do not run a second curve just to gate it.
                        phrase_joint_del_value = float("nan")
                        phrase_joint_ins_value = float("nan")

                    log(f"   -> [phrase-joint mask] phrases={[phrase['text'] for phrase in phrase_candidates]!r}, Del={phrase_joint_del_value:.3f}, Ins={phrase_joint_ins_value:.3f}, runtime={phrase_joint_runtime_sec:.1f}s")

                    # ====================================================
                    # save the independent all-phrases mask before
                    # combining it with the keyword mask.
                    # ====================================================
                    phrase_joint_image_id = (f"{processed_image_id}_{mode_tag}_phrase_joint")

                    phrase_joint_saved_outputs = save_heatmaps(
                        phrase_joint_anomaly_score,
                        visual_tensor,
                        args.size,
                        dataset_name=dataset_name,
                        image_id=phrase_joint_image_id,
                        step_number=step_num,
                        heatmap_dir=target_heatmap_dir,
                        overlay_dir=target_overlay_dir,
                        raw_mask_dir=target_raw_mask_dir,
                        write_files=args.save_intermediate_outputs,
                    )

                    phrase_joint_saved_output = (phrase_joint_saved_outputs[0])

                    with open(jsonl_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"dataset": dataset_name,
                                            "image_id": processed_image_id,
                                            "step": step_num,
                                            "mask_type": "phrase_joint", # identifies this as one run targeting all meaningful phrases.
                                            "mask_mode": ("segmented" if use_segmentation else "dense_patch"),
                                            "segmentation_method": (args.segmentation_method),
                                            "phrases": [{"text": phrase["text"], "positions": phrase["positions"], "importance": phrase["importance"]} for phrase in phrase_candidates],
                                            "target_positions": phrase_positions,
                                            "target_weights": phrase_weights,
                                            "deletion_score": (phrase_joint_del_value),
                                            "insertion_score": (phrase_joint_ins_value),
                                            "runtime_sec": (phrase_joint_runtime_sec),
                                            "raw_mask_path": (phrase_joint_saved_output["raw_mask_path"]),
                                            "raw_preview_path": (phrase_joint_saved_output["raw_preview_path"]),
                                            "heatmap_path": (phrase_joint_saved_output["heatmap_path"]),
                                            "overlay_path": (phrase_joint_saved_output["overlay_path"]),
                                            }) + "\n")

                
                phrase_joint_accepted = (
                    phrase_joint_anomaly_score is not None
                    and (
                        not args.save_intermediate_outputs
                        or (
                            phrase_joint_del_value is not None
                            and phrase_joint_del_value > args.phrase_min_deletion_score
                        )
                    )
                )

                if args.mask_content_mode == "phrase_only":
                    # The old prototype hid keyword artifacts but still fed
                    # the keyword/phrase union into the final image map. This
                    # branch makes the ablation literal: only the phrase target
                    # contributes. If no valid noun phrase exists, publish a
                    # neutral visibility map (zero anomaly score).
                    if phrase_joint_anomaly_score is not None:
                        combined_keyword_phrase_score = phrase_joint_anomaly_score
                        combined_keyword_phrase_patch_visibility = (
                            phrase_joint_patch_visibility.detach()
                        )
                    else:
                        combined_keyword_phrase_score = torch.zeros_like(joint_anomaly_score)
                        combined_keyword_phrase_patch_visibility = torch.ones_like(
                            joint_masks_patch.detach()
                        )
                    metric_positions = phrase_positions or all_positions
                else:
                    combined_keyword_phrase_score = joint_anomaly_score
                    combined_keyword_phrase_patch_visibility = joint_masks_patch.detach()
                    metric_positions = all_positions

                if args.mask_content_mode == "both_combined" and phrase_joint_accepted:
                    # Normalize each component exactly as save_heatmaps() does
                    # before fusing. Raw keyword scores can otherwise be much
                    # larger than raw phrase scores, making the phrase heatmap
                    # disappear from the final max despite being accepted.
                    joint_score_for_fusion = normalize_display_score(joint_anomaly_score)
                    phrase_score_for_fusion = normalize_display_score(phrase_joint_anomaly_score)
                    weighted_phrase_score = (
                        args.phrase_union_weight * phrase_score_for_fusion
                    ).clamp(0.0, 1.0)

                    combined_keyword_phrase_score = torch.maximum(
                        joint_score_for_fusion, weighted_phrase_score
                    )

                    # construct the same weighted anomaly union at
                    # Qwen patch-grid resolution for deletion/insertion metrics.
                    joint_patch_anomaly_score = normalize_display_score(
                        (1.0 - joint_masks_patch.detach()).clamp(0.0, 1.0)
                    )

                    phrase_patch_anomaly_score = normalize_display_score(
                        (1.0 - phrase_joint_patch_visibility.detach()).clamp(0.0, 1.0)
                    )

                    weighted_phrase_patch_score = (args.phrase_union_weight * phrase_patch_anomaly_score).clamp(0.0, 1.0)

                    combined_patch_anomaly_score = torch.maximum(joint_patch_anomaly_score, weighted_phrase_patch_score)

                    # metric() expects visibility:
                    # 1 = candidate visible, 0 = candidate replaced by baseline.
                    combined_keyword_phrase_patch_visibility = (1.0 - combined_patch_anomaly_score).clamp(0.0, 1.0)

                    log(
                        "  [keyword+phrase union accepted] "
                        f"phrase Del={phrase_joint_del_value:.3f}, "
                        f"weight={args.phrase_union_weight:.2f}"
                    )

                elif (
                    args.mask_content_mode == "both_combined"
                    and phrase_joint_anomaly_score is not None
                ):
                    log(
                        "  [keyword+phrase union rejected] "
                        f"phrase Del={phrase_joint_del_value}; "
                        "using keyword-joint map only"
                    )

                # --- Save each seed's own heatmap individually, before any averaging,
                # so per-seed masks can be visually compared later. ---
                for seed_val, seed_result, seed_runtime_sec in (
                    per_seed_results if args.save_intermediate_outputs else []
                ):
                    # mask_content_mode="phrase_only" means no
                    # keyword-derived output at all, including per-seed maps.
                    if args.mask_content_mode == "phrase_only":
                        continue
                    #seed_masks, _, _, _, _, _, _, seed_compact_mask = seed_result
                    seed_final_mask, _, _, _, _, _, _, seed_compact_mask = (seed_result)
                    # Select the native representation produced by this mode.
                    seed_native_mask = (seed_compact_mask if use_segmentation else seed_final_mask)
                    if seed_native_mask is None:
                        continue

                    seed_masks_patch, seed_visibility_mask = (expand_native_visibility_mask(native_mask=seed_native_mask, use_segmentation=use_segmentation, sp_map_patch=sp_map_patch,
                                                                                            sp_map_rgb=sp_map_rgb, target_height=target_height, target_width=target_width,
                                                                                            patch_height=H_p, patch_width=W_p,
                                                                                            ))
                    
                    if use_segmentation and args.mask_smoothing == "gaussian":
                        # Segmented maps need light boundary softening.
                        seed_k = max(3, int((target_width * target_height / num_superpixels) ** 0.5) // 2 | 1)

                        seed_visibility_smooth = TF.gaussian_blur(seed_visibility_mask, kernel_size=[seed_k, seed_k], sigma=[seed_k / 3, seed_k / 3])
                    else:
                        # Dense patch masks were already bilinearly interpolated.
                        # A second large Gaussian blur would unnecessarily spread them.
                        seed_visibility_smooth = seed_visibility_mask

                    # CLEANUP: visualization convention:
                    # 1 = important/anomalous candidate region.
                    seed_anomaly_score = (1.0 - seed_visibility_smooth).clamp(0.0, 1.0)

                    # Each per-seed result comes from the JOINT keyword run,
                    # before phrase refinement.
                    seed_image_id = (f"{processed_image_id}_{mode_tag}_joint_seed{seed_val}")

                    seed_del, seed_ins, _, _, _ = metric(
                        args, image_tensor, baseline_tensor, seed_masks_patch, model, model_family, label, 0, pred_data,
                        size=args.size, prompt=sample.question, image_size=[image.size], positions=all_positions,
                        adapter=adapter, resolution=None
                    )

                    saved_outputs = save_heatmaps(seed_anomaly_score, visual_tensor, args.size,
                                  dataset_name=dataset_name, image_id=seed_image_id, step_number=step_num,
                                  heatmap_dir=target_heatmap_dir, overlay_dir=target_overlay_dir, raw_mask_dir=target_raw_mask_dir,
                                  write_files=args.save_intermediate_outputs)
                    saved_output = saved_outputs[0]
                    seed_minutes, seed_seconds = divmod(seed_runtime_sec, 60)
                    log(f"      -> [seed={seed_val}] Del={seed_del.sum().item():.3f}, Ins={seed_ins.sum().item():.3f}, "
                        f"runtime={int(seed_minutes)}m {seed_seconds:.1f}s")
                    
                    with open(jsonl_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "dataset": dataset_name, "image_id": processed_image_id, "step": step_num,
                            "mask_type": "joint_seed", "mask_mode": ("segmented" if use_segmentation else "dense_patch"),
                            "segmentation_method": args.segmentation_method,
                            "n_seg_requested": (n_seg if use_segmentation else None),
                            "n_seg_actual": (num_superpixels if use_segmentation else None),
                            "seed": seed_val, "iterations": iterations_i, "ig_iter": ig_iter_i,
                            "predicted_token": sample.predicted_token, #Include NO STEP
                            "keywords": keyword_scores,
                            "deletion_score": seed_del.sum().item(), "insertion_score": seed_ins.sum().item(),
                            "raw_mask_path": saved_output["raw_mask_path"],
                            "raw_preview_path": saved_output["raw_preview_path"],
                            "heatmap_path": saved_output["heatmap_path"],
                            "overlay_path": saved_output["overlay_path"],
                            "raw_mask_shape": saved_output["raw_shape"],
                            "raw_mask_min": saved_output["raw_min"],
                            "raw_mask_max": saved_output["raw_max"],
                            "raw_mask_mean": saved_output["raw_mean"],
                            "raw_mask_std": saved_output["raw_std"],
                            "display_percentile_low": (saved_output["display_percentile_low"]),
                            "display_percentile_high": (saved_output["display_percentile_high"]),
                            "runtime": f"{int(seed_minutes)}m {seed_seconds:.1f}s",
                            "runtime_sec": seed_runtime_sec,
                        }) + "\n")


                # Each seed is processed below according to its native representation:
                # compact superpixel mask in segmented mode, or dense patch-grid mask
                # when segmentation_method == "none".
                
                # clean EACH seed's compact mask individually, before averaging,
                # instead of averaging raw seeds and cleaning only the result. A single
                # overconfident/wrong seed (e.g. your seed=1 case) previously got to vote
                # in the average at full, undenoised strength -- so its false region could
                # survive averaging, or the averaging could blur two genuinely-different
                # real regions (seed0 lobe A, seed1 lobe B) into a mask where neither lobe
                # is strong enough to survive suppression afterward. Cleaning per-seed first
                # means each seed's own within-seed noise is stripped on its own terms, and
                # the subsequent average reflects real cross-seed agreement.
                if avg_native_mask is None:
                    log(f"  [SKIP] No valid ensemble mask for step={step_num}, mode={mode_tag}")
                    continue

                # Expand either:
                # evaluate the actual keyword + phrase map,
                # not the earlier keyword-only visibility mask.
                del_scores, ins_scores, del_curve, ins_curve, index = metric(
                    args, image_tensor, baseline_tensor, combined_keyword_phrase_patch_visibility.detach(), model, model_family, label, 0,
                    pred_data, size=args.size, prompt=sample.question, image_size=[image.size], positions=metric_positions, adapter=adapter, resolution=None,
                )

                # these scores now belong to the actual combined map.
                current_del_score = float(del_scores.sum().item())
                current_ins_score = float(ins_scores.sum().item())

                per_seg_del_scores.append(current_del_score)
                per_seg_ins_scores.append(current_ins_score)

                # only mask_content_mode="both_combined" saves the
                # keyword+phrase fused mask -- "keyword_only" would just
                # duplicate keyword_joint, and "phrase_only" explicitly
                # excludes any keyword contribution from the saved outputs.
                # current_del_score/current_ins_score/per_seg_*_scores above
                # are still tracked regardless of mode (used for downstream
                # best-config selection).
                if args.mask_content_mode == "both_combined":
                    # the filename now accurately represents its content.
                    config_image_id = (
                        f"{processed_image_id}_{mode_tag}"
                        "_keyword_phrase_combined"
                    )

                    # save the actual anomaly-space fusion.
                    saved_outputs = save_heatmaps(
                        combined_keyword_phrase_score,
                        visual_tensor,
                        args.size,
                        dataset_name=dataset_name,
                        image_id=config_image_id,
                        step_number=step_num,
                        heatmap_dir=target_heatmap_dir,
                        overlay_dir=target_overlay_dir,
                        raw_mask_dir=target_raw_mask_dir,
                        write_files=args.save_intermediate_outputs,
                    )

                    saved_output = saved_outputs[0]
                    log(f"   -> [keyword+phrase combined, mode={mode_tag}] Del={current_del_score:.3f}, Ins={current_ins_score:.3f}")

                    t1 = time.time()
                    minutes, seconds = divmod(t1-t0, 60)

                    with open(jsonl_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "dataset": dataset_name, "image_id": processed_image_id, "step": step_num,
                            "mask_type": "keyword_phrase_combined",
                            # accurately identify the two possible inputs.
                            "mask_components": (["keyword_joint", "phrase_joint"] if phrase_joint_accepted else ["keyword_joint"]),
                            # fusion now happens after conversion into anomaly-score space.
                            "mask_combination": ("weighted_pixelwise_max_after_per_map_display_normalization" if phrase_joint_accepted else "keyword_joint_fallback"),
                            "mask_mode": ("segmented" if use_segmentation else "dense_patch"),
                            "segmentation_method": args.segmentation_method, 
                            "n_seg_requested": (n_seg if use_segmentation else None),
                            "n_seg_actual": (num_superpixels if use_segmentation else None),
                            "n_mask_variables": num_mask_variables,
                            "iterations": iterations_i, "ig_iter": ig_iter_i,
                            "predicted_token": sample.predicted_token, #Include NO STEP
                            "keywords": keyword_scores,
                            # save the actual phrase-joint targets.
                            "phrases": [{"text": phrase["text"], "positions": phrase["positions"], "importance": phrase["importance"]} for phrase in phrase_candidates],
                            # record whether the phrase map entered fusion.
                            "phrase_joint_accepted": phrase_joint_accepted,
                            # record the configured phrase contribution.
                            "phrase_union_weight": args.phrase_union_weight,
                            "phrase_deletion_score": phrase_joint_del_value,
                            "phrase_insertion_score": phrase_joint_ins_value,
                            "target_positions": all_positions,
                            "deletion_score": del_scores.sum().item(), "insertion_score": ins_scores.sum().item(),
                            "raw_mask_path": saved_output["raw_mask_path"],
                            "raw_preview_path": saved_output["raw_preview_path"],
                            "heatmap_path": saved_output["heatmap_path"],
                            "overlay_path": saved_output["overlay_path"],
                            "raw_mask_shape": saved_output["raw_shape"],
                            "raw_mask_min": saved_output["raw_min"],
                            "raw_mask_max": saved_output["raw_max"],
                            "raw_mask_mean": saved_output["raw_mean"],
                            "raw_mask_std": saved_output["raw_std"],
                            "display_percentile_low": (saved_output["display_percentile_low"]),
                            "display_percentile_high": (saved_output["display_percentile_high"]),
                            "runtime":f"{int(minutes)}m {seconds:.1f}s",
                            "runtime_sec": round(t1-t0,2),
                        }) + "\n")
                else:
                    t1 = time.time()

                # retain the actual metric-compatible combined
                # visibility mask for this configuration.
                per_seg_patch_masks.append(combined_keyword_phrase_patch_visibility.detach())

                # retain the actual RGB keyword + phrase anomaly map.
                per_seg_rgb_heatmaps.append(combined_keyword_phrase_score.detach())

            # ============================================================
            # COMBINE all segmentation levels into one joint mask
            # ============================================================
            if not per_seg_patch_masks:
                continue
            if len(active_mask_configs) == 1:
                # One segmentation level: its anomaly score is already the final map
                # for this YES reasoning step.
                final_step_anomaly_score = per_seg_rgb_heatmaps[0]
                final_step_del_score = per_seg_del_scores[0]
                final_step_ins_score = per_seg_ins_scores[0]

                # NEW: a non-positive deletion score means progressively removing the highlighted region did not reduce the target evidence in the expected direction. Such a map is not reliable enough to enter a pixelwise-max
                # image-level union, where all of its false positives would be preserved.
                if (
                    args.step_map_filter == "none"
                    or final_step_del_score > 0.0
                ):
                    per_step_anomaly_scores.append(final_step_anomaly_score.detach())
                    per_step_metric_records.append({
                        "step": step_num,
                        "deletion_auc": final_step_del_score,
                        "insertion_auc": final_step_ins_score,
                    })
                    #per_step_anomaly_scores.append(combined_keyword_phrase_score.detach())
                    n_successful_yes_heatmaps += 1
                    log(f"  [step-map accepted] Del={final_step_del_score:.3f}, Ins={final_step_ins_score:.3f}")
                else:
                    log(f"  [step-map rejected from final union] non-positive deletion score: Del={final_step_del_score:.3f}, Ins={final_step_ins_score:.3f}")
            else:
                # Dense mode intentionally has exactly one configuration.
                assert args.segmentation_method != "none"
                t0_combined = time.time()
                combined_patch_mask = torch.stack(per_seg_patch_masks, dim=0).mean(dim=0)
                combined_rgb_heatmap = torch.stack(per_seg_rgb_heatmaps, dim=0).mean(dim=0)

                if args.mask_smoothing == "gaussian":
                    k_combined = max(3, (int((target_width * target_height / max(active_mask_configs)) ** 0.5) // 2) | 1)
                    combined_heatmap_smooth = TF.gaussian_blur(combined_rgb_heatmap, kernel_size=[k_combined, k_combined],
                                                            sigma=[k_combined / 3, k_combined / 3])
                else:
                    combined_heatmap_smooth = combined_rgb_heatmap

                combined_del, combined_ins, _, _, _ = metric(
                    args, image_tensor, baseline_tensor, combined_patch_mask.detach(), model, model_family, label, 0, pred_data,
                    size=args.size, prompt=sample.question, image_size=[image.size], positions=metric_positions,
                    adapter=adapter, resolution=None
                )

                saved_outputs = save_heatmaps(combined_heatmap_smooth, visual_tensor, args.size,
                              dataset_name=dataset_name, image_id=f"{processed_image_id}_combined", step_number=step_num,
                              heatmap_dir=target_heatmap_dir, overlay_dir=target_overlay_dir, raw_mask_dir=target_raw_mask_dir,
                              write_files=args.save_intermediate_outputs)
                saved_output = saved_outputs[0]
                log(f" -> [combined over {args.num_superpixels}] Del={combined_del.sum().item():.3f}, Ins={combined_ins.sum().item():.3f}")
                t1_combined = time.time()
                minutes_combined, seconds_combined = divmod(t1_combined-t0_combined, 60)
                with open(jsonl_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "dataset": dataset_name, "image_id": processed_image_id, "step": step_num,
                        "n_seg_requested": "combined", "n_seg_list": active_mask_configs,
                        "keywords": keyword_scores,
                        "deletion_score": combined_del.sum().item(), "insertion_score": combined_ins.sum().item(),
                        "raw_mask_path": saved_output["raw_mask_path"],
                        "raw_preview_path": saved_output["raw_preview_path"],
                        "heatmap_path": saved_output["heatmap_path"],
                        "overlay_path": saved_output["overlay_path"],
                        "raw_mask_shape": saved_output["raw_shape"],
                        "raw_mask_min": saved_output["raw_min"],
                        "raw_mask_max": saved_output["raw_max"],
                        "raw_mask_mean": saved_output["raw_mean"],
                        "raw_mask_std": saved_output["raw_std"],
                        "display_percentile_low": (saved_output["display_percentile_low"]),
                        "display_percentile_high": (saved_output["display_percentile_high"]),
                        "runtime":f"{int(minutes_combined)}m {seconds_combined:.1f}s",
                        "runtime_sec": round(t1_combined-t0_combined,2),
                    }) + "\n")
            
                # final_step_heatmap = combined_heatmap_smooth
                # per_step_final_heatmaps.append(final_step_heatmap)
                final_step_anomaly_score = combined_heatmap_smooth
                # convert the combined metric tensors to Python floats.
                final_step_del_score = float(combined_del.sum().item())
                final_step_ins_score = float(combined_ins.sum().item())
                # include this reasoning-step map in the final image-level union only when its deletion score is positive.
                # The combined per-step heatmap is still saved above regardless;
                # this condition only controls the later cross-step combination.
                if (
                    args.step_map_filter == "none"
                    or final_step_del_score > 0.0
                ):
                    per_step_anomaly_scores.append(final_step_anomaly_score.detach())
                    per_step_metric_records.append({
                        "step": step_num,
                        "deletion_auc": final_step_del_score,
                        "insertion_auc": final_step_ins_score,
                    })
                    # This counts maps actually included in the final union.
                    n_successful_yes_heatmaps += 1

                    log(
                        f"  [combined step-map accepted into final union] "
                        f"Del={final_step_del_score:.3f}, "
                        f"Ins={final_step_ins_score:.3f}"
                    )
                else:
                    log(
                        f"  [combined step-map excluded from final union] "
                        f"Del={final_step_del_score:.3f}, "
                        f"Ins={final_step_ins_score:.3f}"
                    )
                
        # ============================================================
        # Combine all YES-step anomaly maps for this image
        # ============================================================
        if per_step_anomaly_scores:
            if args.final_map_selection == "balanced_auc":
                def balanced_auc_quality(record):
                    # Standard perturbation semantics: insertion is better
                    # when high, deletion is better when low. Harmonic mean
                    # prevents one excellent curve hiding one poor curve.
                    insertion_quality = min(1.0, max(0.0, record["insertion_auc"]))
                    deletion_quality = 1.0 - min(1.0, max(0.0, record["deletion_auc"]))
                    denominator = insertion_quality + deletion_quality
                    return (
                        2.0 * insertion_quality * deletion_quality / denominator
                        if denominator > 1e-12 else 0.0
                    )

                chosen_index = max(
                    range(len(per_step_anomaly_scores)),
                    key=lambda index: balanced_auc_quality(per_step_metric_records[index]),
                )
                final_anomaly_score = per_step_anomaly_scores[chosen_index]
                chosen_record = per_step_metric_records[chosen_index]
                chosen_quality = balanced_auc_quality(chosen_record)
                final_combination_method = "best_step_balanced_insertion_deletion_auc"
            elif args.final_map_selection == "pixelwise_max":
                # Legacy behavior: preserve the union of all accepted regions.
                final_anomaly_score = torch.stack(per_step_anomaly_scores,dim=0,).amax(dim=0)
                chosen_record = None
                chosen_quality = None
                final_combination_method = "pixelwise_max"
            else:
                raise ValueError(
                    f"Unknown final_map_selection={args.final_map_selection!r}; "
                    "expected 'balanced_auc' or 'pixelwise_max'."
                )

            if args.implementation_profile == "paper_original":
                # Native-mask min/max normalization and bilinear expansion
                # already happened above. Retain that same paper-oriented map
                # (hot=important) for rendering and Pixel AUROC.
                final_anomaly_score = final_anomaly_score.clamp(0.0, 1.0)
                final_display_score = final_anomaly_score
                render_profile = "paper_original"
            else:
                final_display_score = TF.gaussian_blur(
                    final_anomaly_score, kernel_size=[5, 5], sigma=[1.0, 1.0]
                )
                render_profile = "ablation"

            saved_outputs = save_heatmaps(
                final_anomaly_score,
                visual_tensor,
                args.size,
                dataset_name=dataset_name,
                image_id=f"{processed_image_id}_final_step_combined",
                step_number="all",
                heatmap_dir=target_heatmap_dir,
                overlay_dir=target_overlay_dir,
                raw_mask_dir=target_raw_mask_dir,
                # Keep the raw map for publishing/metrics, but render only a
                # lightly smoothed version in the human-facing artifacts.
                display_masks=final_display_score,
                render_profile=render_profile,
            )
            saved_output = saved_outputs[0]
            log(
                f" -> [final anomaly map: {final_combination_method}; "
                f"{len(per_step_anomaly_scores)} candidate YES step(s)] "
                f"saved for {processed_image_id}"
            )

            with open(jsonl_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                        "dataset": dataset_name,
                        "image_id": processed_image_id,
                        "step": "final_step_combined",
                        "implementation_profile": args.implementation_profile,
                        "native_mask_resolution": (
                            [args.size, args.size]
                            if args.implementation_profile == "paper_original"
                            else None
                        ),
                        "token_selection": (
                            "paper_eq6_token_llr"
                            if args.implementation_profile == "paper_original"
                            else "ablation_span_selection"
                        ),
                        "render_profile": render_profile,
                        "combination_method": final_combination_method,
                        "selected_step": (chosen_record["step"] if chosen_record else None),
                        "selected_deletion_auc": (chosen_record["deletion_auc"] if chosen_record else None),
                        "selected_insertion_auc": (chosen_record["insertion_auc"] if chosen_record else None),
                        "selected_balanced_quality": chosen_quality,
                        "candidate_step_metrics": per_step_metric_records,
                        "n_candidate_step_maps": len(per_step_anomaly_scores),
                        "n_yes_steps_combined": (
                            len(per_step_anomaly_scores)
                            if final_combination_method == "pixelwise_max"
                            else 1
                        ),
                        # Number of upstream reasoning steps predicting YES.
                        "n_predicted_yes_steps": n_predicted_yes_steps,
                        # Number of YES steps that actually produced heatmaps.
                        "n_successful_yes_heatmaps": (n_successful_yes_heatmaps),
                        "raw_mask_path": saved_output["raw_mask_path"],
                        "raw_preview_path": saved_output["raw_preview_path"],
                        "heatmap_path": saved_output["heatmap_path"],
                        "overlay_path": saved_output["overlay_path"],
                        "raw_mask_shape": saved_output["raw_shape"],
                        "raw_mask_min": saved_output["raw_min"],
                        "raw_mask_max": saved_output["raw_max"],
                        "raw_mask_mean": saved_output["raw_mean"],
                        "raw_mask_std": saved_output["raw_std"],
                        "display_percentile_low": (saved_output["display_percentile_low"]),
                        "display_percentile_high": (saved_output["display_percentile_high"]),
                        }) + "\n")
        else:
            # this branch means no YES heatmap was successfully generated. It does not necessarily mean the planner predicted no YES.
            zero_image, zero_width, zero_height = load_and_resize_image(image_path, args)
            visual_tensor = (F_t.to_tensor(zero_image).unsqueeze(0).cuda())
            final_anomaly_score = torch.zeros((1, 1, zero_height, zero_width),dtype=torch.float32,device=visual_tensor.device,)

            saved_outputs = save_heatmaps(
                        final_anomaly_score,
                        visual_tensor,
                        args.size,
                        dataset_name=dataset_name,
                        image_id=(f"{processed_image_id}_final_step_combined"),
                        step_number="all",
                        heatmap_dir=target_heatmap_dir,
                        overlay_dir=target_overlay_dir,
                        raw_mask_dir=target_raw_mask_dir
                        )
            saved_output = saved_outputs[0]

            # distinguish an actual all-NO result from a failed attribution pipeline for one or more predicted-YES steps.
            if n_predicted_yes_steps == 0:
                zero_reason = "no_predicted_yes_steps"
                log(f" -> [final anomaly map] all reasoning steps predicted NO for {processed_image_id}; saved zero anomaly map")

            else:
                zero_reason = "yes_steps_failed_attribution"
                log(f" -> [final anomaly map] {n_predicted_yes_steps} step(s) predicted YES, but none produced a heatmap for {processed_image_id}; saved zero anomaly map")

            with open(jsonl_file_path,"a", encoding="utf-8") as f:
                f.write(json.dumps({
                                    "dataset": dataset_name,
                                    "image_id": processed_image_id,
                                    "step": "final_step_combined",
                                    # records the real reason for the zero map.
                                    "combination_method": (f"zero_{zero_reason}"),
                                    # No heatmap was available to combine.
                                    "n_yes_steps_combined": 0,
                                    # upstream planner YES count.
                                    "n_predicted_yes_steps": (n_predicted_yes_steps),
                                    # always zero in this branch.
                                    "n_successful_yes_heatmaps": (n_successful_yes_heatmaps),
                                    #"heatmap_path": os.path.join(target_heatmap_dir, f"heatmap_{dataset_name}_{processed_image_id}_final_step_combined_step_all.png"),
                                    #"overlay_path": os.path.join(target_overlay_dir, f"overlay_{dataset_name}_{processed_image_id}_final_step_combined_step_all.png"),
                                    "raw_mask_path": saved_output["raw_mask_path"],
                                    "raw_preview_path": saved_output["raw_preview_path"],
                                    "heatmap_path": saved_output["heatmap_path"],
                                    "overlay_path": saved_output["overlay_path"],
                                    "raw_mask_shape": saved_output["raw_shape"],
                                    "raw_mask_min": saved_output["raw_min"],
                                    "raw_mask_max": saved_output["raw_max"],
                                    "raw_mask_mean": saved_output["raw_mean"],
                                    "raw_mask_std": saved_output["raw_std"],
                                    "display_percentile_low": (saved_output["display_percentile_low"]),
                                    "display_percentile_high": (saved_output["display_percentile_high"]),
                                }) + "\n")

            final_combination_method = f"zero_{zero_reason}"

        if final_mask_callback is not None:
            notify(
                phase="publishing mask",
                image=processed_image_id,
                mask_type=args.segmentation_method,
                step=f"{len(step_group)}/{len(step_group)}",
                iteration="done",
            )
            final_mask_callback(
                image_path,
                saved_output,
                {
                    "dataset": dataset_name,
                    "image_id": processed_image_id,
                    "combination_method": final_combination_method,
                    "implementation_profile": args.implementation_profile,
                    "n_predicted_yes_steps": n_predicted_yes_steps,
                    "n_successful_yes_heatmaps": n_successful_yes_heatmaps,
                    "evidence_sources": sorted(
                        {sample.evidence_source for sample in step_group}
                    ),
                },
            )

        if not args.save_intermediate_outputs:
            # The final float map is retained because the publisher and later
            # AUROC/AUPRO work need lossless scores. Keep its rendered heatmap
            # as the one human-facing diagnostic. Keep the corresponding final
            # overlay as well; only the redundant grayscale raw preview is
            # removed. Intermediate calls did not write files.
            redundant_path = Path(saved_output["raw_preview_path"])
            if redundant_path.is_file():
                redundant_path.unlink()

    num_images = len({sample.image_path for sample in samples})
    # Release the model before the next ablation arm loads, especially when
    # SAM3 preprocessing and multiple checkpoint variants share one GPU.
    del adapter, model, processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "num_samples": len(samples),
        "num_images": num_images,
        "output_dir": args.output_dir,
    }

if __name__ == "__main__":
    run_pipeline()

# %% [markdown]
# # Clear GPU Memory

# %%
torch.cuda.empty_cache()
