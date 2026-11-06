"""
models.py (anomaly_detection)
===============================
Model wrappers for the online anomaly-detection pipeline.

Unlike planner_generation/models.py (which drives the VLM through
`transformers.pipeline` because it only needs generated text), the decision
maker also needs per-step YES/NO log-probabilities, so the model and
processor are loaded and driven directly here instead of through pipeline().

Two backends are supported for the VLM (selected by whether
`modelcfg.vlm.base_url` is set):
  - local (base_url is None): loaded in-process via transformers, exactly as
    before. Works for any checkpoint transformers can load.
  - server (base_url is set): the VLM is served elsewhere (e.g. a vLLM
    instance, possibly a multi-node pipeline/tensor-parallel cluster) behind
    an OpenAI-compatible HTTP API, and this process only sends HTTP requests.
    Required for checkpoints transformers cannot load directly (e.g. FP8
    checkpoints such as Qwen3-VL-32B-Instruct-FP8) and/or models too large
    for a single local GPU.
"""

import re
import gc
import io
import math
import base64
import logging
import os
import time
import requests
import torch
from PIL import Image
import clip
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    __version__ as transformers_version,
)
from transformers.utils import logging as hf_logging
from transformers.utils import is_flash_attn_2_available

from .config import ModelsConfig
from cluster_deployment.server_manager import ServerManager
from pipeline_core.model_output import normalize_model_output

logger = logging.getLogger(__name__)

_RERUN_DECISION_CONTEXT_RE = re.compile(
    r'''["']?decision["']?\s*:\s*["']?\s*$''',
    flags=re.IGNORECASE,
)


def _nearest_step_number(context: str) -> str | None:
    """Return the closest STEP marker preceding a generated YES/NO token.

    Compact responses can place more than one step marker inside the confidence
    extractor's lookback window.  The decision token belongs to the most recent
    marker, not the first marker in that window.
    """
    matches = re.findall(r"\bSTEP\s*(\d+)\b", context, flags=re.IGNORECASE)
    return matches[-1] if matches else None


def _adapt_single_step_bare_decision(text: str, prompt: str) -> tuple[str, bool]:
    """Recover a legacy VLM's bare YES/NO for an unambiguous one-step plan.

    A bare answer cannot be assigned safely when multiple inspection steps
    exist. For exactly one numbered planner step, however, ``no`` and
    ``[STEP 1] NO`` are semantically identical. Canonicalizing it lets the
    normal parser and confidence validation operate without changing the
    model's judgement.
    """
    match = re.fullmatch(
        r"\s*(yes|no)\s*(?:(?:->|:)\s*(.+?))?\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    planner_steps = set(
        re.findall(r"(?im)^[ \t]*step\s+(\d+)\s*:", prompt)
    )
    if match is None or planner_steps != {"1"}:
        return text, False
    answer = match.group(1).upper()
    detail = " ".join((match.group(2) or "").split()).strip()
    if answer == "YES" and detail:
        return f"[STEP 1] YES -> {detail}", True
    return f"[STEP 1] {answer}", True


def _matched_binary_surface(token_surface: str) -> tuple[str, str] | None:
    """Return the semantic label and same-style opposite token surface."""
    match = re.fullmatch(r"(\s*)(yes|no)(\s*)", token_surface, re.IGNORECASE)
    if match is None:
        return None
    prefix, core, suffix = match.groups()
    semantic = core.upper()
    opposite = "NO" if semantic == "YES" else "YES"
    if core.islower():
        opposite = opposite.lower()
    elif core.istitle():
        opposite = opposite.title()
    elif not core.isupper():
        return None
    return semantic, f"{prefix}{opposite}{suffix}"


def _server_binary_token_pairs(tokenizer) -> dict[str, tuple[int, int]]:
    """Resolve common one-token YES/NO surface pairs for server logprobs."""
    pairs: dict[str, tuple[int, int]] = {}
    for prefix in ("", " ", "\n", "\t"):
        for yes_core, no_core in (("YES", "NO"), ("Yes", "No"), ("yes", "no")):
            yes_surface = f"{prefix}{yes_core}"
            no_surface = f"{prefix}{no_core}"
            yes_ids = tokenizer.encode(yes_surface, add_special_tokens=False)
            no_ids = tokenizer.encode(no_surface, add_special_tokens=False)
            if len(yes_ids) == 1 and len(no_ids) == 1:
                pair = (int(yes_ids[0]), int(no_ids[0]))
                pairs[yes_surface] = pair
                pairs[no_surface] = pair
    if not pairs:
        raise RuntimeError(
            "The configured VLM tokenizer has no matched one-token YES/NO "
            "surface pair; exact binary confidence cannot be calculated."
        )
    return pairs


def _validate_deterministic_binary_choice(
    *, chosen_logit: float, opposite_logit: float, token_surface: str
) -> None:
    """Catch a mismatched confidence pair before corrupting run artifacts."""
    if chosen_logit + 1e-7 < opposite_logit:
        raise RuntimeError(
            "Deterministic generation selected binary token surface "
            f"{token_surface!r}, but its matched opposite has a larger logit. "
            "Confidence extraction is inconsistent with generation; refusing "
            "to write the result."
        )

# Keep Transformers' checkpoint/shard progress bars out of the terminal; the
# pipeline's Rich progress display remains the only live progress UI.
hf_logging.set_verbosity(logging.CRITICAL)
#hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def _automatic_attention_kwargs(model_path: str) -> dict[str, str]:
    """Prefer FlashAttention-2 only when this runtime and model support it."""

    if not is_flash_attn_2_available():
        return {}
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_class = AutoModelForImageTextToText._model_mapping[type(config)]
    except (KeyError, TypeError, ValueError):
        return {}
    if not getattr(model_class, "_supports_flash_attn", False):
        return {}
    return {"attn_implementation": "flash_attention_2"}


class Encoder:
    """CLIP encoder used for reference-image nearest-neighbour selection and
    text-image class matching (reference.label_selector)."""

    model = None
    preprocess = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def loadEncoder(cls):
        cls.model, cls.preprocess = clip.load("ViT-B/32", device=cls.device)
        cls.model.eval()
        return cls.model, cls.preprocess


class vlm:
    """VLM wrapper that also extracts per-step YES/NO log-probabilities from
    the decision-maker's generated text, so each planner step's confidence
    can be reported alongside the final decision."""

    model = None
    processor = None
    session: requests.Session | None = None
    base_url: str | None = None
    server_model_name: str | None = None
    server_binary_token_pairs: dict[str, tuple[int, int]] | None = None
    mode: str = "local"

    @classmethod
    def load(cls, modelcfg: ModelsConfig) -> "type[vlm]":
        """Loads the VLM. Uses the server backend (HTTP) when
        modelcfg.vlm.base_url is set, otherwise loads locally via
        transformers from local_path or the Hugging Face hub."""
        if modelcfg.vlm.base_url:
            return cls._load_server(modelcfg)
        return cls._load_local(modelcfg)

    @classmethod
    def _load_local(cls, modelcfg: ModelsConfig) -> "type[vlm]":
        model_path = modelcfg.vlm.selected_model
        logger.info("[VLM] Loading model and processor: %s", model_path)
        logger.info(
            "[VLM] Runtime: transformers=%s torch=%s torch_cuda=%s device=%s",
            transformers_version,
            torch.__version__,
            torch.version.cuda,
            torch.cuda.get_device_name(modelcfg.vlm.device) if torch.cuda.is_available() else "cpu",
        )

        cls.mode = "local"
        cls.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            # Harmless for other tokenizers and fixes the known legacy regex
            # shipped with Mistral/Pixtral-derived checkpoints.
            fix_mistral_regex=True,
        )
        attention_kwargs = _automatic_attention_kwargs(model_path)
        logger.info(
            "[VLM] Attention implementation: %s",
            attention_kwargs.get("attn_implementation", "model default"),
        )
        cls.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=modelcfg.vlm.device,
            trust_remote_code=True,
            **attention_kwargs,
        )
        logger.info("[VLM] Loaded model: %s (device=%s)", model_path, modelcfg.vlm.device)
        return cls

    @classmethod
    def _load_server(cls, modelcfg: ModelsConfig) -> "type[vlm]":
        if modelcfg.vlm.server_launch is not None:
            ServerManager.ensure_running(modelcfg.vlm, "VLM", modelcfg.deployment)
        cls.mode = "server"
        cls.base_url = modelcfg.vlm.base_url.rstrip("/")
        cls.server_model_name = modelcfg.vlm.api_model_name
        cls.session = requests.Session()

        # Request common one-token YES/NO surface pairs. The exact pair is
        # selected after seeing whether generation emitted e.g. `` NO``,
        # ``NO``, or ``No``.
        tokenizer = AutoTokenizer.from_pretrained(
            modelcfg.vlm.selected_model,
            trust_remote_code=True,
        )
        cls.server_binary_token_pairs = _server_binary_token_pairs(tokenizer)

        logger.info("[VLM] Connecting to vLLM server at %s (model=%s)", cls.base_url, cls.server_model_name)
        response = cls.session.get(f"{cls.base_url}/models", timeout=30)
        response.raise_for_status()
        available = [entry.get("id") for entry in response.json().get("data", [])]
        if cls.server_model_name not in available:
            logger.warning(
                "[VLM] Server at %s does not list '%s' among its served models (found: %s). "
                "Requests may fail if this name does not match --served-model-name.",
                cls.base_url, cls.server_model_name, available,
            )
        logger.info("[VLM] Connected to vLLM server: %s (model=%s)", cls.base_url, cls.server_model_name)
        return cls

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def run_vlm(
        cls,
        modelcfg: ModelsConfig,
        images: list[Image.Image],
        prompt: str,
        require_step_context: bool = True,
        do_sample: bool | None = None,
        temperature: float | None = None,
        require_confidence: bool = True,
        min_new_tokens: int | None = None,
    ) -> tuple[str, list[dict], dict[str, int | float]]:
        """
        Runs the VLM on `images` + `prompt` and returns the generated text,
        per-step YES/NO log-probability data, and exact generation statistics
        derived from the generated token IDs.

        To identify which YES/NO tokens correspond to a planner step (rather
        than incidental "yes"/"no" text elsewhere in the response), a token is
        only recorded if the word "STEP" appears in the few preceding tokens
        -- unless `require_step_context=False` (used by rerun.py's
        single-question prompt, which never mentions "STEP"), in which case
        every standalone YES/NO token is recorded instead. Set
        `require_confidence=False` for free-form/structured-text tasks such as
        bounding-box localization; those calls return an empty probability
        list and do not ask the server to produce YES/NO token logprobs.
        """
        if cls.mode == "server":
            return cls._run_vlm_server(
                modelcfg, images, prompt, require_step_context, do_sample,
                temperature, require_confidence, min_new_tokens,
            )
        return cls._run_vlm_local(
            modelcfg, images, prompt, require_step_context, do_sample,
            temperature, require_confidence, min_new_tokens,
        )

    @classmethod
    def _run_vlm_server(
        cls,
        modelcfg: ModelsConfig,
        images: list[Image.Image],
        prompt: str,
        require_step_context: bool,
        do_sample: bool | None,
        temperature: float | None,
        require_confidence: bool = True,
        min_new_tokens: int | None = None,
    ) -> tuple[str, list[dict], dict[str, int | float]]:
        content = [{"type": "image_url", "image_url": {"url": cls._encode_image(img)}} for img in images]
        content.append({"type": "text", "text": prompt})

        do_sample_effective = modelcfg.vlm.generation.do_sample if do_sample is None else do_sample
        if require_confidence and cls.server_binary_token_pairs is None:
            raise RuntimeError(
                "Server YES/NO token IDs are unavailable; load the VLM server "
                "backend before calling run_vlm()."
            )
        payload = {
            "model": cls.server_model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": modelcfg.vlm.generation.max_new_tokens,
            "temperature": (
                (modelcfg.vlm.generation.temperature if temperature is None else temperature)
                if do_sample_effective else 0
            ),
            "repetition_penalty": 1.1,
        }
        if min_new_tokens is not None:
            payload["min_tokens"] = min_new_tokens
        chat_template_kwargs = getattr(modelcfg.vlm, "chat_template_kwargs", {}) or {}
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(
                chat_template_kwargs
            )
        if require_confidence:
            payload.update({
                "logprobs": True,
                # vLLM 0.28+ returns these exact logits at every generated
                # token. This is cheaper and more reliable than hoping both
                # labels appear in an arbitrary natural top-k.
                "top_logprobs": 0,
                "logprob_token_ids": list(dict.fromkeys(
                    token_id
                    for pair in cls.server_binary_token_pairs.values()
                    for token_id in pair
                )),
            })
        generation_start = time.perf_counter()
        response = cls.session.post(f"{cls.base_url}/chat/completions", json=payload, timeout=1800)
        response.raise_for_status()
        generation_time_sec = time.perf_counter() - generation_start
        data = response.json()

        choice = data["choices"][0]
        full_text = normalize_model_output(choice["message"]["content"])
        full_text, adapted_bare_decision = _adapt_single_step_bare_decision(
            full_text, prompt
        )
        token_entries = (
            (choice.get("logprobs") or {}).get("content") or []
            if require_confidence else []
        )

        probs_data: list[dict] = []
        recorded_steps: set[str] = set()

        for i, entry in enumerate(token_entries):
            token_surface = entry["token"]
            matched_surface = _matched_binary_surface(token_surface)
            if matched_surface is None:
                continue
            token_text, opposite_surface = matched_surface

            preceding_text = "".join(e["token"] for e in token_entries[max(0, i - 12):i])
            step_num = _nearest_step_number(preceding_text)
            if require_step_context:
                if step_num is None and adapted_bare_decision:
                    step_num = "1"
                elif step_num is None:
                    continue
                if step_num in recorded_steps:
                    continue
            else:
                # The rerun prompt returns JSON. Reasons/details often contain
                # incidental words such as "no", so confidence must come only
                # from the value of its top-level decision field.
                if probs_data or _RERUN_DECISION_CONTEXT_RE.search(preceding_text) is None:
                    continue
                step_num = "1"

            candidates = {
                c["token"]: c["logprob"]
                for c in (entry.get("top_logprobs") or [])
            }
            candidates[token_surface] = entry["logprob"]
            chosen_logprob = candidates.get(token_surface)
            opposite_logprob = candidates.get(opposite_surface)
            if chosen_logprob is None or opposite_logprob is None:
                raise RuntimeError(
                    "vLLM did not return the matched binary token pair "
                    f"{token_surface!r}/{opposite_surface!r} "
                    f"(step={step_num}). This server must support the vLLM "
                    "logprob_token_ids request extension; the pipeline refuses "
                    "to write incomplete confidence data."
                )
            if not do_sample_effective:
                _validate_deterministic_binary_choice(
                    chosen_logit=chosen_logprob,
                    opposite_logit=opposite_logprob,
                    token_surface=token_surface,
                )
            if require_step_context:
                recorded_steps.add(step_num)

            highest = max(chosen_logprob, opposite_logprob)
            log_denominator = (
                math.log(
                    math.exp(chosen_logprob - highest)
                    + math.exp(opposite_logprob - highest)
                )
                + highest
            )
            logprob = chosen_logprob - log_denominator
            confidence = math.exp(logprob)

            probs_data.append({
                "step": step_num,
                "token": token_text,
                "logprob": logprob,
                "confidence": f"{confidence:.2%}",
                "token_surface": token_surface,
                "opposite_token_surface": opposite_surface,
                "confidence_basis": "matched_generated_token_variant_v1",
            })

        if require_confidence and require_step_context:
            answered_steps = {
                step
                for step, _ in re.findall(
                    r"\[\s*STEP\s*(\d+)\s*\]\s*(YES|NO)\b",
                    full_text,
                    flags=re.IGNORECASE,
                )
            }
            missing_steps = answered_steps - recorded_steps
            if missing_steps:
                raise RuntimeError(
                    "vLLM generated step decisions without matching confidence "
                    f"records for steps: {sorted(missing_steps, key=int)}."
                )
        elif require_confidence and not probs_data:
            raise RuntimeError(
                "vLLM rerun response contained no usable YES/NO confidence record."
            )

        generation_stats = {"generation_time_sec": generation_time_sec}
        if os.environ.get("ANOMALY_ABLATION_PROFILE") == "1":
            usage = data.get("usage") or {}
            generation_stats["generated_token_count"] = int(
                usage.get("completion_tokens", len(token_entries))
            )

        return full_text, probs_data, generation_stats

    @classmethod
    def _run_vlm_local(
        cls,
        modelcfg: ModelsConfig,
        images: list[Image.Image],
        prompt: str,
        require_step_context: bool,
        do_sample: bool | None,
        temperature: float | None,
        require_confidence: bool = True,
        min_new_tokens: int | None = None,
    ) -> tuple[str, list[dict], dict[str, int | float]]:
        with torch.inference_mode():
            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }]

            # Let the selected checkpoint's AutoProcessor render the prompt,
            # place image tokens and build every architecture-specific tensor
            # (e.g. image_sizes/image_grid_thw). This standard HF contract is
            # what keeps this path free of model-family branches.
            chat_template_kwargs = (
                getattr(modelcfg.vlm, "chat_template_kwargs", {}) or {}
            )
            inputs = cls.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **chat_template_kwargs,
            ).to(modelcfg.vlm.device)

            generation_kwargs = {
                "max_new_tokens": modelcfg.vlm.generation.max_new_tokens,
                "do_sample": modelcfg.vlm.generation.do_sample if do_sample is None else do_sample,
                "output_scores": require_confidence,
                "repetition_penalty": 1.1,
                "return_dict_in_generate": True,
            }
            if min_new_tokens is not None:
                generation_kwargs["min_new_tokens"] = min_new_tokens
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = (
                    modelcfg.vlm.generation.temperature if temperature is None else temperature
                )

            torch.cuda.synchronize()
            generation_start = time.perf_counter()
            outputs = cls.model.generate(
                **inputs,
                **generation_kwargs,
            )
            torch.cuda.synchronize()
            generation_time_sec = time.perf_counter() - generation_start

            generated_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
            generation_stats = {"generation_time_sec": generation_time_sec}
            if os.environ.get("ANOMALY_ABLATION_PROFILE") == "1":
                generation_stats["generated_token_count"] = int(generated_ids.numel())

            del inputs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            full_text = normalize_model_output(
                cls.processor.decode(generated_ids, skip_special_tokens=True)
            )
            full_text, adapted_bare_decision = _adapt_single_step_bare_decision(
                full_text, prompt
            )

            probs_data = []
            recorded_steps: set[str] = set()
            scores_list = outputs.scores if require_confidence else ()

            for i, (token_id, score_tensor) in enumerate(zip(generated_ids, scores_list)):
                token_surface = cls.processor.decode([token_id])
                matched_surface = _matched_binary_surface(token_surface)

                if matched_surface is not None:
                    token_text, opposite_surface = matched_surface
                    context_ids = generated_ids[max(0, i - 12):i]
                    context_text = cls.processor.decode(context_ids)

                    step_num = _nearest_step_number(context_text)
                    if require_step_context:
                        if step_num is None and adapted_bare_decision:
                            step_num = "1"
                        elif step_num is None:
                            del score_tensor
                            continue
                        if step_num in recorded_steps:
                            del score_tensor
                            continue
                    else:
                        # See the server path above: only the JSON decision
                        # field represents the rerun outcome.
                        if probs_data or _RERUN_DECISION_CONTEXT_RE.search(context_text) is None:
                            del score_tensor
                            continue
                        step_num = "1"

                    opposite_ids = cls.processor.tokenizer.encode(
                        opposite_surface, add_special_tokens=False
                    )
                    if len(opposite_ids) != 1:
                        raise RuntimeError(
                            "The generated binary token surface "
                            f"{token_surface!r} has no one-token matched "
                            f"opposite {opposite_surface!r}; exact confidence "
                            "cannot be calculated."
                        )
                    chosen_id = int(token_id.item())
                    opposite_id = int(opposite_ids[0])
                    logits_pair = score_tensor[0, [chosen_id, opposite_id]]
                    if not torch.isfinite(logits_pair).all():
                        logger.warning(
                            "Skipping non-finite YES/NO logits for generated token %s",
                            token_text,
                        )
                        del score_tensor
                        continue
                    if not generation_kwargs["do_sample"]:
                        _validate_deterministic_binary_choice(
                            chosen_logit=float(logits_pair[0].item()),
                            opposite_logit=float(logits_pair[1].item()),
                            token_surface=token_surface,
                        )
                    if require_step_context:
                        recorded_steps.add(step_num)

                    log_probs_pair = logits_pair.log_softmax(dim=0)
                    logprob = log_probs_pair[0].item()
                    confidence = torch.exp(log_probs_pair[0]).item()

                    probs_data.append({
                        "step": step_num,
                        "token": token_text,
                        "logprob": logprob,
                        "confidence": f"{confidence:.2%}",
                        "token_surface": token_surface,
                        "opposite_token_surface": opposite_surface,
                        "confidence_basis": "matched_generated_token_variant_v1",
                    })

                del score_tensor  # free each score tensor immediately after use

            # NOTE: `outputs.scores` retains a reference to EVERY per-token
            # score tensor (up to max_new_tokens of them) until this del --
            # the per-iteration `del score_tensor` above only drops a local
            # name binding, not the tuple's own reference. This is the actual
            # point where that memory is released.
            del outputs, scores_list, generated_ids
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        # Final aggressive cleanup before returning to caller
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        return full_text, probs_data, generation_stats

    @classmethod
    def release(cls) -> None:
        """Release model references after normal completion or interruption."""
        if cls.mode == "server":
            if cls.session is not None:
                cls.session.close()
            cls.session = None
            cls.server_binary_token_pairs = None
            ServerManager.stop(tag="VLM:release")
            return

        cls.model = None
        cls.processor = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except RuntimeError:
            # CUDA context is already corrupted; reset silently
            pass
