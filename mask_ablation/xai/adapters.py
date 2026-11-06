"""Differentiable model adapters used by the XAI optimizer."""

from typing import Any, Callable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..igos_core_methods_helper import grid_to_blocks

class Qwen3VLPrecomputedAdapter:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def prepare_inputs(self, image, question):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        return inputs.to(self.model.device) 
    
    def prepare_reference_context_inputs(self, candidate, reference, question):
        # two-image context.
        # Candidate is intentionally first because candidate patches
        # will occupy the first visual-patch range.
        messages = [{
                    "role": "user",
                    "content": [{"type": "image",
                                 "image": candidate},
                                {"type": "image",
                                 "image": reference},
                                {"type": "text",
                                 "text": question},
                               ],
                    }]

        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")

        inputs.pop("token_type_ids", None)
        return inputs.to(self.model.device)

    def encode_answer(self, output_text):
        encoded = self.tokenizer(output_text.strip(), add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"].to(self.model.device)

    def pack_visual_grid(self, pixel_grid):
        """Convert raster visual-grid values to this model's native order."""
        return grid_to_blocks(
            pixel_grid, self.processor.image_processor.merge_size
        )

    def _token_probs_from_prepared_inputs(self, inputs, output_ids,):
    # shared implementation for one-image and two-image inputs.
        full_input_ids = torch.cat([inputs["input_ids"], output_ids.to(self.model.device),], dim=1)

        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_input_ids

        if "attention_mask" in model_inputs:
            extra_mask = torch.ones_like(output_ids, device=self.model.device)
            model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], extra_mask], dim=1)

        if "mm_token_type_ids" in model_inputs:
            extra_type_ids = torch.zeros_like(output_ids, device=self.model.device)
            model_inputs["mm_token_type_ids"] = torch.cat([model_inputs["mm_token_type_ids"], extra_type_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        logits = outputs.logits
        n_output_tokens = output_ids.shape[-1]

        target_logits = logits[:,-n_output_tokens - 1:-1,:]

        probs = torch.softmax(target_logits, dim=-1)

        return torch.gather(probs, dim=-1, index=output_ids.to(self.model.device).unsqueeze(-1)).squeeze(-1)[0]

    def token_probs(self, image, question, output_ids):
    # Existing candidate-only probability.
        inputs = self.prepare_inputs(image=image,question=question,)

        return self._token_probs_from_prepared_inputs(inputs,output_ids)

    def token_probs_with_reference(self, candidate, reference, question, output_ids):
    # probability conditioned on the unchanged reference.
        inputs = self.prepare_reference_context_inputs(candidate=candidate, reference=reference, question=question,)

        return self._token_probs_from_prepared_inputs(inputs,output_ids)
    
    def inject_candidate_pixels(self, candidate_pixel_values):
    # candidate-only mode uses the whole pixel sequence.
        if self.candidate_patch_slice is None:
            return candidate_pixel_values

        base_pixel_values = (self.current_base_inputs["pixel_values"])

        patch_slice = self.candidate_patch_slice
        start = patch_slice.start
        stop = patch_slice.stop

        # concatenate instead of doing an in-place assignment.
        # This preserves the autograd connection to candidate_pixel_values
        # while keeping reference pixels constant.
        return torch.cat([base_pixel_values[:start].detach(), candidate_pixel_values, base_pixel_values[stop:].detach(),], dim=0)
    
    def token_probs_from_dict(self, model_inputs, output_ids):
        """Computes probabilities directly from pre-packaged tensors, preserving gradients."""
        # 1. Protect the original base prompt keys
        final_inputs = dict(model_inputs)
        
        # 2. Concatenate base question text token IDs with target answer token IDs
        full_input_ids = torch.cat([final_inputs["input_ids"], output_ids.to(self.model.device)], dim=1)
        final_inputs["input_ids"] = full_input_ids
        
        # 3. Extend the attention mask to account for the answer tokens
        if "attention_mask" in final_inputs:
            extra_mask = torch.ones_like(output_ids, device=self.model.device)
            final_inputs["attention_mask"] = torch.cat([final_inputs["attention_mask"], extra_mask], dim=1)
        
        # 4. Pad mm_token_type_ids to avoid sequence misalignment in M-RoPE
        if "mm_token_type_ids" in final_inputs:
            extra_type_ids = torch.zeros_like(output_ids, device=self.model.device)
            final_inputs["mm_token_type_ids"] = torch.cat([final_inputs["mm_token_type_ids"], extra_type_ids], dim=1)

        # 5. Forward pass tracking gradients exclusively for the input patches
        # use_cache=False prevents massive VRAM hoarding during gradient tracking
        outputs = self.model(**final_inputs, use_cache=False)
        
        logits = outputs.logits
        n = output_ids.shape[-1]
        target_logits = logits[:, -n - 1:-1, :]
        probs = torch.softmax(target_logits, dim=-1)
        
        return torch.gather(probs, dim=-1, index=output_ids.to(self.model.device).unsqueeze(-1)).squeeze(-1)[0]


# Model-family registry -------------------------------------------------------
#
# iGOS requires gradients through the visual input, token probabilities for a
# fixed output, and knowledge of the model's visual-grid packing.  Those are
# architecture-specific capabilities; a generic ``generate()`` wrapper cannot
# provide them safely.  The study is therefore model-agnostic at the adapter
# boundary: selecting another registered family is config-only, while adding a
# new family is localized to one loader/adapter implementation here.
MODEL_ADAPTER_REGISTRY: dict[
    str, Callable[[str], tuple[Any, Any, Any, str]]
] = {}


def register_model_adapter(name: str):
    def decorator(loader: Callable[[str], tuple[Any, Any, Any, str]]):
        key = name.strip().lower()
        if key in MODEL_ADAPTER_REGISTRY:
            raise ValueError(f"Duplicate XAI model adapter: {name}")
        MODEL_ADAPTER_REGISTRY[key] = loader
        return loader
    return decorator


@register_model_adapter("qwen3_vl")
def load_qwen3_vl_adapter(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor, Qwen3VLPrecomputedAdapter(model, processor), "qwen3_vl"
    
