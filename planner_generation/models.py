"""Load and run the planner pipeline's VLM, LLM, and CLIP encoder."""

import io
import time
import base64
import logging
import torch
import requests
from transformers import AutoProcessor, pipeline
from transformers.utils import logging as hf_logging
from PIL import Image
import clip
from .config import ModelConfig, ModelsConfig
from .server_manager import ServerManager

logger = logging.getLogger(__name__)

# Silence transformers' own console output (verbosity + tqdm weight-loading /
# checkpoint-shard progress bars) so the terminal only shows our Rich
# progress bar; full detail still goes through our logger to the log file.
hf_logging.set_verbosity(logging.CRITICAL)
#hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _connect_server(base_url: str, model_name: str, tag: str) -> requests.Session:
    """Open an HTTP session to an OpenAI-compatible server (e.g. vLLM) and
    verify it's reachable and serving the expected model."""
    session = requests.Session()
    response = session.get(f"{base_url}/models", timeout=30)
    response.raise_for_status()
    available = [entry.get("id") for entry in response.json().get("data", [])]
    if model_name not in available:
        logger.warning(
            "[%s] Server at %s does not list '%s' among its served models (found: %s). "
            "Requests may fail if this name does not match --served-model-name.",
            tag, base_url, model_name, available,
        )
    logger.info("[%s] Connected to server: %s (model=%s)", tag, base_url, model_name)
    return session


class Encoder:
    """Description: This class initiates the CLIP encoder."""

    @classmethod
    def loadEncoder(cls):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model, cls.preprocess = clip.load("ViT-B/32", device=DEVICE)
        #cls.model.eval()
        return cls.model, cls.preprocess
#####################################################################################################
class vlm:
    """Description: This class initiates the VLM and load and run function for the VLM."""

    vlm_pipe_line = None
    mode = "local"
    session: requests.Session | None = None
    base_url: str | None = None
    server_model_name: str | None = None

    @classmethod
    def load(cls, cfg: ModelsConfig):
        """Loads the VLM. Uses the server backend (HTTP) when cfg.vlm.base_url
        is set, otherwise loads locally via transformers.pipeline()."""
        if cfg.vlm.base_url:
            cls.mode = "server"
            cls.base_url = cfg.vlm.base_url.rstrip("/")
            cls.server_model_name = cfg.vlm.api_model_name
            if cfg.vlm.server_launch is not None:
                ServerManager.ensure_running(cfg.vlm, "VLM", cfg.deployment)
            cls.session = _connect_server(cls.base_url, cls.server_model_name, "VLM")
            return cls

        cls.mode = "local"
        logger.info("[VLM] loading pipeline model=%s device=%s", cfg.vlm.modelname, cfg.vlm.device)
        if cfg.vlm.source == "local":
            model_path = cfg.vlm.local_path
        elif cfg.vlm.source == "huggingface":
            model_path = cfg.vlm.huggingface_name
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        # Code for local model instance intialisation
        cls.vlm_pipe_line = pipeline('image-text-to-text',
                                     model = model_path,
                                     processor=processor,
                                     device = cfg.vlm.device,
                                     trust_remote_code=True,
                                     )
        logger.info("[VLM] Loaded model: %s (device=%s)", model_path, cfg.vlm.device)
        return cls

    @classmethod
    def run_vlm(
        cls,
        cfg: ModelConfig,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> tuple[list[dict], float]:

        """ Description: This function takes the prompt and list of image as input and infer them to output a text

        Inputs:  
                cfg: the vlm ModelConfig (e.g. modelcfg.vlm), 
                images: list of image object, 
                prompt: defines the behaviour of llm
        
        Output: (generated chat messages, elapsed inference time in seconds)"""

        if cls.mode == "server":
            return cls._run_vlm_server(cfg, images, prompt, max_new_tokens=max_new_tokens)

        if cls.vlm_pipe_line is None:
            raise RuntimeError("VLM is not loaded. Call vlm.load(models_config) before run_vlm().")

        # Construct the message structure
        messages = [
            {
                "role": "user",
                "content": [
                    # Add all images first (or interleave them with text)
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Run the pipeline
        start = time.perf_counter()
        generation_limit = max_new_tokens or cfg.generation.max_new_tokens
        outputs = cls.vlm_pipe_line(
            messages,
            max_new_tokens=generation_limit,
            do_sample=cfg.generation.do_sample,
            temperature=cfg.generation.temperature,  # Low temperature for factual image analysis
            **(getattr(cfg, "chat_template_kwargs", {}) or {}),
        )
        elapsed = time.perf_counter() - start
        logger.info("[VLM] Inference completed in %.3fs (max_new_tokens=%d)", elapsed, generation_limit)
        return outputs[0]["generated_text"], elapsed

    @classmethod
    def _run_vlm_server(
        cls,
        cfg: ModelConfig,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> tuple[list[dict], float]:
        generation_limit = max_new_tokens or cfg.generation.max_new_tokens
        content = [{"type": "image_url", "image_url": {"url": _encode_image(img)}} for img in images]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": cls.server_model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": generation_limit,
            "temperature": cfg.generation.temperature if cfg.generation.do_sample else 0,
        }
        chat_template_kwargs = getattr(cfg, "chat_template_kwargs", {}) or {}
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        start = time.perf_counter()
        response = cls.session.post(f"{cls.base_url}/chat/completions", json=payload, timeout=1800)
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        generated_text = response.json()["choices"][0]["message"]["content"]
        logger.info("[VLM] Inference completed in %.3fs (max_new_tokens=%d)", elapsed, generation_limit)

        # Match transformers.pipeline('image-text-to-text')'s return shape:
        # the full message list with the assistant's reply appended, so
        # existing callers' out[-1]['content'] keeps working unchanged.
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": generated_text},
        ]
        return messages, elapsed

#####################################################################################################

class llm:

  """Description: This class initiates the LLM and load and run functions for the LLM."""

  llm_pipe_line = None
  mode = "local"
  session: requests.Session | None = None
  base_url: str | None = None
  server_model_name: str | None = None

  @classmethod
  def load(cls, cfg:ModelsConfig):
      if cfg.llm.base_url:
          cls.mode = "server"
          cls.base_url = cfg.llm.base_url.rstrip("/")
          cls.server_model_name = cfg.llm.api_model_name
          if cfg.llm.server_launch is not None:
              ServerManager.ensure_running(cfg.llm, "LLM", cfg.deployment)
          cls.session = _connect_server(cls.base_url, cls.server_model_name, "LLM")
          return cls

      cls.mode = "local"
      logger.info("[LLM] loading pipeline model=%s device=%s", cfg.llm.modelname, cfg.llm.device)
      if cfg.llm.source == "local":
          model_path = cfg.llm.local_path
      elif cfg.llm.source == "huggingface":
          model_path = cfg.llm.huggingface_name
      cls.llm_pipe_line = pipeline(
              'text-generation',
              model=model_path,
              device=cfg.llm.device,
          )
      logger.info("[LLM] Loaded model: %s (device=%s)", model_path, cfg.llm.device)
      return cls
  
  @classmethod
  def run_llm(cls, cfg:ModelConfig, system_message:str, prompt:str) -> tuple[list[dict], float]:
      
      """ Description: This function takes the prompt and list of text as input and infer them to output a text
        
        Inputs:
                cfg: the llm ModelConfig (e.g. modelcfg.llm),
                system_message: defines the behaviour of llm , 
                prompt: input text to llm
        
        Output: (generated chat messages, elapsed inference time in seconds)"""

      if cls.mode == "server":
          return cls._run_llm_server(cfg, system_message, prompt)

      if cls.llm_pipe_line is None:
          raise RuntimeError("LLM is not loaded. Call llm.load(models_config) before run_llm().")

      # Chat-template schemas differ across checkpoints.  LLaVA 1.5 accepts
      # structured content blocks, while LLaVA-OneVision's Qwen template
      # concatenates content as strings.  Probe the active tokenizer first so
      # the pipeline never receives an incompatible message representation.
      structured_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
      plain_messages = [
          {"role": "system", "content": system_message},
          {"role": "user", "content": prompt},
      ]

      # Some templates intentionally omit the system role.  Detect that from
      # the active tokenizer rather than hard-coding a model name, then retain
      # the instructions by folding them into the user turn.
      tokenizer = cls.llm_pipe_line.tokenizer
      try:
          rendered = tokenizer.apply_chat_template(
              structured_messages, tokenize=False, add_generation_prompt=True
          )
          messages = structured_messages
          structured_content = True
      except Exception as structured_exc:
          try:
              rendered = tokenizer.apply_chat_template(
                  plain_messages, tokenize=False, add_generation_prompt=True
              )
          except Exception as plain_exc:
              raise RuntimeError(
                  "The active tokenizer rejected both structured and plain-string "
                  "chat messages."
              ) from plain_exc
          logger.info(
              "[LLM] Active chat template requires plain-string message content "
              "(structured content rejected: %s).",
              structured_exc,
          )
          messages = plain_messages
          structured_content = False

      if system_message and system_message not in rendered:
          logger.warning(
              "[LLM] Active chat template omits system messages; folding system instructions into the user prompt."
          )
          combined_prompt = f"{system_message}\n\n{prompt}"
          messages = [
              {
                  "role": "user",
                  "content": (
                      [{"type": "text", "text": combined_prompt}]
                      if structured_content else combined_prompt
                  ),
              }
          ]
      
      # Run the pipeline
      start = time.perf_counter()
      outputs = cls.llm_pipe_line(
          messages,
          max_new_tokens=cfg.generation.max_new_tokens,
          do_sample=cfg.generation.do_sample,
          temperature=cfg.generation.temperature,  # Low temperature for factual image analysis
          # TextGenerationPipeline exposes chat-template variables through
          # tokenizer_encode_kwargs rather than as top-level generation args.
          tokenizer_encode_kwargs=(
              getattr(cfg, "chat_template_kwargs", {}) or None
          ),
      )
      elapsed = time.perf_counter() - start
      logger.info("[LLM] Inference completed in %.3fs (max_new_tokens=%d)", elapsed, cfg.generation.max_new_tokens)
      #print(outputs)
      return outputs[0]["generated_text"], elapsed

  @classmethod
  def _run_llm_server(cls, cfg: ModelConfig, system_message: str, prompt: str) -> tuple[list[dict], float]:
      payload = {
          "model": cls.server_model_name,
          "messages": [
              {"role": "system", "content": system_message},
              {"role": "user", "content": prompt},
          ],
          "max_tokens": cfg.generation.max_new_tokens,
          "temperature": cfg.generation.temperature if cfg.generation.do_sample else 0,
      }
      chat_template_kwargs = getattr(cfg, "chat_template_kwargs", {}) or {}
      if chat_template_kwargs:
          payload["chat_template_kwargs"] = dict(chat_template_kwargs)
      start = time.perf_counter()
      response = cls.session.post(f"{cls.base_url}/chat/completions", json=payload, timeout=1800)
      response.raise_for_status()
      elapsed = time.perf_counter() - start
      generated_text = response.json()["choices"][0]["message"]["content"]
      logger.info("[LLM] Inference completed in %.3fs (max_new_tokens=%d)", elapsed, cfg.generation.max_new_tokens)

      messages = [
          {"role": "system", "content": system_message},
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": generated_text},
      ]
      return messages, elapsed
