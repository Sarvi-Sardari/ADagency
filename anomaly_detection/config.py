"""
config.py (anomaly_detection)
==============================

Behavior change (2026-08): a local segmentation checkpoint is now validated
as a FILE during configuration loading.  Older behavior accepted any existing
path, including a directory, and therefore allowed mask generation to perform
localization work before failing much later when the segmentation backend
tried to open that directory as checkpoint weights.

Artifact grouping is also centralized here: ``merge_anomalies`` determines
the saved folder tree, while user-defined multi-class names are used only to
select a planner.  A source subfolder is added only when an inference group
label differs from the image's real test-folder label.  This preserves the
flat MVTec LOCO layout and prevents filename collisions in merged MVTec AD
groups.

Multi-class ``class_examples`` filenames are resolved from ``train/good``.
This keeps class anchors in the training/reference split rather than reading
examples from the evaluation split at ``test/good``.

Behavior change (2026-09): every dataset entry may carry an optional
``object_crop`` block (see ``ObjectCropConfig`` and
``anomaly_detection/object_crop.py``). When enabled, the VLM first locates the
whole inspected object from that dataset's object labels and the anomaly
decision runs on that crop. An absent block means "disabled", so existing
configuration files (and their ablation fingerprints) are unchanged.

Configuration loading for the online anomaly-detection pipeline.

This module is self-contained (mirrors planner_generation/config.py) rather
than importing from planner_generation, so anomaly_detection/ can be run,
tested and reasoned about independently.

Two config files are loaded:
  - configs/anomaly_detection_configs.yaml  -> AnomalyDetectionConfig (new,
    this pipeline's own schema — see that file's header comment for why it
    isn't dataset_configs.yaml).
  - configs/model_configs.yaml              -> ModelsConfig (REUSED as-is;
    only the `models.vlm` section is parsed since this pipeline never loads
    an LLM).
"""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import yaml

from cluster_deployment.config import (
    DeploymentConfig,
    apply_model_profile,
    load_model_config_bundle,
)
from .secondary_classification import (
    SecondaryClassificationProfile,
    parse_profile as parse_secondary_profile,
)

logger = logging.getLogger(__name__)

DeviceType = int | str

# ---------------------------------------------------------------------
# Configuration data classes - Model (vlm only — see module docstring)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    do_sample: bool


@dataclass(frozen=True)
class ImageProcessingConfig:
    target_w: tuple[int, int]   # (min, max)
    target_h: tuple[int, int]   # (min, max)
    resample: str


@dataclass(frozen=True)
class ServerLaunchConfig:
    """A managed OpenAI-compatible vLLM server command."""

    command: list[str]
    env: dict[str, str]
    ready_timeout: float


@dataclass(frozen=True)
class ModelConfig:
    modelname: str
    source: str
    local_path: Path | None
    huggingface_name: str | None
    device: DeviceType
    generation: GenerationConfig
    image_processing: ImageProcessingConfig
    # Extra Jinja variables understood by the selected checkpoint's own chat
    # template, e.g. {"enable_thinking": False}. Empty is the universal
    # default and requires no model-family branching in Python.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional: when set, the VLM is NOT loaded in-process via transformers.
    # Instead models.vlm connects to this OpenAI-compatible base URL (e.g. a
    # vLLM server, possibly a multi-node cluster) and calls it over HTTP.
    # Required for checkpoints transformers cannot load directly (e.g. FP8
    # checkpoints such as Qwen3-VL-32B-Instruct-FP8).
    base_url: str | None = None
    server_launch: ServerLaunchConfig | None = None
    served_model_name: str | None = None

    @property
    def selected_model(self) -> str:
        """Return either the resolved local model path or the Hugging Face
        repository identifier, depending on the configured source."""
        if self.source == "local":
            if self.local_path is None:
                raise ValueError("local_path is required when source='local'.")
            return str(self.local_path)

        if self.source == "huggingface":
            if not self.huggingface_name:
                raise ValueError("huggingface_name is required when source='huggingface'.")
            return self.huggingface_name

        raise ValueError(f"Unsupported model source: {self.source}")

    @property
    def api_model_name(self) -> str:
        """Stable OpenAI API model id, independent of machine-local paths."""

        return self.served_model_name or self.modelname


@dataclass(frozen=True)
class SegmentationModelConfig:
    """
    Generic segmentation-backend config, mirroring ModelConfig's vlm/llm
    shape so all three model kinds are configured the same way in
    model_configs.yaml. Used only by the standalone mask-generation stage
    (mask_generation.py) for methods that perform actual pixel segmentation.

    `modelname` does double duty:
      1. It selects which backend loader mask_generation.py uses (looked up
         in its SEGMENTATION_BACKENDS registry, e.g. "sam3").
      2. It literally names the output folders under masks/ (masks/<modelname>
         and masks/bounding_box_<modelname>) -- so switching to a different
         segmentation model only requires editing this section, never
         mask_generation_configs.yaml's `methods:` list.
    """
    modelname: str
    source: str                    # "local" | "huggingface"
    local_path: Path | None        # a checkpoint FILE for source=local (not necessarily a directory)
    huggingface_name: str | None
    device: str
    repo_dir: Path | None          # optional local package checkout to add to sys.path before import
    confidence_threshold: float    # generic per-detection/per-mask confidence cutoff; meaning is backend-specific


@dataclass(frozen=True)
class ModelsConfig:
    vlm: ModelConfig
    # Optional: only required when a mask-generation method needs pixel
    # segmentation (see SegmentationModelConfig above). None when
    # models.segmentation_model is absent from model_configs.yaml.
    segmentation_model: SegmentationModelConfig | None = None
    deployment: DeploymentConfig | None = None

# ---------------------------------------------------------------------
# Configuration data classes - Dataset
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectCropConfig:
    """Optional per-dataset "locate the object, then inspect the crop" step.

    ``object_labels`` are read from ``labels_file`` (one label per line, the
    same cache written by planner generation under ``prompts/object_labels``).
    They are always loaded when the file exists, even while ``enabled`` is
    false, so the live app can offer the crop checkbox for every recipe.
    ``padding_ratio`` widens the located box on each side by that fraction of
    the box width/height so object edges are not cut off.
    """

    enabled: bool
    labels_file: Path
    object_labels: tuple[str, ...]
    padding_ratio: float = 0.05
    prompt_template: str = "image_crop.j2"
    save_cropped_png: bool = False


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    root_dir: Path
    merge_anomalies: bool
    classes: Literal["single", "multi"]
    class_examples: dict[str, Path] | None
    # None only for entries constructed directly in code/tests; the YAML
    # loader always supplies a (possibly disabled) ObjectCropConfig.
    object_crop: ObjectCropConfig | None = None


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    runs_dir: Path

@dataclass(frozen=True)
class AnomalyDetectionConfig:
    datasets: list[DatasetEntry]
    planners_dir: Path
    prompts_dir: Path
    output: OutputConfig
    # "model" preserves the VLM's explicit final label. "steps" retains the
    # legacy behavior where any YES forces anomaly and all-NO forces good.
    decision_label_policy: Literal["model", "steps"] = "model"
    # Optional filename relative to prompts_dir. None delegates selection of
    # the sole runtime default to decision.py.
    decision_prompt_template: str | None = None
    secondary_classification: SecondaryClassificationProfile | None = None

# ---------------------------------------------------------------------
# Project and path helpers
# ---------------------------------------------------------------------


def get_project_root() -> Path:
    """
    Return the absolute project root directory.

    Assumes this file is at project_folder/anomaly_detection/config.py, so
    the project root is two levels up.
    """
    return Path(__file__).resolve().parent.parent


def windows_safe_path_alias(path: Path | str) -> Path:
    """Return the colon-to-hyphen alias used by the filename migration.

    The filesystem root/drive anchor is preserved so a Windows drive such as
    ``C:\\`` is never rewritten. Only ordinary path components are changed.
    """
    original = Path(path).expanduser()
    parts = original.parts
    if not parts:
        return original
    if original.anchor:
        return Path(original.anchor, *(part.replace(":", "-") for part in parts[1:]))
    return Path(*(part.replace(":", "-") for part in parts))


def resolve_portable_file(raw_path: Path | str, *base_dirs: Path | str) -> Path:
    """Resolve an artifact, including its migrated Windows-safe path alias.

    Absolute paths are checked directly. Relative paths are checked against
    each supplied base in order. For every candidate, the exact recorded path
    wins; the colon-to-hyphen alias is used only when the original file is
    absent. If nothing exists, return the first exact candidate so callers
    retain their normal, informative file-not-found behavior.
    """
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path(base) / path for base in base_dirs]
    if not candidates:
        candidates = [path]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        safe_alias = windows_safe_path_alias(candidate)
        if safe_alias != candidate and safe_alias.is_file():
            return safe_alias.resolve()
    return candidates[0].resolve()


def resolve_path(value: str | None, project_root: Path) -> Path | None:
    """Resolve a configured path. Relative paths are interpreted relative to
    project_root; absolute paths and ~ are preserved/expanded."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(f"Path value must be a string, got {type(value).__name__}.")

    value = value.strip()
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def artifact_group_dir(
    base_dir: Path,
    label_folder: str,
    source_label_folder: str | None,
    *,
    prefix: str = "",
) -> Path:
    """Return the one canonical directory for a persisted image artifact.

    ``label_folder`` is the output group created by ``merge_anomalies``:
    ``good``/``anomalies`` for a merged dataset, or the real test-folder name
    for a non-merged dataset.  ``source_label_folder`` is the image's real
    test-folder name.  User-defined semantic class names (for example Orange,
    Banana, and Cherry) deliberately are not accepted here because they only
    select planners and must never alter the output layout.

    Examples:
      - non-merged LOCO: ``texts/logical_anomalies``
      - merged AD: ``texts/anomalies/broken_large``
    """
    label = str(label_folder)
    source = str(source_label_folder or label)
    directory = Path(base_dir) / f"{prefix}{label}"
    if source != label:
        directory /= source
    return directory


def resolve_input_file(value: str | Path, project_root: Path) -> Path:
    """Resolve a config file path supplied to a loader function, regardless
    of the terminal's current working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and ensure that its root value is a mapping."""
    yaml_path = Path(path).expanduser().resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    if not yaml_path.is_file():
        raise ValueError(f"Configuration path is not a file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Configuration file is empty: {yaml_path}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in: {yaml_path}")

    return data


def require_mapping(raw: dict[str, Any], key: str, section: str) -> dict[str, Any]:
    """Return a required nested YAML mapping."""
    if key not in raw:
        raise ValueError(f"Missing required configuration section: {section}.{key}")

    value = raw[key]
    if not isinstance(value, dict):
        raise TypeError(f"{section}.{key} must be a YAML mapping.")

    return value


def require_existing_directory(path: Path | None, field_name: str) -> Path:
    """Ensure that a required directory exists."""
    if path is None:
        raise ValueError(f"{field_name} is required.")

    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{field_name} is not a directory: {path}")

    return path


def require_existing_file(path: Path | None, field_name: str) -> Path:
    """Ensure that a required file exists.

    Segmentation backends receive ``local_path`` as checkpoint weights, not as
    a model directory.  Checking ``is_file()`` here makes a folder-vs-``.pt``
    mistake fail before any bounding boxes, zero masks, or cache files are
    written by the standalone mask-generation stage.
    """
    if path is None:
        raise ValueError(f"{field_name} is required.")

    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")

    if not path.is_file():
        raise ValueError(
            f"{field_name} must be a checkpoint file, not a directory: {path}"
        )

    return path


def parse_device(value: Any, section: str) -> DeviceType:
    """Validate a model device. Supported examples: 0, 1, -1, "cpu", "cuda", "cuda:0", "auto"."""
    if isinstance(value, bool):
        raise TypeError(f"{section}.device cannot be a Boolean.")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{section}.device cannot be empty.")
        return value

    raise TypeError(f"{section}.device must be an integer or string.")


def parse_generation_config(raw: Any, section: str) -> GenerationConfig:
    """Parse and validate generation settings."""
    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise TypeError(f"{section}.generation must be a YAML mapping.")

    max_new_tokens = raw.get("max_new_tokens", 2048)
    temperature = raw.get("temperature", 0.2)
    do_sample = raw.get("do_sample", True)

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError(f"{section}.generation.max_new_tokens must be an integer.")
    if max_new_tokens <= 0:
        raise ValueError(f"{section}.generation.max_new_tokens must be greater than zero.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(f"{section}.generation.temperature must be numeric.")
    if temperature < 0:
        raise ValueError(f"{section}.generation.temperature cannot be negative.")

    if not isinstance(do_sample, bool):
        raise TypeError(f"{section}.generation.do_sample must be true or false.")

    if not do_sample and temperature != 0:
        logger.warning("%s.generation.temperature may be ignored because do_sample=false.", section)

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        do_sample=do_sample,
    )


def parse_range(value: Any, field: str, section: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{section}.image_processing.{field} must be a [min, max] list of 2 integers.")

    lo, hi = value
    for v in (lo, hi):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{section}.image_processing.{field} values must be integers.")
        if v <= 0:
            raise ValueError(f"{section}.image_processing.{field} values must be greater than zero.")
    if lo > hi:
        raise ValueError(f"{section}.image_processing.{field}: min ({lo}) cannot exceed max ({hi}).")

    return (lo, hi)


def parse_image_processing_config(raw: Any, section: str) -> ImageProcessingConfig:
    """Parse and validate models.vlm.image_processing. target_w / target_h
    are [min, max] pairs of positive ints with min <= max."""
    if not isinstance(raw, dict):
        raise TypeError(f"{section}.image_processing must be a YAML mapping.")

    target_w = parse_range(raw.get("target_w"), "target_w", section)
    target_h = parse_range(raw.get("target_h"), "target_h", section)

    _ALLOWED_RESAMPLE = {"nearest", "bilinear", "bicubic", "lanczos"}
    resample = raw.get("resample")
    if not isinstance(resample, str) or resample.lower() not in _ALLOWED_RESAMPLE:
        raise ValueError(
            f"{section}.image_processing.resample must be one of {sorted(_ALLOWED_RESAMPLE)}, got: {resample!r}"
        )

    return ImageProcessingConfig(target_w=target_w, target_h=target_h, resample=resample.lower())


# ---------------------------------------------------------------------
# Model parser (vlm section only)
# ---------------------------------------------------------------------


def parse_server_launch_config(raw: Any, section: str) -> ServerLaunchConfig | None:
    """Parse an optional managed ``vllm serve`` command."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"{section}.server_launch must be a YAML mapping.")
    command = raw.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise TypeError(
            f"{section}.server_launch.command must be a non-empty list of strings."
        )
    env_raw = raw.get("env", {})
    if not isinstance(env_raw, dict) or not all(isinstance(key, str) for key in env_raw):
        raise TypeError(f"{section}.server_launch.env must be a string-keyed YAML mapping.")
    ready_timeout = raw.get("ready_timeout", 1800)
    if isinstance(ready_timeout, bool) or not isinstance(ready_timeout, (int, float)):
        raise TypeError(f"{section}.server_launch.ready_timeout must be numeric.")
    if ready_timeout <= 0:
        raise ValueError(f"{section}.server_launch.ready_timeout must be greater than zero.")
    return ServerLaunchConfig(
        command=list(command),
        env={key: str(value) for key, value in env_raw.items()},
        ready_timeout=float(ready_timeout),
    )


def parse_model_config(raw: dict[str, Any], model_name: str, project_root: Path) -> ModelConfig:
    """Parse and validate the models.vlm section."""
    section = f"models.{model_name}"

    modelname = raw.get("model_name")
    if not modelname or not isinstance(modelname, str):
        raise ValueError(f"{section}.model_name is required and must be a non-empty string.")

    source = raw.get("source")
    if source not in {"local", "huggingface"}:
        raise ValueError(f"{section}.source must be either 'local' or 'huggingface'.")

    image_processing_raw = raw.get("image_processing")
    if image_processing_raw is None:
        raise ValueError(f"{section}.image_processing is required for the vlm model.")
    image_processing = parse_image_processing_config(image_processing_raw, section)

    local_path = resolve_path(raw.get("local_path"), project_root)
    huggingface_name = raw.get("huggingface_name")
    if huggingface_name is not None:
        if not isinstance(huggingface_name, str):
            raise TypeError(f"{section}.huggingface_name must be a string.")
        huggingface_name = huggingface_name.strip() or None

    if source == "local":
        local_path = require_existing_directory(local_path, f"{section}.local_path")

    if source == "huggingface" and not huggingface_name:
        raise ValueError(f"{section}.huggingface_name is required when source='huggingface'.")

    device = parse_device(raw.get("device", 0), section)
    generation = parse_generation_config(raw.get("generation"), section)

    chat_template_kwargs = raw.get("chat_template_kwargs", {})
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    if not isinstance(chat_template_kwargs, dict) or not all(
        isinstance(key, str) for key in chat_template_kwargs
    ):
        raise TypeError(f"{section}.chat_template_kwargs must be a string-keyed YAML mapping.")

    base_url = raw.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str):
            raise TypeError(f"{section}.base_url must be a string.")
        base_url = base_url.strip() or None

    server_launch = parse_server_launch_config(raw.get("server_launch"), section)
    if server_launch is not None and base_url is None:
        raise ValueError(f"{section}.server_launch requires {section}.base_url.")

    served_model_name = raw.get("served_model_name")
    if served_model_name is not None:
        if not isinstance(served_model_name, str):
            raise TypeError(f"{section}.served_model_name must be a string.")
        served_model_name = served_model_name.strip() or None

    return ModelConfig(
        modelname=modelname,
        source=source,
        local_path=local_path,
        huggingface_name=huggingface_name,
        device=device,
        generation=generation,
        image_processing=image_processing,
        chat_template_kwargs=dict(chat_template_kwargs),
        base_url=base_url,
        server_launch=server_launch,
        served_model_name=served_model_name,
    )


def parse_segmentation_model_config(raw: dict[str, Any], project_root: Path) -> SegmentationModelConfig:
    """Parse and validate the optional models.segmentation_model section."""
    section = "models.segmentation_model"
    if not isinstance(raw, dict):
        raise TypeError(f"{section} must be a YAML mapping.")

    modelname = raw.get("model_name")
    if not isinstance(modelname, str) or not modelname.strip():
        raise ValueError(f"{section}.model_name is required and must be a non-empty string.")
    modelname = modelname.strip()

    source = raw.get("source")
    if source not in {"local", "huggingface"}:
        raise ValueError(f"{section}.source must be either 'local' or 'huggingface'.")

    local_path = resolve_path(raw.get("local_path"), project_root)
    huggingface_name = raw.get("huggingface_name")
    if huggingface_name is not None:
        if not isinstance(huggingface_name, str):
            raise TypeError(f"{section}.huggingface_name must be a string.")
        huggingface_name = huggingface_name.strip() or None

    if source == "local":
        # A checkpoint FILE, not the directory that contains it (for example,
        # use models/sam3_model/sam3.pt rather than models/sam3_model/).
        local_path = require_existing_file(local_path, f"{section}.local_path")

    if source == "huggingface" and not huggingface_name:
        logger.info(
            "%s.huggingface_name not set; the backend loader is expected to "
            "resolve its own default Hugging Face checkpoint.", section,
        )

    device_raw = parse_device(raw.get("device", "cuda"), section)
    device = str(device_raw)

    repo_dir_raw = raw.get("repo_dir")
    repo_dir = (
        require_existing_directory(resolve_path(repo_dir_raw, project_root), f"{section}.repo_dir")
        if repo_dir_raw else None
    )

    confidence_threshold = raw.get("confidence_threshold", 0.5)
    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
        raise TypeError(f"{section}.confidence_threshold must be numeric.")
    if not 0 <= float(confidence_threshold) <= 1:
        raise ValueError(f"{section}.confidence_threshold must be between 0 and 1.")

    return SegmentationModelConfig(
        modelname=modelname,
        source=source,
        local_path=local_path,
        huggingface_name=huggingface_name,
        device=device,
        repo_dir=repo_dir,
        confidence_threshold=float(confidence_threshold),
    )


def load_models_config(config_path: str | Path = "configs/model_configs.yaml") -> ModelsConfig:
    """
    Read and validate configs/model_configs.yaml, parsing `models.vlm`
    (always) and `models.segmentation_model` (only if present).

    This deliberately reuses the SAME file as planner_generation (same model,
    same schema) but ignores `models.llm` entirely — anomaly detection never
    loads an LLM, so that section is irrelevant here and isn't validated.
    `models.segmentation_model` is only consumed by the standalone mask-
    generation stage; callers that don't need it (e.g. run_anomaly_detection.py)
    simply ignore the returned field.
    """
    project_root = get_project_root()
    resolved_config_path = resolve_input_file(config_path, project_root)
    raw_config, deployment = load_model_config_bundle(resolved_config_path)
    models_raw = require_mapping(raw_config, key="models", section="root")
    vlm_raw = apply_model_profile(
        require_mapping(models_raw, key="vlm", section="models"),
        deployment.mode,
        "models.vlm",
    )
    vlm_config = parse_model_config(raw=vlm_raw, model_name="vlm", project_root=project_root)

    # deployment.mode is the single switch between running everything on this
    # machine's GPU and using the multi-node vLLM cluster. In "local" mode any
    # configured base_url is deliberately dropped, so the VLM is loaded
    # in-process via transformers exactly as it was before cluster support
    # existed -- no other config edits needed to switch back and forth.
    if deployment.mode == "local" and vlm_config.base_url:
        logger.info(
            "deployment.mode is 'local' -- ignoring models.vlm.base_url (%s) and "
            "loading the VLM in-process instead.",
            vlm_config.base_url,
        )
        vlm_config = replace(vlm_config, base_url=None, server_launch=None)
    if deployment.mode == "cluster" and not vlm_config.base_url:
        raise ValueError(
            "deployment.mode is 'cluster', but models.vlm.base_url is not configured."
        )

    segmentation_raw = models_raw.get("segmentation_model")
    segmentation_config = (
        parse_segmentation_model_config(segmentation_raw, project_root)
        if segmentation_raw is not None else None
    )

    return ModelsConfig(
        vlm=vlm_config,
        segmentation_model=segmentation_config,
        deployment=deployment,
    )


# ---------------------------------------------------------------------
# Dataset-entry parser
# ---------------------------------------------------------------------


def parse_class_examples(raw: Any, classes_mode: str, dataset_root: Path, section: str) -> dict[str, Path] | None:
    """
    Parse optional per-class example image filenames for multi-class datasets.

    For classes == "single", class_examples must be empty/absent.
    For classes == "multi", class_examples is optional, but if present each
    value is resolved against <dataset_root>/train/good and must exist.
    """
    
    if classes_mode == "single":
        if raw:
            raise ValueError(f"{section}.class_examples must be empty when {section}.classes == 'single'.")
        return None

    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise TypeError(f"{section}.class_examples must be a YAML mapping.")

    # Class anchors belong to the same normal training pool used for reference
    # selection. Do not read them from test/good (the evaluation split).
    good_dir = dataset_root / "train" / "good"
    resolved: dict[str, Path] = {}
    for label, filename in raw.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{section}.class_examples has an invalid label: {label!r}")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"{section}.class_examples.{label} must be a non-empty filename string.")

        image_path = (good_dir / filename).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"{section}.class_examples.{label} does not exist: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"{section}.class_examples.{label} is not a file: {image_path}")

        resolved[label] = image_path

    return resolved


def load_object_labels(path: Path) -> tuple[str, ...]:
    """Read one object label per line, ignoring blank and ``#`` comment lines.

    Duplicate labels are dropped (first occurrence wins) so the crop prompt
    never repeats a part name.
    """
    labels: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        label = raw_line.strip()
        if label and not label.startswith("#") and label not in labels:
            labels.append(label)
    return tuple(labels)


def parse_object_crop_config(
    raw: Any, dataset_name: str, project_root: Path, section: str,
) -> ObjectCropConfig:
    """Parse the optional ``datasets[].object_crop`` mapping.

    Supported keys (all optional)::

        object_crop:
          enabled: false            # true = crop every test image on the fly
          labels_file: "prompts/object_labels/<name>_object_labels.txt"
          padding_ratio: 0.05       # fraction of box size added on each side
          prompt_template: "image_crop.j2"   # relative to prompts_dir
          save_cropped_png: false     # save the unletterboxed crop under the run

    An enabled crop without any usable object label is rejected at load time
    instead of failing on the first image of a long run.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{section}.object_crop must be a YAML mapping.")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{section}.object_crop.enabled must be true or false.")
    save_cropped_png = raw.get("save_cropped_png", False)
    if not isinstance(save_cropped_png, bool):
        raise TypeError(f"{section}.object_crop.save_cropped_png must be true or false.")

    labels_raw = raw.get("labels_file")
    if labels_raw is None:
        labels_file = (
            project_root / "prompts" / "object_labels" / f"{dataset_name}_object_labels.txt"
        ).resolve()
    else:
        if not isinstance(labels_raw, str) or not labels_raw.strip():
            raise TypeError(f"{section}.object_crop.labels_file must be a non-empty path string.")
        labels_file = resolve_path(labels_raw, project_root)

    padding_ratio = raw.get("padding_ratio", 0.05)
    if isinstance(padding_ratio, bool) or not isinstance(padding_ratio, (int, float)):
        raise TypeError(f"{section}.object_crop.padding_ratio must be numeric.")
    if not 0.0 <= float(padding_ratio) <= 1.0:
        raise ValueError(f"{section}.object_crop.padding_ratio must be between 0 and 1.")

    prompt_template = raw.get("prompt_template", "image_crop.j2")
    prompt_path = Path(str(prompt_template))
    if (
        not isinstance(prompt_template, str)
        or not prompt_template.strip()
        or prompt_path.is_absolute()
        or ".." in prompt_path.parts
    ):
        raise ValueError(
            f"{section}.object_crop.prompt_template must be a filename relative to prompts_dir."
        )

    object_labels = load_object_labels(labels_file) if labels_file.is_file() else ()
    if enabled and not object_labels:
        raise ValueError(
            f"{section}.object_crop is enabled, but no object labels were found in "
            f"{labels_file}. Create that file (one label per line) or set "
            f"{section}.object_crop.labels_file."
        )

    return ObjectCropConfig(
        enabled=enabled,
        labels_file=labels_file,
        object_labels=object_labels,
        padding_ratio=float(padding_ratio),
        prompt_template=prompt_template.strip(),
        save_cropped_png=save_cropped_png,
    )


def parse_dataset_entry(raw: dict[str, Any], project_root: Path, section: str) -> DatasetEntry:
    """Parse and validate one entry of the top-level `datasets:` list."""
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"{section}.name is required and must be a non-empty string.")

    root_dir_raw = raw.get("root_dir")
    if not root_dir_raw:
        raise ValueError(f"{section}.root_dir is required.")
    root_dir = require_existing_directory(resolve_path(root_dir_raw, project_root), f"{section}.root_dir")

    merge_anomalies = raw.get("merge_anomalies", False)

    classes_mode = raw.get("classes", "single")
    if classes_mode not in ("single", "multi"):
        raise ValueError(f"{section}.classes must be 'single' or 'multi', got: {classes_mode!r}")

    class_examples = parse_class_examples(raw.get("class_examples"), classes_mode, root_dir, section)
    object_crop = parse_object_crop_config(raw.get("object_crop"), name, project_root, section)

    return DatasetEntry(
        name=name,
        root_dir=root_dir,
        merge_anomalies=merge_anomalies,
        classes=classes_mode,
        class_examples=class_examples,
        object_crop=object_crop,
    )


# ---------------------------------------------------------------------
# Public anomaly-detection-configuration loader
# ---------------------------------------------------------------------


def load_anomaly_detection_config(config_path: str | Path = "configs/anomaly_detection_configs.yaml") -> AnomalyDetectionConfig:
    """Read and validate the complete anomaly-detection configuration file."""
    project_root = get_project_root()
    resolved_config_path = resolve_input_file(config_path, project_root)
    raw_config = read_yaml(resolved_config_path)

    datasets_raw = raw_config.get("datasets")
    if not datasets_raw or not isinstance(datasets_raw, list):
        raise ValueError("root.datasets is required and must be a non-empty list.")

    datasets = [parse_dataset_entry(entry, project_root, f"datasets[{i}]") for i, entry in enumerate(datasets_raw)]
    names = [d.name for d in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"datasets[].name must be unique; got duplicates in: {names}")

    planners_raw = raw_config.get("planners_dir")
    if not isinstance(planners_raw, str) or not planners_raw.strip():
        raise ValueError("planners_dir is required.")
    planners_dir = resolve_path(planners_raw, project_root)

    prompts_dir_raw = raw_config.get("prompts_dir")
    if not prompts_dir_raw:
        raise ValueError("prompts_dir is required.")
    prompts_dir = resolve_path(prompts_dir_raw, project_root)

    decision_raw = raw_config.get("decision", {})
    if not isinstance(decision_raw, dict):
        raise ValueError("decision must be a mapping when provided.")
    decision_label_policy = str(decision_raw.get("label_policy", "model")).strip().lower()
    if decision_label_policy not in {"model", "steps"}:
        raise ValueError(
            "decision.label_policy must be either 'model' (preserve the VLM final label) "
            "or 'steps' (legacy step-derived override)."
        )

    decision_prompt_raw = decision_raw.get("prompt_template")
    decision_prompt_template: str | None = None
    if decision_prompt_raw is not None:
        if not isinstance(decision_prompt_raw, str):
            raise TypeError("decision.prompt_template must be a string when provided.")
        decision_prompt_template = decision_prompt_raw.strip()
        prompt_path = Path(decision_prompt_template)
        if (
            not decision_prompt_template
            or prompt_path.is_absolute()
            or ".." in prompt_path.parts
        ):
            raise ValueError(
                "decision.prompt_template must be a non-empty filename relative to prompts_dir."
            )

    secondary_classification = parse_secondary_profile(
        raw_config.get("secondary_classification")
    )
    selected_prompt = (
        secondary_classification.prompt_template
        if secondary_classification is not None else decision_prompt_template
    )
    if selected_prompt is not None:
        selected_prompt_path = prompts_dir / selected_prompt
        if not selected_prompt_path.is_file():
            raise FileNotFoundError(
                f"Configured decision prompt does not exist: {selected_prompt_path}"
            )
    for dataset in datasets:
        crop = dataset.object_crop
        if crop is not None and crop.enabled and not (prompts_dir / crop.prompt_template).is_file():
            raise FileNotFoundError(
                f"{dataset.name}: object-crop prompt does not exist: "
                f"{prompts_dir / crop.prompt_template}"
            )

    out_raw = require_mapping(raw_config, key="output", section="root")
    out_root_raw = out_raw.get("root_dir")
    if not out_root_raw:
        raise ValueError("output.root_dir is required.")
    out_root = resolve_path(out_root_raw, project_root)

    runs_dir_raw = out_raw.get("runs_dir")
    if not runs_dir_raw:
        raise ValueError("output.runs_dir is required.")
    output_config = OutputConfig(root_dir=out_root, runs_dir=resolve_path(runs_dir_raw, out_root))


    return AnomalyDetectionConfig(
        datasets=datasets,
        planners_dir=planners_dir,
        prompts_dir=prompts_dir,
        output=output_config,
        decision_label_policy=decision_label_policy,
        decision_prompt_template=decision_prompt_template,
        secondary_classification=secondary_classification,
    )


# ---------------------------------------------------------------------
# Configuration data classes - standalone metrics (configs/metrics_configs.yaml)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SecondaryMetricsConfig:
    ground_truth_file: Path
    ground_truth_masks: dict[str, Path | None]


@dataclass(frozen=True)
class MetricsDatasetEntry:
    name: str
    include_rerun: bool
    root_dir: Path | None  # original dataset root - only required for pixel AUROC/AUPRO
    ground_truth_dir: Path | None = None  # optional override for non-standard mask locations
    secondary_metrics: SecondaryMetricsConfig | None = None


@dataclass(frozen=True)
class MetricsConfig:
    run_dir: Path
    datasets: list[MetricsDatasetEntry]
    output_subdir: str


# ---------------------------------------------------------------------
# Configuration data classes - standalone mask generation
# ---------------------------------------------------------------------

# Technique-level method names (dataset-facing, used in mask_generation_configs.yaml).
# NOTE these are deliberately backend-agnostic: "segmentation" and
# "bounding_box_segmentation" don't hardcode a model name (e.g. "sam3") --
# WHICH segmentation model actually implements them is configured once in
# model_configs.yaml's models.segmentation_model section, and its model_name
# is what names the output folder (see mask_generation.py's
# `_method_folder_name`). Renaming/swapping the backend model never requires
# touching this list or any dataset's `methods:` entry.
MASK_METHODS = ("bounding_box", "segmentation", "bounding_box_segmentation")


@dataclass(frozen=True)
class MaskGenerationDatasetEntry:
    name: str
    methods: tuple[str, ...]
    include_rerun: bool
    generate_secondary_masks: bool = False
    secondary_methods: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaskGenerationConfig:
    run_dir: Path
    datasets: list[MaskGenerationDatasetEntry]
    # configs/model_configs.yaml is ALWAYS used (same file as the rest of the
    # pipeline) -- there is deliberately no per-config override here anymore,
    # since the segmentation backend now lives inside that same file too.
    prompts_dir: Path
    overwrite: bool


def parse_metrics_dataset_entry(raw: dict[str, Any], project_root: Path, section: str) -> MetricsDatasetEntry:
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"{section}.name is required and must be a non-empty string.")
    include_rerun = raw.get("include_rerun", False)
    if not isinstance(include_rerun, bool):
        raise TypeError(f"{section}.include_rerun must be true or false.")

    root_dir_raw = raw.get("root_dir")
    root_dir = None
    if root_dir_raw:
        root_dir = require_existing_directory(resolve_path(root_dir_raw, project_root), f"{section}.root_dir")

    ground_truth_dir_raw = raw.get("ground_truth_dir")
    ground_truth_dir = None
    if ground_truth_dir_raw:
        ground_truth_dir = require_existing_directory(
            resolve_path(ground_truth_dir_raw, project_root),
            f"{section}.ground_truth_dir",
        )

    secondary_metrics = None
    secondary_raw = raw.get("secondary_metrics")
    if secondary_raw is not None:
        if not isinstance(secondary_raw, dict):
            raise TypeError(f"{section}.secondary_metrics must be a YAML mapping.")
        enabled = secondary_raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError(f"{section}.secondary_metrics.enabled must be true or false.")
        if enabled:
            if root_dir is None:
                raise ValueError(f"{section}.root_dir is required for secondary metrics.")
            ground_truth_file_raw = secondary_raw.get("ground_truth_file")
            if not isinstance(ground_truth_file_raw, str) or not ground_truth_file_raw.strip():
                raise ValueError(f"{section}.secondary_metrics.ground_truth_file is required.")
            ground_truth_file_path = Path(ground_truth_file_raw).expanduser()
            if not ground_truth_file_path.is_absolute():
                ground_truth_file_path = root_dir / ground_truth_file_path
            ground_truth_file = require_existing_file(
                ground_truth_file_path.resolve(),
                f"{section}.secondary_metrics.ground_truth_file",
            )
            masks_raw = secondary_raw.get("ground_truth_masks")
            if not isinstance(masks_raw, dict) or not masks_raw:
                raise ValueError(f"{section}.secondary_metrics.ground_truth_masks must be a non-empty mapping.")
            ground_truth_masks: dict[str, Path | None] = {}
            for class_name, path_raw in masks_raw.items():
                if not isinstance(class_name, str) or not class_name.strip():
                    raise ValueError(f"{section}.secondary_metrics.ground_truth_masks has an invalid class name.")
                if path_raw is None:
                    ground_truth_masks[class_name.strip()] = None
                    continue
                if not isinstance(path_raw, str) or not path_raw.strip():
                    raise TypeError(f"{section}.secondary_metrics.ground_truth_masks.{class_name} must be a path or null.")
                mask_path = Path(path_raw).expanduser()
                if not mask_path.is_absolute():
                    mask_path = root_dir / mask_path
                ground_truth_masks[class_name.strip()] = require_existing_directory(
                    mask_path.resolve(),
                    f"{section}.secondary_metrics.ground_truth_masks.{class_name}",
                )
            secondary_metrics = SecondaryMetricsConfig(
                ground_truth_file=ground_truth_file,
                ground_truth_masks=ground_truth_masks,
            )

    return MetricsDatasetEntry(
        name=name,
        include_rerun=include_rerun,
        root_dir=root_dir,
        ground_truth_dir=ground_truth_dir,
        secondary_metrics=secondary_metrics,
    )


def load_metrics_config(
    config_path: str | Path = "configs/metrics_configs.yaml",
    *,
    run_dir_override: str | Path | None = None,
) -> MetricsConfig:
    """
    Read and validate configs/metrics_configs.yaml.

    This is intentionally decoupled from anomaly_detection_configs.yaml:
    metrics are computed AFTER a run (and optionally after a rerun), purely
    from the files already written under <run_dir>/<dataset_name>/, so this
    config only needs to say WHICH run + WHICH datasets + WHETHER to fold in
    rerun results.
    """
    project_root = get_project_root()
    resolved_config_path = resolve_input_file(config_path, project_root)
    raw_config = read_yaml(resolved_config_path)

    run_dir_raw = run_dir_override or raw_config.get("run_dir")
    if not run_dir_raw:
        raise ValueError("root.run_dir is required.")
    run_dir = require_existing_directory(resolve_path(run_dir_raw, project_root), "run_dir")

    datasets_raw = raw_config.get("datasets")
    if not datasets_raw or not isinstance(datasets_raw, list):
        raise ValueError("root.datasets is required and must be a non-empty list.")
    datasets = [parse_metrics_dataset_entry(entry, project_root, f"datasets[{i}]") for i, entry in enumerate(datasets_raw)]

    output_raw = raw_config.get("output", {}) or {}
    if not isinstance(output_raw, dict):
        raise TypeError("root.output must be a YAML mapping.")
    output_subdir = output_raw.get("subdir", "metrics")
    if not isinstance(output_subdir, str) or not output_subdir.strip():
        raise TypeError("root.output.subdir must be a non-empty string.")

    return MetricsConfig(
        run_dir=run_dir,
        datasets=datasets,
        output_subdir=output_subdir,
    )


def _canonical_mask_method(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} entries must be non-empty strings.")
    normalized = value.strip().lower().replace("-", "_").replace("+", "_")
    # "sam3" / "bounding_box_sam3" are kept as accepted legacy spellings so
    # existing configs/scripts referencing the old, backend-specific names
    # keep working -- they now map onto the generic technique names.
    aliases = {
        "boundingbox": "bounding_box",
        "bounding_boxes": "bounding_box",
        "sam3": "segmentation",
        "seg": "segmentation",
        "boundingbox_sam3": "bounding_box_segmentation",
        "bbox_sam3": "bounding_box_segmentation",
        "bounding_box_sam3": "bounding_box_segmentation",
        "boundingbox_segmentation": "bounding_box_segmentation",
        "bbox_segmentation": "bounding_box_segmentation",
    }
    normalized = aliases.get(normalized, normalized)
    if not normalized.replace("_", "").isalnum():
        raise ValueError(f"Invalid mask method name {value!r}; use letters, numbers, and underscores.")
    return normalized


def parse_mask_generation_dataset_entry(
    raw: Any, project_root: Path, section: str,
) -> MaskGenerationDatasetEntry:
    if not isinstance(raw, dict):
        raise TypeError(f"{section} must be a YAML mapping.")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{section}.name is required and must be a non-empty string.")

    methods_raw = raw.get("methods")
    if not isinstance(methods_raw, list) or not methods_raw:
        raise ValueError(f"{section}.methods must be a non-empty list.")
    methods = tuple(dict.fromkeys(
        _canonical_mask_method(method, f"{section}.methods") for method in methods_raw
    ))

    include_rerun = raw.get("include_rerun", False)
    if not isinstance(include_rerun, bool):
        raise TypeError(f"{section}.include_rerun must be true or false.")

    generate_secondary_masks = raw.get("generate_secondary_masks", False)
    if not isinstance(generate_secondary_masks, bool):
        raise TypeError(f"{section}.generate_secondary_masks must be true or false.")
    secondary_methods_raw = raw.get("secondary_methods", [])
    if not isinstance(secondary_methods_raw, list):
        raise TypeError(f"{section}.secondary_methods must be a list.")
    secondary_methods = tuple(dict.fromkeys(
        _canonical_mask_method(method, f"{section}.secondary_methods")
        for method in secondary_methods_raw
    ))
    if generate_secondary_masks and not secondary_methods:
        raise ValueError(f"{section}.secondary_methods must be non-empty when secondary masks are enabled.")

    return MaskGenerationDatasetEntry(
        name=name.strip(), methods=methods, include_rerun=include_rerun,
        generate_secondary_masks=generate_secondary_masks,
        secondary_methods=secondary_methods,
    )


def load_mask_generation_config(
    config_path: str | Path = "configs/mask_generation_configs.yaml",
    *,
    run_dir_override: str | Path | None = None,
    dataset_names: set[str] | None = None,
    methods_override: tuple[str, ...] | None = None,
) -> MaskGenerationConfig:
    """Load the standalone post-inference mask-generation configuration.

    Overrides are applied before path validation so the checked-in example
    config can retain placeholders while still supporting one-off CLI runs.
    """
    project_root = get_project_root()
    resolved_config_path = resolve_input_file(config_path, project_root)
    raw = read_yaml(resolved_config_path)

    run_dir_raw = str(run_dir_override) if run_dir_override is not None else raw.get("run_dir")
    if not run_dir_raw:
        raise ValueError("root.run_dir is required.")
    run_dir = require_existing_directory(resolve_path(run_dir_raw, project_root), "run_dir")

    datasets_raw = raw.get("datasets")
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ValueError("root.datasets must be a non-empty list.")
    if dataset_names is not None:
        configured_names = {
            entry.get("name") for entry in datasets_raw if isinstance(entry, dict)
        }
        missing = dataset_names - configured_names
        if missing:
            raise ValueError(f"Datasets not present in config: {sorted(missing)}")
        datasets_raw = [
            entry for entry in datasets_raw
            if isinstance(entry, dict) and entry.get("name") in dataset_names
        ]

    datasets = [
        parse_mask_generation_dataset_entry(entry, project_root, f"datasets[{index}]")
        for index, entry in enumerate(datasets_raw)
    ]
    if methods_override is not None:
        canonical_methods = tuple(dict.fromkeys(
            _canonical_mask_method(method, "methods_override") for method in methods_override
        ))
        datasets = [
            MaskGenerationDatasetEntry(
                name=dataset.name,
                methods=canonical_methods,
                include_rerun=dataset.include_rerun,
                generate_secondary_masks=dataset.generate_secondary_masks,
                secondary_methods=dataset.secondary_methods,
            )
            for dataset in datasets
        ]
    names = [entry.name for entry in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"datasets[].name must be unique; got duplicates in: {names}")
    for dataset in datasets:
        dataset_run_dir = run_dir / dataset.name
        if not dataset_run_dir.is_dir():
            raise FileNotFoundError(f"Run output for dataset {dataset.name!r} not found: {dataset_run_dir}")

    # NOTE: model_config is intentionally no longer a configurable field here.
    # configs/model_configs.yaml (the same file used everywhere else in this
    # project) is always used -- see load_models_config()'s default argument.
    prompts_raw = raw.get("prompts_dir", "prompts/anomaly_detection_pipeline")
    prompts_dir = require_existing_directory(resolve_path(prompts_raw, project_root), "prompts_dir")

    overwrite = raw.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise TypeError("root.overwrite must be true or false.")

    return MaskGenerationConfig(
        run_dir=run_dir,
        datasets=datasets,
        prompts_dir=prompts_dir,
        overwrite=overwrite,
    )
