"""Configuration loading for offline planner generation.

Behavior change (2026-08): multi-class ``class_examples`` are resolved from
``train/good`` instead of ``test/good``. They act as class anchors for
partitioning the normal training-reference pool and should not come from the
evaluation split.
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

logger = logging.getLogger(__name__)

DeviceType = int | str

# ---------------------------------------------------------------------
# Configuration data classes - Dataset
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceSelectionConfig:
    number_of_images: int
    random_seed: int
    explicit_images: list[Path]

@dataclass(frozen=True)
class ClassExamplesConfig:
    examples: dict[str, Path] 

@dataclass(frozen=True)
class ObjectLabelsConfig:
    label_example: str
    use_previous_labels: bool


@dataclass(frozen=True)
class AnomalyRulesConfig:
    file: Path
    key: str | None
    method: Literal["manual", "automatic"]
    overwrite: bool


@dataclass(frozen=True)
class OutputConfig:
    reference_images_dir: Path
    preprocessed_reference_dir: Path
    descriptions_dir: Path
    master_descriptions_dir: Path
    planners_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root_dir: Path
    classes: Literal["single", "multi"] | str
    class_examples: ClassExamplesConfig | None
    reference_selection: ReferenceSelectionConfig
    anomaly_rules: AnomalyRulesConfig
    object_labels: ObjectLabelsConfig
    output: OutputConfig
    prompts_dir: Path

    @property
    def class_labels_filepath(self) -> dict[str, Path]:
        """Drop-in for your existing reference_image_creator()'s
        class_labels_filepath argument — same {label: path} shape,
        empty dict when single-class so callers don't need an is_class branch
        just to avoid a None."""
        if self.class_examples is None:
            return {}
        return self.class_examples.examples

    @property
    def class_labels(self) -> list[str]:
        """Class-folder names shared by descriptions, planners, and inference."""
        if self.classes == "single":
            return ["good"]
        if self.class_examples is None:
            return []
        return list(self.class_examples.examples)

# ---------------------------------------------------------------------
# Configuration data classes - Model
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
    """Describes how to auto-start/stop this model's own `vllm serve`
    subprocess (see server_manager.ServerManager). Only used when base_url
    is set and automated swapping is desired -- otherwise leave
    server_launch unset and start `vllm serve` manually."""
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
    image_processing: ImageProcessingConfig | None = None
    # Extra Jinja variables for the checkpoint's own chat template. This is
    # deliberately generic: models that need no options use an empty mapping.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional: when set, this model is NOT loaded in-process via
    # transformers.pipeline(). Instead it connects to this OpenAI-compatible
    # base URL (e.g. a vLLM server, possibly a multi-node cluster) over HTTP.
    base_url: str | None = None
    # Optional: when set alongside base_url, vlm.load()/llm.load() will
    # automatically start/stop a local `vllm serve` subprocess for this
    # model (see server_manager.ServerManager), instead of assuming a
    # server is already running. Required to support VLM/LLM swapping when
    # they are two distinct models that can't both fit in GPU memory at once.
    server_launch: ServerLaunchConfig | None = None
    served_model_name: str | None = None

    @property
    def selected_model(self) -> str:
        """
        Return either the resolved local model path or the Hugging Face
        repository identifier, depending on the configured source.
        """
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
        return self.served_model_name or self.modelname


@dataclass(frozen=True)
class ModelsConfig:
    vlm: ModelConfig
    llm: ModelConfig
    deployment: DeploymentConfig | None = None

# ---------------------------------------------------------------------
# Project and path helpers
# ---------------------------------------------------------------------


def get_project_root() -> Path:
    """
    Return the absolute project root directory.

    This assumes config.py is located at:

        project_folder/planner_generation/config.py

    so the project root is two levels up from this file.
    """
    #print("Project root is:", Path(__file__).resolve().parent.parent)
    return Path(__file__).resolve().parent.parent #Check once if parent alone is enough


def resolve_path(value: str | None, project_root: Path) -> Path | None:
    """
    Resolve a configured path.

    Relative paths are interpreted relative to the project root.
    Absolute Windows and Linux paths are preserved.
    User-home paths such as ~/models are expanded.
    """
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


def resolve_input_file(value: str | Path, project_root: Path) -> Path:
    """
    Resolve a file supplied to the configuration loader.

    This allows calls such as:

        load_models_config("configs/models_config.yaml")

    regardless of the current terminal working directory.
    """
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


# ---------------------------------------------------------------------
# YAML reader
# ---------------------------------------------------------------------


def read_yaml(path: str | Path) -> dict[str, Any]:
    """
    Read a YAML file and ensure that its root value is a mapping.
    """
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


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def require_mapping(raw: dict[str, Any], key: str, section: str) -> dict[str, Any]:
    """
    Return a required nested YAML mapping.
    """
    if key not in raw:
        raise ValueError(f"Missing required configuration section: {section}.{key}")

    value = raw[key]

    if not isinstance(value, dict):
        raise TypeError(f"{section}.{key} must be a YAML mapping.")

    return value


def parse_device(value: Any, section: str) -> DeviceType:
    """
    Validate a model device.

    Supported examples: 0, 1, -1, "cpu", "cuda", "cuda:0", "auto"
    """
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
    """
    Parse and validate generation settings.
    """
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
        # This is allowed by Transformers, but temperature is normally ignored when sampling is disabled.
        logger.warning("%s.generation.temperature may be ignored because do_sample=false.", section)

    return GenerationConfig(max_new_tokens=max_new_tokens, temperature=float(temperature), do_sample=do_sample)

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
    """
    Parse and validate models.vlm.image_processing.
    target_w / target_h are [min, max] pairs of positive ints with min <= max.
    """
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

def require_existing_directory(path: Path | None, field_name: str) -> Path:
    """
    Ensure that a required directory exists.
    """
    if path is None:
        raise ValueError(f"{field_name} is required.")

    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{field_name} is not a directory: {path}")

    return path

# ---------------------------------------------------------------------
# Dataset parser
# ---------------------------------------------------------------------

def parse_class_examples(raw: Any, classes_mode: str, dataset_root: Path, section: str = "dataset") -> ClassExamplesConfig | None:
    """
    Parse and validate dataset.class_examples.

    Required + non-empty when classes_mode == "multi".
    Must be absent/empty when classes_mode == "single".
    Each value is resolved against dataset_root/train/good and must exist.
    """
    if classes_mode not in ("single", "multi"):
        raise ValueError(f"{section}.classes must be 'single' or 'multi', got: {classes_mode!r}")

    if classes_mode == "single":
        if raw:
            raise ValueError(f"{section}.class_examples must be empty when {section}.classes == 'single'.")
        return None

    # classes_mode == "multi" from here on
    if not raw:
        raise ValueError(f"{section}.class_examples is required and must be non-empty when {section}.classes == 'multi'.")

    if not isinstance(raw, dict):
        raise TypeError(f"{section}.class_examples must be a YAML mapping.")

    # Class anchors and selected planner references both come from the normal
    # training split. test/good remains part of the evaluation dataset only.
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

    return ClassExamplesConfig(examples=resolved)

def parse_dataset_config(raw: dict[str, Any], project_root: Path) -> DatasetConfig:
    """Parse and validate dataset section into a structured dataclass."""

    # 1. Dataset Root
    dataset_name = raw.get("name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset.name is required and must be a non-empty string.")
    root_dir_raw = raw.get("root_dir")
    if not root_dir_raw:
        raise ValueError("dataset.root_dir is required.")
    dataset_root = require_existing_directory(resolve_path(root_dir_raw, project_root), "dataset.root_dir")
    train_good_dir = require_existing_directory(dataset_root / "train" / "good", "dataset train/good directory")
    # Planner class examples no longer come from this directory, but validate
    # the standard evaluation split because later planner stages may use it.
    require_existing_directory(dataset_root / "test" / "good", "dataset test/good directory")

    classes_mode = raw.get("classes", "single")
    class_examples_config = parse_class_examples(raw.get("class_examples"), classes_mode, dataset_root)

    # 2. Reference Selection
    ref_raw = require_mapping(raw, key="reference_selection", section="dataset")
    explicit_imgs = ref_raw.get("explicit_images", [])
    if not isinstance(explicit_imgs, list):
        raise TypeError("dataset.reference_selection.explicit_images must be a YAML list.")
    random_seed = ref_raw.get("random_seed", 42)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("dataset.reference_selection.random_seed must be an integer.")
    number_of_images = ref_raw.get("number_of_images", 3)
    if isinstance(number_of_images, bool) or not isinstance(number_of_images, int) or number_of_images <= 0:
        raise ValueError("dataset.reference_selection.number_of_images must be a positive integer.")
    available_reference_images = [
        path for path in train_good_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    ]
    if not explicit_imgs and number_of_images > len(available_reference_images):
        raise ValueError(
            f"dataset.reference_selection.number_of_images={number_of_images}, but only "
            f"{len(available_reference_images)} images exist in {train_good_dir}."
        )
    ref_config = ReferenceSelectionConfig(
        number_of_images=number_of_images,
        random_seed=random_seed,
        explicit_images=[resolve_path(p, dataset_root) for p in explicit_imgs]
    )

    # 3. Anomaly Rules
    anomaly_raw = require_mapping(raw, key="anomaly_rules", section="dataset")
    anomaly_file = resolve_path(anomaly_raw.get("file"), project_root)
    if anomaly_file is None or not anomaly_file.is_file():
        raise FileNotFoundError(f"dataset.anomaly_rules.file does not exist: {anomaly_file}")
    anomaly_key = anomaly_raw.get("key")
    if anomaly_key is not None and (not isinstance(anomaly_key, str) or not anomaly_key.strip()):
        raise ValueError("dataset.anomaly_rules.key must be a non-empty string when provided.")
    anomaly_key = anomaly_key.strip() if isinstance(anomaly_key, str) else None

    anomaly_method = anomaly_raw.get("method", "automatic")
    if anomaly_method not in ("manual", "automatic"):
        raise ValueError(f"dataset.anomaly_rules.method must be 'manual' or 'automatic', got: {anomaly_method!r}")

    anomaly_overwrite = anomaly_raw.get("overwrite", False)
    if not isinstance(anomaly_overwrite, bool):
        raise TypeError("dataset.anomaly_rules.overwrite must be true or false.")

    if anomaly_method == "manual" and not anomaly_key:
        raise ValueError("dataset.anomaly_rules.key is required when dataset.anomaly_rules.method == 'manual'.")

    anomaly_config = AnomalyRulesConfig(
        file=anomaly_file,
        key=anomaly_key,
        method=anomaly_method,
        overwrite=anomaly_overwrite,
    )

    # 4. Object Labels
    labels_raw = require_mapping(raw, key="object_labels", section="dataset")
    label_example = labels_raw.get("label_example", "")
    if not isinstance(label_example, str) or not label_example.strip():
        raise ValueError("dataset.object_labels.label_example must be a non-empty filename.")
    label_example_path = train_good_dir / label_example
    if not label_example_path.is_file():
        raise FileNotFoundError(f"dataset.object_labels.label_example does not exist: {label_example_path}")
    labels_config = ObjectLabelsConfig(label_example=label_example, use_previous_labels=labels_raw.get("use_previous_labels", True))

    # 5. Prompts Directory
    prompts_dir_raw = raw.get("prompts_dir")
    if not prompts_dir_raw:
        raise ValueError("prompts_dir is required.")
    prompts_dir = require_existing_directory(resolve_path(prompts_dir_raw, project_root), "dataset.prompts_dir")
    required_templates = {
        "anomaly_rules_from_description.j2",
        "anomaly_rules_from_objects.j2",
        "anomaly_rules_from_user_input.j2",
        "normal_descriptor.j2",
        "object_namer.j2",
        "object_summarizer.j2",
        "planner.j2",
    }
    missing_templates = sorted(name for name in required_templates if not (prompts_dir / name).is_file())
    if missing_templates:
        raise FileNotFoundError(f"Missing planner prompt templates in {prompts_dir}: {missing_templates}")

    # 6. Outputs
    out_raw = require_mapping(raw, key="output", section="dataset")

    output_paths = {
        key: resolve_path(out_raw.get(key), project_root)
        for key in (
            "reference_images_dir",
            "preprocessed_reference_dir",
            "descriptions_dir",
            "master_descriptions_dir",
            "planners_dir",
        )
    }
    missing_outputs = sorted(key for key, path in output_paths.items() if path is None)
    if missing_outputs:
        raise ValueError(f"Missing required dataset.output paths: {missing_outputs}")

    output_config = OutputConfig(
        reference_images_dir=output_paths["reference_images_dir"],
        preprocessed_reference_dir=output_paths["preprocessed_reference_dir"],
        descriptions_dir=output_paths["descriptions_dir"],
        master_descriptions_dir=output_paths["master_descriptions_dir"],
        planners_dir=output_paths["planners_dir"],
    )

    return DatasetConfig( name=dataset_name.strip(),
                          root_dir=dataset_root,
                          classes=classes_mode,
                          class_examples=class_examples_config,
                          reference_selection=ref_config,
                          anomaly_rules=anomaly_config,
                          object_labels=labels_config,
                          output=output_config,
                          prompts_dir=prompts_dir,
                        )

# ---------------------------------------------------------------------
# Model parser
# ---------------------------------------------------------------------


def parse_server_launch_config(raw: Any, section: str) -> ServerLaunchConfig | None:
    """
    Parse and validate an optional server_launch block (see
    ServerLaunchConfig / server_manager.ServerManager). Returns None when
    the block is absent, meaning the server is assumed to already be
    running and managed manually.
    """
    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise TypeError(f"{section}.server_launch must be a YAML mapping.")

    command = raw.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise TypeError(f"{section}.server_launch.command must be a non-empty list of strings.")

    env_raw = raw.get("env", {})
    if not isinstance(env_raw, dict):
        raise TypeError(f"{section}.server_launch.env must be a YAML mapping.")
    env: dict[str, str] = {}
    for key, value in env_raw.items():
        if not isinstance(key, str):
            raise TypeError(f"{section}.server_launch.env keys must be strings.")
        env[key] = str(value)

    ready_timeout = raw.get("ready_timeout", 1800)
    if isinstance(ready_timeout, bool) or not isinstance(ready_timeout, (int, float)):
        raise TypeError(f"{section}.server_launch.ready_timeout must be numeric.")
    if ready_timeout <= 0:
        raise ValueError(f"{section}.server_launch.ready_timeout must be greater than zero.")

    return ServerLaunchConfig(command=command, env=env, ready_timeout=float(ready_timeout))


def parse_model_config(raw: dict[str, Any], model_name: str, project_root: Path, requires_image_processing: bool) -> ModelConfig:
    """
    Parse and validate one model section.
    """
    section = f"models.{model_name}"

    modelname = raw.get("model_name")
    if not modelname or not isinstance(modelname, str):
        raise ValueError(f"{section}.model_name is required and must be a non-empty string.")

    source = raw.get("source")

    image_processing_raw = raw.get("image_processing")

    if requires_image_processing:
        if image_processing_raw is None:
            raise ValueError(f"{section}.image_processing is required for the vlm model.")
        image_processing = parse_image_processing_config(image_processing_raw, section)
    else:
        if image_processing_raw is not None:
            raise ValueError(
                f"{section}.image_processing is not allowed here — it is vlm-only. "
                f"Remove it from {section} (did you mean to put it under models.vlm?)."
            )
        image_processing = None

    if source not in {"local", "huggingface"}:
        raise ValueError(f"{section}.source must be either 'local' or 'huggingface'.")

    local_path = resolve_path(raw.get("local_path"), project_root)

    huggingface_name = raw.get("huggingface_name")

    if huggingface_name is not None:
        if not isinstance(huggingface_name, str):
            raise TypeError(f"{section}.huggingface_name must be a string.")

        huggingface_name = huggingface_name.strip() or None

    if source == "local":
        local_path = require_existing_directory(local_path, f"{section}.local_path",)

    if source == "huggingface":
        if not huggingface_name:
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
    if base_url and server_launch is None:
        logger.warning(
            "%s.base_url is set without server_launch -- assuming a server is "
            "already running and managed manually (no automatic start/stop).",
            section,
        )

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


# ---------------------------------------------------------------------
# Public model-configuration loader
# ---------------------------------------------------------------------


def load_models_config(config_path: str | Path = "configs/models_config.yaml") -> ModelsConfig:
    """
    Read and validate the complete model configuration file.
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
    llm_raw = apply_model_profile(
        require_mapping(models_raw, key="llm", section="models"),
        deployment.mode,
        "models.llm",
    )

    vlm_config = parse_model_config(raw=vlm_raw, model_name="vlm", project_root=project_root, requires_image_processing=True)
    llm_config = parse_model_config(raw=llm_raw, model_name="llm", project_root=project_root, requires_image_processing=False)

    # deployment.mode is the single switch between running everything on this
    # machine's GPU and using the multi-node vLLM cluster. In "local" mode any
    # configured base_url/server_launch is deliberately dropped, so models are
    # loaded in-process via transformers.pipeline() and swapped by clear_vram()
    # exactly as they were before cluster support existed.
    if deployment.mode == "local":
        if vlm_config.base_url or llm_config.base_url:
            logger.info(
                "deployment.mode is 'local' -- ignoring models.vlm.base_url/"
                "models.llm.base_url and loading both models in-process instead."
            )
        vlm_config = replace(vlm_config, base_url=None, server_launch=None)
        llm_config = replace(llm_config, base_url=None, server_launch=None)

    if deployment.mode == "cluster":
        missing = [
            name
            for name, config in (("vlm", vlm_config), ("llm", llm_config))
            if not config.base_url
        ]
        if missing:
            raise ValueError(
                "deployment.mode is 'cluster', but base_url is missing for: "
                + ", ".join(f"models.{name}" for name in missing)
            )

    return ModelsConfig(vlm=vlm_config, llm=llm_config, deployment=deployment)

def load_dataset_config(config_path: str | Path = "configs/dataset_config.yaml") -> DatasetConfig:
    """
    Read and validate the complete dataset configuration file.
    """
    project_root = get_project_root()
    resolved_config_path = resolve_input_file(config_path, project_root)
    raw_config = read_yaml(resolved_config_path)

    dataset_raw = require_mapping(raw_config, key="dataset", section="root")

    return parse_dataset_config(dataset_raw, project_root)
