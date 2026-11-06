"""Configuration for the standalone XAI mask-ablation stage.

The stage consumes a completed anomaly-detection run and writes one standard
mask folder per ``(model, method)`` arm.  It deliberately has no dependency on
``anomaly_detection.config``: the existing inference, rerun, mask-generation,
and metrics scripts remain untouched.

Minimal YAML shape::

    run_dir: outputs/runs/<existing-run>
    datasets: [pushpins, screw_bag]
    include_rerun: true
    models: [...]
    methods: [...]

Relative paths are resolved from ``pipeline_project/`` rather than from the
caller's current working directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"XAI mask-ablation config not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {yaml_path}")
    return data


def _require_safe_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{field} must contain only letters, numbers, '.', '_' or '-' and "
            f"must start with a letter or number; got {value!r}."
        )
    return value


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    include_rerun: bool


@dataclass(frozen=True)
class ModelVariant:
    """One differentiable LVLM plus the adapter that understands its inputs.

    The orchestration is adapter-agnostic.  The bundled implementation is
    ``qwen3_vl``; another model family is added by registering an adapter in
    ``xai_engine.py`` rather than changing this config or the study pipeline.
    """

    name: str
    adapter: str
    source: str
    local_path: Path | None
    huggingface_name: str | None

    @property
    def selected_model(self) -> str:
        if self.source == "local":
            if self.local_path is None:
                raise ValueError(f"Model {self.name!r} requires local_path.")
            return str(self.local_path)
        if self.source == "huggingface":
            if not self.huggingface_name:
                raise ValueError(f"Model {self.name!r} requires huggingface_name.")
            return self.huggingface_name
        raise ValueError(
            f"Model {self.name!r} has unsupported source {self.source!r}; "
            "expected 'local' or 'huggingface'."
        )


@dataclass(frozen=True)
class MethodVariant:
    name: str
    params: dict[str, Any]
    output_name: str | None = None

    def folder_name(self, model: ModelVariant) -> str:
        explicit = self.output_name
        if explicit:
            return explicit
        return f"xai_{model.name}_{self.name}"


@dataclass(frozen=True)
class MaskAblationConfig:
    run_dir: Path
    datasets: list[DatasetEntry]
    models: list[ModelVariant]
    methods: list[MethodVariant]
    diagnostics_subdir: str
    overwrite: bool
    continue_on_error: bool
    pilot_images_by_dataset: dict[str, tuple[str, ...]]
    evaluate_after_generation: bool = True
    # A small, deterministic attribution-only pilot. ``None`` means all
    # images. Good/zero-mask images are intentionally excluded from a pilot;
    # a later full run resumes the pilot masks and fills the complete dataset.
    max_attribution_images_per_dataset: int | None = None


def _parse_datasets(raw: Any, default_include_rerun: bool) -> list[DatasetEntry]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("'datasets' must be a non-empty YAML list.")

    datasets: list[DatasetEntry] = []
    for index, value in enumerate(raw):
        if isinstance(value, str):
            datasets.append(
                DatasetEntry(
                    name=_require_safe_name(value, f"datasets[{index}]"),
                    include_rerun=default_include_rerun,
                )
            )
            continue
        if not isinstance(value, dict) or "name" not in value:
            raise TypeError(
                f"datasets[{index}] must be a dataset name or a mapping with 'name'."
            )
        include_rerun = value.get("include_rerun", default_include_rerun)
        if not isinstance(include_rerun, bool):
            raise TypeError(f"datasets[{index}].include_rerun must be true or false.")
        datasets.append(
            DatasetEntry(
                name=_require_safe_name(value["name"], f"datasets[{index}].name"),
                include_rerun=include_rerun,
            )
        )

    names = [entry.name for entry in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate dataset names are not allowed: {names}")
    return datasets


def _parse_models(raw: Any, project_root: Path) -> list[ModelVariant]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("'models' must be a non-empty YAML list.")
    models: list[ModelVariant] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise TypeError(f"models[{index}] must be a mapping.")
        name = _require_safe_name(value.get("name"), f"models[{index}].name")
        adapter = _require_safe_name(
            value.get("adapter", "qwen3_vl"), f"models[{index}].adapter"
        )
        source = value.get("source", "local")
        if source not in ("local", "huggingface"):
            raise ValueError(
                f"models[{index}].source must be 'local' or 'huggingface'."
            )
        local_raw = value.get("local_path")
        model = ModelVariant(
            name=name,
            adapter=adapter,
            source=source,
            local_path=_resolve_path(local_raw, project_root) if local_raw else None,
            huggingface_name=value.get("huggingface_name"),
        )
        # Accessing the property validates the selected source now rather than
        # after a progress bar has started and a GPU may already be occupied.
        selected = model.selected_model
        if source == "local" and not Path(selected).is_dir():
            raise FileNotFoundError(
                f"models[{index}].local_path is not a model directory: {selected}"
            )
        models.append(model)

    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names are not allowed: {names}")
    return models


def _parse_methods(raw: Any) -> list[MethodVariant]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("'methods' must be a non-empty YAML list.")
    methods: list[MethodVariant] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise TypeError(f"methods[{index}] must be a mapping.")
        name = _require_safe_name(value.get("name"), f"methods[{index}].name")
        params = value.get("args", value.get("params", {}))
        if not isinstance(params, dict):
            raise TypeError(f"methods[{index}].args must be a mapping.")
        output_name = value.get("output_name")
        if output_name is not None:
            output_name = _require_safe_name(
                output_name, f"methods[{index}].output_name"
            )
        methods.append(
            MethodVariant(name=name, params=dict(params), output_name=output_name)
        )

    names = [method.name for method in methods]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate method names are not allowed: {names}")
    return methods


def _parse_pilot_images(
    raw: Any, configured_dataset_names: set[str]
) -> dict[str, tuple[str, ...]]:
    """Validate basenames or source-qualified image identities for a pilot."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("'pilot_images' must be a mapping of dataset names to lists.")

    parsed: dict[str, tuple[str, ...]] = {}
    for dataset_name, image_names in raw.items():
        dataset_name = _require_safe_name(dataset_name, "pilot_images dataset")
        if dataset_name not in configured_dataset_names:
            raise ValueError(
                f"pilot_images contains unknown dataset {dataset_name!r}; "
                f"configured datasets: {sorted(configured_dataset_names)}"
            )
        if not isinstance(image_names, list) or not image_names:
            raise ValueError(
                f"pilot_images.{dataset_name} must be a non-empty list."
            )
        cleaned: list[str] = []
        for index, image_name in enumerate(image_names):
            if (
                not isinstance(image_name, str)
                or not image_name.strip()
                or image_name != image_name.strip()
                or "\\" in image_name
                or len(image_name.split("/")) not in (1, 2, 3)
                or any(part in ("", ".", "..") for part in image_name.split("/"))
            ):
                raise ValueError(
                    f"pilot_images.{dataset_name}[{index}] must be one image "
                    "basename, source/basename, or label/source/basename "
                    f"without absolute paths or traversal; got {image_name!r}."
                )
            cleaned.append(image_name)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(
                f"pilot_images.{dataset_name} contains duplicate names: {cleaned}"
            )
        parsed[dataset_name] = tuple(cleaned)
    return parsed


def load_mask_ablation_config(
    config_path: str | Path,
    *,
    run_dir_override: str | Path | None = None,
    dataset_names: set[str] | None = None,
    model_names: set[str] | None = None,
    method_names: set[str] | None = None,
    include_rerun_override: bool | None = None,
    overwrite_override: bool | None = None,
    full_run: bool = False,
) -> MaskAblationConfig:
    project_root = get_project_root()
    raw = _read_yaml(config_path)

    raw_run_dir = run_dir_override if run_dir_override is not None else raw.get("run_dir")
    if not raw_run_dir:
        raise ValueError("'run_dir' is required.")
    run_dir = _resolve_path(raw_run_dir, project_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Existing anomaly-detection run_dir not found: {run_dir}")

    default_include_rerun = raw.get("include_rerun", False)
    if not isinstance(default_include_rerun, bool):
        raise TypeError("'include_rerun' must be true or false.")
    if include_rerun_override is not None:
        default_include_rerun = include_rerun_override

    datasets = _parse_datasets(raw.get("datasets"), default_include_rerun)
    pilot_images_by_dataset = _parse_pilot_images(
        raw.get("pilot_images"), {dataset.name for dataset in datasets}
    )
    if include_rerun_override is not None:
        datasets = [
            replace(dataset, include_rerun=include_rerun_override)
            for dataset in datasets
        ]
    models = _parse_models(raw.get("models"), project_root)
    methods = _parse_methods(raw.get("methods"))

    def selected(values: list[Any], wanted: set[str] | None, kind: str) -> list[Any]:
        if not wanted:
            return values
        available = {value.name for value in values}
        missing = wanted - available
        if missing:
            raise ValueError(
                f"Unknown --{kind} value(s): {sorted(missing)}; configured: {sorted(available)}"
            )
        return [value for value in values if value.name in wanted]

    def default_names(field: str) -> set[str] | None:
        value = raw.get(field)
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError(f"'{field}' must be a non-empty YAML list when provided.")
        return {
            _require_safe_name(name, f"{field}[{index}]")
            for index, name in enumerate(value)
        }

    if not full_run:
        if dataset_names is None:
            dataset_names = default_names("default_datasets")
        if method_names is None:
            method_names = default_names("default_methods")

    datasets = selected(datasets, dataset_names, "dataset")
    models = selected(models, model_names, "model")
    methods = selected(methods, method_names, "method")

    for dataset in datasets:
        summary = run_dir / dataset.name / "all_data_summary.jsonl"
        if not summary.is_file():
            raise FileNotFoundError(f"Dataset summary not found: {summary}")

    diagnostics_subdir = _require_safe_name(
        raw.get("diagnostics_subdir", "xai_mask_ablation"), "diagnostics_subdir"
    )
    overwrite = raw.get("overwrite", False)
    continue_on_error = raw.get("continue_on_error", True)
    evaluate_after_generation = raw.get("evaluate_after_generation", True)
    max_attribution_images = raw.get("max_attribution_images_per_dataset")
    if full_run:
        max_attribution_images = None
        pilot_images_by_dataset = {}
    if overwrite_override is not None:
        overwrite = overwrite_override
    if not isinstance(overwrite, bool):
        raise TypeError("'overwrite' must be true or false.")
    if not isinstance(continue_on_error, bool):
        raise TypeError("'continue_on_error' must be true or false.")
    if not isinstance(evaluate_after_generation, bool):
        raise TypeError("'evaluate_after_generation' must be true or false.")
    if max_attribution_images is not None and (
        not isinstance(max_attribution_images, int)
        or isinstance(max_attribution_images, bool)
        or max_attribution_images < 1
    ):
        raise TypeError(
            "'max_attribution_images_per_dataset' must be null or a positive integer."
        )
    # Partial method folders must never be ranked as complete datasets. Keep
    # the YAML's evaluation preference for ``--full``, but suppress metrics
    # automatically for the default limited pilot.
    if max_attribution_images is not None or pilot_images_by_dataset:
        evaluate_after_generation = False

    # Validate local segmentation dependencies before loading the LVLM or
    # allocating GPU memory. Relative method paths follow the same project-root
    # convention as model paths.
    for method in methods:
        segmentation = method.params.get("segmentation_method", "none")
        if segmentation in {"sam3", "hybrid"}:
            sam3_value = method.params.get("sam3_model_id", "models/sam3_model")
            sam3_path = _resolve_path(sam3_value, project_root)
            if not sam3_path.is_dir():
                raise FileNotFoundError(
                    f"Method {method.name!r} requires a SAM3 model directory: {sam3_path}"
                )
            checkpoint = sam3_path / "sam3.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Method {method.name!r} requires the local SAM3 checkpoint: {checkpoint}"
                )
            repo_value = method.params.get("sam3_repo_dir", "../sam3")
            repo_path = _resolve_path(repo_value, project_root)
            if not (repo_path / "sam3" / "model_builder.py").is_file():
                raise FileNotFoundError(
                    f"Method {method.name!r} requires the local SAM3 checkout: {repo_path}"
                )
            prompts = method.params.get(
                "sam3_text_prompts", ["object", "part", "component"]
            )
            if (
                not isinstance(prompts, list)
                or not prompts
                or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
            ):
                raise TypeError(
                    f"Method {method.name!r}: sam3_text_prompts must be a non-empty list of strings."
                )

    folders = [method.folder_name(model) for model in models for method in methods]
    if len(folders) != len(set(folders)):
        raise ValueError(
            "Method output folders collide across the configured model/method grid: "
            f"{folders}. Give colliding methods distinct output_name values."
        )

    return MaskAblationConfig(
        run_dir=run_dir,
        datasets=datasets,
        models=models,
        methods=methods,
        diagnostics_subdir=diagnostics_subdir,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
        pilot_images_by_dataset=pilot_images_by_dataset,
        evaluate_after_generation=evaluate_after_generation,
        max_attribution_images_per_dataset=max_attribution_images,
    )
