# Configurations

All pipeline behavior is controlled by the YAML files in this folder. The scripts never contain dataset paths, model names or experiment settings. Relative paths inside the files are resolved from the repository root. The exception is `deployment.cluster_config`, which is resolved from the folder of the model configuration.

## Start here: templates

[`templates/`](templates/README.md) contains a fully commented template for every configuration family. Each key is described with:

- whether it is required, optional (with its default), or required only when another key has a certain value;
- the allowed values and the path-resolution base;
- its effect on resuming and on experiment fingerprints.

To create a new configuration, copy the matching template and change the values marked `<-- CHANGE`.

## Provided configurations

The files below reproduce the authors' experiments. They expect the datasets under `datasets/` and the planners under `reference_images/planners/`. Only some planner folders are part of the repository (see [`reference_images/README.md`](../reference_images/README.md)); for the other datasets, generate the planners first or adjust `planners_dir`.

| File(s) | Stage | Datasets |
|---|---|---|
| `model_configs.yaml` | all stages (VLM, LLM, segmentation model, local/cluster) | – |
| `dataset_configs.yaml` | planner generation | one dataset per file |
| `anomaly_detection_configs.yaml` | detection | custom box dataset with object cropping |
| `anomaly_detection_configs_mvtec_ad.yaml` | detection | MVTec AD (15 categories) |
| `anomaly_detection_configs_mvtec_loco.yaml` | detection | MVTec LOCO AD (5 categories) |
| `anomaly_detection_configs_ibadas_task_01.yaml`, `_task_02.yaml` | detection | custom box datasets; task 02 shows secondary classification |
| `mask_generation_configs_*.yaml` | mask generation | matching detection run |
| `metrics_configs*.yaml` | metrics | matching detection run; `_ibadas_task_02` shows secondary metrics |
| `ablation_configs/*.yaml` | repeated experiments | one file per dataset family; `anomaly_detection_ablation.example.yaml` is a commented example |
| `mask_ablation_configs.yaml` | XAI mask study | – |

The `run_dir` values in the mask, metrics and XAI configurations point to run folders on the authors' machines. Replace them with a local run folder, or pass `--run-dir`. Ablation runs fill them in automatically.

## Files that are not part of the repository

| File | Reason |
|---|---|
| `model_cluster_configs.yaml` | contains machine addresses and user names; create it from `templates/model_cluster_configs.template.yaml` |
| `experiments/` and other local variants | machine-specific experiment configurations |

## Rules shared by all configurations

- A dataset `name` must be identical in planner generation, detection, masks, metrics and the XAI study. Ablations verify that the mask and metrics dataset lists exactly match the detection list.
- Changing a model, prompt, planner, image size, crop setting or rule changes the experiment. Use a new `output.runs_dir` / `experiment.runs_dir` instead of resuming an old run.
- Validate ablation configurations with `python run_anomaly_detection_ablation.py --config <file> --dry-run` before starting them.
