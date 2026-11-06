# Configuration templates

Commented reference copies of every configuration family. Each template explains every key: whether it is required, its default, allowed values, and which other keys it depends on. The explanations come from the loader code, not from older comments.

Copy a template next to the real configs (`configs/`), fill in the values marked `<-- CHANGE`, and pass it with the stage's `--...-config` flag. The scripts never read these template files themselves.

| Template | Stage / entry point | Loader |
|---|---|---|
| `dataset_configs.template.yaml` | planner generation (`run_planner_generation.py --dataset-config`) | `planner_generation/config.py` |
| `model_configs.template.yaml` | every stage (`--model-config`) | `planner_generation/config.py`, `anomaly_detection/config.py` |
| `model_cluster_configs.template.yaml` | cluster mode only (via `deployment.cluster_config`) | `cluster_deployment/config.py` |
| `anomaly_detection_configs.template.yaml` | detection (`run_anomaly_detection.py --dataset-config`) | `anomaly_detection/config.py` |
| `mask_generation_configs.template.yaml` | masks (`run_mask_generation.py --config`) | `anomaly_detection/config.py` |
| `metrics_configs.template.yaml` | metrics (`run_metrics.py --metrics-config`) | `anomaly_detection/config.py` |
| `anomaly_detection_ablation.template.yaml` | repeated experiments (`run_anomaly_detection_ablation.py --config`) | `anomaly_detection/ablation/orchestration.py` |
| `mask_ablation_configs.template.yaml` | XAI mask study (`run_mask_ablation.py --config`) | `mask_ablation/config.py` |

## How the files connect

```
dataset_configs ──(planner generation)──► planners_dir/<dataset>/<class>/
                                                │
anomaly_detection_configs.planners_dir ◄────────┘
anomaly_detection_configs ──(detection)──► <output.root_dir>/<output.runs_dir>/<timestamp>_/   = run_dir
                                                │
mask_generation_configs.run_dir ◄───────────────┤   (rerun runs on the same run_dir)
metrics_configs.run_dir ◄───────────────────────┤
mask_ablation_configs.run_dir ◄─────────────────┘

model_configs ─ used by all of the above; in cluster mode it merges model_cluster_configs
ablation config ─ points to anomaly_detection + model + mask_generation + metrics configs
                  and fills in run_dir for every run itself
```

Rules shared by all families:

- Dataset `name` values must be the same everywhere: planner generation, detection, masks, metrics, and the XAI study. Ablations check that the mask and metrics dataset lists exactly match the detection list.
- Relative paths are resolved from the project root. Two exceptions: `deployment.cluster_config` is resolved from the model config's own folder, and the metrics config's `secondary_metrics` paths are resolved from that dataset's `root_dir`.
- Changing a model, prompt, planner, image size, rule, or stage setting changes the experiment. Use a new `output.runs_dir` / `experiment.runs_dir` instead of resuming an old run.
- `model_cluster_configs.template.yaml` uses placeholders (`192.168.XX.YY`, `<ssh-user>`, `<net-interface>`). Keep the filled-in copy private.
