# `anomaly_detection` package

Online (batch) anomaly detection and every post-inference stage that works on a detection run: rerun, mask generation, metrics, and the repeated-experiment orchestrator. The package uses planners that were produced beforehand by [`planner_generation`](../planner_generation/README.md).

## Entry points

| Command (from the repository root) | Module | Purpose |
|---|---|---|
| `python run_anomaly_detection.py` | `runner.py` | Detect anomalies in every dataset listed in an anomaly-detection config |
| `python -m anomaly_detection.rerun <run_dir> <dataset>` | `rerun.py` | Calibrate a confidence threshold and re-ask uncertain planner steps |
| `python run_mask_generation.py` | `masking/orchestration.py` | Create pixel masks for a finished run |
| `python run_metrics.py` | `metrics.py` | Compute image- and pixel-level metrics for a finished run |
| `python run_anomaly_detection_ablation.py --config <yaml>` | `ablation/orchestration.py` | Run detection → rerun → masks → metrics repeatedly and aggregate |
| `python -m anomaly_detection.crop_train_good` | `crop_train_good.py` | Optional: crop `train/good` references to the object |

Each command prints all options with `--help`. Configuration keys are documented in [`configs/templates/`](../configs/templates/README.md).

## Detection workflow

For each dataset in the configuration:

1. **Reference embedding.** All `train/good` images are embedded with CLIP (`ViT-B/32`).
2. **Optional object crop** (`object_crop.enabled`). The VLM locates the whole object, and the test image is cropped to it. If the object is not found, the run stops with `ObjectNotFoundError`; the pipeline never falls back silently to the uncropped image.
3. **Nearest reference.** The most similar normal image is selected for every test image.
4. **Planner selection.** Single-class datasets use `<planners_dir>/<dataset>/good/`. Multi-class datasets use the planner of the class matched for the nearest reference. When a folder contains several planner files, the newest one is used.
5. **Decision.** The VLM receives the candidate and reference images and the planner. It answers every step with YES/NO, a confidence value and evidence, and then gives a final label. Responses with missing steps or confidence records are retried. `decision.label_policy: model` keeps the VLM's final label; `steps` applies the legacy rule (any YES → anomaly).
6. **Persistence.** Per-image artifacts are written atomically. A completed dataset receives `all_data_summary.jsonl`.

Resume a run with `--resume-dir "<run_dir>"`. A completed dataset is skipped only after its summary and all referenced artifacts have been validated; inside an interrupted dataset, images that were already decided are reused.

## Optional features

| Feature | Configuration | Module |
|---|---|---|
| Merged anomaly folders (`good` vs `anomalies/<source>/`) | `datasets[].merge_anomalies` | `config.py` (`artifact_group_dir`) |
| Multi-class datasets | `classes: multi`, `class_examples` | `class_labeler.py`, `reference_selector.py` |
| Object-centred cropping | `datasets[].object_crop` | `object_crop.py` |
| Secondary classification (several named classes from YES/NO steps) | `secondary_classification` | `secondary_classification.py` |
| Remote OpenAI-compatible model server / Ray cluster | model config `deployment` | `models.py`, [`cluster_deployment`](../cluster_deployment/README.md) |

**Multi-class label cache.** Class descriptions are stored in `prompts/class_labels/<dataset>_data_labels.txt`. If the file is missing, they are generated from `class_examples` and confirmed interactively (accept, regenerate, or quit), and the confirmed descriptions are cached.

**Secondary classification.** Python rules map the final set of YES steps to exactly one configured class (for example `non_empty`, `empty_dirty`, `empty_clean`). Each class has a binary label. The rule set is validated so that it covers every YES/NO combination without overlaps. The resolved profile is saved with the run, and later stages reject a run whose profile does not match.

## Post-inference stages

### Rerun (`rerun.py`, `rerun_stage/`)

The rerun stage computes one anomaly score per image from the step confidences and selects the optimum threshold from the ROC curve of the run itself. It then re-asks the lowest-confidence steps (`--num-steps`) for eligible images. The policies are:

- `binary-gate` (default): only images predicted `good` whose score crosses the threshold are rerun. The first re-asked step answered YES flips the image to `anomaly`.
- `class-aware` and `global-secondary`: variants for runs that use secondary classification. `global-secondary` needs a secondary ground truth (`--metrics-config` or `--secondary-ground-truth`).

Output: `<run_dir>/<dataset>/rerun_results/` (`rerun_result.jsonl`, `rerun_meta.json`, threshold ROC plot, per-image rerun steps). `--threshold-only` stops after threshold selection.

### Mask generation (`mask_generation.py`, `masking/`)

| Method | Output folder | Mask source |
|---|---|---|
| `bounding_box` | `masks/bounding_box/` | VLM boxes (0–1000 relative coordinates) |
| `segmentation` | `masks/<segmentation model>/` | segmentation model, text-prompted with the saved anomaly reasoning |
| `bounding_box_segmentation` | `masks/bounding_box_<segmentation model>/` | segmentation model guided by the VLM boxes |

Images predicted `good` receive an empty mask. Box coordinates are cached as JSON and shared between methods. Each dataset/method combination is fingerprinted under `mask_generation/methods/`: with changed settings the stage stops instead of mixing masks, unless `--overwrite` is given. Invalid box output is retried with corrective feedback, and the rejected responses are kept for diagnosis.

New techniques can be added without touching the configuration schema:

- `@register_method("<name>", needs_boxes=..., needs_segmentation=...)` registers a mask generator.
- `@register_segmentation_backend("<model_name>")` registers a loader for `models.segmentation_model.model_name`. A processor that has the SAM3 interface (`set_image`, `set_text_prompt`, `add_geometric_prompt`) reuses the existing generators.

### Metrics (`metrics.py`, `scoring.py`)

Metrics are computed from files only; no model is loaded. Per anomaly class and overall, the stage reports accuracy, F1, the confusion matrix and image AUROC. When ground-truth masks resolve and `masks/<method>/` exists, it adds pixel AUROC and AUPRO for each mask method. Optional secondary metrics are one-vs-rest per class. Rerun predictions replace first-pass predictions per image when `include_rerun` is set; records are matched by `(label_folder, source_label_folder, image_name)`.

### Repeated experiments (`ablation/orchestration.py`)

The orchestrator runs the other stages as separate processes, `num_runs` times, each in its own run folder. It records progress in `ablation_manifest.json` and aggregates metrics, timing, token counts and GPU telemetry as mean and standard deviation. The stage configurations are fingerprinted, so a resume after a configuration change is rejected. Always validate a new configuration with `--dry-run` first.

### Reference cropping (`crop_train_good.py`)

This optional manual step, run when test-time cropping is enabled, makes the normal references object-centred too:

```bash
python -m anomaly_detection.crop_train_good --dataset-config <config> --datasets <name> --dry-run
python -m anomaly_detection.crop_train_good --dataset-config <config> --datasets <name>
python -m anomaly_detection.crop_train_good --dataset-config <config> --datasets <name> --restore
```

- The first run copies `train/good` to `train/good_uncropped` and verifies the copy byte by byte. Every later run crops from that backup.
- The cropped set replaces `train/good` only if every image succeeded. The step writes `train/good_object_crop_manifest.json`, and detection logs a warning when train and test crop settings differ.
- `train/good` is shared by planner generation and all experiments, so cropping it affects every later run on that dataset.

## Module reference

| Module | Responsibility |
|---|---|
| `runner.py` | CLI, run-directory creation/resume, hardware monitor lifecycle |
| `pipeline.py` | per-dataset orchestration, resume validation, summary writing |
| `inference_agent.py` | per-image inference, planner lookup, artifact persistence |
| `decision.py` | decision prompt rendering, final-label parsing, step-evidence validation |
| `models.py` | VLM wrapper (Transformers or OpenAI-compatible server) and CLIP encoder |
| `reference_selector.py` | CLIP reference embeddings, nearest-reference search, class matching |
| `class_labeler.py` | multi-class label cache |
| `object_crop.py` | object localization, crop geometry, crop profiles and timing |
| `secondary_classification.py` | rule engine and profile fingerprint for secondary classes |
| `scoring.py` | lightweight parsing of step confidences and image scores |
| `rerun.py`, `rerun_stage/` | threshold selection, step selection, rerun execution and artifacts |
| `mask_generation.py`, `masking/` | mask methods, segmentation backends, caching, fingerprints |
| `metrics.py` | image/pixel/secondary metrics and CSV writing |
| `ablation/orchestration.py` | repeated-experiment orchestration and aggregation |
| `crop_train_good.py` | optional reference cropping |
| `config.py` | loading and validation of detection, model, mask and metrics configurations |
| `hardware_monitor.py` | separate-process CPU/GPU/thermal/kernel telemetry recorder |
| `logging_utils.py`, `utils.py` | logging setup, image and file helpers |
