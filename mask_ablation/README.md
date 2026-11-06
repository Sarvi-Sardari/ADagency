# `mask_ablation` package: XAI attribution-mask study

A post-inference study that explains the VLM's anomaly evidence with iGOS+-style attribution and turns the resulting heatmaps into pixel masks. Every configured model/method combination ("arm") is published in the standard mask layout of a detection run, so the regular metrics stage can rank the arms by pixel AUROC and AUPRO.

The study reads a completed anomaly-detection run. It never calls or modifies detection, planner generation, rerun, standard mask generation, or the metrics code.

## Requirements

- The environment from the root `requirements.txt`, on Linux (or Ubuntu under WSL2).
- A differentiable VLM checkpoint. A Qwen3-VL adapter is included; any Qwen3-VL size works without code changes.
- For the SAM3 and hybrid arms: the SAM3 source checkout and `sam3.pt` checkpoint (see the root README).
- GPU memory: gradients flow through the full VLM, so an 8B model needs most of a 24 GB GPU. Stop other GPU jobs first.

## Configuration

`configs/mask_ablation_configs.yaml` defines:

- `run_dir`: the detection run to explain;
- `datasets`: the datasets to process, with optional per-dataset `include_rerun`;
- `models`: the differentiable models and their adapters;
- `methods`: the arms, each with its iGOS settings;
- the pilot selection (`default_datasets`, `default_methods`, `pilot_images`, `max_attribution_images_per_dataset`).

Every key, and every method setting with its default value, is documented in [`configs/templates/mask_ablation_configs.template.yaml`](../configs/templates/mask_ablation_configs.template.yaml).

## Running

```bash
# Pilot: default datasets x default methods, limited to the pilot images.
python run_mask_ablation.py

# Full study: every configured dataset and method, all images, then metrics.
python run_mask_ablation.py --full
```

Inspect the pilot heatmaps and masks before starting a full run. The pilot never runs metrics on partial folders.

| Option | Effect |
|---|---|
| `--run-dir PATH` | explain another detection run |
| `--dataset NAME`, `--model NAME`, `--method NAME` | restrict the grid (repeatable) |
| `--include-rerun` / `--base-results` | use rerun-corrected or first-pass evidence for all datasets |
| `--overwrite` | regenerate the selected arms (required after changing an arm's settings) |
| `--no-evaluate` | skip the final metrics ranking |
| `--pixel-only` | rank with pixel AUROC only (skip AUPRO) |
| `--presentation`, `--presentation-image`, `--presentation-dir` | capture ordered frames for presentations |
| `--no-hardware-monitor`, `--hardware-monitor-interval` | telemetry control |

Running the same command again resumes per image. Each arm is fingerprinted by its model, method settings, prediction source and evidence. A changed fingerprint stops that arm instead of mixing incompatible masks; pass `--overwrite` or give the arm a new name.

For repeated filenames across defect folders, qualify `pilot_images` entries by source, for example `breakfast_box: [structural_anomalies/018.png]`. A merged reporting group can also be specified as `label_folder/source_label_folder/018.png`. Plain basenames remain supported but ambiguous matches are rejected. Absolute paths and traversal are not accepted. Stale absolute artifact paths are relocated only when their complete `rerun_results`, `texts`, or `preproc` suffix exists under the active dataset run; files are never matched by basename alone.

The terminal shows one progress bar for the images of the current method and one for the current optimization (image, mask type, reasoning step, iteration, elapsed time). Model-load messages, token diagnostics and warnings go to the study log. After a complete run, the unchanged metrics stage evaluates the new mask folders and the terminal prints the best arm by AUPRO, then by pixel AUROC.

## Evidence

| `mask_generation_mode` | What is explained |
|---|---|
| `yes_steps` | each planner step answered YES; the best step map is kept |
| `final_reason_details` | the final reason and details of the decision |
| `decision_output` | the complete decision output (used by the paper profile) |

With `include_rerun: true`, rerun results replace first-pass evidence image by image, matched by `(label_folder, source_label_folder, image_name)`; they are never appended as duplicates. An image flipped by the rerun is explained with the rerun step that flipped it. Because the rerun stops at the first YES and writes no new summary, `final_reason_details` uses that step's `reason` and `details`. An image confirmed `good` by the rerun, or an image without evidence, receives an explicitly recorded empty map; no evidence text is invented.

## Methods

- **Experimental arms** (`implementation_profile: ablation`) combine a region representation (dense patch grid, SLIC superpixels, SAM3 regions, or SAM3 + SLIC hybrid) with text targets (keywords, noun phrases, or both). They use a stabilized iGOS+ variant with mean L2 regularization, bounded masks, gradient clipping and an L2 decay floor. SAM3 region maps are cached per image and shared between SAM3 and hybrid arms. Published masks are usually sharp binary maps of the hot (yellow/red) heatmap regions (`published_mask_mode: heatmap_hot_regions`).
- **`paper_original`** (`implementation_profile: paper_original`) reproduces the published method: token-level LLR selection, cumulative selected-token log likelihood, one 32×32 mask, a deletion and insertion objective, L1/BTV regularization, a NAG update, and min/max rendering. Its parameters are locked and validated as a group. The coarse mask is bilinearly expanded to the model's visual grid and to the output image. It publishes the continuous anomaly map, because thresholding would discard the ranking information that pixel AUROC needs.

Supporting another VLM family requires one differentiable adapter registered with `@register_model_adapter("<name>")` in `mask_ablation/xai/adapters.py`. The adapter must give gradient access to the visual input, fixed-output token probabilities and the visual patch layout. Configuration, evidence handling and outputs stay unchanged.

## Outputs

```text
<run_dir>/<dataset>/masks/<arm>/<label_folder>/[<source_label_folder>/]<image_stem>.png   # published masks
<run_dir>/<dataset>/xai_mask_ablation/<arm>/       # float maps, heatmaps, overlays, token metadata
<run_dir>/xai_mask_ablation/                       # logs, performance_*.json, ablation_summary_*.csv,
                                                    # pixel_auroc_per_image_*.csv, pixel_auroc_best_images_*.csv
<run_dir>/presentation/<dataset>/<arm>/frames/<image>/   # --presentation frames and manifest.jsonl
```

`<arm>` is the method's `output_name`, or `xai_<model>_<method>`. Evaluation uses a symbolic-link view that contains only the arms of the current configuration, so older pilot folders do not enter the ranking. Normal images have no single-image ROC and are recorded as `--` in the per-image AUROC tables.

Example of capturing presentation frames for one image:

```bash
python run_mask_ablation.py --dataset <dataset> --method <method> --presentation --presentation-image processed_00007.png
```

## Module reference

| Module | Responsibility |
|---|---|
| `config.py` | configuration loading and validation (models, methods, pilot selection) |
| `pipeline.py` | resumable study orchestration, mask publication, fingerprints, summaries |
| `evidence.py` | one rerun-aware evidence record per image |
| `engine.py` | lazy bridge to the XAI engine |
| `xai_engine.py` | engine entry point and experiment loop (`Args` holds every method setting and its default) |
| `xai/adapters.py` | differentiable model adapters and their registry |
| `xai/evidence.py` | attribution samples from run artifacts |
| `xai/segmentation.py` | image baselines, SLIC and SAM3 preprocessing |
| `xai/optimization.py` | mask ensembling, expansion and display normalization |
| `xai/reporting.py`, `reports.py` | per-image reports |
| `paper_original.py` | the locked paper method |
| `igos_core_methods.py`, `igos_core_methods_helper.py`, `igos_core_utils.py` | iGOS / iGOS++ optimizers (adapted third-party code, see below) |
| `logging_utils.py` | study logging |

## Attribution

`igos_core_methods.py`, `igos_core_methods_helper.py` and `igos_core_utils.py` are adapted from the IGOS / iGOS++ explanation methods (© 2024 Bytedance Ltd. and/or its affiliates; based on [saeed-khorram/IGOS](https://github.com/saeed-khorram/IGOS)). They remain under their original license. The adaptations are described in the module docstrings.
