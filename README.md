# ADagency Pipeline

A configuration-driven pipeline that detects and localizes visual anomalies in industrial images with a vision-language model (VLM). The VLM gets no anomaly training. Each dataset gets an *inspection planner*: a short list of visually answerable YES/NO questions, generated once from a few normal reference images and a list of anomaly rules. At inspection time the VLM compares a test image with its closest normal reference and answers every planner step. The answers carry confidence scores and evidence, and the VLM gives a final `good` / `anomaly` decision.

The repository covers the full research workflow:

- offline planner generation from normal images;
- batch anomaly detection over several datasets, with resumable runs;
- confidence-based reruns of uncertain planner steps;
- pixel-mask generation (VLM bounding boxes and SAM3 segmentation);
- image- and pixel-level metrics (accuracy, F1, image AUROC, pixel AUROC, AUPRO);
- repeated-run experiment orchestration with hardware telemetry;
- an explainability (XAI) study that builds attribution masks with iGOS-style optimization;
- optional multi-PC execution through Ray and vLLM.

Tested datasets: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco), and custom industrial box datasets that follow the same folder layout.

---

## Contents

1. [How the pipeline works](#how-the-pipeline-works)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Model assets](#model-assets)
6. [Dataset preparation](#dataset-preparation)
7. [Configuration](#configuration)
8. [Running the pipeline](#running-the-pipeline)
9. [Repeated experiments (ablations)](#repeated-experiments-ablations)
10. [XAI mask study](#xai-mask-study)
11. [Multi-PC cluster execution](#multi-pc-cluster-execution)
12. [Output layout](#output-layout)
13. [Resuming and reproducibility](#resuming-and-reproducibility)
14. [Troubleshooting](#troubleshooting)
15. [Further documentation](#further-documentation)
16. [Related branches](#related-branches)
17. [Third-party components](#third-party-components)

---

## How the pipeline works

```text
                 ┌──────────────────────── OFFLINE (once per dataset) ────────────────────────┐
 train/good ──►  │ CLIP reference selection → VLM object names → VLM per-image descriptions    │
 anomaly rules ─►│ → LLM master description → VLM planner (numbered YES/NO inspection steps)  │
                 └──────────────────────────────────────┬─────────────────────────────────────┘
                                                        │ reference_images/planners/<dataset>/
                 ┌──────────────────────── ONLINE (per test image) ───────────────────────────┐
 test image ──►  │ [optional object crop] → nearest normal reference (CLIP) → planner          │
                 │ → VLM answers every step (YES/NO + confidence + evidence) → final decision  │
                 └──────────────────────────────────────┬─────────────────────────────────────┘
                                                        │ outputs/<runs_dir>/<run>/<dataset>/
            ┌──────────────┬────────────────────────────┼───────────────────────┬─────────────────┐
            ▼              ▼                            ▼                       ▼                 ▼
     rerun (re-ask   mask generation           metrics (image +          XAI mask study    repeated
     uncertain       (boxes / SAM3)            pixel level)              (iGOS attribution) experiments
     steps)
```

| Stage | Entry point | Main input | Main output |
|---|---|---|---|
| Planner generation | `run_planner_generation.py` | `train/good` images, anomaly rules | planner per dataset/class |
| Anomaly-rule curation (optional) | `run_anomaly_rule_generation.py` | descriptions, object names | compiled rules file |
| Anomaly detection | `run_anomaly_detection.py` | `test/` images, planners | `all_data_summary.jsonl` + per-image evidence |
| Rerun | `python -m anomaly_detection.rerun` | a detection run | `rerun_results/rerun_result.jsonl` |
| Mask generation | `run_mask_generation.py` | a detection run | `masks/<method>/...png` |
| Metrics | `run_metrics.py` | a detection run (+ masks, rerun) | `metrics/*.csv` |
| Repeated experiments | `run_anomaly_detection_ablation.py` | the four stage configs | runs + mean/std aggregates |
| XAI mask study | `run_mask_ablation.py` | a detection run | `masks/xai_<model>_<method>/...` |
| Reference cropping (optional) | `python -m anomaly_detection.crop_train_good` | `train/good` | object-cropped references |

Every stage is a separate command. Later stages read only the files that earlier stages wrote, so any stage can be repeated or resumed on its own.

---

## Repository layout

```text
.
├── run_planner_generation.py        # offline planner generation
├── run_anomaly_rule_generation.py   # standalone anomaly-rule curation
├── run_anomaly_detection.py         # batch anomaly detection
├── run_mask_generation.py           # post-inference pixel masks
├── run_metrics.py                   # post-inference metrics
├── run_anomaly_detection_ablation.py# repeated-experiment orchestrator
├── run_mask_ablation.py             # XAI attribution-mask study
├── planner_generation/              # planner-generation package
├── anomaly_detection/               # detection, rerun, masks, metrics, ablation
├── mask_ablation/                   # XAI (iGOS) mask study
├── cluster_deployment/              # Ray / vLLM / remote telemetry support
├── pipeline_core/                   # shared, model-independent helpers
├── configs/                         # YAML configurations and documented templates
├── prompts/                         # Jinja prompt templates, rules, label caches
├── reference_images/                # generated planners, rules and descriptions
└── requirements.txt                 # Python runtime for this branch (Linux, Python 3.12)
```

Each package folder contains its own `README.md` describing its modules. Not included in the repository and created locally:

| Folder | Content |
|---|---|
| `datasets/` | image datasets (see [Dataset preparation](#dataset-preparation)) |
| `models/` | model checkpoints (see [Model assets](#model-assets)) |
| `outputs/` | all run results, logs, and telemetry |

---

## Requirements

| Item | Requirement |
|---|---|
| Operating system | Linux. On Windows, use Ubuntu under WSL2; native Windows is not supported (FlashAttention, vLLM and the `/proc`-based telemetry need Linux). |
| Python | 3.12 (Conda environment recommended) |
| GPU | NVIDIA GPU with a current driver. An 8B VLM (e.g. Qwen3-VL-8B-Instruct) in BF16 runs on a 24 GB GPU such as an RTX 4090. The XAI study and SAM3 need most of that memory. |
| CUDA toolkit | 13.0 (needed to compile FlashAttention) |
| Disk | roughly 20 GB per 8B checkpoint, plus datasets and outputs |
| Optional | SSH + Ray on every PC (cluster mode only) |

---

## Installation

Run all commands from the repository root.

```bash
# 1. Environment
conda create -n xai python=3.12 pip -y
conda activate xai
python -m pip install --upgrade pip

# 2. CUDA toolkit for compiling FlashAttention, then verify driver and compiler
conda install -c nvidia cuda-toolkit=13.0 -y
nvidia-smi
nvcc --version

# 3. PyTorch first (FlashAttention compiles against it), then everything else
python -m pip install torch==2.13.0 torchvision==0.28.0
python -m pip install --no-build-isolation -r requirements.txt
```

Notes:

- `requirements.txt` pins every package used by all stages of this branch, including Ray and vLLM for cluster mode. Ray and vLLM are only used when `deployment.mode: cluster`.
- TensorRT-LLM is not supported: it conflicts with the pinned environment, was not faster than vLLM, and does not return the per-token log-probabilities used for confidence.
- Run every command with the environment's interpreter (`conda activate xai`, or its absolute path in scripts and `tmux` sessions). A system Python without these packages fails at import time.

---

## Model assets

Checkpoints are not stored in the repository. The default configurations expect them under `models/`:

```text
models/
├── Qwen3-VL-8B-Instruct/   # VLM (and default LLM) checkpoint directory
├── sam3/                   # SAM3 source checkout (segmentation masks)
└── sam3_model/sam3.pt      # SAM3 checkpoint file
```

```bash
# VLM (any Transformers-compatible VLM can be configured; this is the default)
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct

# SAM3 source at the tested commit
git clone https://github.com/facebookresearch/sam3.git models/sam3
git -C models/sam3 checkout 11dec2936de97f2857c1f76b66d982d5a001155d

# SAM3 weights (gated on Hugging Face: request access first)
hf auth login
hf download facebook/sam3 sam3.pt --local-dir models/sam3_model
```

- The CLIP encoder (`ViT-B/32`, used for reference selection) is downloaded automatically on first use.
- SAM3 is needed only for the `segmentation` / `bounding_box_segmentation` mask methods and the SAM3-based XAI arms.
- To use a different checkpoint, change `models.vlm` / `models.llm` in the model configuration (see [Configuration](#configuration)). Thinking models such as Qwen3/Qwen3.5 need `chat_template_kwargs: {enable_thinking: false}`.
- Planners, prompts and evaluation results depend on the model. Regenerate the planners when changing the VLM family, and store them in a separate `planners_dir`.

---

## Dataset preparation

Datasets live in `datasets/` (git-ignored) and use the MVTec folder convention:

```text
datasets/<family>/<dataset>/
├── train/
│   └── good/                 # normal images: planner references and inference references
├── test/
│   ├── good/                 # normal test images
│   └── <anomaly_folder>/     # one folder per anomaly type, e.g. broken_large/
└── ground_truth/             # optional pixel masks for pixel AUROC / AUPRO
    └── <anomaly_folder>/...  # as distributed with the dataset
```

- **MVTec AD**: extract each category to `datasets/mvtec_ad/<category>/`.
- **MVTec LOCO AD**: extract each category to `datasets/mvtec_loco/<category>/`. Its test folders are `good`, `logical_anomalies`, `structural_anomalies`.
- **Custom data**: follow the same layout. Ground-truth masks may also be stored as `<dataset>/mask_all/` (all masks in one folder), or in any directory named by `ground_truth_dir` in the metrics configuration.

Dataset names must be identical in every configuration file (planner generation, detection, masks, metrics, XAI), because they become output folder names.

---

## Configuration

All behavior is controlled by YAML files in [`configs/`](configs/). Relative paths in these files are resolved from the repository root.

| File family | Used by |
|---|---|
| `dataset_configs.yaml` | planner generation (one dataset per file) |
| `model_configs.yaml` | every stage (VLM, LLM, segmentation model, local/cluster switch) |
| `anomaly_detection_configs_*.yaml` | detection (many datasets per file) |
| `mask_generation_configs_*.yaml` | mask generation |
| `metrics_configs_*.yaml` | metrics |
| `ablation_configs/*.yaml` | repeated experiments |
| `mask_ablation_configs.yaml` | XAI mask study |

**Start from the templates in [`configs/templates/`](configs/templates/).** They are fully commented copies of every configuration family. Each key is marked as required, optional (with its default), or required only when another option is set, and the allowed values are listed. [`configs/README.md`](configs/README.md) gives an overview of the provided configurations.

The configurations shipped in `configs/` reproduce the authors' experiments. They refer to datasets in `datasets/` and to planners in `reference_images/planners/`. Only some planner folders are included (see [`reference_images/README.md`](reference_images/README.md)); for other datasets, generate the planners first.

---

## Running the pipeline

The commands below use the default configuration paths. Every entry point accepts explicit paths (`--help` lists all options).

### 1. Generate planners (offline, once per dataset)

Edit a copy of [`configs/templates/dataset_configs.template.yaml`](configs/templates/dataset_configs.template.yaml) (dataset name, root, anomaly rules, output `planners_dir`), then run:

```bash
python run_planner_generation.py \
    --dataset-config configs/dataset_configs.yaml \
    --model-config   configs/model_configs.yaml
```

Steps: CLIP selects representative `train/good` references; the VLM names object parts (cached in `prompts/object_labels/`); the VLM describes each reference; the LLM merges these into a master description; anomaly rules are loaded or curated; the VLM writes the numbered planner. The result is `<planners_dir>/<dataset>/<good|class>/<good|class>_<timestamp>_planner.txt`; logs are written to `reference_images/planner_generation_logs/`. When several planner files exist in a folder, detection uses the newest one.

Anomaly rules come either from `prompts/anomaly_definition.json` (`method: manual`, recommended) or from an interactive review (`method: automatic`). The rules can also be compiled on their own:

```bash
python run_anomaly_rule_generation.py --dataset-config configs/dataset_configs.yaml
```

### 2. Detect anomalies

List the datasets and the matching `planners_dir` in an anomaly-detection configuration, then run:

```bash
python run_anomaly_detection.py \
    --dataset-config configs/anomaly_detection_configs_mvtec_loco.yaml \
    --model-config   configs/model_configs.yaml
```

A new timestamped run folder is created under `outputs/<output.runs_dir>/`. This folder is referred to as `<run_dir>` below. To continue an interrupted run instead of starting a new one:

```bash
python run_anomaly_detection.py --dataset-config <config> --resume-dir "<run_dir>"
```

Completed datasets are validated and skipped; already-decided images are reused. A hardware recorder starts automatically (`--no-hardware-monitor` disables it).

### 3. Rerun uncertain decisions (optional)

```bash
python -m anomaly_detection.rerun "<run_dir>" <dataset_name> --num-steps 3
```

The rerun stage calibrates a confidence threshold on the run's own results (ROC). It then re-asks the lowest-confidence planner steps for borderline `good` predictions. If any re-asked step is answered YES, the image is flipped to `anomaly`. Use `--threshold-only` to compute the threshold without loading the VLM.

### 4. Generate pixel masks (optional, needed for pixel metrics)

Set `run_dir` and the methods in a mask-generation configuration, then run:

```bash
python run_mask_generation.py --config configs/mask_generation_configs_mvtec_loco.yaml
```

| Method | How the mask is made | Needs |
|---|---|---|
| `bounding_box` | filled VLM bounding boxes | VLM |
| `segmentation` | SAM3, text-prompted with the VLM's anomaly reasoning | SAM3 |
| `bounding_box_segmentation` | SAM3 guided by the VLM boxes | VLM + SAM3 |

The command can be repeated safely: masks that already exist are skipped. Changed method settings stop the stage until `--overwrite` is passed.

### 5. Compute metrics

Set `run_dir` and the datasets in a metrics configuration, then run:

```bash
python run_metrics.py --metrics-config configs/metrics_configs_mvtec_loco.yaml --combine
```

No model is loaded. The stage writes per-dataset CSVs to `<run_dir>/<dataset>/metrics/`, and `--combine` adds one combined CSV in `<run_dir>`. Rows contain accuracy, F1, the confusion matrix and image AUROC per anomaly class. When ground-truth and predicted masks exist, pixel AUROC and AUPRO are added for each mask method. Options:

- `--include-rerun` / `--base-results` select rerun-corrected or first-pass predictions for all datasets.
- `--omit aupro pixel-auroc image-auroc` skips expensive calculations.

---

## Repeated experiments (ablations)

`run_anomaly_detection_ablation.py` runs detection → rerun → masks → metrics several times. Each run gets its own folder, and the orchestrator aggregates the mean and standard deviation of metrics, timing, generated tokens and GPU telemetry.

```bash
# Validate configuration and paths, print the commands, use no GPU
python run_anomaly_detection_ablation.py --config configs/ablation_configs/ablation_mvtec_loco.yaml --dry-run

# Run (use tmux for long experiments)
python run_anomaly_detection_ablation.py --config configs/ablation_configs/ablation_mvtec_loco.yaml
```

- Progress is stored in `<experiment.runs_dir>/ablation_manifest.json`. Running the same command again resumes from that point.
- The experiment is fingerprinted: changing a stage configuration, prompt sequence or rerun policy blocks resuming. Use a new `experiment.runs_dir` for a changed experiment.
- Useful options: `--runs N`, `--stages detection rerun`, `--skip masks`, `--prediction-source base`, `--aggregate-only`.

See [`configs/templates/anomaly_detection_ablation.template.yaml`](configs/templates/anomaly_detection_ablation.template.yaml) for every option. Hardware telemetry is written to `<run_dir>/logs/hardware_telemetry_*/`.

---

## XAI mask study

`run_mask_ablation.py` explains the VLM's anomaly evidence with iGOS+-style attribution on a differentiable VLM (Qwen3-VL adapter included). It publishes each configured model/method combination as an additional mask folder, which the metrics stage can rank.

```bash
python run_mask_ablation.py          # pilot: selected datasets/methods/images
python run_mask_ablation.py --full   # every configured dataset and method, then metrics
```

The study needs gradients through the full VLM and therefore most of a 24 GB GPU. Details: [`mask_ablation/README.md`](mask_ablation/README.md).

---

## Multi-PC cluster execution

By default (`deployment.mode: local`) every model runs in-process on the local GPU through Transformers. In cluster mode the pipeline serves the model with vLLM over several PCs that are joined in a Ray cluster.

1. Create the same environment on every PC, and make the model directories available at the same paths on every PC.
2. Configure password-less SSH from the head PC to every worker (`ssh-copy-id <user>@<worker-ip>`; test with `ssh -o BatchMode=yes <user>@<worker-ip> true`).
3. Start Ray (the pipeline never starts or stops Ray itself):

   ```bash
   # head PC
   export NCCL_SOCKET_IFNAME=<interface> GLOO_SOCKET_IFNAME=<interface> VLLM_HOST_IP=<head-ip>
   ray start --head --port=6379 --node-ip-address=<head-ip>

   # every worker PC
   export NCCL_SOCKET_IFNAME=<interface> GLOO_SOCKET_IFNAME=<interface> VLLM_HOST_IP=<worker-ip>
   ray start --address='<head-ip>:6379' --node-ip-address=<worker-ip>
   ```

4. Copy [`configs/templates/model_cluster_configs.template.yaml`](configs/templates/model_cluster_configs.template.yaml) to `configs/model_cluster_configs.yaml`, fill in the placeholders, and set `deployment.mode: cluster` in the model configuration. This file contains addresses and user names, so keep it private; it is git-ignored.
5. Validate the cluster without loading a model:

   ```bash
   python -m cluster_deployment.preflight --model-config configs/model_configs.yaml
   python -m cluster_deployment.preflight --model-config configs/model_configs.yaml --telemetry-smoke-seconds 5
   ```

6. Run any stage as usual. The pipeline starts its own vLLM server across the Ray GPUs, records telemetry on every node, and stops the server at the end.

See [`cluster_deployment/README.md`](cluster_deployment/README.md).

---

## Output layout

```text
outputs/<runs_dir>/<YYYY-MM-DD HH-MM-SS>_/          # one detection run = <run_dir>
├── <dataset>/
│   ├── all_data_summary.jsonl        # one record per test image (label, score, paths)
│   ├── preproc/                      # letterboxed candidate/reference images sent to the VLM
│   ├── texts/                        # per image: decision text, step YES/NO probabilities, timing
│   ├── performance/                  # timing and token statistics
│   ├── rerun_results/                # rerun_result.jsonl, threshold ROC plot, rerun steps
│   ├── masks/<method>/               # predicted masks (standard and XAI methods)
│   ├── mask_generation/              # mask fingerprints and performance
│   └── metrics/                      # metrics_<timestamp>.csv
├── logs/                             # run logs and hardware telemetry
└── metrics_*.csv                     # combined metrics (run_metrics.py --combine)
```

Merged anomaly groups (`merge_anomalies: true`) keep the original test folder as a sub-folder (e.g. `anomalies/broken_large/000.png`), so same-named images never overwrite one another. Records are identified by `(label_folder, source_label_folder, image_name)`.

---

## Resuming and reproducibility

- **Resume, do not restart.** Detection (`--resume-dir`), rerun, masks, the XAI study and ablations all skip valid finished work and redo only missing or invalid items. Results are written atomically, so an interrupted write never looks complete.
- **Fingerprints protect experiments.** Mask methods, crop settings, secondary-classification rules and ablation configurations are fingerprinted. A changed setting stops the stage instead of mixing results. Start a new `output.runs_dir` / `experiment.runs_dir` when the method, model, prompt or planner changes.
- **Use `--overwrite` only on purpose.** It regenerates outputs that may have taken hours to produce.
- **Keep base and rerun predictions consistent.** Masks and metrics must use the same prediction source (`include_rerun`).
- **Long runs:** start them inside `tmux`. Stop them with `Ctrl+C` (SIGINT), which shuts down cleanly, and resume them later with the same command.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` at start | Wrong interpreter. Activate the `xai` environment or call its Python by absolute path. |
| `FileNotFoundError` for a planner | `planners_dir` or the dataset `name` does not match the planner folder. Generate planners for that dataset first. |
| `CUDA out of memory` | Another process is using the GPU (`nvidia-smi`), or the image size is too large. Reduce `image_processing.target_w/target_h`, or stop the other job. |
| First inference is very slow | Normal for FP8 checkpoints and for vLLM: kernel compilation and warm-up happen on the first call. Compare steady-state speed only from the second image on. |
| Fingerprint / profile mismatch on resume | A configuration changed since the run started. Use a new runs directory. |
| Segmentation fails immediately | `segmentation_model.local_path` must be the `sam3.pt` file, not its folder, and `repo_dir` must point to the SAM3 checkout. |
| Malformed VLM output retried repeatedly | The model does not follow the step/confidence format. Check `chat_template_kwargs` (disable thinking), `max_new_tokens`, and the planner length. |
| Cluster preflight fails | Ray not started on every PC, mismatched package versions between PCs, missing model directory on a worker, or SSH not password-less. |

---

## Further documentation

| Document | Content |
|---|---|
| [`configs/templates/README.md`](configs/templates/README.md) | annotated templates for every configuration family |
| Package READMEs | `planner_generation/`, `anomaly_detection/`, `mask_ablation/`, `cluster_deployment/`, `pipeline_core/`, `configs/`, `prompts/`, `reference_images/` |

---

## Related branches

| Branch | Content |
|---|---|
| `ad_agency` (this branch) | planner generation, batch anomaly detection, masks, metrics, ablations, XAI study, cluster execution |
| `live_inference` | the semi-live web application for single-image inspection with cameras (local GPU only) |
| `finetuning` | synthetic data generation and LoRA/QLoRA fine-tuning of the VLM |

---

## Third-party components

- `mask_ablation/igos_core_*.py` are adapted from the IGOS / iGOS++ explanation methods (© 2024 Bytedance Ltd. and/or its affiliates; based on [saeed-khorram/IGOS](https://github.com/saeed-khorram/IGOS)). They remain under their original license.
- [OpenAI CLIP](https://github.com/openai/CLIP), [SAM3](https://github.com/facebookresearch/sam3), the configured VLM/LLM checkpoints, and the MVTec datasets are subject to their own licenses and terms of use. They are downloaded separately and are not redistributed here.
