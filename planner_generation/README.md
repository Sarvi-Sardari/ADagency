# `planner_generation` package

Offline generation of *inspection planners*: numbered, visually answerable YES/NO questions that the anomaly-detection stage asks the VLM for every test image. Planners are created once per dataset (and per class for multi-class datasets) from a few normal images and a list of anomaly rules.

## Usage

```bash
python run_planner_generation.py \
    --dataset-config configs/dataset_configs.yaml \
    --model-config   configs/model_configs.yaml
```

One dataset is processed per invocation. Start from [`configs/templates/dataset_configs.template.yaml`](../configs/templates/dataset_configs.template.yaml), which documents every key. The model configuration must define both `models.vlm` and `models.llm`; they may point to the same checkpoint.

To compile only the anomaly rules, without generating planners:

```bash
python run_anomaly_rule_generation.py --dataset-config configs/dataset_configs.yaml
```

## Workflow

| Step | Model | Module | Output |
|---|---|---|---|
| 1. Select representative normal references from `train/good` (per class for multi-class datasets, or an explicit list) | CLIP | `reference.py` | `<preprocessed_reference_dir>/<dataset>/` |
| 2. Name the visible object parts (cached and reusable) | VLM | `object_namer.py` | `prompts/object_labels/<dataset>_object_labels.txt` |
| 3. Describe each reference image | VLM | `descriptor.py` | `<descriptions_dir>/<dataset>/<class>/` |
| 4. Merge the descriptions into one master description per class | LLM | `descriptor.py` | `<master_descriptions_dir>/<dataset>/<class>/` |
| 5. Resolve the anomaly rules | – / VLM | `anomaly_rules.py` | rules text |
| 6. Write the planner, covering every anomaly rule | VLM | `planner.py` | `<planners_dir>/<dataset>/<class>/<class>_<timestamp>_planner.txt` |

The VLM and LLM are loaded one after the other and released between steps, so they never have to share GPU memory. In cluster mode the managed vLLM server is swapped VLM → LLM → VLM in the same way.

### Anomaly rules

| `anomaly_rules.method` | Source | Interaction |
|---|---|---|
| `manual` (recommended) | `anomaly_rules.file[anomaly_rules.key]`, e.g. `prompts/anomaly_definition.json` | none |
| `automatic` | candidates generated from the master description, object names and optional user input | interactive review; the result is compiled to `reference_images/anomaly_rules/<dataset>.txt` and reused unless `overwrite: true` |

### Planner quality guidelines

- One atomic, visually answerable question per step. Do not put several numbered sub-questions in one step.
- Every approved anomaly rule must be covered, without duplicate steps.
- Overlong or overconstrained planners slow down inference and can reduce recall. Review the generated planner text before running an experiment.
- Planners depend on the VLM that wrote them. Use a separate `planners_dir` for each model so planners are never overwritten.

## Outputs

| Path (configurable under `dataset.output`) | Content |
|---|---|
| `reference_images_dir` | `anomaly_rules/<dataset>.txt` (automatic method) and `planner_generation_logs/` |
| `preprocessed_reference_dir` | the selected references, letterboxed as sent to the models |
| `descriptions_dir` | per-image normal descriptions (JSON) |
| `master_descriptions_dir` | consolidated description per class (JSON) |
| `planners_dir` | `<dataset>/<class>/*_planner.txt` and `<dataset>/performance/*.json` |

The anomaly-detection configuration must set `planners_dir` to the same folder. Detection always uses the newest planner file in each class folder.

## Module reference

| Module | Responsibility |
|---|---|
| `config.py` | loading and validation of the dataset and model configurations (including the cluster merge) |
| `models.py` | VLM, LLM and CLIP wrappers (local Transformers or an OpenAI-compatible server) |
| `reference.py` | CLIP-based reference selection and class partitioning |
| `object_namer.py` | object-part naming and the label cache |
| `descriptor.py` | per-image and master descriptions |
| `anomaly_rules.py` | loading, generating, reviewing and compiling anomaly rules |
| `planner.py` | planner prompting, step-count checks and retries, planner files |
| `utils.py` | prompt rendering, image preprocessing, model lifecycle |
| `logging_utils.py` | planner-generation log configuration |
| `server_manager.py` | compatibility import of the shared vLLM lifecycle manager in `cluster_deployment` |

Prompt templates for this stage are in [`prompts/planner_pipeline/`](../prompts/README.md).
