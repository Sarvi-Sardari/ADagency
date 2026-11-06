# Planner-generation artifacts

This folder stores the outputs of planner generation (`run_planner_generation.py`). The anomaly-detection stage reads only the **planners**; the other folders document how each planner was produced.

## Layout

```text
reference_images/
├── planners/                     # planners_dir values used by the detection configs
│   └── <planner set>/<dataset>/
│       ├── <class>/<class>_<timestamp>_planner.txt   # numbered YES/NO inspection steps
│       └── performance/performance_<timestamp>.json   # generation timing
├── anomaly_rules/<dataset>.txt   # compiled rules (anomaly_rules.method: automatic)
├── descriptions/<dataset>/<class>/          # per-image normal descriptions (VLM)
├── master_descriptions/<dataset>/<class>/   # merged description per class (LLM)
├── preprocessed/<dataset>/                  # selected references as sent to the models
└── planner_generation_logs/<date>/          # planner-generation logs
```

`<class>` is `good` for single-class datasets, or the class label for multi-class datasets. When a class folder contains several planner files, detection uses the newest one.

## Included planner sets

| Folder | Content | Used by |
|---|---|---|
| `planners/mvtec_loco_8B_planners_v2/` | MVTec LOCO AD, Qwen3-VL-8B | `configs/anomaly_detection_configs_mvtec_loco.yaml` |
| `planners/ibadas_8B_planners_v2/` | custom box datasets, Qwen3-VL-8B | `configs/anomaly_detection_configs_ibadas_task_02.yaml` |
| `planners/mvtec_ad_4B_planners/` | MVTec AD, 4B model (earlier experiment) | – |
| `planners/<dataset>/` | earlier planners written directly into `planners/` | – |

Other planner sets referenced by the provided configurations (for example the MVTec AD 8B planners) are not included. Generate them with `run_planner_generation.py`, or point `planners_dir` to an included set.

## Notes

- Planners depend on the VLM that wrote them and on the anomaly rules. Keep one planner set per model and rule version, and never overwrite a set that existing results were produced with.
- Generated logs and performance files may contain absolute paths from the machine that produced them. They are informational only and are not read by the pipeline.
