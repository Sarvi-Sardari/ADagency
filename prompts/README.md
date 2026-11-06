# Prompts, anomaly rules and label caches

Every instruction sent to a model is a Jinja template in this folder. No prompt text is hard-coded in Python. Changing a template changes the experiment, so run a changed prompt in a new runs directory.

## Layout

```text
prompts/
├── anomaly_definition.json        # curated anomaly rules per dataset (manual rules)
├── planner_pipeline/              # templates for offline planner generation
├── anomaly_detection_pipeline/    # templates for detection, rerun, cropping, localization
├── object_labels/                 # cached object-part names per dataset
└── class_labels/                  # cached class descriptions for multi-class datasets
```

## `anomaly_definition.json`

A JSON object with one entry per dataset key (for example `breakfast_box`, `bottle`, `toothbrush`). Each entry lists the anomaly types the planner must cover. Planner generation reads it with `anomaly_rules.method: manual` and `anomaly_rules.key: <key>`. With `method: automatic` it serves only as a fallback for user input.

## `planner_pipeline/` (used by `run_planner_generation.py`)

| Template | Purpose |
|---|---|
| `object_namer.j2` | name the visible parts of the object in a normal example image |
| `normal_descriptor.j2` | describe one normal reference image (VLM) |
| `object_summarizer.j2` | merge the per-image descriptions into one master description per class (LLM) |
| `description_text_summarizer.j2` | legacy summarizer; not referenced by the current code |
| `anomaly_rules_from_description.j2` | propose anomaly rules from the master description |
| `anomaly_rules_from_objects.j2` | propose anomaly rules from the object names |
| `anomaly_rules_from_user_input.j2` | turn free-text user input into structured rules |
| `planner.j2` | write the numbered YES/NO inspection planner (VLM) |

All templates listed above except `description_text_summarizer.j2` must exist in the configured `prompts_dir`; the loader checks this at start-up.

## `anomaly_detection_pipeline/` (used by detection and later stages)

| Template | Used by | Purpose |
|---|---|---|
| `decision_maker_v2_original_compact.j2` | detection (default) | answer every planner step and give the final decision |
| `decision_maker_secondary.j2` | detection with `secondary_classification` | step answers only; the secondary class is derived from them in Python |
| `rerun_step.j2` | rerun | re-ask one planner step |
| `image_crop.j2` | object cropping | locate the whole object (`bbox_2d`) |
| `bounding_box_localizer.j2` | mask generation | locate the anomaly with bounding boxes |
| `label_generator.j2` | multi-class datasets | generate class descriptions from `class_examples` |

A different decision template is selected with `decision.prompt_template` in the detection configuration, with `--decision-prompt-template` on the command line, or with `experiment.decision_prompt_sequence` in an ablation. Templates must keep the output contract the parsers expect: one `[STEP n] YES|NO` record per planner step, with confidence evidence, followed by the final decision line.

## Caches

| Folder | File | Written by | Read by |
|---|---|---|---|
| `object_labels/` | `<dataset>_object_labels.txt` (one label per line) | planner generation (`object_labels.use_previous_labels: false`, or no cache yet) | planner generation, object cropping |
| `class_labels/` | `<dataset>_data_labels.txt` | detection on first use of a multi-class dataset (interactive confirmation) | detection |

Delete a cache file, or set `use_previous_labels: false`, to generate it again.
