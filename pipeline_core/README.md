# `pipeline_core` package

Small, model-independent helpers shared by all pipeline stages. The modules import no model framework and contain no stage-specific logic, so the same behavior (atomic file writes, image letterboxing, prompt rendering, output cleanup) applies everywhere.

| Module | Public functions | Purpose |
|---|---|---|
| `artifacts.py` | `atomic_write_text`, `read_jsonl` | Write text through a temporary file and an atomic rename, so an interrupted write never leaves a partial file at the final path. Read JSONL files, skipping malformed lines with a warning. |
| `imaging.py` | `list_images`, `preprocess_image_paths`, `letterbox` | Find images (`.jpg .jpeg .png .gif .bmp .webp`), batch images for an encoder, and resize within configured `[min, max]` bounds with aspect-preserving black padding (`models.vlm.image_processing`). |
| `prompting.py` | `get_jinja_env`, `build_prompt`, `build_prompt_blocks` | Load and cache Jinja templates from a prompt directory and render them. |
| `model_output.py` | `normalize_model_output`, `normalize_step_decision_layout` | Remove model-family-specific formatting (e.g. reasoning blocks, layout variations) before strict parsers read step answers. |
| `logging.py` | `configure_logging`, `open_terminal_stream`, `capture_terminal_output` | Common log handlers and capture of terminal output into run log files. |

Other packages import these helpers directly, for example `from pipeline_core.artifacts import atomic_write_text`. New stages should reuse them instead of re-implementing file publication or image preprocessing, so that results stay comparable across stages.
