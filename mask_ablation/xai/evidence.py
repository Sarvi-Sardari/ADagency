"""Build first-pass and rerun XAI samples from persisted artifacts."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

@dataclass
class HeatmapSample:
    image_path: str
    question: str
    output_text: str
    step: int
    predicted_token: str    # "YES" or "NO" -- which answer this reasoning step gave #Include NO STEP
    row_index: int
    decision_filepath: str
    probs_data_filepath: str
    planner_filepath: str
    dataset_name: str        # from row["dataset"] -- needed to resolve the reference path the same way resolve_image_path() resolves the candidate path
    label: str                # from row.get("label"), "anomaly" or "good" -- picks ref_anomaly vs ref_good
    # Fields used only for consolidated final-reason samples.
    is_final_summary: bool = False  # True if this is the consolidated final reason/details, False for individual YES steps
    original_yes_steps: List[int] = None  # When is_final_summary=True, stores which YES steps were consolidated
    # The pipeline integration supplies the exact reference selected upstream.
    # Older notebook callers may omit it and retain the legacy lookup fallback.
    reference_image_path: str | None = None
    # Auditable provenance: "first_pass" or "rerun".
    evidence_source: str = "first_pass"

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows

def parse_prob_data_file(path: str) -> List[int]:
    """Extracts step numbers where the token is 'YES'.
       Returns a list of the step numbers which had the 'YES'."""
    step_tokens = {}                                            #Include NO STEP
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line.strip())
                token = str(obj.get("token", "")).strip().upper()
                if token in ("YES", "NO"):
                    step_tokens[int(obj["step"])] = token
    return step_tokens

def parse_planner_file(path: str) -> Dict[int, str]:
    """Maps step numbers to planner questions."""
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(r"Step\s+(\d+)\s*:\s*(.*?)(?=\n\s*Step\s+\d+\s*:|\Z)", flags=re.IGNORECASE | re.DOTALL)
    return {int(m.group(1)): m.group(2).strip() for m in pattern.finditer(text)}            #group(1) is the text matched with (\d+) in regex and group(2) is text matched with (.*?) in regex

def parse_output_file(path: str) -> Dict[int, str]:
    """Maps step numbers to their output explanation reason."""
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"\[?\s*STEP\s+(\d+)\s*\]?\s+(YES|NO)\s*(?:->|:)\s*(.*?)(?=\n\s*\[?\s*STEP|\n\s*Anomaly;|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {int(m.group(1)): m.group(3).strip() for m in pattern.finditer(text)}

def parse_final_reason_details(path: str) -> Dict[str, str]:
    """Extract the consolidated final anomaly reason and details.

    Returns an empty dictionary when the expected section is absent.
    
    Example format:
    Anomaly; reason=The candidate image shows clear anomalies...; details=Dirt/grime, discoloration/liquid marks...
    """
    text = Path(path).read_text(encoding="utf-8")
    
    # Pattern to match: Anomaly; reason=<text>; details=<text>
    pattern = re.compile(
        r"Anomaly\s*;\s*reason\s*=\s*(.*?)(?=\s*;\s*details\s*=|\Z).*?;\s*details\s*=\s*(.*?)(?=\n|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    
    match = pattern.search(text)
    if match:
        reason = match.group(1).strip()
        details = match.group(2).strip()
        return {"reason": reason, "details": details}
    return {}

def extract_yes_steps_from_file(path: str) -> List[int]:
    """Return the sorted step numbers whose saved prediction is ``YES``."""
    step_tokens = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line.strip())
                token = str(obj.get("token", "")).strip().upper()
                if token in ("YES", "NO"):
                    step_tokens[int(obj["step"])] = token
    return sorted([step for step, token in step_tokens.items() if token == "YES"])

def resolve_image_path(decision_filepath: str, dataset_name: str, base_dir: str, label: str = "anomaly") -> str:
    """Transforms text file path to the processed image path. `label` picks the
    correct preproc subfolder -- anomaly rows live under cand_anomaly, good
    rows live under cand_good; using the wrong one gives a FileNotFoundError
    since the two subfolders never share filenames."""
    decision_path = Path(base_dir) / decision_filepath
    parts = decision_path.parts
    dataset_index = parts.index(dataset_name)
    dataset_root = Path(*parts[: dataset_index + 1])
    
    output_name = decision_path.name
    image_name = output_name[:-4] if output_name.endswith(".txt") else output_name
    cand_subdir = "cand_anomaly" if label == "anomaly" else "cand_good"
    return str(dataset_root / "preproc" / cand_subdir / f"processed_{image_name}")      #Include NO STEPS (GOOD Files)

def get_reference_path_for_category(decision_filepath: str, dataset_name: str, base_dir: str, label: str = "anomaly", log = print) -> str | None:
    """Mirrors resolve_image_path()'s path derivation, but resolves the
    REFERENCE image instead of the candidate. Reference filenames encode the
    candidate name they were chosen for (<cand_name>.png_ref_<ref_name>.png),
    so this can't be a direct path join like resolve_image_path() -- it has
    to glob for the candidate's name as a prefix and read back whichever
    reference file the upstream retrieval step actually picked. Returns None
    if no matching reference file exists, which is the expected/safe case
    make_baseline_image() already falls back on."""
    #decision_path = Path(base_dir) / decision_filepath
    decision_path = Path(decision_filepath)
    parts = decision_path.parts
    dataset_index = parts.index(dataset_name)
    dataset_root = Path(*parts[: dataset_index + 1])

    output_name = decision_path.name
    image_name = output_name[:-4] if output_name.endswith(".txt") else output_name
    processed_image_name = f"processed_{image_name}"

    ref_subdir = "ref_anomaly" if label == "anomaly" else "ref_good"
    ref_dir = dataset_root / "preproc" / ref_subdir

    if not ref_dir.is_dir():
        log(f"  [ref_lookup] MISSING DIR: {ref_dir} does not exist")
        return None

    #matches = list(ref_dir.glob(f"{processed_image_name}*_ref_*"))
    pattern = f"{processed_image_name}*_ref_*"
    matches = list(ref_dir.glob(pattern))
    sample_files = [p.name for p in list(ref_dir.iterdir())[:5]]
    log(f"  [ref_lookup] dir={ref_dir} pattern={pattern!r} -> {len(matches)} matches; "
        f"sample files in dir: {sample_files}")
    if not matches:
        return None
    
    # A reference file named "..._ref_<candidate_name>" means the retrieval step
    # picked the CANDIDATE ITSELF as its own reference (can happen for "good"
    # rows when the nearest-neighbor search returns the same image). Using it
    # as a baseline gives phi(image, baseline, mask) ~= image regardless of mask
    # value, which silently zeroes out the importance signal in
    # find_keywords_from_precomputed and causes every keyword to get skipped --
    # exactly the "no keywords cleared threshold" case with zero explanation.
    # Catch it here and fall back to None (-> blur baseline) instead.
    ref_name_part = matches[0].name.split("_ref_", 1)[-1]
    if ref_name_part == image_name:
        log(f"  [ref_lookup] REJECTED -- reference is the candidate itself "
            f"({matches[0].name}); falling back to blur baseline")
        return None
    
    if len(matches) > 1:
        # Shouldn't happen if retrieval picked exactly one reference per candidate,
        # but don't silently guess -- surface it so a naming collision gets noticed
        # rather than quietly using the wrong reference.
        print(f"  [WARN] multiple reference matches for {processed_image_name}, using first: {[m.name for m in matches]}")

    return str(matches[0])


# %%
def build_heatmap_samples(summary_jsonl_path: str, base_dir: str, mask_generation_mode: str = "yes_steps") -> List[HeatmapSample]:
    """
    MODIFIED: Returns a list of HeatmapSample objects.
    
    Parameters:
    -----------
    mask_generation_mode : str, default "yes_steps"
        - "yes_steps": Original behavior - generates one sample per YES step
        - "final_reason_details": ADDED - generates one synthetic sample per image
                                  using the consolidated final reason/details instead
                                  of individual YES steps
    """
    
    rows = read_jsonl(summary_jsonl_path)
    samples = []

    for row_index, row in enumerate(rows):
        if row.get("label") in ("anomaly", "good"): #Include NO STEPS (GOOD FILES)
            base_p = Path(base_dir)
            decision_path = base_p / row["decision_filepath"]
            probs_path = base_p / row["probs_data_filepath"]
            planner_path = base_p / row["planner_filepath"]

            #yes_steps = parse_prob_data_file(str(probs_path))
            step_tokens = parse_prob_data_file(str(probs_path))     #Include NO STEP
            planner_questions = parse_planner_file(str(planner_path))
            output_by_step = parse_output_file(str(decision_path))
            image_path = resolve_image_path(row["decision_filepath"], row["dataset"], base_dir, label=row.get("label"))

            if mask_generation_mode == "final_reason_details":
                # New mode - extract final reason/details instead of YES steps
                final_info = parse_final_reason_details(str(decision_path))
                if not final_info or not final_info.get("reason") or not final_info.get("details"):
                    print(f"  [row {row_index}] SKIPPED -- no valid final reason/details in {decision_path.name}")
                    continue
                
                # Extract which YES steps were consolidated into this final decision
                yes_steps = extract_yes_steps_from_file(str(probs_path))
                
                if not yes_steps:
                    print(f"  [row {row_index}] SKIPPED -- no YES steps found in {probs_path.name}")
                    continue
                
                # Build clean output text:
                # 1. Concatenate reason + details (without structural prefixes)
                # 2. Strip boilerplate patterns that add noise without visual grounding
                raw_output = f"{final_info['reason']} {final_info['details']}"
                final_output = raw_output
                
                # Build the question from the actual YES-step planner questions
                # instead of a hardcoded string. This grounds the VLM's attention
                # on the same visual regions it was inspecting when it answered YES.
                yes_step_questions = [
                    planner_questions[step]
                    for step in yes_steps
                    if step in planner_questions
                ]
                if yes_step_questions:
                    # Join all YES-step questions with a separator so the model sees
                    # the full set of anomaly checks that were positive.
                    final_question = " ".join(yes_step_questions)
                else:
                    # Fallback if planner questions are somehow missing
                    final_question = "Comprehensive Anomaly Analysis"
                
                samples.append(HeatmapSample(
                    image_path=image_path,
                    question=final_question,
                    output_text=final_output,
                    step=999,  # Special step marker for "final_reason_details" mode
                    predicted_token="YES",  # By definition, final summary only created for anomaly cases
                    row_index=row_index,
                    decision_filepath=str(decision_path),
                    probs_data_filepath=str(probs_path),
                    planner_filepath=str(planner_path),
                    dataset_name=row["dataset"],
                    label=row.get("label"),
                    is_final_summary=True,
                    original_yes_steps=yes_steps,
                ))
                print(f"  [row {row_index}] Created final_reason_details sample (consolidated from YES steps: {yes_steps})")
                print(f"    question: \"{final_question[:100]}{'...' if len(final_question) > 100 else ''}\"")
                print(f"    output_text: \"{final_output[:120]}{'...' if len(final_output) > 120 else ''}\"")
            
            else:
                # ORIGINAL MODE: yes_steps - process each YES step individually
                missing = [s for s, tok in step_tokens.items() if tok == "NO" and not (s in planner_questions and s in output_by_step)]
                if missing:
                    print(f"  [row {row_index}] NO steps dropped for missing planner/output entries: {missing}") #Include NO STEPS

                for step, token in step_tokens.items():             #Include NO STEP
                    if step in planner_questions and step in output_by_step:
                        samples.append(HeatmapSample(
                            image_path=image_path,
                            question=planner_questions[step],
                            output_text=output_by_step[step],
                            step=step,
                            predicted_token=token,
                            row_index=row_index,
                            decision_filepath=str(decision_path),
                            probs_data_filepath=str(probs_path),
                            planner_filepath=str(planner_path),
                            dataset_name=row["dataset"],
                            label=row.get("label"),
                            is_final_summary=False,
                            original_yes_steps=None,
                         ))
    
    print(f"\n[build_heatmap_samples] Created {len(samples)} samples using mode='{mask_generation_mode}'")
    return samples


def _img_key(entry: Dict[str, Any]) -> str:
    """Mirrors pipeline_project/anomaly_detection/rerun.py's _img_key(): the
    same image identity used across all_data_summary.jsonl, rerun_result.jsonl
    and metrics.py, so joins between them line up."""
    return entry.get("img_name") or Path(entry["decision_filepath"]).name.removesuffix(".txt")


def build_rerun_heatmap_samples(run_dir: str, dataset: str, mask_generation_mode: str = "yes_steps") -> List[HeatmapSample]:
    """Build heatmap samples from rerun-flipped evidence.

    images the FIRST pass predicted "good" but that
    pipeline_project/anomaly_detection/rerun.py's rerun_low_confidence_predictions()
    flipped to "anomaly" after re-asking their lowest-confidence planner steps.

    This does NOT invent or randomly assign evidence: for each flipped image it
    reads back the exact per-step {decision, reason, details, step, is_anomaly}
    JSON that rerun.py wrote to
    <run_dir>/<dataset>/rerun_results/texts/<label_folder>/<img_name>_rerun_steps.json,
    and only keeps the step(s) that actually triggered the flip (is_anomaly=True).
    rerun.py's _rerun_image() stops at the FIRST "Yes", so there is normally
    exactly one such step per image, but this handles multiple defensively.

    rerun_result.jsonl does not carry decision_filepath/probs_data_filepath/
    planner_filepath directly (see rerun.py's module docstring) -- these, plus
    the image's ORIGINAL (pre-rerun) "label", are recovered by joining back to
    all_data_summary.jsonl via the (label_folder, img_name) key rerun.py itself
    uses. The ORIGINAL label is what must be used for cand_good/cand_anomaly
    (and ref_good/ref_anomaly) folder resolution below -- the image was
    preprocessed into that folder BEFORE rerun ever ran, so the post-rerun
    "anomaly" outcome would resolve to a folder the file was never written to.
    """
    dataset_dir = Path(run_dir) / dataset
    summary_path = dataset_dir / "all_data_summary.jsonl"
    rerun_result_path = dataset_dir / "rerun_results" / "rerun_result.jsonl"

    if not rerun_result_path.is_file():
        print(f"[build_rerun_heatmap_samples] No rerun_result.jsonl at {rerun_result_path}; "
              f"include_rerun=True but there is nothing to add.")
        return []

    base_rows = read_jsonl(str(summary_path))
    base_by_key = {(row.get("label_folder"), _img_key(row)): (row_index, row) for row_index, row in enumerate(base_rows)}

    rerun_rows = read_jsonl(str(rerun_result_path))
    base_dir_str = str(Path(run_dir))

    samples: List[HeatmapSample] = []
    for rerun_row in rerun_rows:
        if rerun_row.get("rerun_status") != "flipped_to_anomaly":
            continue

        key = (rerun_row.get("label_folder"), rerun_row.get("img_name"))
        base_entry = base_by_key.get(key)
        if base_entry is None:
            print(f"  [rerun join] SKIPPED -- no matching all_data_summary.jsonl row for {key}")
            continue
        row_index, base_row = base_entry

        rerun_steps_filepath = rerun_row.get("rerun_steps_filepath")
        if not rerun_steps_filepath:
            print(f"  [rerun join] SKIPPED -- flipped entry {key} has no rerun_steps_filepath")
            continue

        step_results = json.loads(Path(rerun_steps_filepath).read_text(encoding="utf-8"))
        # Only the step(s) that actually flipped the decision carry real anomaly
        # evidence -- earlier bottom-N steps that came back "No" are not
        # anomaly evidence and must not be fabricated into a sample.
        anomaly_steps = [s for s in step_results if s.get("is_anomaly")]
        if not anomaly_steps:
            print(f"  [rerun join] SKIPPED -- {key} is flipped_to_anomaly but no step in "
                  f"{rerun_steps_filepath} has is_anomaly=True (unexpected)")
            continue

        original_label = base_row.get("label")   # pre-rerun label; decides cand_good/cand_anomaly
        image_path = resolve_image_path(base_row["decision_filepath"], base_row["dataset"], base_dir_str, label=original_label)
        decision_filepath = base_row["decision_filepath"]
        probs_data_filepath = base_row["probs_data_filepath"]
        planner_filepath = base_row["planner_filepath"]

        if mask_generation_mode == "final_reason_details":
            # Consolidate every rerun step that fired (normally just one) into a
            # single synthetic sample, exactly like build_heatmap_samples()'s
            # final_reason_details mode does for the first-pass decision text.
            reasons = [s.get("reason", "").strip() for s in anomaly_steps if s.get("reason")]
            details = [s.get("details", "").strip() for s in anomaly_steps if s.get("details")]
            questions = [s.get("step", "").strip() for s in anomaly_steps if s.get("step")]
            if not reasons and not details:
                print(f"  [rerun join] SKIPPED -- {key} flipped step(s) have no reason/details text")
                continue
            samples.append(HeatmapSample(
                image_path=image_path,
                question=" ".join(questions) if questions else "Comprehensive Anomaly Analysis (rerun)",
                output_text=f"{' '.join(reasons)} {' '.join(details)}".strip(),
                step=999,
                predicted_token="YES",
                row_index=row_index,
                decision_filepath=decision_filepath,
                probs_data_filepath=probs_data_filepath,
                planner_filepath=planner_filepath,
                dataset_name=base_row["dataset"],
                label=original_label,
                is_final_summary=True,
                original_yes_steps=[s["step_number"] for s in anomaly_steps if "step_number" in s],
            ))
            print(f"  [rerun join] Created final_reason_details sample for {key} "
                  f"(rerun steps: {[s.get('step_number') for s in anomaly_steps]})")
        else:
            # yes_steps mode: one sample per rerun step that fired, using the
            # ACTUAL rerun question/reason/details -- never randomly assigned.
            for s in anomaly_steps:
                reason = (s.get("reason") or "").strip()
                details_text = (s.get("details") or "").strip()
                output_text = f"{reason} {details_text}".strip()
                if not output_text:
                    print(f"  [rerun join] SKIPPED step {s.get('step_number')} for {key} -- empty reason/details")
                    continue
                samples.append(HeatmapSample(
                    image_path=image_path,
                    question=s.get("step", ""),
                    output_text=output_text,
                    step=int(s.get("step_number", 0)),
                    predicted_token="YES",
                    row_index=row_index,
                    decision_filepath=decision_filepath,
                    probs_data_filepath=probs_data_filepath,
                    planner_filepath=planner_filepath,
                    dataset_name=base_row["dataset"],
                    label=original_label,
                    is_final_summary=False,
                    original_yes_steps=None,
                ))

    print(f"\n[build_rerun_heatmap_samples] Created {len(samples)} rerun-derived samples from {rerun_result_path}")
    return samples


def resolve_dataset_jsonl_and_base_dir(run_dir: str, dataset: str) -> tuple[str, str]:
    """Resolves (summary_jsonl_path, base_dir) from a run_dir + dataset name,
    matching the convention configs/mask_generation_configs.yaml and
    configs/metrics_configs.yaml already use: <run_dir>/<dataset>/all_data_summary.jsonl."""
    dataset_dir = Path(run_dir) / dataset
    summary_jsonl_path = str(dataset_dir / "all_data_summary.jsonl")
    return summary_jsonl_path, str(Path(run_dir))


def build_heatmap_samples_v2(
    run_dir: str,
    dataset: str,
    mask_generation_mode: str = "yes_steps",
    include_rerun: bool = False,
) -> List[HeatmapSample]:
    """
    Primary sample-loading entry point for this engine (used by run_pipeline()
    below). Wraps build_heatmap_samples() for the first-pass
    all_data_summary.jsonl evidence, and -- when include_rerun=True -- appends
    the rerun-derived samples from build_rerun_heatmap_samples(), which are
    built strictly from the real rerun_result.jsonl / *_rerun_steps.json
    evidence (see that function's docstring), never fabricated.
    """
    summary_jsonl_path, base_dir = resolve_dataset_jsonl_and_base_dir(run_dir, dataset)
    samples = build_heatmap_samples(summary_jsonl_path, base_dir, mask_generation_mode=mask_generation_mode)

    if include_rerun:
        rerun_samples = build_rerun_heatmap_samples(run_dir, dataset, mask_generation_mode=mask_generation_mode)
        samples.extend(rerun_samples)
        print(f"[build_heatmap_samples_v2] include_rerun=True -> added {len(rerun_samples)} rerun-derived samples "
              f"(total now {len(samples)})")

    return samples


# %%
