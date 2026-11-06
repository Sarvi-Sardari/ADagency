"""Load, generate, review, and persist anomaly rules for planner creation."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console

from .config import DatasetConfig, ModelsConfig
from .models import vlm
from .utils import build_prompt, clear_vram
from pipeline_core.model_output import normalize_model_output

logger = logging.getLogger(__name__)

RULE_FIELDS = (
    "id",
    "object_name",
    "category",
    "part_or_region",
    "anomaly_name",
    "visual_description",
)
STATUS_MESSAGE_SECONDS = 3


def _show_status(console: Console, message: str) -> None:
    """Print a persistent stage transition and give an interactive user time to read it."""
    suffix = f" Continuing in {STATUS_MESSAGE_SECONDS} second(s)."
    console.print(f"[yellow]{message}{suffix}[/yellow]")
    if console.is_terminal:
        time.sleep(STATUS_MESSAGE_SECONDS)


def parse_candidate_anomalies(raw_output: Any, source: str) -> list[dict[str, str]]:
    """Parse and validate candidate-anomaly JSON returned by a VLM."""
    if isinstance(raw_output, dict):
        payload = raw_output
    else:
        text = normalize_model_output(str(raw_output))
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} did not return valid anomaly-rule JSON: {exc}") from exc

    rules = payload.get("candidate_anomalies") if isinstance(payload, dict) else None
    if not isinstance(rules, list):
        raise ValueError(f"{source} output must contain a candidate_anomalies list.")

    normalized: list[dict[str, str]] = []
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"{source} rule {index} must be a JSON object.")
        missing = [field for field in RULE_FIELDS[1:] if not str(rule.get(field, "")).strip()]
        if missing:
            raise ValueError(f"{source} rule {index} is missing required fields: {missing}")
        normalized.append({field: str(rule.get(field, "")).strip() for field in RULE_FIELDS})
    return normalized


def load_approved_anomaly_rules(path: Path, key: str | None = None) -> tuple[Any, int | None]:
    """Load canonical candidate rules or a legacy keyed JSON rule value."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Anomaly rules file is not valid JSON: {path}: {exc}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("candidate_anomalies"), list):
        rules = parse_candidate_anomalies(payload, source=str(path))
        if not rules:
            raise ValueError(f"Approved anomaly rules file contains no rules: {path}")
        return {"candidate_anomalies": rules}, len(rules)

    if key and isinstance(payload, dict) and key in payload:
        logger.warning(
            "Using legacy keyed anomaly rules from %s[%r]. Generate a curated canonical rule file to enable exact planner coverage checks.",
            path,
            key,
        )
        return payload[key], None

    available = sorted(payload) if isinstance(payload, dict) else []
    raise ValueError(
        f"Expected a canonical candidate_anomalies document in {path}"
        + (f" or legacy key {key!r}. Available keys: {available}" if key else ".")
    )


def deduplicate_and_renumber(rules: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove semantic duplicates using stable identifying fields and assign A001... IDs."""
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for rule in rules:
        signature = tuple(
            " ".join(rule[field].lower().split())
            for field in ("object_name", "part_or_region", "anomaly_name", "visual_description")
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(dict(rule))

    for index, rule in enumerate(unique, start=1):
        rule["id"] = f"A{index:03d}"
    return unique


def _parse_selection(raw: str, count: int) -> list[int] | None:
    """Parse a selection string into 1-based indices, or None if invalid."""
    value = raw.strip().lower()
    if value in {"", "all"}:
        return list(range(1, count + 1))
    if value in {"none", "skip"}:
        return []

    indices: list[int] = []
    for token in [part.strip() for part in value.split(",") if part.strip()]:
        if "-" in token:
            start, separator, end = token.partition("-")
            if not separator or not start.isdigit() or not end.isdigit() or int(start) > int(end):
                return None
            indices.extend(range(int(start), int(end) + 1))
        elif token.isdigit():
            indices.append(int(token))
        else:
            return None

    if not indices or any(index < 1 or index > count for index in indices):
        return None
    return list(dict.fromkeys(indices))


def _prompt_rule_selection(
    rules: list[dict[str, str]],
    stage_number: int,
    source: str,
    stage_context: str,
    console: Console,
) -> list[dict[str, str]]:
    """
    Show one candidate source in a compact form (anomaly_name + visual_description
    only) and let the user pick which to keep. Mirrors object_namer.py's
    interactive convention: the prompt only appears in an alternate screen
    while input is needed (so it never corrupts the pipeline's live progress
    bar) and disappears immediately after. Every question asked and answer
    given is recorded in the log file, never printed to the console.
    """
    if not rules:
        _show_status(
            console,
            f"Stage {stage_number}/3: {source} returned no valid candidates. Moving to the next method.",
        )
        logger.warning(
            "[Anomaly Rules] Stage %d/3 (%s) returned no valid candidates; moving to the next method.",
            stage_number,
            source,
        )
        return []

    lines = [
        f"  {index:>2}. {rule['anomaly_name']} - {rule['visual_description']}"
        for index, rule in enumerate(rules, start=1)
    ]
    question = f"Select rules to keep (Enter=all, none=skip, 1,3, 2-4): "

    def _ask() -> list[dict[str, str]]:
        while True:
            console.rule(f"[bold cyan]Stage {stage_number}/3: {source}[/bold cyan]")
            console.print(stage_context)
            console.print(f"\n{len(rules)} generated candidate rule(s):")
            for line in lines:
                console.print(line)
            answer = console.input(f"\n{question}").strip()
            indices = _parse_selection(answer, len(rules))
            if indices is None:
                console.print(f"Invalid selection. Choose indices between 1 and {len(rules)}.")
                logger.debug("[Anomaly Rules] Q: %s | Invalid A: %r (reprompting)", question, answer)
                continue
            logger.info(
                "[Anomaly Rules] Q: %s | A: %r -> kept %d/%d", question, answer, len(indices), len(rules)
            )
            return [rules[index - 1] for index in indices]

    if console.is_terminal:
        with console.screen():
            return _ask()
    return _ask()


def _run_rule_prompt(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlm_model: vlm,
    image: Image.Image,
    template_name: str,
    source: str,
    **context: Any,
) -> list[dict[str, str]]:
    prompt = build_prompt(datacfg.prompts_dir, template_name, **context)
    output, elapsed = vlm_model.run_vlm(modelcfg.vlm, [image], prompt)
    logger.info("Generated %s anomaly candidates in %.3fs", source, elapsed)
    response_text = output[1]["content"]
    logger.debug("%s VLM assistant response:\n%s", source, response_text)
    candidates = parse_candidate_anomalies(response_text, source=source)
    logger.info("%s returned %d valid candidate rule(s).", source, len(candidates))
    return candidates


def generate_object_name_candidates(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlm_model: vlm,
    image: Image.Image,
    object_names: list[str],
) -> list[dict[str, str]]:
    return _run_rule_prompt(
        datacfg,
        modelcfg,
        vlm_model,
        image,
        "anomaly_rules_from_objects.j2",
        "Object-name generator",
        object_names=object_names,
    )


def generate_description_candidates(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlm_model: vlm,
    image: Image.Image,
    master_description: dict[str, Any],
) -> list[dict[str, str]]:
    return _run_rule_prompt(
        datacfg,
        modelcfg,
        vlm_model,
        image,
        "anomaly_rules_from_description.j2",
        "Master-description generator",
        master_description=master_description,
    )


def generate_user_input_candidates(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlm_model: vlm,
    image: Image.Image,
    user_input: str,
) -> list[dict[str, str]]:
    return _run_rule_prompt(
        datacfg,
        modelcfg,
        vlm_model,
        image,
        "anomaly_rules_from_user_input.j2",
        "User-input generator",
        user_input=user_input,
    )


def write_approved_anomaly_rules(path: Path, rules: list[dict[str, str]]) -> Path:
    approved = deduplicate_and_renumber(rules)
    if not approved:
        raise ValueError("At least one anomaly rule must be approved before saving.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"candidate_anomalies": approved}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def anomaly_rules_output_path(datacfg: DatasetConfig, dataset_name: str) -> Path:
    """Location of the dataset's compiled (automatic-method) anomaly-rules file."""
    return datacfg.output.reference_images_dir / "anomaly_rules" / f"{dataset_name}.txt"


def _load_latest_master_descriptions(master_root: Path, dataset_name: str, class_labels: list[str]) -> dict[str, Any]:
    descriptions: dict[str, Any] = {}
    for label in class_labels:
        class_dir = master_root / dataset_name / label
        files = sorted(
            (path for path in class_dir.glob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not files:
            raise FileNotFoundError(f"No master-description JSON found in {class_dir}")
        descriptions[label] = json.loads(files[0].read_text(encoding="utf-8"))
    return descriptions


def _resolve_user_input_text(console: Console) -> str | None:
    """
    Ask the user for the optional user-intent text used by the third
    (user-input) rule generator. A blank answer skips this optional third
    method. The configured JSON is deliberately reserved for the final
    fallback only, after all automatic methods have yielded no approved rules.
    """
    question = "Optional user-input method: describe the required normal condition or use-case logic (leave blank to skip this method): "

    def _ask() -> str:
        return console.input(f"\n{question}").strip()

    def _show_and_ask() -> str:
        console.rule("[bold cyan]Stage 3/3: User-input-based rule generation[/bold cyan]")
        console.print(
            "This method creates rules only from the condition you enter. "
            "Leave it blank to skip this method; JSON is used only if no automatic method produces approved rules."
        )
        return _ask()

    if console.is_terminal:
        with console.screen():
            answer = _show_and_ask()
    else:
        answer = _show_and_ask()

    # Arrow keys can be sent as terminal escape sequences if the user tries to
    # scroll while this prompt is active. They are not meaningful user intent.
    if answer and not re.fullmatch(r"(?:\x1b\[[A-D])+", answer):
        logger.info("[Anomaly Rules] Q: %s | A: %r", question, answer)
        return answer

    if answer:
        logger.info("[Anomaly Rules] Q: %s | A: %r (terminal navigation input ignored)", question, answer)
    else:
        logger.info("[Anomaly Rules] Q: %s | A: '' (blank) -> skipped user-input method.", question)
    _show_status(
        console,
        "Stage 3/3: User-input-based generation was skipped because no text was entered. No rules were generated by this method.",
    )
    logger.warning("[Anomaly Rules] Stage 3/3 (User-input generator) was blank; no rules generated by this method.")
    return None


def _generate_and_select(
    stage_number: int,
    source: str,
    stage_context: str,
    generator,
    console: Console,
) -> list[dict[str, str]]:
    console.print(f"[cyan]Stage {stage_number}/3: trying {source}...[/cyan]")
    logger.info("[Anomaly Rules] Stage %d/3: starting %s.", stage_number, source)
    try:
        candidates = generator()
    except ValueError as exc:
        _show_status(
            console,
            f"Stage {stage_number}/3: {source} produced no valid output. Moving to the next method.",
        )
        logger.warning(
            "[Anomaly Rules] Stage %d/3 (%s) returned no valid candidates (%s); moving to the next method.",
            stage_number,
            source,
            exc,
        )
        return []
    return _prompt_rule_selection(candidates, stage_number, source, stage_context, console)


def _load_json_fallback(datacfg: DatasetConfig) -> tuple[str, int | None]:
    """Load the legacy configured JSON rule text after automatic generation has no approval."""
    if not datacfg.anomaly_rules.key:
        raise ValueError(
            "Automatic anomaly-rule generation produced no approved rules and "
            "dataset.anomaly_rules.key is missing, so the configured JSON fallback cannot be used."
        )

    try:
        payload = json.loads(datacfg.anomaly_rules.file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Automatic anomaly-rule generation produced no approved rules and JSON fallback "
            f"{datacfg.anomaly_rules.file} could not be read: {exc}"
        ) from exc

    if not isinstance(payload, dict) or datacfg.anomaly_rules.key not in payload:
        raise KeyError(
            f"Automatic anomaly-rule generation produced no approved rules and fallback key "
            f"{datacfg.anomaly_rules.key!r} was not found in {datacfg.anomaly_rules.file}."
        )

    value = payload[datacfg.anomaly_rules.key]
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return text, None


def _persist_if_any(output_path: Path, approved: list[dict[str, str]]) -> list[dict[str, str]]:
    """Dedup/renumber and write only when there's something to write, so an
    empty intermediate stage never trips write_approved_anomaly_rules()'s
    empty-list guard."""
    deduped = deduplicate_and_renumber(approved)
    if deduped:
        write_approved_anomaly_rules(output_path, deduped)
    return deduped


def resolve_anomaly_rules(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    dataset_name: str,
    img_names: list[str],
    console: Console,
) -> tuple[str, int | None]:
    """
    Resolve the ANOMALY_RULES text (and, when known, the exact expected
    planner-step count) consumed by planner_generator(). This is the single
    integration point called from the planner-generation pipeline right
    after master descriptions are synthesised -- anomaly-rule curation is
    not a separate step run by the user.

    method == "manual": use datacfg.anomaly_rules.file[key] directly. No VLM
    calls, no compiled txt file, no exact-coverage check (expected_rule_count
    is None).

    method == "automatic": reuse the dataset's compiled anomaly-rules txt
    file unless datacfg.anomaly_rules.overwrite is True or it doesn't exist
    yet. Otherwise, candidates are generated and interactively reviewed (in
    a compact anomaly_name/visual_description view) in order -- master
    description, then object names, then optional user input -- and
    incrementally compiled into that txt file with sequential, non-duplicate
    ids in the order rules were added.
    """
    output_path = anomaly_rules_output_path(datacfg, dataset_name)

    if datacfg.anomaly_rules.method == "manual":
        logger.info(
            "[Anomaly Rules] method=manual -> using %s[%r] directly (no generation, no compiled file).",
            datacfg.anomaly_rules.file, datacfg.anomaly_rules.key,
        )
        payload = json.loads(datacfg.anomaly_rules.file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or datacfg.anomaly_rules.key not in payload:
            raise KeyError(
                f"Anomaly rule key {datacfg.anomaly_rules.key!r} was not found in "
                f"{datacfg.anomaly_rules.file}."
            )
        value = payload[datacfg.anomaly_rules.key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        return text, None

    # method == "automatic"
    if not datacfg.anomaly_rules.overwrite and output_path.is_file():
        logger.info(
            "[Anomaly Rules] method=automatic, overwrite=False, existing compiled file found -> reusing %s",
            output_path,
        )
        payload, count = load_approved_anomaly_rules(output_path)
        text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False)
        return text, count

    logger.info(
        "[Anomaly Rules] method=automatic -> generating rules (overwrite=%s, existing_present=%s) -> %s",
        datacfg.anomaly_rules.overwrite, output_path.is_file(), output_path,
    )

    master_descriptions = _load_latest_master_descriptions(
        datacfg.output.master_descriptions_dir, dataset_name, datacfg.class_labels,
    )
    reference_path = datacfg.root_dir / "train" / "good" / datacfg.object_labels.label_example
    image = Image.open(reference_path).convert("RGB")

    clear_vram()
    vlm_model = vlm.load(modelcfg)

    approved: list[dict[str, str]] = []

    approved.extend(_generate_and_select(
        1,
        "Master-description-based generation",
        "Candidate anomaly rules are inferred from the synthesised normal master description.",
        lambda: generate_description_candidates(datacfg, modelcfg, vlm_model, image, master_descriptions),
        console,
    ))
    approved = _persist_if_any(output_path, approved)

    approved.extend(_generate_and_select(
        2,
        "Object-name-based generation",
        "Candidate anomaly rules are inferred from the object labels selected earlier in this pipeline.",
        lambda: generate_object_name_candidates(datacfg, modelcfg, vlm_model, image, img_names),
        console,
    ))
    approved = _persist_if_any(output_path, approved)

    user_input_text = _resolve_user_input_text(console)
    if user_input_text:
        approved.extend(_generate_and_select(
            3,
            "User-input-based generation",
            "Candidate anomaly rules are inferred only from the normal-condition or use-case text you supplied.",
            lambda: generate_user_input_candidates(datacfg, modelcfg, vlm_model, image, user_input_text),
            console,
        ))
        approved = _persist_if_any(output_path, approved)

    if not approved:
        logger.warning(
            "[Anomaly Rules] No automatic method produced approved rules; reverting to configured JSON fallback %s[%r].",
            datacfg.anomaly_rules.file,
            datacfg.anomaly_rules.key,
        )
        console.print(
            "[yellow]No automatic anomaly-rule method produced approved rules. "
            "Reverting to anomaly_definition.json for this planner run.[/yellow]"
        )
        return _load_json_fallback(datacfg)

    logger.info("[Anomaly Rules] Compiled %d approved rules -> %s", len(approved), output_path)
    return json.dumps({"candidate_anomalies": approved}, indent=2, ensure_ascii=False), len(approved)
