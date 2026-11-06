"""Standalone entry point to (re)resolve a dataset's anomaly rules without
running the rest of the planner-generation pipeline.

Delegates to the exact same anomaly_rules.resolve_anomaly_rules() used inside
run_planner_generation.py, so behaviour (method/overwrite handling, compact
interactive review, logging) is identical however it's invoked. Anomaly-rule
curation is normally just one step of that pipeline -- use this only when you
want to (re)generate the compiled rules file on its own, e.g. while iterating
on class descriptions between full pipeline runs.
"""

import argparse
from pathlib import Path

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent

from planner_generation.anomaly_rules import resolve_anomaly_rules
from planner_generation.config import load_dataset_config, load_models_config
from planner_generation.logging_utils import setup_logging

CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_object_names(dataset_name: str) -> list[str]:
    path = PROJECT_ROOT / "prompts" / "object_labels" / f"{dataset_name}_object_labels.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Object-label cache not found: {path}. Run planner generation through "
            "object-name selection at least once before generating anomaly rules standalone."
        )
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Object-label cache is empty: {path}")
    return names


def run(dataset_config_path: str | None, model_config_path: str | None) -> None:
    datacfg = load_dataset_config(dataset_config_path or CONFIG_DIR / "dataset_configs.yaml")
    modelcfg = load_models_config(model_config_path or CONFIG_DIR / "model_configs.yaml")
    console = Console()
    log_file = setup_logging(
        f"anomaly_rules_{datacfg.name}",
        log_dir=datacfg.output.reference_images_dir / "planner_generation_logs",
        console=console,
    )

    img_names = _load_object_names(datacfg.name)
    anomaly_rules_text, expected_rule_count = resolve_anomaly_rules(
        datacfg=datacfg,
        modelcfg=modelcfg,
        dataset_name=datacfg.name,
        img_names=img_names,
        console=console,
    )

    console.rule(f"[bold green]Anomaly rules resolved for {datacfg.name}")
    console.print(f"[bold]Method:[/bold] {datacfg.anomaly_rules.method}")
    console.print(
        "[bold]Expected planner steps:[/bold] "
        f"{expected_rule_count if expected_rule_count is not None else 'legacy-untracked'}"
    )
    console.print(f"[bold]Full log saved to:[/bold] {log_file}\n")
    del anomaly_rules_text  # only the compiled file / log matter for standalone runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-config", default=None, metavar="PATH")
    parser.add_argument("--model-config", default=None, metavar="PATH")
    args = parser.parse_args()
    run(args.dataset_config, args.model_config)
