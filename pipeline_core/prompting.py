"""Cached Jinja template loading and prompt rendering."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_ENVIRONMENTS: dict[Path, Environment] = {}


def get_jinja_env(prompts_dir: Path | str) -> Environment:
    """Return the cached Jinja environment for an existing prompt directory."""
    directory = Path(prompts_dir).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Prompts directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Prompts path is not a directory: {directory}")
    if directory not in _ENVIRONMENTS:
        _ENVIRONMENTS[directory] = Environment(
            loader=FileSystemLoader(str(directory)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _ENVIRONMENTS[directory]


def build_prompt(
    prompts_dir: Path | str, template_name: str, **context: Any
) -> str:
    """Render one template with the supplied context."""
    return get_jinja_env(prompts_dir).get_template(template_name).render(**context)


def build_prompt_blocks(
    prompts_dir: Path | str,
    template_name: str,
    blocks: tuple[str, ...] = ("system", "user"),
    **context: Any,
) -> dict[str, str]:
    """Render selected named blocks from one Jinja template."""
    template = get_jinja_env(prompts_dir).get_template(template_name)
    template_context = template.new_context(context)
    return {
        name: "".join(template.blocks[name](template_context))
        for name in blocks
        if name in template.blocks
    }
