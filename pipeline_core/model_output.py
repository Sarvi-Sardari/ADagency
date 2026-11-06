"""Model-family-neutral cleanup for text consumed by strict parsers."""

from __future__ import annotations

import re


_THINKING_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_STRUCTURAL_TOKEN_RE = re.compile(
    r"<\|(?:begin|end)_of_box\|>", flags=re.IGNORECASE
)
_STEP_MARKER_BOUNDARY_RE = re.compile(
    r"[ \t]+(?=\[step\s+\d+\]\s*(?:yes|no)\b)", flags=re.IGNORECASE
)
_FINAL_DECISION_BOUNDARY_RE = re.compile(
    r"[ \t]+(?=<?(?:good|anomaly)>?\s*;\s*reason\s*=)", flags=re.IGNORECASE
)


def normalize_model_output(text: str) -> str:
    """Remove non-semantic wrappers while preserving the model's answer.

    Some multimodal checkpoints expose hidden reasoning or grounding control
    tokens in decoded text.  Those tokens are not part of the requested
    response contract and can prevent anchored parsers from seeing an
    otherwise valid first or last line.
    """
    cleaned = _THINKING_BLOCK_RE.sub("", str(text))
    cleaned = _STRUCTURAL_TOKEN_RE.sub("", cleaned)
    return cleaned.strip()


def normalize_step_decision_layout(text: str) -> str:
    """Restore record boundaries in a concatenated step-decision response.

    Some checkpoints emit the requested records with spaces instead of line
    breaks.  This changes layout only: it does not add, remove, or reinterpret
    any step answer, evidence, or final decision field.  Strict downstream
    checks still reject missing, duplicate, and unexpected step numbers.
    """
    cleaned = normalize_model_output(text)
    cleaned = _STEP_MARKER_BOUNDARY_RE.sub("\n", cleaned)
    cleaned = _FINAL_DECISION_BOUNDARY_RE.sub("\n", cleaned)
    return cleaned.strip()
