"""Canonical transition envelope for the processing pipeline.

The pipeline passes a single mutable ``dict[str, Any]`` (a "transition")
between steps. Using a plain dict with shared string-key constants keeps the
data model model-agnostic: a vision step writes ``PIXEL_VALUES``, a tokenizer
step writes ``INPUT_IDS`` / ``LANG_LENS``, a normalizer touches ``STATE`` /
``ACTION``, and steps that don't care about a key leave it untouched. This
mirrors lerobot's ``EnvTransition`` but without the env-specific reward/done
machinery, which the inference path here never needs.

Key groups:

* **Raw inputs** (set by the caller): :data:`IMAGES`, :data:`STATE`,
  :data:`TASK`.
* **Processed inputs** (produced by steps): :data:`PIXEL_VALUES`,
  :data:`INPUT_IDS`, :data:`LANG_LENS`, and the intermediate :data:`PROMPT`.
* **Action** (postprocess): :data:`ACTION`.
"""

from __future__ import annotations

from typing import Any

# Raw inputs (caller-provided).
IMAGES = "images"  # list of per-camera (B, C, H, W) tensors, or stacked (B, n, C, H, W)
STATE = "state"  # (B, state_dim) proprioceptive state
TASK = "task"  # list[str] of task descriptions (one per sample)

# Intermediate / processed inputs (produced by steps).
PROMPT = "prompt"  # list[str] assembled prompt, consumed by the tokenizer step
PIXEL_VALUES = "pixel_values"  # canonical (B, n, C, target, target)
INPUT_IDS = "input_ids"  # (B, max_length) int64
LANG_LENS = "lang_lens"  # (B,) int64 real token lengths

# Action (postprocess).
ACTION = "action"  # (B, chunk, action_dim)

Transition = dict[str, Any]


def identity_adapter(x: Any) -> Any:
    """Default ``to_transition`` / ``to_output`` — pass the value through.

    Lets a :class:`~phyai_utils_tools.processing.pipeline.ProcessorPipeline`
    be driven directly with a transition dict, or wrapped with model-specific
    adapters that build the dict from a raw payload and extract a typed result.
    """
    return x


__all__ = [
    "ACTION",
    "IMAGES",
    "INPUT_IDS",
    "LANG_LENS",
    "PIXEL_VALUES",
    "PROMPT",
    "STATE",
    "TASK",
    "Transition",
    "identity_adapter",
]
