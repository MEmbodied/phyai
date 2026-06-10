"""Generic tokenization step.

:class:`TokenizerStep` encodes a ``PROMPT`` (falling back to ``TASK``) with a
HuggingFace tokenizer into ``INPUT_IDS`` / ``LANG_LENS``. It is model-agnostic —
how the prompt string is assembled (state binning, templates) is a model
concern done by an earlier step (for pi0.5,
:class:`~phyai_utils_tools.models.pi05.steps_pi05.StateTokenizerPrepareStep`).

The tokenizer is dependency-injected (passed in as an object), so the step has
no load-order constraint and the framework stays free of a hard ``transformers``
call site beyond the loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    INPUT_IDS,
    LANG_LENS,
    PROMPT,
    TASK,
    Transition,
)


@ProcessorStepRegistry.register("tokenizer_processor")
@dataclass
class TokenizerStep(ProcessorStep):
    """Tokenize ``PROMPT`` (or ``TASK``) into ``INPUT_IDS`` / ``LANG_LENS``.

    ``tokenizer`` is a HuggingFace tokenizer object (dependency-injected — the
    json carries only ``tokenizer_name``, the live object is supplied via
    ``step_kwargs`` on load). The encode call uses right padding to
    ``max_length`` and truncation. Produces ``INPUT_IDS`` ``(B, max_length)``
    int64 and ``LANG_LENS`` ``(B,)`` int64 (real lengths from the attention
    mask). Config schema matches lerobot's ``tokenizer_processor``.
    """

    tokenizer: Any = field(repr=False, default=None)
    max_length: int = 200
    tokenizer_name: str | None = None
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    def __call__(self, transition: Transition) -> Transition:
        if self.tokenizer is None:
            raise ValueError("TokenizerStep requires a `tokenizer` object.")
        prompts = transition.get(PROMPT)
        if prompts is None:
            prompts = transition.get(TASK)
        if prompts is None:
            raise ValueError("TokenizerStep requires a PROMPT or TASK entry.")
        if isinstance(prompts, str):
            prompts = [prompts]

        encoded = self.tokenizer(
            list(prompts),
            max_length=self.max_length,
            padding=self.padding,
            padding_side=self.padding_side,
            truncation=self.truncation,
            return_tensors="pt",
        )
        out = transition.copy()
        out[INPUT_IDS] = encoded["input_ids"].to(torch.int64)
        out[LANG_LENS] = encoded["attention_mask"].sum(dim=-1).to(torch.int64)
        return out

    def get_config(self) -> dict[str, Any]:
        # Matches lerobot's tokenizer_processor config keys. The tokenizer
        # object is never serialized; tokenizer_name lets a loader re-fetch it.
        return {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
            "tokenizer_name": self.tokenizer_name,
        }


__all__ = ["TokenizerStep"]
