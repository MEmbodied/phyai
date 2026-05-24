"""pi05 PaliGemma prompt assembly and tokenization.

Mirrors openpi's ``PaligemmaTokenizer.tokenize()``: state is assumed to
have been normalized to ``[-1, 1]`` upstream, here it is discretized into
256 bins and embedded in a natural-language prompt of the form::

    "Task: <cleaned_task>, State: <bin_0> <bin_1> ...;\\nAction: "

The prompt is tokenized with the PaliGemma tokenizer to produce
``(input_ids, lang_lens)`` shaped to feed :class:`Pi05BatchRequest`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from phyai.tokenizer import get_tokenizer

DEFAULT_TOKENIZER_NAME = "google/paligemma-3b-pt-224"
STATE_NUM_BINS = 256


def get_pi05_tokenizer(
    name_or_path: str = DEFAULT_TOKENIZER_NAME, **kwargs: Any
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load the PaliGemma tokenizer pi05 was trained against."""
    return get_tokenizer(name_or_path, **kwargs)


def discretize_state(state: torch.Tensor, num_bins: int = STATE_NUM_BINS) -> np.ndarray:
    """Discretize a [-1, 1]-normalized state vector into ``num_bins`` integer bins.

    Mirrors openpi's PaligemmaTokenizer state binning. Caller must ensure
    ``state`` already lies in ``[-1, 1]``; values outside the range are
    clipped to the nearest bin by ``np.digitize``.
    """
    state_np = state.detach().cpu().numpy()
    bins = np.linspace(-1.0, 1.0, num_bins + 1)[:-1]
    return np.digitize(state_np, bins=bins) - 1


def build_pi05_prompts(tasks: list[str], discretized_states: np.ndarray) -> list[str]:
    """Assemble pi05 prompt strings from task descriptions and binned states."""
    if len(tasks) != len(discretized_states):
        raise ValueError(
            f"len(tasks)={len(tasks)} must match "
            f"len(discretized_states)={len(discretized_states)}"
        )
    prompts: list[str] = []
    for task, state_bins in zip(tasks, discretized_states):
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, state_bins))
        prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
    return prompts


def tokenize_pi05_inputs(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tasks: list[str],
    states: torch.Tensor,
    max_length: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build pi05 prompts from ``(tasks, states)`` and tokenize.

    ``states`` must already be normalized to ``[-1, 1]`` (shape
    ``(B, state_dim)``). Returns ``(input_ids, lang_lens)`` matching the
    layout expected by :class:`Pi05BatchRequest`:

    * ``input_ids``: ``(B, max_length)`` int64, right-padded
    * ``lang_lens``: ``(B,)`` int64, real (unpadded) length per sample
    """
    if states.dim() != 2:
        raise ValueError(
            f"states must be (B, state_dim), got shape {tuple(states.shape)}"
        )
    if states.shape[0] != len(tasks):
        raise ValueError(
            f"states batch dim {states.shape[0]} != len(tasks) {len(tasks)}"
        )

    discretized = discretize_state(states)
    prompts = build_pi05_prompts(tasks, discretized)

    encoded = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        padding_side="right",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(torch.int64)
    lang_lens = encoded["attention_mask"].sum(dim=-1).to(torch.int64)
    return input_ids, lang_lens
