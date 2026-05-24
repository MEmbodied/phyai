"""HuggingFace tokenizer loading with optional fastokens acceleration."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from phyai.utils import this_rank_log

logger = logging.getLogger(__name__)

_FASTOKENS_PATCHED = False


def fastokens_available() -> bool:
    """Return True if the optional ``fastokens`` package is importable.

    Driven by whether the user installed ``phyai[fastokenizer]`` extras.
    """
    return find_spec("fastokens") is not None


def try_enable_fastokens() -> bool:
    """Patch transformers to use the fastokens BPE backend if installed.

    Idempotent — patches at most once per process. Returns True when patched
    (or already patched), False when fastokens is not installed.

    Must be called BEFORE ``AutoTokenizer.from_pretrained``: transformers
    instantiates its tokenizer backend on first load, so a later patch does
    not retroactively switch backends on existing tokenizer instances.
    """
    global _FASTOKENS_PATCHED
    if _FASTOKENS_PATCHED:
        return True
    if not fastokens_available():
        return False
    import fastokens

    fastokens.patch_transformers()
    _FASTOKENS_PATCHED = True
    this_rank_log(logger, logging.INFO, "fastokens backend enabled")
    return True


def get_tokenizer(
    name_or_path: str, **kwargs: Any
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a HuggingFace tokenizer, optionally accelerated by fastokens.

    Calls :func:`try_enable_fastokens` before instantiation so if the user
    installed ``phyai[fastokenizer]`` the returned tokenizer's BPE backend is
    the fastokens shim. Without the extras this falls back transparently to
    the default HF Rust ``tokenizers`` backend.
    """
    try_enable_fastokens()
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)
