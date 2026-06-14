"""HuggingFace tokenizer loading with optional fastokens acceleration."""

from __future__ import annotations

from phyai_utils_tools.tokenizer.loader import (
    fastokens_available,
    get_tokenizer,
    try_enable_fastokens,
)

__all__ = ["fastokens_available", "get_tokenizer", "try_enable_fastokens"]
