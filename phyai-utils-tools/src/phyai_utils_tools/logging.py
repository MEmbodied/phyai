"""Minimal rank-aware logging for phyai-utils-tools.

This package is a dependency-free *leaf* in the workspace — it must not
import the main ``phyai`` package (that would create a cycle, since
``phyai`` depends on the processors here). ``phyai.utils`` owns the
canonical rank-aware loggers; this module vendors the one helper the
moved tokenizer loader needs (:func:`rank0_log`) with the same rank-0
semantics, guarding the ``torch.distributed`` access so it is a no-op in
single-process runs.
"""

from __future__ import annotations

import logging
from typing import Any


def _rank_prefix() -> str:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return f"[rank {dist.get_rank()}/{dist.get_world_size()}]"
    except Exception:  # torch missing or dist not usable — fall through
        pass
    return "[rank -/-]"


def rank0_log(
    logger: logging.Logger,
    level: int,
    msg: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Log ``msg`` only on distributed rank 0 (or always, single-process).

    Mirrors :func:`phyai.utils.this_rank_log` for rank 0 but keeps this
    package free of any ``phyai`` import. Outside a distributed context it
    behaves like a plain :meth:`logging.Logger.log` call.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    logger.log(level, f"{_rank_prefix()} {msg}", *args, **kwargs)


__all__ = ["rank0_log"]
