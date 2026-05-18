"""Process-level engine config: shared backend / dtype / device defaults pulled by models.

:class:`EngineConfig` is a frozen dataclass holding the runtime knobs
every phyai model constructor consults when it isn't given explicit
overrides — currently the attention backend, norm kernel backend, the
default parameter dtype, the default parameter device, and the log
level. It is exposed through a process-level singleton; callers read
it via :func:`get_engine_config` and replace it via
:func:`set_engine_config`, typically once at program startup or in
test setup.

Defaults target the production CUDA stack (flashinfer for both attn
and norm, bf16 params, ``"cuda"`` device, ``INFO`` logs). To run on a
dev box without flashinfer, or to load fp32 checkpoints without a
cast, override at the top of ``main`` / ``conftest`` ::

    set_engine_config(get_engine_config().replace(
        attn_backend="sdpa",
        norm_backend="phyai-kernel",
        params_dtype=torch.float32,
        device="cpu",
        log_level=logging.DEBUG,
    ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from threading import Lock

import torch


@dataclass(frozen=True)
class EngineConfig:
    """Shared backend / dtype / device defaults consulted by every phyai model constructor.

    Subclasses are not expected — every field is a global, single-valued
    default. New fields land here as the engine grows (linear backend
    override, etc.).
    """

    attn_backend: str = "flashinfer"
    norm_backend: str = "flashinfer"
    params_dtype: torch.dtype = field(default=torch.bfloat16)
    device: str = "cuda"
    log_level: int = logging.INFO

    def replace(self, **changes: object) -> "EngineConfig":
        """Return a new EngineConfig with the given fields overridden."""
        return replace(self, **changes)


_config: EngineConfig | None = None
_lock = Lock()


def get_engine_config() -> EngineConfig:
    """Return the process-level :class:`EngineConfig`.

    Lazily allocates with defaults on first call, so importing
    ``phyai.models`` without an explicit init Just Works on a
    CUDA + flashinfer box.
    """
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = EngineConfig()
    return _config


def set_engine_config(cfg: EngineConfig) -> None:
    """Replace the process-level :class:`EngineConfig`.

    Pass a freshly-constructed instance (or the result of
    ``get_engine_config().replace(...)``). The change is global; every
    subsequent model constructor that consults the singleton picks up
    the new values.
    """
    global _config
    with _lock:
        _config = cfg


__all__ = ["EngineConfig", "get_engine_config", "set_engine_config"]
