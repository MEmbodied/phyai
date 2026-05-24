"""GreenCtxBackend Protocol + global registry + auto-resolution.

A "backend" wraps the SM partitioning capability of one driver-level
implementation (flashinfer, torch). Backends are registered on import of
:mod:`phyai.vgpu.backends`. ``resolve`` picks one according to the
priority ``explicit > env (PHYAI_VGPU_BACKEND) > auto`` (auto order:
flashinfer -> torch with a fallback warning).
"""

from __future__ import annotations

import warnings
from typing import Callable, Protocol, TypeVar, runtime_checkable

import torch

from phyai.env import envs
from phyai.vgpu._spec import ShardSpec
from phyai.vgpu.exceptions import VGPUNotApplicableError


@runtime_checkable
class GreenCtxBackend(Protocol):
    """Backend protocol for SM partitioning.

    Contract:
      - ``split_by_count`` / ``split_by_sm_counts`` return ``N + 1`` specs
        where the last is the remainder (trailing SMs not allocated to a
        group).
      - ``create_single`` returns a single non-remainder spec.
      - ``destroy`` is best-effort; backends document their own caveats
        (e.g. driver-level memory the underlying API does not release).
    """

    name: str

    def split_by_count(
        self,
        device: torch.device,
        num_groups: int,
        min_count: int,
    ) -> list[ShardSpec]: ...

    def split_by_sm_counts(
        self,
        device: torch.device,
        sm_counts: list[int],
    ) -> list[ShardSpec]: ...

    def create_single(
        self,
        device: torch.device,
        num_sms: int,
    ) -> ShardSpec: ...

    def destroy(self, spec: ShardSpec) -> None: ...


_BACKENDS: dict[str, type] = {}
_CURRENT: GreenCtxBackend | None = None

_BackendT = TypeVar("_BackendT", bound=type)


def register(name: str, cls: type) -> None:
    """Register a backend class under ``name``.

    Re-registration overwrites the previous mapping (intended for tests
    or user customisation). Built-in backends use the decorator form
    :func:`register_vgpu_backend`; this functional shape stays for
    callers who want explicit (name, class) pairs without binding to a
    class definition.
    """
    _BACKENDS[name] = cls


def register_vgpu_backend(name: str) -> Callable[[_BackendT], _BackendT]:
    """Class decorator: register a :class:`GreenCtxBackend` under ``name``.

    Equivalent to :func:`register` but applied at class-definition time:

        @register_vgpu_backend("torch")
        class TorchBackend:
            ...
    """

    def deco(cls: _BackendT) -> _BackendT:
        _BACKENDS[name] = cls
        return cls

    return deco


def known_backends() -> list[str]:
    """Return the sorted list of registered backend names."""
    _ensure_builtins_loaded()
    return sorted(_BACKENDS)


def set_backend(name_or_obj: "str | GreenCtxBackend") -> "GreenCtxBackend":
    """Set the process-level backend explicitly.

    Args:
        name_or_obj: Either a registered backend name or an already
            instantiated backend object.

    Returns:
        The active backend instance.

    Raises:
        VGPUNotApplicableError: when the name is unknown.
    """
    global _CURRENT
    _ensure_builtins_loaded()
    if isinstance(name_or_obj, str):
        if name_or_obj not in _BACKENDS:
            raise VGPUNotApplicableError(
                f"unknown backend {name_or_obj!r} (known: {sorted(_BACKENDS)})"
            )
        _CURRENT = _BACKENDS[name_or_obj]()
    else:
        _CURRENT = name_or_obj
    return _CURRENT


def get_backend() -> GreenCtxBackend:
    """Return the currently active backend.

    Raises:
        RuntimeError: if neither :func:`phyai.vgpu.init` nor
            :func:`set_backend` was called yet.
    """
    if _CURRENT is None:
        raise RuntimeError(
            "phyai.vgpu has no active backend. "
            "Call phyai.vgpu.init(...) or phyai.vgpu.set_backend(...) first."
        )
    return _CURRENT


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_builtins_loaded() -> None:
    """Idempotently import the built-in backends so they self-register."""
    import phyai.vgpu.backends  # noqa: F401


def resolve(name: str | None) -> GreenCtxBackend:
    """Resolve a backend choice. Priority: explicit > env > auto.

    Auto order: ``flashinfer -> torch``. When auto falls through to torch
    because flashinfer is not importable a ``UserWarning`` is emitted —
    explicit ``backend='torch'`` does not warn.
    """
    _ensure_builtins_loaded()
    if name is not None:
        return set_backend(name)
    env = envs.PHYAI_VGPU_BACKEND.get()
    if env:
        return set_backend(env)
    if _flashinfer_available():
        return set_backend("flashinfer")
    warnings.warn(
        "phyai.vgpu: flashinfer not available, falling back to torch "
        "backend (only single-vGPU create_single is supported; multi-shard "
        "disjoint split unavailable). Install flashinfer for full capability.",
        stacklevel=3,
    )
    return set_backend("torch")


def _reset_for_tests() -> None:
    """Internal: reset the active backend pointer.

    Test-only helper. Does not touch the registry.
    """
    global _CURRENT
    _CURRENT = None
