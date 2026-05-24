"""Shared utilities for ``phyai.layers.attention``.

The flashinfer split-k scratch is process-global and per-device. Every
attention backend that uses flashinfer (the no-cache
:class:`~phyai.layers.attention.attention.Attention` stack, the AR
:class:`~phyai.layers.attention.ar.ARAttention` stack, and the
:class:`~phyai.layers.attention.diffusion.DiffusionAttention` stack)
falls back to this buffer when the caller doesn't pass an explicit
``fi_workspace``. Sharing one scratch across every layer keeps memory
flat regardless of model depth.

Sizing
------
* Default: ``RuntimeConfig.flashinfer_workspace_bytes`` (128 MiB out of
  the box, 1x flashinfer's recommendation).
* Override the engine-level value via ``PHYAI_FLASHINFER_WORKSPACE_BYTES``
  (overlaid onto :class:`~phyai.engine_config.RuntimeConfig` by
  :meth:`EngineConfig.from_env`) or by passing a bespoke
  :class:`~phyai.engine_config.EngineConfig` to the engine. The
  resolver consults the :class:`EngineConfig` singleton — no direct
  env reads here.
* The first caller for a given device may also pass ``workspace_bytes``
  to :func:`get_global_fi_workspace`; once the buffer for that device
  exists, the parameter is ignored.

External pools
--------------
:func:`register_global_fi_workspace` lets a runtime hand in its own
:class:`torch.Tensor` (own allocator, pinned region, deterministic test
bytes, etc.) and the registry will treat it as the canonical buffer for
that device. The registry is keyed by ``(device.type, device.index)`` so
multi-GPU processes get one buffer per device rather than one total.
"""

from __future__ import annotations

import torch

from phyai.engine_config import get_engine_config


def resolve_workspace_bytes(override: int | None = None) -> int:
    """Resolve the flashinfer scratch size.

    Order of precedence: explicit ``override`` ->
    :class:`~phyai.engine_config.RuntimeConfig.flashinfer_workspace_bytes`
    on the :class:`EngineConfig` singleton (which has already absorbed
    any ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env override). Raises
    :class:`ValueError` for non-positive ``override``; the engine config
    is validated at construction time so the singleton value is always
    a positive int.
    """
    if override is not None:
        if override <= 0:
            raise ValueError(f"workspace_bytes={override} must be positive.")
        return override
    return get_engine_config().runtime.flashinfer_workspace_bytes


# Process-global flashinfer scratch. Keyed on
# ``(device.type, device.index)`` so a multi-GPU process gets one
# buffer per device rather than one buffer total.
_global_fi_workspaces: dict[tuple[str, int | None], torch.Tensor] = {}


def _device_key(device: torch.device | str) -> tuple[str, int | None]:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return (dev.type, dev.index)


def get_global_fi_workspace(
    device: torch.device | str, *, workspace_bytes: int | None = None
) -> torch.Tensor:
    """Get-or-create the process-global flashinfer scratch on ``device``.

    Allocated lazily on first call for each device. Size comes from
    ``workspace_bytes`` if given, else
    :attr:`~phyai.engine_config.RuntimeConfig.flashinfer_workspace_bytes`
    on the :class:`EngineConfig` singleton (which absorbs the
    ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env override). Subsequent calls
    for the same device return the same tensor.

    ``workspace_bytes`` only applies when the buffer for ``device`` has
    not been allocated yet; it is *not* a per-instance override. To
    swap in your own pre-allocated tensor, use
    :func:`register_global_fi_workspace`.
    """
    key = _device_key(device)
    ws = _global_fi_workspaces.get(key)
    if ws is None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        ws = torch.empty(
            resolve_workspace_bytes(workspace_bytes), dtype=torch.uint8, device=dev
        )
        _global_fi_workspaces[key] = ws
    return ws


def register_global_fi_workspace(
    device: torch.device | str, workspace: torch.Tensor
) -> None:
    """Inject a pre-allocated tensor as the global scratch for ``device``.

    Useful when the runtime owns the GPU memory pool itself (custom
    allocator, pinned scratch shared with another subsystem, or a
    deterministic-bytes test harness). Replaces any previous binding for
    ``device``. The tensor must be 1-D ``uint8`` and live on a device
    matching ``device``.
    """
    if workspace.dtype != torch.uint8 or workspace.ndim != 1:
        raise ValueError(
            f"workspace must be a 1-D uint8 tensor, got "
            f"shape={tuple(workspace.shape)}, dtype={workspace.dtype}."
        )
    key = _device_key(device)
    if (workspace.device.type, workspace.device.index) != key:
        raise ValueError(
            f"workspace.device={workspace.device} does not match device="
            f"{torch.device(device)}."
        )
    _global_fi_workspaces[key] = workspace


def _reset_global_fi_workspaces() -> None:
    """Drop the global workspace registry. Tests only."""
    _global_fi_workspaces.clear()


__all__ = [
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "resolve_workspace_bytes",
]
