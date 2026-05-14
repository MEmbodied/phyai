"""Shared utilities for ``phyai.layers.attention``.

The flashinfer split-k scratch is process-global and per-device. Every
:class:`~phyai.layers.attention.NoStateAttention` instance — and any
future attention layer that uses flashinfer — falls back to this buffer
when the caller doesn't pass an explicit ``fi_workspace``. Sharing one
scratch across every layer keeps memory flat regardless of model depth.

Sizing
------
* Default: 128 MiB (1× flashinfer's recommendation).
* Override via the ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env var (positive
  integer, in bytes). Malformed values are rejected at lookup time.
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

import os

import torch


# flashinfer split-k scratch. Default is 1× the upstream-recommended
# 128 MiB; bump it via the ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env var
# when larger head counts or long-context prefill push split-k off the
# fast path.
_FI_WORKSPACE_BYTES_DEFAULT: int = 128 * 1024 * 1024
_FI_WORKSPACE_BYTES_ENV: str = "PHYAI_FLASHINFER_WORKSPACE_BYTES"


def resolve_workspace_bytes(override: int | None = None) -> int:
    """Resolve the flashinfer scratch size.

    Order of precedence: explicit ``override`` → env var → default.
    Raises :class:`ValueError` for non-positive or malformed inputs.
    """
    if override is not None:
        if override <= 0:
            raise ValueError(f"workspace_bytes={override} must be positive.")
        return override
    raw = os.environ.get(_FI_WORKSPACE_BYTES_ENV)
    if raw is None:
        return _FI_WORKSPACE_BYTES_DEFAULT
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(
            f"{_FI_WORKSPACE_BYTES_ENV}={raw!r} must be an integer (bytes)."
        ) from e
    if n <= 0:
        raise ValueError(f"{_FI_WORKSPACE_BYTES_ENV}={n} must be positive.")
    return n


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
    ``workspace_bytes`` if given, else the
    ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env var, else ``128 MiB``.
    Subsequent calls for the same device return the same tensor.

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
