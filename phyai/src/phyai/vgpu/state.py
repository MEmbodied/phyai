"""Process-level :func:`init` guard + at-exit cleanup hook.

Module state is intentionally minimal: a flag that ``init`` has run, an
idempotent ``atexit`` handler that closes any vGPU registered through
:class:`vGPU` / :func:`create_vgpus`, and a single switch
(:func:`resolve`) that selects the active backend.
"""

from __future__ import annotations

import atexit
import weakref

import torch
import torch.distributed as dist

from phyai.utils.cuda import device_capability
from phyai.vgpu.backend import resolve
from phyai.vgpu.exceptions import VGPUNotApplicableError


_INITIALIZED: bool = False
_ATEXIT_REGISTERED: bool = False
# Weak references so user-side ``close()`` + GC still works the normal way.
_REGISTERED_VGPUS: "list[weakref.ref]" = []


def init(
    device: "str | torch.device" = "cuda:0",
    *,
    backend: str | None = None,
) -> None:
    """Validate the environment, then select a backend.

    Refuses (raises :class:`VGPUNotApplicableError`) when:
      - ``torch.distributed`` is initialised with ``world_size > 1``;
      - the target device is not a CUDA device;
      - CUDA is unavailable; or
      - the device's compute capability is below 8.0.

    May be called more than once; the latest call wins for backend
    selection.
    """
    global _INITIALIZED, _ATEXIT_REGISTERED
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise VGPUNotApplicableError(
            f"phyai.vgpu only supports world_size == 1 "
            f"(got {dist.get_world_size()}). For multi-GPU use phyai.parallel."
        )
    dev = torch.device(device)
    if dev.type != "cuda":
        raise VGPUNotApplicableError(f"vgpu requires a cuda device, got {dev}")
    if not torch.cuda.is_available():
        raise VGPUNotApplicableError(
            "vgpu requires CUDA, but torch.cuda.is_available() is False"
        )
    cc = device_capability(dev)
    if cc < (8, 0):
        raise VGPUNotApplicableError(
            f"green ctx requires compute capability >= 8.0, got {cc}"
        )
    resolve(backend)
    _INITIALIZED = True
    if not _ATEXIT_REGISTERED:
        atexit.register(_close_all_atexit)
        _ATEXIT_REGISTERED = True


def register_vgpu(vgpu: object) -> None:
    """Track a vGPU instance for atexit cleanup."""
    _REGISTERED_VGPUS.append(weakref.ref(vgpu))


def _close_all_atexit() -> None:
    """Close any still-live vGPU at process exit. Best-effort."""
    for ref in list(_REGISTERED_VGPUS):
        v = ref()
        if v is None:
            continue
        try:
            v.close()
        except Exception:
            pass
    _REGISTERED_VGPUS.clear()


def _is_initialized() -> bool:
    return _INITIALIZED


def _reset_for_tests() -> None:
    """Test-only: clear init flag and the registered-vGPU list."""
    global _INITIALIZED
    _INITIALIZED = False
    _REGISTERED_VGPUS.clear()
