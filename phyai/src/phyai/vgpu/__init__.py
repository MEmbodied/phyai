"""phyai.vgpu — single-card SM partitioning via CUDA Green Context.

Quick start::

    import phyai.vgpu as V

    V.init(device="cuda:0")              # default backend = flashinfer
    a, b = V.create_vgpus(
        device="cuda:0",
        sm_counts=[64, 64],
        names=["a", "b"],
    )
    with a.activate():
        y_a = model_a(x)
    with b.activate():
        y_b = model_b(x)
"""

from __future__ import annotations

# Import the backends package up front so its registrations happen even
# when callers reach ``set_backend`` / ``split_device`` before any other
# ``phyai.vgpu.*`` symbol.
import phyai.vgpu.backends as _backends  # noqa: F401
from phyai.vgpu.backend import (
    GreenCtxBackend,
    get_backend,
    known_backends,
    register,
    register_vgpu_backend,
    set_backend,
)
from phyai.vgpu.exceptions import (
    BackendCapabilityError,
    VGPUDriverError,
    VGPUError,
    VGPUNotApplicableError,
    VGPURuntimeError,
)
from phyai.vgpu.partition import Shard, split_device, split_device_by_sm_count
from phyai.vgpu.state import init
from phyai.vgpu.vgpu import create_vgpus, vGPU

__all__ = [
    # init / state
    "init",
    "set_backend",
    "get_backend",
    "known_backends",
    "register",
    "register_vgpu_backend",
    # split / shard
    "split_device",
    "split_device_by_sm_count",
    "Shard",
    # vgpu
    "vGPU",
    "create_vgpus",
    # protocol
    "GreenCtxBackend",
    # errors
    "VGPUError",
    "VGPUNotApplicableError",
    "VGPURuntimeError",
    "VGPUDriverError",
    "BackendCapabilityError",
]
