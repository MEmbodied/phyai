"""phyai.layers.linear — parallel linear layers with declarative kernel dispatch.

Quick start::

    import torch.distributed as dist
    import phyai.parallel as P
    import phyai.layers.linear as L

    dist.init_process_group("nccl")
    P.init(layout=(8,), mesh_dim_names=("tp",))
    L.init()

    qkv = L.QKVParallelLinear(
        hidden_size=4096, head_dim=128, num_heads=32, num_kv_heads=8,
        axis="tp", spec=L.Bf16Spec(),
    )
    o_proj = L.RowParallelLinear(
        in_features=4096, out_features=4096,
        axis="tp", sp_axis="sp",
        spec=L.Fp8Spec(granularity=L.Granularity.PER_CHANNEL),
    )
"""

from __future__ import annotations

from phyai.layers.linear.backend import Granularity, KernelProbe, LinearKernel
from phyai.layers.linear.backends import FlashInferKernel, TorchKernel
from phyai.layers.linear.dispatch import (
    KernelDispatcher,
    _set_linear_dispatcher,
    get_linear_dispatcher,
)
from phyai.layers.linear.layers import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.linear.registry import (
    DefaultPolicy,
    ForcedPolicy,
    LinearKernelRegistry,
    Policy,
)
from phyai.layers.linear.spec import ActivationView, Bf16Spec, Fp8Spec
from phyai.layers.loaders import (
    ColumnShardLoader,
    QKVShardLoader,
    ReplicatedLoader,
    RowShardLoader,
)
from phyai.utils.cuda import sm_arch


def init(
    *,
    register_flashinfer: bool = True,
    validate: bool = True,
    sample_specs: list[str] | None = None,
) -> KernelDispatcher:
    """Build the process-level :class:`KernelDispatcher` and register defaults.

    Call once after :func:`phyai.parallel.init`. Subsequent calls replace
    the dispatcher — useful in tests, harmless otherwise.
    """
    reg = LinearKernelRegistry()

    if register_flashinfer:
        # FlashInfer is preferred for bf16 (cuBLASLt/cuDNN autopick) and
        # for block-FP8 prefill/decode on Blackwell. The point of
        # ``prefer_for`` is that the per-regime winner can be re-tuned in
        # one place without touching kernel code.
        reg.register(
            FlashInferKernel(),
            prefer_for={
                ("bf16", "prefill"),
                ("fp8_block_128_128", "prefill"),
                ("fp8_block_128_128", "decode"),
            },
        )

    reg.register(TorchKernel())  # always last, always a fallback

    if validate:
        default_specs = ["bf16", "fp8_per_tensor", "fp8_per_channel"]
        # Block-FP8 only validates on Blackwell; skip on older hardware so
        # developer laptops don't fail init.
        sm = sm_arch()
        if sm >= 100:
            default_specs.append("fp8_block_128_128")
        reg.validate(
            sample_specs=sample_specs if sample_specs is not None else default_specs,
            sm=sm,
        )

    d = KernelDispatcher(reg)
    _set_linear_dispatcher(d)
    return d


def _reset_for_test() -> None:
    """Drop the dispatcher singleton. Tests only."""
    _set_linear_dispatcher(None)


__all__ = [
    "init",
    # layers
    "LinearBase",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    # specs
    "Bf16Spec",
    "Fp8Spec",
    "ActivationView",
    "Granularity",
    # loaders
    "ColumnShardLoader",
    "RowShardLoader",
    "QKVShardLoader",
    "ReplicatedLoader",
    # dispatcher / registry
    "KernelDispatcher",
    "get_linear_dispatcher",
    "LinearKernelRegistry",
    "DefaultPolicy",
    "ForcedPolicy",
    "Policy",
    "LinearKernel",
    "KernelProbe",
    # backends
    "TorchKernel",
    "FlashInferKernel",
]
