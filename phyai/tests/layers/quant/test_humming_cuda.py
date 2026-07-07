"""HummingKernel numeric smoke test (CUDA + humming required).

Skipped on CPU / dev hosts and when humming-kernels is unavailable. On a real
GPU (int4 needs sm>=80) this exercises the full allocate -> fill ->
process_after_loading -> apply path against a dequant + F.linear reference.

This is a first integration smoke with a loose tolerance; tighten the reference
comparison once validated on target hardware (Ampere/Ada/Hopper/Blackwell).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.quant import HummingWeightSpec
from phyai.layers.quant.base import AllocationRequest
from phyai.utils.humming import has_humming


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    maj, mnr = torch.cuda.get_device_capability()
    return maj * 10 + mnr


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_humming() and _sm() >= 80),
    reason="humming int4 GEMM requires CUDA sm>=80 and humming-kernels",
)


def test_int4_group_w4a16_forward_matches_reference():
    from humming.utils.weight import dequantize_weight  # noqa: F401

    from phyai.layers.linear.backends.humming import HummingKernel

    N, K, M, group = 1024, 1024, 64, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    weight = torch.randn(N, K, dtype=dtype, device=device) * 0.1
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.1

    # Asymmetric int4 (AWQ/GPTQ-style): unsigned int4 + zero-point, group-128.
    spec = HummingWeightSpec(
        w_dtype="int4",
        scale_type="group",
        group_size=group,
        has_zero_point=True,
        scale_dtype="bfloat16",
    )
    layer = torch.nn.Module()
    layer.spec = spec
    layer.bias = None
    spec.allocate(
        layer,
        AllocationRequest(weight_shape=(N, K), logical_widths=[N], device=device),
    )

    # Quantize via humming's own schema so signed/unsigned/zero-point stay
    # self-consistent between the loaded params and the reference dequant.
    schema = spec._weight_schema()
    tensors = type(schema).quant_tensor(weight, schema, dtype)
    for name, t in tensors.items():
        getattr(layer, name).data.copy_(t)

    spec.process_after_loading(layer)
    out = HummingKernel().apply(layer, x, None)

    deq_weight = schema.dequant_tensors(tensors).to(dtype)  # (N, K)
    ref = F.linear(x, deq_weight)

    assert out.shape == (M, N)
    assert torch.isfinite(out).all()
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.98, f"humming int4 GEMM vs dequant reference cosine={cos.item():.4f}"
