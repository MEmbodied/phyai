"""WeightSpec tests — allocate shapes, process_after_loading, activation quant.

No CUDA required; float8_e4m3fn tensors allocate on CPU in newer PyTorch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from phyai.layers.quant import (
    ActivationView,
    AllocationRequest,
    Bf16Spec,
    Fp8Spec,
    Granularity,
)
from phyai.layers.quant.fp8 import _convert_to_channelwise


# ---------------------------------------------------------------------------
# Bf16Spec
# ---------------------------------------------------------------------------


def _request(
    *,
    weight_shape: tuple[int, ...],
    logical_widths: list[int] | None = None,
    params_dtype: torch.dtype = torch.bfloat16,
) -> AllocationRequest:
    return AllocationRequest(
        weight_shape=weight_shape,
        logical_widths=logical_widths
        if logical_widths is not None
        else [weight_shape[0]],
        fused_dim=0,
        params_dtype=params_dtype,
    )


def test_bf16_allocate_plain():
    layer = nn.Module()
    Bf16Spec().allocate(layer, _request(weight_shape=(64, 32)))
    assert layer.weight.shape == (64, 32)
    assert layer.weight.dtype == torch.bfloat16
    assert not hasattr(layer, "weight_scale")
    assert layer.logical_widths == [64]


def test_bf16_allocate_fused_sizes():
    layer = nn.Module()
    Bf16Spec().allocate(
        layer,
        _request(weight_shape=(64, 16), logical_widths=[32, 16, 16]),
    )
    assert layer.weight.shape == (64, 16)
    assert layer.logical_widths == [32, 16, 16]


def test_bf16_respects_params_dtype():
    layer = nn.Module()
    Bf16Spec().allocate(
        layer,
        _request(weight_shape=(16, 16), params_dtype=torch.float16),
    )
    assert layer.weight.dtype == torch.float16


def test_bf16_process_after_loading_noop():
    layer = nn.Module()
    Bf16Spec().allocate(layer, _request(weight_shape=(16, 16)))
    # Fill with deterministic values so torch.equal doesn't trip over NaNs
    # that torch.empty may leave behind.
    layer.weight.data.fill_(0.5)
    before = layer.weight.data.clone()
    Bf16Spec().process_after_loading(layer)
    assert torch.equal(layer.weight.data, before)


def test_bf16_does_not_set_size_attrs():
    """``layer.input_size_per_partition`` etc. are linear-specific now —
    the spec is op-agnostic and must not write them."""
    layer = nn.Module()
    Bf16Spec().allocate(layer, _request(weight_shape=(64, 32)))
    assert not hasattr(layer, "input_size_per_partition")
    assert not hasattr(layer, "output_size_per_partition")
    assert not hasattr(layer, "input_size_global")
    assert not hasattr(layer, "output_size_global")


# ---------------------------------------------------------------------------
# Fp8Spec
# ---------------------------------------------------------------------------


def test_fp8_per_tensor_shapes_pre_and_post_loading():
    spec = Fp8Spec(granularity=Granularity.PER_TENSOR)
    layer = nn.Module()
    spec.allocate(
        layer,
        _request(weight_shape=(48, 64), logical_widths=[32, 16]),
    )
    assert layer.weight.shape == (48, 64)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (2,)  # one per logical matrix
    assert layer.input_scale.shape == (1,)
    assert spec.spec_id == "fp8_per_tensor"

    # After loading, weight_scale is fanned out to per-channel.
    spec.process_after_loading(layer)
    assert layer.weight_scale.shape == (48,)


def test_fp8_per_channel_shapes():
    spec = Fp8Spec(granularity=Granularity.PER_CHANNEL)
    layer = nn.Module()
    spec.allocate(layer, _request(weight_shape=(128, 64)))
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (128,)
    assert not hasattr(layer, "input_scale")  # computed at runtime
    assert spec.spec_id == "fp8_per_channel"


def test_fp8_block_shapes():
    spec = Fp8Spec(granularity=Granularity.BLOCK, block_shape=(128, 128))
    layer = nn.Module()
    spec.allocate(layer, _request(weight_shape=(384, 256)))
    assert layer.weight.shape == (384, 256)
    assert layer.weight_scale.shape == (3, 2)  # 384/128 x 256/128
    assert spec.spec_id == "fp8_block_128_128"


def test_fp8_block_requires_block_shape():
    with pytest.raises(ValueError, match="block_shape"):
        Fp8Spec(granularity=Granularity.BLOCK)


def test_fp8_block_enforces_divisibility():
    spec = Fp8Spec(granularity=Granularity.BLOCK, block_shape=(128, 128))
    layer = nn.Module()
    with pytest.raises(ValueError, match="not divisible"):
        spec.allocate(layer, _request(weight_shape=(256, 100)))


def test_fp8_rejects_non_2d_shape():
    spec = Fp8Spec(granularity=Granularity.PER_CHANNEL)
    layer = nn.Module()
    with pytest.raises(ValueError, match="2-D weight_shape"):
        spec.allocate(
            layer,
            AllocationRequest(
                weight_shape=(8, 16, 32),
                logical_widths=[8],
            ),
        )


def test_convert_to_channelwise_basic():
    scales = torch.tensor([0.25, 0.5, 1.0])
    out = _convert_to_channelwise(scales, [2, 1, 3])
    assert out.tolist() == [0.25, 0.25, 0.5, 1.0, 1.0, 1.0]


def test_fp8_quantize_activation_per_tensor():
    spec = Fp8Spec(granularity=Granularity.PER_TENSOR)
    layer = nn.Module()
    spec.allocate(layer, _request(weight_shape=(32, 16)))
    x = torch.randn(4, 16)
    act = spec.quantize_activation(x, layer)
    assert isinstance(act, ActivationView)
    assert act.x.dtype == torch.float8_e4m3fn
    assert act.x_scale is layer.input_scale
    assert act.granularity == Granularity.PER_TENSOR


def test_fp8_quantize_activation_per_channel_rowwise():
    spec = Fp8Spec(granularity=Granularity.PER_CHANNEL)
    layer = nn.Module()
    spec.allocate(layer, _request(weight_shape=(32, 16)))
    x = torch.randn(4, 16) * 5.0
    act = spec.quantize_activation(x, layer)
    assert act.x.dtype == torch.float8_e4m3fn
    assert act.x_scale.shape == (4, 1)
    assert act.granularity == Granularity.PER_CHANNEL


def test_fp8_needs_act_quant_true():
    assert Fp8Spec(granularity=Granularity.PER_CHANNEL).needs_act_quant is True


def test_fp8_spec_id_format():
    assert Fp8Spec(granularity=Granularity.PER_TENSOR).spec_id == "fp8_per_tensor"
    assert Fp8Spec(granularity=Granularity.PER_CHANNEL).spec_id == "fp8_per_channel"
    assert (
        Fp8Spec(granularity=Granularity.BLOCK, block_shape=(128, 128)).spec_id
        == "fp8_block_128_128"
    )
    assert (
        Fp8Spec(granularity=Granularity.BLOCK, block_shape=(64, 256)).spec_id
        == "fp8_block_64_256"
    )
