"""HummingWeightSpec + HummingKernel: CPU-testable surface.

These tests never call humming (it is CUDA-only and optional). They cover the
parts that must work on a CPU/dev host: stable ``spec_id`` strings, the
kernel's capability predicates (SM gate + ``humming_`` prefix filter, with
``_HAS_HUMMING`` monkeypatched), ``materialize`` routing INT4 to humming, and
that the kernel registers after flashinfer but before the torch fallback.
"""

from __future__ import annotations

import pytest
import torch

from phyai.layers.linear.backend import KernelProbe
from phyai.layers.linear.backends import humming as humming_backend
from phyai.layers.linear.backends.humming import HummingKernel, _parse_dtypes
from phyai.layers.linear.registry import list_registered_linear_kernels
from phyai.layers.quant import HummingWeightSpec
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.materialize import materialize
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant
from phyai.parallel.state import Mode


def _probe(spec_id: str, *, sm: int = 90) -> KernelProbe:
    return KernelProbe(
        spec_id=spec_id,
        M_bucket=1,
        N=512,
        K=512,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        sm=sm,
        mode=Mode.EAGER,
    )


# --------------------------------------------------------------------------- #
# spec_id
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(w_dtype="int4", scale_type="channel"), "humming_wint4_abfloat16_channel"),
        (
            dict(w_dtype="int4", scale_type="group", group_size=128),
            "humming_wint4_abfloat16_group_g128",
        ),
        (
            dict(
                w_dtype="int4", scale_type="group", group_size=128, has_zero_point=True
            ),
            "humming_wint4_abfloat16_group_g128_zp",
        ),
        (
            dict(w_dtype="int8", a_dtype="int8", scale_type="channel"),
            "humming_wint8_aint8_channel",
        ),
        (
            dict(w_dtype="float8e4m3", scale_type="channel"),
            "humming_wfloat8e4m3_abfloat16_channel",
        ),
        (
            dict(
                w_dtype="int4",
                scale_type="block",
                group_size=128,
                group_size_n=128,
            ),
            "humming_wint4_abfloat16_block_g128x128",
        ),
    ],
)
def test_spec_id(kwargs, expected):
    assert HummingWeightSpec(**kwargs).spec_id == expected


def test_weight_dtype_is_int32():
    assert HummingWeightSpec(w_dtype="int4").weight_dtype is torch.int32


# --------------------------------------------------------------------------- #
# _parse_dtypes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec_id,w,a",
    [
        ("humming_wint4_abfloat16_group_g128", "int4", "bfloat16"),
        ("humming_wint8_aint8_channel", "int8", "int8"),
        ("humming_wfloat8e4m3_abfloat16_channel", "float8e4m3", "bfloat16"),
        ("humming_wfloat4e2m1_abfloat16_group_g32", "float4e2m1", "bfloat16"),
    ],
)
def test_parse_dtypes(spec_id, w, a):
    assert _parse_dtypes(spec_id) == (w, a)


# --------------------------------------------------------------------------- #
# HummingKernel.can_handle capability predicates
# --------------------------------------------------------------------------- #


def test_can_handle_false_without_humming(monkeypatch):
    monkeypatch.setattr(humming_backend, "_HAS_HUMMING", False)
    k = HummingKernel()
    assert not k.can_handle(_probe("humming_wint4_abfloat16_group_g128", sm=100))


def test_can_handle_rejects_non_humming_spec(monkeypatch):
    monkeypatch.setattr(humming_backend, "_HAS_HUMMING", True)
    k = HummingKernel()
    for spec_id in (
        "bf16",
        "fp8_block_128_128",
        "nvfp4_block_16_128x4",
        "fp8_per_channel",
    ):
        assert not k.can_handle(_probe(spec_id, sm=100))


@pytest.mark.parametrize(
    "spec_id,sm,ok",
    [
        # SM gate is set by the ACTIVATION dtype (humming check_dtype).
        # bf16 activation (weight-only) => sm80, regardless of weight dtype.
        ("humming_wint4_abfloat16_group_g128", 80, True),
        ("humming_wint4_abfloat16_group_g128", 75, False),
        ("humming_wint8_abfloat16_channel", 75, False),  # bf16 act needs sm80
        ("humming_wfloat8e4m3_abfloat16_channel", 80, True),  # fp8 *weight*, bf16 act
        (
            "humming_wfloat4e2m1_abfloat16_group_g32",
            100,
            True,
        ),  # mxfp4 weight, bf16 act
        # int8 activation (W8A8 / W4A8) => sm75.
        ("humming_wint8_aint8_channel", 75, True),
        ("humming_wint8_aint8_channel", 70, False),
        # fp8 activation (W8A8 fp8) => sm89.
        ("humming_wfloat8e4m3_afloat8e4m3_channel", 89, True),
        ("humming_wfloat8e4m3_afloat8e4m3_channel", 80, False),
        # fp4 activation (W4A4 mxfp4) => sm120.
        ("humming_wfloat4e2m1_afloat4e2m1_channel", 120, True),
        ("humming_wfloat4e2m1_afloat4e2m1_channel", 100, False),
    ],
)
def test_can_handle_sm_gate(monkeypatch, spec_id, sm, ok):
    monkeypatch.setattr(humming_backend, "_HAS_HUMMING", True)
    assert HummingKernel().can_handle(_probe(spec_id, sm=sm)) is ok


# --------------------------------------------------------------------------- #
# materialize routing
# --------------------------------------------------------------------------- #


def _int4(gran=Granularity.PER_CHANNEL, symmetric=True, block_shape=None):
    return QuantScheme(
        weight=TensorQuant(
            QDType.INT4, gran, symmetric=symmetric, block_shape=block_shape
        )
    )


def test_materialize_int4_channel():
    spec = materialize(_int4(), sm=90)
    assert isinstance(spec, HummingWeightSpec)
    assert spec.spec_id == "humming_wint4_abfloat16_channel"
    assert spec.has_zero_point is False


def test_materialize_int4_asymmetric_sets_zero_point():
    spec = materialize(_int4(symmetric=False), sm=90)
    assert spec.has_zero_point is True
    assert spec.spec_id == "humming_wint4_abfloat16_channel_zp"


def test_materialize_int4_block_carries_group():
    spec = materialize(_int4(gran=Granularity.BLOCK, block_shape=(128, 128)), sm=90)
    assert spec.scale_type == "block"
    assert (spec.group_size_n, spec.group_size) == (128, 128)


def test_materialize_bf16_and_nvfp4_are_backend_independent():
    from phyai.layers.quant import Bf16Spec, Nvfp4Spec

    bf16 = materialize(
        QuantScheme(weight=TensorQuant(QDType.BF16, Granularity.PER_TENSOR)), sm=90
    )
    assert isinstance(bf16, Bf16Spec)
    nvfp4 = materialize(
        QuantScheme(weight=TensorQuant(QDType.NVFP4, Granularity.BLOCK)), sm=100
    )
    assert isinstance(nvfp4, Nvfp4Spec)


def test_materialize_fp8_auto_leans_humming():
    # Default (PHYAI_LINEAR_QUANT_BACKEND unset -> auto) sends fp8, incl. block, to humming.
    block = materialize(
        QuantScheme(
            weight=TensorQuant(
                QDType.FP8_E4M3, Granularity.BLOCK, block_shape=(128, 128)
            )
        ),
        sm=100,
    )
    assert isinstance(block, HummingWeightSpec)
    chan = materialize(
        QuantScheme(weight=TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL)),
        sm=100,
    )
    assert isinstance(chan, HummingWeightSpec)


def test_materialize_fp8_flashinfer_backend(monkeypatch):
    from phyai.layers.quant import Fp8Spec

    monkeypatch.setenv("PHYAI_LINEAR_QUANT_BACKEND", "flashinfer")
    block = materialize(
        QuantScheme(
            weight=TensorQuant(
                QDType.FP8_E4M3, Granularity.BLOCK, block_shape=(128, 128)
            )
        ),
        sm=100,
    )
    assert isinstance(block, Fp8Spec)
    chan = materialize(
        QuantScheme(weight=TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL)),
        sm=100,
    )
    assert isinstance(chan, Fp8Spec)
    assert chan.spec_id == "fp8_per_channel"


def test_materialize_humming_only_ignores_backend_pref(monkeypatch):
    # int4/mxfp4/... only humming can run -> stay humming even if flashinfer asked.
    monkeypatch.setenv("PHYAI_LINEAR_QUANT_BACKEND", "flashinfer")
    spec = materialize(
        QuantScheme(weight=TensorQuant(QDType.INT4, Granularity.PER_CHANNEL)), sm=90
    )
    assert isinstance(spec, HummingWeightSpec)


def test_materialize_new_weight_dtypes():
    cases = [
        (QDType.INT2, "humming_wint2_abfloat16_channel"),
        (QDType.INT3, "humming_wint3_abfloat16_channel"),
        (QDType.INT6, "humming_wint6_abfloat16_channel"),
        (QDType.FP6_E2M3, "humming_wfloat6e2m3_abfloat16_channel"),
        (QDType.FP6_E3M2, "humming_wfloat6e3m2_abfloat16_channel"),
    ]
    for dt, sid in cases:
        spec = materialize(
            QuantScheme(weight=TensorQuant(dt, Granularity.PER_CHANNEL)), sm=90
        )
        assert isinstance(spec, HummingWeightSpec)
        assert spec.spec_id == sid


def test_materialize_bad_backend_raises(monkeypatch):
    monkeypatch.setenv("PHYAI_LINEAR_QUANT_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="PHYAI_LINEAR_QUANT_BACKEND"):
        materialize(
            QuantScheme(weight=TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL)),
            sm=90,
        )


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_humming_registered_before_torch_fallback():
    names = [cls.name for cls, _ in list_registered_linear_kernels()]
    assert "humming" in names
    assert names.index("humming") < names.index("torch")


# --------------------------------------------------------------------------- #
# materialize dtype breadth (Phase 3)
# --------------------------------------------------------------------------- #


def test_materialize_int8_channel():
    scheme = QuantScheme(weight=TensorQuant(QDType.INT8, Granularity.PER_CHANNEL))
    spec = materialize(scheme, sm=90)
    assert isinstance(spec, HummingWeightSpec)
    assert spec.spec_id == "humming_wint8_abfloat16_channel"


def test_materialize_fp8_per_channel_routes_to_humming():
    scheme = QuantScheme(weight=TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL))
    spec = materialize(scheme, sm=90)
    assert isinstance(spec, HummingWeightSpec)
    assert spec.spec_id == "humming_wfloat8e4m3_abfloat16_channel"


def test_materialize_fp8_w8a8_per_tensor():
    scheme = QuantScheme(
        weight=TensorQuant(QDType.FP8_E4M3, Granularity.PER_TENSOR),
        input=TensorQuant(QDType.FP8_E4M3, Granularity.PER_TENSOR, dynamic=True),
    )
    spec = materialize(scheme, sm=90)
    assert isinstance(spec, HummingWeightSpec)
    assert spec.spec_id == "humming_wfloat8e4m3_afloat8e4m3_tensor"


def test_materialize_mxfp4():
    scheme = QuantScheme(weight=TensorQuant(QDType.MXFP4, Granularity.PER_CHANNEL))
    spec = materialize(scheme, sm=120)
    assert isinstance(spec, HummingWeightSpec)
    assert spec.spec_id == "humming_wfloat4e2m1_abfloat16_channel"
    # MXFP4's defining e8m0 microscale — this is what distinguishes it from NVFP4.
    assert spec.scale_dtype == "float8e8m0"


def test_nvfp4_and_mxfp4_are_distinct():
    from phyai.layers.quant import Nvfp4Spec

    # NVFP4 (e4m3 scale) -> flashinfer, never humming.
    nvfp4 = materialize(
        QuantScheme(weight=TensorQuant(QDType.NVFP4, Granularity.BLOCK)), sm=100
    )
    assert isinstance(nvfp4, Nvfp4Spec)
    # MXFP4 (e8m0 scale) -> humming, never flashinfer.
    mxfp4 = materialize(
        QuantScheme(weight=TensorQuant(QDType.MXFP4, Granularity.PER_CHANNEL)), sm=100
    )
    assert isinstance(mxfp4, HummingWeightSpec)
    assert mxfp4.scale_dtype == "float8e8m0"


def test_materialize_int8_w8a8_and_w4a8():
    w8a8 = materialize(
        QuantScheme(
            weight=TensorQuant(QDType.INT8, Granularity.PER_CHANNEL),
            input=TensorQuant(QDType.INT8, Granularity.PER_CHANNEL, dynamic=True),
        ),
        sm=90,
    )
    assert w8a8.spec_id == "humming_wint8_aint8_channel"
    w4a8 = materialize(
        QuantScheme(
            weight=TensorQuant(QDType.INT4, Granularity.PER_CHANNEL),
            input=TensorQuant(QDType.INT8, Granularity.PER_CHANNEL, dynamic=True),
        ),
        sm=90,
    )
    assert w4a8.spec_id == "humming_wint4_aint8_channel"


def test_materialize_fp8_e5m2_weight():
    spec = materialize(
        QuantScheme(weight=TensorQuant(QDType.FP8_E5M2, Granularity.PER_CHANNEL)), sm=90
    )
    assert isinstance(spec, HummingWeightSpec)
    assert spec.spec_id == "humming_wfloat8e5m2_abfloat16_channel"


def test_supported_specs_humming_guarded(monkeypatch):
    import phyai.layers.linear as linmod

    monkeypatch.setattr(linmod, "has_humming", lambda: False)
    assert not any(s.startswith("humming_") for s in linmod.supported_specs_for_sm(120))

    monkeypatch.setattr(linmod, "has_humming", lambda: True)
    # int8 activation (W8A8 / W4A8) => sm75; bf16-activation weight-only not yet.
    specs75 = linmod.supported_specs_for_sm(75)
    assert "humming_wint8_aint8_channel" in specs75
    assert "humming_wint4_aint8_channel" in specs75
    assert "humming_wint8_abfloat16_channel" not in specs75  # bf16 act needs sm80
    # bf16 activation (weight-only, incl. fp8/mxfp4 weight) => sm80.
    specs80 = linmod.supported_specs_for_sm(80)
    assert "humming_wint4_abfloat16_channel" in specs80
    assert "humming_wfloat8e4m3_abfloat16_channel" in specs80  # fp8 weight, bf16 act
    assert "humming_wfloat4e2m1_abfloat16_channel" in specs80  # mxfp4 weight, bf16 act
    assert "humming_wfloat8e4m3_afloat8e4m3_channel" not in specs80  # fp8 act needs 89
    # fp8 activation => sm89; fp4 activation => sm120.
    assert "humming_wfloat8e4m3_afloat8e4m3_channel" in linmod.supported_specs_for_sm(
        89
    )
    assert "humming_wfloat4e2m1_afloat4e2m1_channel" in linmod.supported_specs_for_sm(
        120
    )
    assert (
        "humming_wfloat4e2m1_afloat4e2m1_channel"
        not in linmod.supported_specs_for_sm(100)
    )


# --------------------------------------------------------------------------- #
# group-scale fidelity: TensorQuant.group_size flows to a humming group spec
# --------------------------------------------------------------------------- #


def test_materialize_int4_group_from_group_size():
    scheme = QuantScheme(
        weight=TensorQuant(QDType.INT4, Granularity.PER_CHANNEL, group_size=128)
    )
    spec = materialize(scheme, sm=90)
    assert spec.scale_type == "group"
    assert spec.group_size == 128
    assert spec.spec_id == "humming_wint4_abfloat16_group_g128"


def test_compressed_tensors_group_int4_carries_group_size():
    from phyai.layers.quant.importers import ConfigSources, build_quant_plan

    cfg = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "g0": {
                "targets": ["re:.*"],
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": False,
                },
            }
        },
    }
    plan = build_quant_plan(ConfigSources(hf_quant_config=cfg))
    scheme = plan.resolve("model.layers.0.mlp.gate_proj", None)
    assert scheme.weight.dtype is QDType.INT4
    assert scheme.weight.group_size == 128
    assert scheme.weight.symmetric is False
    spec = materialize(scheme, sm=90)
    assert spec.spec_id == "humming_wint4_abfloat16_group_g128_zp"
