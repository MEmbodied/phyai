"""Lower a semantic :class:`QuantScheme` to a physical ``WeightSpec``."""

from __future__ import annotations

from phyai.layers.quant.bf16 import Bf16Spec
from phyai.layers.quant.fp8 import Fp8Spec
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.humming import HummingWeightSpec
from phyai.layers.quant.nvfp4 import Nvfp4Spec
from phyai.layers.quant.scheme import QDType, QuantScheme
from phyai.utils.humming import (
    has_humming,
    humming_supports_sm,
    require_humming,
    require_humming_supports_sm,
)

# QDType -> humming canonical dtype string (humming/dtypes.py to_str form).
_HUMMING_WDTYPE = {
    QDType.INT8: "int8",
    QDType.INT6: "int6",
    QDType.INT4: "int4",
    QDType.INT3: "int3",
    QDType.INT2: "int2",
    QDType.FP8_E4M3: "float8e4m3",
    QDType.FP8_E5M2: "float8e5m2",
    QDType.MXFP4: "float4e2m1",
    QDType.FP6_E2M3: "float6e2m3",
    QDType.FP6_E3M2: "float6e3m2",
}
# note(chenghua): Humming owns weight dtypes with no FlashInfer path.
_HUMMING_ONLY = (
    QDType.FP8_E5M2,
    QDType.INT8,
    QDType.INT6,
    QDType.INT4,
    QDType.INT3,
    QDType.INT2,
    QDType.MXFP4,
    QDType.FP6_E2M3,
    QDType.FP6_E3M2,
)
_HUMMING_ADTYPE = {
    QDType.BF16: "bfloat16",
    QDType.INT8: "int8",
    QDType.INT4: "int4",
    QDType.FP8_E4M3: "float8e4m3",
    QDType.FP8_E5M2: "float8e5m2",
    QDType.MXFP4: "float4e2m1",
}
_HUMMING_ACT_SM_GATE = {
    "int8": 75,
    "float16": 75,
    "int4": 80,
    "bfloat16": 80,
    "float8e4m3": 89,
    "float8e5m2": 89,
    "float4e2m1": 120,
}
_GRAN_TO_SCALE = {
    Granularity.PER_TENSOR: "tensor",
    Granularity.PER_CHANNEL: "channel",
    Granularity.BLOCK: "block",
}


def _humming_spec(scheme: QuantScheme) -> HummingWeightSpec:
    """Build a :class:`HummingWeightSpec` from a semantic scheme.

    Scale strategy is derived from ``granularity`` + ``group_size``:
    a positive ``group_size`` (AWQ/GPTQ, llm-compressor ``group``) becomes a
    ``group`` scale; ``BLOCK`` becomes a 2-D ``block`` scale; ``PER_TENSOR`` a
    ``tensor`` scale; otherwise ``channel``.

    MXFP4 is the OCP microscaling format: E2M1 weight with an **e8m0 (ue8m0)**
    block scale. That e8m0 scale is what distinguishes it from NVFP4 (E2M1 with
    an e4m3 scale), which is routed separately to ``Nvfp4Spec`` / flashinfer and
    never reaches this function.
    """
    w = scheme.weight
    w_dtype = _HUMMING_WDTYPE[w.dtype]
    if scheme.input is not None and not scheme.input.symmetric:
        raise ValueError("Humming activation quantization requires symmetric inputs")
    if scheme.input is None:
        a_dtype = "bfloat16"
    else:
        try:
            a_dtype = _HUMMING_ADTYPE[scheme.input.dtype]
        except KeyError as exc:
            raise ValueError(
                "Humming does not support activation dtype "
                f"{scheme.input.dtype.value!r}"
            ) from exc
    group_size = int(w.group_size)
    group_size_n = 0
    if w.granularity is Granularity.BLOCK and w.block_shape is not None:
        scale_type = "block"
        group_size_n, group_size = int(w.block_shape[0]), int(w.block_shape[1])
    elif group_size > 0:
        scale_type = "group"
    else:
        scale_type = _GRAN_TO_SCALE[w.granularity]
    # MXFP4's defining microscale is e8m0; this is what separates it from
    # NVFP4 (e4m3 scale, flashinfer-only). Other dtypes leave humming to pick
    # the scale dtype from the schema default.
    scale_dtype = "float8e8m0" if w.dtype is QDType.MXFP4 else None
    return HummingWeightSpec(
        w_dtype=w_dtype,
        a_dtype=a_dtype,
        scale_type=scale_type,
        group_size=group_size,
        group_size_n=group_size_n,
        has_zero_point=not w.symmetric,
        scale_dtype=scale_dtype,
        input_group_size=(
            int(scheme.input.group_size) if scheme.input is not None else 0
        ),
        input_dynamic=(scheme.input.dynamic if scheme.input is not None else True),
        raw_config=scheme.raw_config,
        input_raw_config=scheme.input_raw_config,
        online=scheme.online,
        native_storage=scheme.pack_format == "humming",
    )


def _validate_fp8_runtime_scheme(scheme: QuantScheme) -> None:
    w = scheme.weight
    a = scheme.input
    if not w.symmetric or w.dynamic or w.group_size:
        raise ValueError("FP8 E4M3 production weights must be static and symmetric")
    if a is None:
        return
    if a.dtype is not QDType.FP8_E4M3 or not a.symmetric:
        raise ValueError(
            "FP8 E4M3 production schemes require symmetric FP8 input activations"
        )
    if w.granularity is Granularity.PER_TENSOR:
        if w.block_shape is not None:
            raise ValueError("FP8 tensorwise weights must not set block_shape")
        if a.granularity is not Granularity.PER_TENSOR or a.group_size:
            raise ValueError("FP8 tensorwise weights require tensorwise activations")
        return
    if w.granularity is Granularity.PER_CHANNEL:
        if w.block_shape is not None:
            raise ValueError("FP8 channelwise weights must not set block_shape")
        if (
            a.granularity is not Granularity.PER_CHANNEL
            or a.group_size
            or not a.dynamic
        ):
            raise ValueError(
                "FP8 channelwise weights require dynamic per-token activations"
            )
        return
    if w.granularity is Granularity.BLOCK:
        if w.block_shape != (128, 128):
            raise ValueError("FP8 block weights require block_shape=(128, 128)")
        if a.granularity is not Granularity.PER_CHANNEL or a.group_size != 128:
            raise ValueError(
                "FP8 block weights require dynamic K-grouped activations with group_size=128"
            )
        return
    raise ValueError(
        "FP8 E4M3 production only supports per-tensor, per-channel, or "
        "block-128 weights; "
        f"got {w.granularity.value}"
    )


def _require_humming_runtime(scheme: QuantScheme, sm: int) -> None:
    require_humming()
    require_humming_supports_sm(sm)
    spec = _humming_spec(scheme)
    required_sm = _HUMMING_ACT_SM_GATE.get(spec.a_dtype)
    if required_sm is None:
        raise ValueError(f"Humming does not support activation dtype {spec.a_dtype!r}")
    if sm < required_sm:
        raise RuntimeError(
            f"Humming activation dtype {spec.a_dtype!r} requires sm_{required_sm}+; "
            f"got sm_{sm}"
        )


def _quant_backend() -> str:
    """Backend preference for quantized Linear (``PHYAI_LINEAR_QUANT_BACKEND``).

    ``auto`` (default) leans humming for every format humming can run.
    """
    from phyai.env import envs

    backend = (envs.PHYAI_LINEAR_QUANT_BACKEND.get() or "auto").lower()
    if backend not in ("auto", "humming", "flashinfer"):
        raise ValueError(
            f"PHYAI_LINEAR_QUANT_BACKEND must be auto/humming/flashinfer, got "
            f"{backend!r}"
        )
    return backend


def _flashinfer_supported_specs_for_sm(sm: int) -> set[str]:
    from phyai.layers.linear.backends.flashinfer import (
        flashinfer_supported_specs_for_sm,
    )

    return flashinfer_supported_specs_for_sm(sm)


def _fp8_spec(scheme: QuantScheme) -> Fp8Spec:
    return Fp8Spec(
        granularity=scheme.weight.granularity,
        block_shape=scheme.weight.block_shape,
        activation=scheme.input,
        raw_config=scheme.raw_config,
        online=scheme.online,
    )


def _is_modelopt_pbwo(scheme: QuantScheme) -> bool:
    raw = scheme.raw_config or {}
    return (
        str(raw.get("quant_method", "")).lower() == "modelopt"
        and str(raw.get("quant_algo", "")).upper() == "FP8_PB_WO"
    )


def _require_flashinfer_spec(spec_id: str, sm: int) -> None:
    supported = _flashinfer_supported_specs_for_sm(sm)
    if spec_id not in supported:
        available = ", ".join(sorted(supported)) or "none"
        raise RuntimeError(
            f"FlashInfer cannot run {spec_id!r} on sm_{sm}; "
            f"available FlashInfer specs: {available}"
        )


def materialize(scheme: QuantScheme, sm: int) -> object:
    """Return the physical ``WeightSpec`` for ``scheme`` on an SM-``sm`` device.

    humming is a kernel *and* a weight layout; because that layout is fixed at
    load time, the backend choice is made here (not per-forward). The semantic
    ``scheme`` stays backend-agnostic; ``PHYAI_LINEAR_QUANT_BACKEND`` decides which
    physical spec (and thus kernel) serves formats more than one backend can do
    (fp8). Formats only humming can run always go to humming; NVFP4 (e4m3 scale,
    128x4) is flashinfer-only; bf16 is unquantized.
    """
    w = scheme.weight
    if w.dtype is QDType.FP8_E4M3:
        _validate_fp8_runtime_scheme(scheme)
    if scheme.pack_format == "humming":
        if w.dtype in (QDType.MXFP4, QDType.NVFP4):
            raise ValueError(
                f"{w.dtype.value} checkpoints must use their standard "
                "compressed-tensors pack format, not pack_format='humming'"
            )
        _require_humming_runtime(scheme, sm)
        return _humming_spec(scheme)
    if w.dtype is QDType.BF16:
        return Bf16Spec()
    if w.dtype is QDType.NVFP4:
        if (
            scheme.input is None
            or scheme.input.dtype is not QDType.NVFP4
            or not scheme.input.dynamic
            or w.group_size != 16
            or scheme.input.group_size != 16
        ):
            raise ValueError(
                "NVFP4 runtime requires group-16 weights and dynamic group-16 "
                "NVFP4 input activations (W4A4)"
            )
        spec = Nvfp4Spec(
            scale_layout="128x4",
            raw_config=scheme.raw_config,
            input_raw_config=scheme.input_raw_config,
            online=scheme.online,
        )
        _require_flashinfer_spec(spec.spec_id, sm)
        return spec
    if w.dtype is QDType.FP8_E4M3:
        backend = _quant_backend()
        if scheme.input is None:
            if backend == "flashinfer":
                raise RuntimeError(
                    "FlashInfer FP8 dense GEMM requires quantized FP8 activations; "
                    "weight-only FP8 requires Humming"
                )
            _require_humming_runtime(scheme, sm)
            return _humming_spec(scheme)
        spec = _fp8_spec(scheme)
        if backend == "flashinfer":
            _require_flashinfer_spec(spec.spec_id, sm)
            return spec
        if backend == "humming":
            if _is_modelopt_pbwo(scheme):
                raise RuntimeError(
                    "ModelOpt FP8_PB_WO checkpoint layout requires the FlashInfer "
                    "FP8 block backend; use PHYAI_LINEAR_QUANT_BACKEND=auto or "
                    "flashinfer"
                )
            _require_humming_runtime(scheme, sm)
            return _humming_spec(scheme)

        flashinfer_specs = _flashinfer_supported_specs_for_sm(sm)
        # note(chenghua): Humming wins SM90 tensor latency; FlashInfer owns
        # block FP8 and the probed Blackwell/Thor tensor paths.
        if w.granularity is Granularity.BLOCK and spec.spec_id in flashinfer_specs:
            return spec
        if sm in (100, 103, 110, 120, 121) and spec.spec_id in flashinfer_specs:
            return spec
        if has_humming() and humming_supports_sm(sm):
            _require_humming_runtime(scheme, sm)
            return _humming_spec(scheme)
        if spec.spec_id in flashinfer_specs:
            return spec
        raise RuntimeError(
            f"no runtime backend can run {spec.spec_id!r} on sm_{sm}; "
            "install a compatible FlashInfer/Humming build or choose a supported scheme"
        )
    if w.dtype in _HUMMING_ONLY:
        if (
            w.dtype is QDType.MXFP4
            and scheme.input is not None
            and scheme.input.dtype is QDType.MXFP4
        ):
            raise RuntimeError(
                "MXFP4 W4A4 group-32 is unsupported by Humming 0.1.10; "
                "use MXFP4 W4A16 or W4A8 with FP8 E4M3 activations"
            )
        # note(chenghua): no FlashInfer path for these dtypes, so an SM Humming
        # can't serve has no fallback — fail loudly instead of KeyError'ing in humming.
        _require_humming_runtime(scheme, sm)
        return _humming_spec(scheme)
    raise NotImplementedError(f"materialize: unsupported weight dtype {w.dtype!r}")


__all__ = ["materialize"]
