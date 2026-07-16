"""Numerical core for weight quantization.

Weight layout is ``(out_features, in_features)`` or ``(N, K)`` and grouping is
always along K. Integer and FP8 qparams follow compressed-tensors. MXFP4 and
NVFP4 additionally return their final packed E2M1 bytes and storage-ready scale
tensors so serialization never has to infer codes from a fake-quantized weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from compressed_tensors.compressors.mx_utils import compress_mx_scale
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import (
    fake_quantize as ct_fake_quantize,
)
from compressed_tensors.quantization.lifecycle.forward import quantize as ct_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0
_NVFP4_BLOCK_SIZE = 16
_MXFP4_BLOCK_SIZE = 32
_FP8_BLOCK_SIZE = 128


class QuantDType(str, Enum):
    """Explicit quantized data type supported by the optimizer."""

    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    INT6 = "int6"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"

    @property
    def is_integer(self) -> bool:
        return self.value.startswith("int")

    @property
    def is_fp8(self) -> bool:
        return self in (QuantDType.FP8_E4M3, QuantDType.FP8_E5M2)

    @property
    def is_fp4(self) -> bool:
        return self in (QuantDType.MXFP4, QuantDType.NVFP4)

    @property
    def supports_activation(self) -> bool:
        return self in (
            QuantDType.INT4,
            QuantDType.INT8,
            QuantDType.FP8_E4M3,
            QuantDType.FP8_E5M2,
            QuantDType.MXFP4,
            QuantDType.NVFP4,
        )


class FP8Scheme(str, Enum):
    """Supported FP8 scale layouts."""

    TENSORWISE = "tensorwise"
    BLOCK_128 = "block-128"


@dataclass(frozen=True)
class WeightQuant:
    """Semantic weight-quant specification with an explicit element dtype."""

    dtype: QuantDType | str
    symmetric: bool = True
    group_size: int = 0
    fp8_scheme: FP8Scheme | str | None = None

    def __post_init__(self) -> None:
        try:
            dtype = QuantDType(self.dtype)
        except ValueError as exc:
            raise ValueError(f"unsupported quantized dtype {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", dtype)

        if self.group_size < 0:
            raise ValueError(f"group_size must be non-negative, got {self.group_size}")
        if self.is_fp8:
            if not self.symmetric:
                raise ValueError("fp8 weight quant must be symmetric")
            if self.group_size != 0:
                raise ValueError(
                    "fp8 weight quant uses fp8_scheme instead of group_size"
                )
            if self.fp8_scheme is None:
                raise ValueError(
                    "fp8 weight quant requires fp8_scheme='tensorwise' or 'block-128'"
                )
            try:
                fp8_scheme = FP8Scheme(self.fp8_scheme)
            except ValueError as exc:
                raise ValueError(f"unsupported fp8_scheme {self.fp8_scheme!r}") from exc
            object.__setattr__(self, "fp8_scheme", fp8_scheme)
        elif self.fp8_scheme is not None:
            raise ValueError("fp8_scheme is only valid for fp8 weight quant")
        if self.is_fp4:
            expected = _MXFP4_BLOCK_SIZE if self.is_mxfp4 else _NVFP4_BLOCK_SIZE
            if not self.symmetric:
                raise ValueError(f"{self.dtype.value} weight quant must be symmetric")
            if self.group_size != expected:
                raise ValueError(
                    f"{self.dtype.value} weight quant requires group_size={expected}, "
                    f"got {self.group_size}"
                )

    @property
    def strategy(self) -> str:
        if self.is_fp8:
            return "tensor" if self.fp8_scheme is FP8Scheme.TENSORWISE else "block"
        if self.is_nvfp4:
            return "tensor_group"
        return "group" if self.group_size > 0 else "channel"

    @property
    def block_structure(self) -> tuple[int, int] | None:
        if self.is_fp8 and self.fp8_scheme is FP8Scheme.BLOCK_128:
            return (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE)
        return None

    @property
    def num_bits(self) -> int:
        if self.is_fp8 or self.dtype is QuantDType.INT8:
            return 8
        if self.is_fp4 or self.dtype is QuantDType.INT4:
            return 4
        return int(self.dtype.value.removeprefix("int"))

    @property
    def is_integer(self) -> bool:
        return self.dtype.is_integer

    @property
    def is_fp8(self) -> bool:
        return self.dtype.is_fp8

    @property
    def is_fp4(self) -> bool:
        return self.dtype.is_fp4

    @property
    def is_mxfp4(self) -> bool:
        return self.dtype is QuantDType.MXFP4

    @property
    def is_nvfp4(self) -> bool:
        return self.dtype is QuantDType.NVFP4

    @property
    def is_float(self) -> bool:
        return self.is_fp8 or self.is_fp4

    @property
    def is_e5m2(self) -> bool:
        return self.dtype is QuantDType.FP8_E5M2

    def int_range(self) -> tuple[int, int]:
        if not self.is_integer:
            raise ValueError(f"int_range is undefined for {self.dtype.value}")
        return -(1 << (self.num_bits - 1)), (1 << (self.num_bits - 1)) - 1


def to_ct_args(q: WeightQuant) -> QuantizationArgs:
    strategy = QuantizationStrategy(q.strategy)
    kwargs: dict = {
        "num_bits": q.num_bits,
        "type": QuantizationType.FLOAT if q.is_float else QuantizationType.INT,
        "symmetric": q.symmetric,
        "strategy": strategy,
    }
    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        kwargs["group_size"] = q.group_size
    if strategy is QuantizationStrategy.BLOCK:
        kwargs["block_structure"] = list(q.block_structure or ())
    if q.is_mxfp4:
        kwargs.update(scale_dtype=torch.uint8, zp_dtype=torch.uint8)
    elif q.is_nvfp4:
        kwargs.update(
            scale_dtype=torch.float8_e4m3fn,
            zp_dtype=torch.float8_e4m3fn,
        )
    return QuantizationArgs(**kwargs)


def activation_args(
    dtype: QuantDType | str,
    *,
    fp8_scheme: FP8Scheme | str | None = None,
) -> QuantizationArgs:
    """Build the dynamic activation schema paired with an RTN weight dtype."""
    dtype = QuantDType(dtype)
    if not dtype.supports_activation:
        raise ValueError(f"{dtype.value} is not supported for activation quantization")
    if dtype is QuantDType.MXFP4:
        return QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP,
            group_size=_MXFP4_BLOCK_SIZE,
            dynamic=True,
            scale_dtype=torch.uint8,
            zp_dtype=torch.uint8,
        )
    if dtype is QuantDType.NVFP4:
        return QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=_NVFP4_BLOCK_SIZE,
            dynamic=True,
            scale_dtype=torch.float8_e4m3fn,
            zp_dtype=torch.float8_e4m3fn,
        )
    if dtype.is_fp8 and fp8_scheme is not None:
        scheme = FP8Scheme(fp8_scheme)
        if scheme is FP8Scheme.BLOCK_128:
            return QuantizationArgs(
                num_bits=8,
                type=QuantizationType.FLOAT,
                symmetric=True,
                strategy=QuantizationStrategy.GROUP,
                group_size=_FP8_BLOCK_SIZE,
                dynamic=True,
            )
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR,
            dynamic=True,
        )
    return QuantizationArgs(
        num_bits=8 if dtype.is_fp8 else int(dtype.value.removeprefix("int")),
        type=QuantizationType.FLOAT if dtype.is_fp8 else QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )


def _grouped_view(w: torch.Tensor, group_size: int) -> torch.Tensor:
    n, k = w.shape
    if group_size <= 0 or group_size == k:
        return w.view(n, 1, k)
    if k % group_size != 0:
        raise ValueError(f"in_features={k} not divisible by group_size={group_size}")
    return w.view(n, k // group_size, group_size)


def _minmax(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = _grouped_view(w.float(), group_size)
    return grouped.amin(dim=-1), grouped.amax(dim=-1)


def compute_scale_zp(
    w: torch.Tensor, q: WeightQuant
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return scale and optional zero point in compressed-tensors convention."""
    if q.is_fp4:
        raise ValueError("FP4 uses quantize_fp4 to produce authoritative artifacts")
    if q.is_fp8:
        if w.ndim != 2:
            raise ValueError(f"FP8 weight must be 2-D, got shape={tuple(w.shape)}")
        fp8_max = _FP8_E5M2_MAX if q.is_e5m2 else _FP8_E4M3_MAX
        wf = w.float()
        if q.fp8_scheme is FP8Scheme.TENSORWISE:
            scale = (wf.abs().amax() / fp8_max).clamp_min(1e-12).reshape(1)
        else:
            n, k = wf.shape
            block_n, block_k = q.block_structure or (0, 0)
            if n % block_n != 0 or k % block_k != 0:
                raise ValueError(
                    "fp8 block-128 requires out_features and in_features divisible "
                    f"by 128, got ({n}, {k})"
                )
            blocks = wf.view(n // block_n, block_n, k // block_k, block_k)
            scale = (blocks.abs().amax(dim=(1, 3)) / fp8_max).clamp_min(1e-12)
        return scale, None
    wmin, wmax = _minmax(w, q.group_size)
    scale, zp = calculate_qparams(wmin, wmax, to_ct_args(q))
    if q.symmetric:
        return scale.float(), None
    return scale.float(), zp.float()


def _repeat_to_k(tensor: torch.Tensor, k: int) -> torch.Tensor:
    n, groups = tensor.shape
    return tensor.unsqueeze(-1).expand(n, groups, k // groups).reshape(n, k)


def _expand_fp8_scale(
    scale: torch.Tensor, shape: tuple[int, int], q: WeightQuant
) -> torch.Tensor:
    n, k = shape
    if q.fp8_scheme is FP8Scheme.TENSORWISE:
        if scale.numel() != 1:
            raise ValueError(
                f"fp8 tensorwise scale must contain one value, got shape={tuple(scale.shape)}"
            )
        return scale.reshape(1, 1).expand(n, k)
    block_n, block_k = q.block_structure or (0, 0)
    expected = (n // block_n, k // block_k)
    if n % block_n != 0 or k % block_k != 0 or tuple(scale.shape) != expected:
        raise ValueError(
            f"fp8 block-128 scale must have shape {expected} for weight {(n, k)}, "
            f"got {tuple(scale.shape)}"
        )
    return scale.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)


def apply_fake_quant(
    w: torch.Tensor, q: WeightQuant, scale: torch.Tensor, zp: torch.Tensor | None
) -> torch.Tensor:
    """Elementwise integer or FP8 fake quantization."""
    if q.is_fp4:
        raise ValueError("FP4 uses quantize_fp4 to produce authoritative artifacts")
    wf = w.float()
    _, k = wf.shape
    if q.is_fp8:
        expanded_scale = _expand_fp8_scale(scale, tuple(wf.shape), q)
        fp8_dtype = torch.float8_e5m2 if q.is_e5m2 else torch.float8_e4m3fn
        return (wf / expanded_scale).to(fp8_dtype).float() * expanded_scale
    expanded_scale = _repeat_to_k(scale, k)
    qmin, qmax = q.int_range()
    if zp is None:
        codes = torch.clamp(torch.round(wf / expanded_scale), qmin, qmax)
        return codes * expanded_scale
    expanded_zp = _repeat_to_k(zp, k)
    codes = torch.clamp(torch.round(wf / expanded_scale) + expanded_zp, qmin, qmax)
    return (codes - expanded_zp) * expanded_scale


def fake_quantize(
    w: torch.Tensor,
    q: WeightQuant,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale is None:
        scale, zero_point = compute_scale_zp(w, q)
    return apply_fake_quant(w, q, scale, zero_point).to(w.dtype)


def quantize_to_codes(
    w: torch.Tensor,
    q: WeightQuant,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
) -> torch.Tensor:
    """Return signed integer codes in compressed-tensors convention."""
    if not q.is_integer:
        raise ValueError("quantize_to_codes is int-only")
    wf = w.float()
    _, k = wf.shape
    expanded_scale = _repeat_to_k(scale, k)
    qmin, qmax = q.int_range()
    if zero_point is None:
        return torch.clamp(torch.round(wf / expanded_scale), qmin, qmax).to(torch.int32)
    expanded_zp = _repeat_to_k(zero_point, k)
    return torch.clamp(torch.round(wf / expanded_scale) + expanded_zp, qmin, qmax).to(
        torch.int32
    )


def _quantize_mxfp4(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    q = WeightQuant(QuantDType.MXFP4, group_size=_MXFP4_BLOCK_SIZE)
    args = to_ct_args(q)
    group_min, group_max = _minmax(w, _MXFP4_BLOCK_SIZE)
    scale, zero_point = calculate_qparams(group_min, group_max, args)
    quantized = ct_quantize(
        x=w.float(),
        scale=scale,
        zero_point=zero_point,
        args=args,
    )
    fake = ct_fake_quantize(
        x=w.float(),
        scale=scale,
        zero_point=zero_point,
        args=args,
    ).to(w.dtype)
    packed = pack_fp4_to_uint8(quantized)
    encoded_scale = compress_mx_scale(scale, torch.uint8)
    return fake, packed.contiguous(), encoded_scale.contiguous(), None


def _quantize_nvfp4(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = WeightQuant(QuantDType.NVFP4, group_size=_NVFP4_BLOCK_SIZE)
    args = to_ct_args(q)
    wf = w.float()
    global_scale = generate_gparam(wf.amin(), wf.amax())
    group_min, group_max = _minmax(wf, _NVFP4_BLOCK_SIZE)
    scale, zero_point = calculate_qparams(
        group_min,
        group_max,
        args,
        global_scale=global_scale,
    )
    quantized = ct_quantize(
        x=wf,
        scale=scale,
        zero_point=zero_point,
        args=args,
        global_scale=global_scale,
    )
    fake = ct_fake_quantize(
        x=wf,
        scale=scale,
        zero_point=zero_point,
        args=args,
        global_scale=global_scale,
    ).to(w.dtype)
    packed = pack_fp4_to_uint8(quantized)
    encoded_scale = scale.to(torch.float8_e4m3fn)
    return (
        fake,
        packed.contiguous(),
        encoded_scale.contiguous(),
        global_scale.to(torch.float32).contiguous(),
    )


def quantize_fp4(
    w: torch.Tensor, q: WeightQuant
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return fake weight, packed E2M1 bytes, encoded scale, and CT global scale."""
    if w.ndim != 2:
        raise ValueError(f"FP4 weight must be 2-D, got shape={tuple(w.shape)}")
    if not q.is_fp4:
        raise ValueError(f"quantize_fp4 requires an FP4 dtype, got {q.dtype.value}")
    if w.shape[-1] % q.group_size != 0:
        raise ValueError(
            f"in_features={w.shape[-1]} not divisible by {q.dtype.value} "
            f"group_size={q.group_size}"
        )
    if q.is_mxfp4:
        return _quantize_mxfp4(w)
    return _quantize_nvfp4(w)


__all__ = [
    "QuantDType",
    "FP8Scheme",
    "WeightQuant",
    "to_ct_args",
    "activation_args",
    "compute_scale_zp",
    "fake_quantize",
    "apply_fake_quant",
    "quantize_to_codes",
    "quantize_fp4",
]
