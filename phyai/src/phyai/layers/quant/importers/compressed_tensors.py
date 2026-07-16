"""CompressedTensorsImporter — llm-compressor ``config_groups`` 2 QuantPlan."""

from __future__ import annotations

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.importers.base import ConfigSources
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant

_STRATEGY_TO_GRAN = {
    "tensor": Granularity.PER_TENSOR,
    "channel": Granularity.PER_CHANNEL,
    "block": Granularity.BLOCK,
    "token": Granularity.PER_CHANNEL,  # per-token rowwise activation
    "group": Granularity.PER_CHANNEL,  # group size carried on TensorQuant.group_size
    "tensor_group": Granularity.PER_CHANNEL,
}

_FP4_FORMAT_TO_DTYPE = {
    "mxfp4-pack-quantized": QDType.MXFP4,
    "nvfp4-pack-quantized": QDType.NVFP4,
}


def _dtype_of(
    qtype: str,
    num_bits: int,
    compression_format: str | None = None,
) -> QDType:
    if qtype == "float" and num_bits == 4:
        dtype = _FP4_FORMAT_TO_DTYPE.get(compression_format or "")
        if dtype is not None:
            return dtype
        raise ValueError(
            "compressed-tensors: float4 requires group format "
            "'mxfp4-pack-quantized' or 'nvfp4-pack-quantized'"
        )
    if qtype == "float" and num_bits == 8:
        return QDType.FP8_E4M3
    if qtype == "int" and num_bits in (2, 3, 4, 6, 8):
        return QDType(f"int{num_bits}")
    raise NotImplementedError(
        f"compressed-tensors: unsupported type={qtype!r} num_bits={num_bits}"
    )


def _weight_dtype(args: dict, compression_format: str | None = None) -> QDType:
    dtype = _dtype_of(
        args.get("type", "int"),
        int(args.get("num_bits", 8)),
        compression_format,
    )
    humming_dtype = args.get("humming_dtype")
    if humming_dtype is None:
        return dtype
    if dtype not in (QDType.FP8_E4M3, QDType.FP8_E5M2):
        raise ValueError("compressed-tensors: humming_dtype is only valid for FP8")
    if humming_dtype == "float8e4m3":
        return QDType.FP8_E4M3
    if humming_dtype == "float8e5m2":
        return QDType.FP8_E5M2
    raise ValueError(f"compressed-tensors: unsupported humming_dtype {humming_dtype!r}")


def _validate_fp4_args(dtype: QDType, args: dict) -> None:
    if dtype not in (QDType.MXFP4, QDType.NVFP4):
        return
    expected_group_size = 32 if dtype is QDType.MXFP4 else 16
    expected_strategy = "group" if dtype is QDType.MXFP4 else "tensor_group"
    if args.get("strategy") != expected_strategy:
        raise ValueError(
            f"compressed-tensors: {dtype.value} requires strategy={expected_strategy!r}"
        )
    if int(args.get("group_size", 0) or 0) != expected_group_size:
        raise ValueError(
            f"compressed-tensors: {dtype.value} requires "
            f"group_size={expected_group_size}"
        )
    if not bool(args.get("symmetric", True)):
        raise ValueError(f"compressed-tensors: {dtype.value} requires symmetric=True")


def _to_tensorquant(
    args: dict,
    compression_format: str | None = None,
) -> TensorQuant:
    strategy = args.get("strategy", "tensor")
    gran = _STRATEGY_TO_GRAN.get(strategy)
    if gran is None:
        raise NotImplementedError(
            f"compressed-tensors: unsupported strategy {strategy!r}"
        )
    block = args.get("block_structure")
    if strategy == "block" and (
        not isinstance(block, (list, tuple)) or len(block) != 2
    ):
        raise ValueError(
            "compressed-tensors: strategy='block' requires a two-element block_structure"
        )
    block_shape = (int(block[0]), int(block[1])) if strategy == "block" else None
    # ``group`` strategy carries its K-group size; keep 0 for channel/tensor.
    group_size = (
        int(args.get("group_size", 0) or 0)
        if strategy in ("group", "tensor_group")
        else 0
    )
    dtype = _weight_dtype(args, compression_format)
    _validate_fp4_args(dtype, args)
    return TensorQuant(
        dtype=dtype,
        granularity=gran,
        symmetric=bool(args.get("symmetric", True)),
        dynamic=args.get("dynamic", False) not in (False, None),
        micro_scaled=dtype in (QDType.MXFP4, QDType.NVFP4),
        block_shape=block_shape,
        group_size=group_size,
    )


def _target_matcher(target: str) -> Matcher:
    if target.startswith("re:"):
        return Matcher("regex", target[3:])
    if any(c in target for c in "*?["):
        return Matcher("glob", target)
    # A bare CamelCase token (no dot, leading uppercase) is an nn.Module class name.
    if "." not in target and target[:1].isupper():
        return Matcher("module_cls", target)
    return Matcher("name", target)


def _raw_weight_config(config: dict, group: dict) -> dict | None:
    if config.get("pack_format") == "humming":
        return None
    weight = dict(group["weights"])
    fmt = group.get("format") or config.get("format")
    if fmt in (None, "mixed-precision"):
        fmt = "float-quantized" if weight.get("type") == "float" else "pack-quantized"
    weight.update(
        quant_method="compressed-tensors",
        format=fmt,
    )
    return weight


def _raw_input_config(config: dict, group: dict) -> dict | None:
    input_args = group.get("input_activations")
    if input_args is None:
        return None
    activation = dict(input_args)
    fmt = group.get("format") or config.get("format")
    if fmt in (None, "mixed-precision"):
        fmt = (
            "float-quantized" if activation.get("type") == "float" else "int-quantized"
        )
    activation.update(
        quant_method="compressed-tensors",
        format=fmt,
    )
    return activation


def _validate_fp8_scheme(
    weight: TensorQuant,
    weight_args: dict,
    activation: TensorQuant | None,
    activation_args: dict | None,
) -> None:
    """Reject FP8 layouts that have no supported production kernel.

    The runtime supports serialized W8A16 through Humming, plus tensorwise
    W8A8, channel-weight/token-activation W8A8 through Humming, and 128x128
    block-weight with K-grouped dynamic activations through FlashInfer.
    """
    if weight.dtype is not QDType.FP8_E4M3:
        return
    if not weight.symmetric or weight_args.get("dynamic", False) not in (False, None):
        raise ValueError("compressed-tensors FP8 weights must be static and symmetric")
    weight_strategy = weight_args.get("strategy", "tensor")
    if activation is None or activation_args is None:
        if weight_strategy == "tensor":
            valid = (
                weight_args.get("block_structure") is None
                and int(weight_args.get("group_size", 0) or 0) == 0
            )
        elif weight_strategy == "channel":
            valid = (
                weight_args.get("block_structure") is None
                and int(weight_args.get("group_size", 0) or 0) == 0
            )
        elif weight_strategy == "block":
            valid = (
                weight.block_shape == (128, 128)
                and int(weight_args.get("group_size", 0) or 0) == 0
            )
        else:
            valid = False
        if not valid:
            raise ValueError(
                "compressed-tensors FP8 W8A16 supports tensor, channel, or "
                "block_structure=[128, 128] weights"
            )
        return
    if activation.dtype is not QDType.FP8_E4M3:
        raise ValueError(
            "compressed-tensors FP8 production schemes require FP8 input activations"
        )
    if not activation.symmetric:
        raise ValueError("compressed-tensors FP8 activations must be symmetric")

    activation_strategy = activation_args.get("strategy", "tensor")
    if weight_strategy == "tensor":
        if (
            weight_args.get("block_structure") is not None
            or int(weight_args.get("group_size", 0) or 0) != 0
            or activation_strategy != "tensor"
        ):
            raise ValueError(
                "compressed-tensors FP8 tensor weights require strategy='tensor' "
                "tensor activations"
            )
        if (
            activation_args.get("block_structure") is not None
            or int(activation_args.get("group_size", 0) or 0) != 0
        ):
            raise ValueError(
                "compressed-tensors FP8 tensor activations must not set group_size"
            )
        return
    if weight_strategy == "channel":
        if (
            weight_args.get("block_structure") is not None
            or int(weight_args.get("group_size", 0) or 0) != 0
        ):
            raise ValueError(
                "compressed-tensors FP8 channel weights must not set block/group size"
            )
        if activation_strategy != "token" or not activation.dynamic:
            raise ValueError(
                "compressed-tensors FP8 channel weights require dynamic token activations"
            )
        if (
            activation_args.get("block_structure") is not None
            or int(activation_args.get("group_size", 0) or 0) != 0
        ):
            raise ValueError(
                "compressed-tensors FP8 token activations must not set block/group size"
            )
        return
    if weight_strategy == "block":
        if weight.block_shape != (128, 128):
            raise ValueError(
                "compressed-tensors FP8 block weights require block_structure=[128, 128]"
            )
        if int(weight_args.get("group_size", 0) or 0) != 0:
            raise ValueError(
                "compressed-tensors FP8 block weights must not set group_size"
            )
        if (
            activation_strategy != "group"
            or activation.group_size != 128
            or not activation.dynamic
        ):
            raise ValueError(
                "compressed-tensors FP8 block weights require dynamic group-128 activations"
            )
        if activation_args.get("block_structure") is not None:
            raise ValueError(
                "compressed-tensors FP8 group activations must not set block_structure"
            )
        return
    raise ValueError(
        "compressed-tensors FP8 only supports tensor weights or block[128,128] weights; "
        f"got strategy={weight_strategy!r}"
    )


class CompressedTensorsImporter:
    name = "compressed-tensors"

    def detect(self, src: ConfigSources) -> bool:
        cfg = src.hf_quant_config
        return bool(cfg) and cfg.get("quant_method") == "compressed-tensors"

    def build_plan(self, src: ConfigSources) -> QuantPlan:
        cfg = src.hf_quant_config or {}
        rules: list[Rule] = []
        for name in cfg.get("ignore") or []:
            rules.append(Rule(_target_matcher(name), None))
        for group in (cfg.get("config_groups") or {}).values():
            compression_format = group.get("format") or cfg.get("format")
            weight_args = group["weights"]
            weight = _to_tensorquant(weight_args, compression_format)
            if (
                weight.dtype in (QDType.INT3, QDType.INT6)
                and cfg.get("pack_format") != "humming"
            ):
                raise NotImplementedError(
                    "compressed-tensors INT3/INT6 packing differs from Humming's "
                    "continuous bit layout; use pack_format='humming'"
                )
            input_args = group.get("input_activations")
            act = (
                _to_tensorquant(input_args, compression_format) if input_args else None
            )
            _validate_fp8_scheme(weight, weight_args, act, input_args)
            scheme = QuantScheme(
                weight=weight,
                input=act,
                raw_config=_raw_weight_config(cfg, group),
                input_raw_config=_raw_input_config(cfg, group),
                pack_format=cfg.get("pack_format"),
            )
            for target in group.get("targets") or []:
                rules.append(Rule(_target_matcher(target), scheme))
        return QuantPlan(rules=tuple(rules), default=None)


__all__ = ["CompressedTensorsImporter"]
