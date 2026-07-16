"""ModelOptImporter — NVIDIA ModelOpt fp8/nvfp4 checkpoints 2 QuantPlan.

ModelOpt stores its quant config either in a standalone
``hf_quant_config.json`` (``{"quantization": {"quant_algo": "FP8", ...}}``)
or inline as ``config.json``'s ``quantization_config`` with
``quant_method`` in ``modelopt`` / ``modelopt_fp8`` / ``modelopt_fp4``. In
both shapes a single ``quant_algo`` applies to every linear layer except
``exclude_modules``.
"""

from __future__ import annotations

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.importers.base import ConfigSources
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant

_MODELOPT_METHODS = {"modelopt", "modelopt_fp8", "modelopt_fp4"}


def _quant_block(src: ConfigSources) -> dict | None:
    """Return the dict holding ``quant_algo``, from either config shape."""
    if src.standalone and isinstance(src.standalone.get("quantization"), dict):
        return src.standalone["quantization"]
    cfg = src.hf_quant_config
    if cfg and cfg.get("quant_method") in _MODELOPT_METHODS:
        return cfg
    return None


def _modelopt_raw(block: dict) -> dict:
    raw = dict(block)
    raw["quant_method"] = "modelopt"
    return raw


def _ct_fp8_channel_raw() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "type": "float",
        "num_bits": 8,
        "strategy": "channel",
        "symmetric": True,
    }


def _ct_fp8_token_input_raw() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "type": "float",
        "num_bits": 8,
        "strategy": "token",
        "group_size": 0,
        "symmetric": True,
        "dynamic": True,
    }


def _scheme_for_algo(quant_algo: str, block: dict) -> QuantScheme:
    algo = quant_algo.upper()
    raw = _modelopt_raw(block)
    if algo in ("NVFP4", "FP4"):
        group_size = int(block.get("group_size", 16) or 16)
        if group_size != 16:
            raise ValueError(f"ModelOpt NVFP4 requires group_size=16, got {group_size}")
        weight = TensorQuant(
            QDType.NVFP4,
            Granularity.BLOCK,
            micro_scaled=True,
            block_shape=(1, 16),
            group_size=16,
        )
        act = TensorQuant(
            QDType.NVFP4,
            Granularity.PER_CHANNEL,
            micro_scaled=True,
            dynamic=True,
            group_size=16,
        )
        return QuantScheme(
            weight=weight,
            input=act,
            raw_config=raw,
            input_raw_config=raw,
        )
    if algo == "FP8":
        weight = TensorQuant(QDType.FP8_E4M3, Granularity.PER_TENSOR)
        act = TensorQuant(QDType.FP8_E4M3, Granularity.PER_TENSOR, dynamic=False)
        fp8_raw = {
            "quant_method": "fp8",
            "activation_scheme": "static",
        }
        return QuantScheme(
            weight=weight,
            input=act,
            raw_config=fp8_raw,
            input_raw_config=fp8_raw,
        )
    if algo == "FP8_PER_CHANNEL_PER_TOKEN":
        weight = TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL)
        act = TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL, dynamic=True)
        return QuantScheme(
            weight=weight,
            input=act,
            raw_config=_ct_fp8_channel_raw(),
            input_raw_config=_ct_fp8_token_input_raw(),
        )
    if algo == "FP8_PB_WO":
        weight = TensorQuant(
            QDType.FP8_E4M3,
            Granularity.BLOCK,
            block_shape=(128, 128),
        )
        act = TensorQuant(
            QDType.FP8_E4M3,
            Granularity.PER_CHANNEL,
            dynamic=True,
            group_size=128,
        )
        return QuantScheme(weight=weight, input=act, raw_config=raw)
    if algo == "MXFP8":
        raise NotImplementedError(
            "ModelOpt MXFP8 uses FP8 values with UE8M0 group-32 scales; "
            "it needs a distinct runtime dtype and must not be treated as plain FP8"
        )
    raise NotImplementedError(f"modelopt: unsupported quant_algo {quant_algo!r}")


def _exclude_matchers(pattern: str) -> tuple[Matcher, ...]:
    kind = "glob" if any(c in pattern for c in "*?[") else "name"
    patterns = {pattern}
    fused_names = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    for source, fused in fused_names.items():
        if source in pattern:
            patterns.add(pattern.replace(source, fused))
    return tuple(Matcher(kind, value) for value in sorted(patterns))


class ModelOptImporter:
    name = "modelopt"

    def detect(self, src: ConfigSources) -> bool:
        block = _quant_block(src)
        return block is not None and block.get("quant_algo") is not None

    def build_plan(self, src: ConfigSources) -> QuantPlan:
        block = _quant_block(src) or {}
        scheme = _scheme_for_algo(block["quant_algo"], block)
        excluded = block.get("exclude_modules") or block.get("ignore") or []
        rules = tuple(
            Rule(matcher, None)
            for name in excluded
            for matcher in _exclude_matchers(str(name))
        )
        return QuantPlan(rules=rules, default=scheme)


__all__ = ["ModelOptImporter"]
