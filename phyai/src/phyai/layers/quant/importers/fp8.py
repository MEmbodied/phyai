"""Fp8Importer — HF flat fp8 ``quantization_config`` 2 QuantPlan."""

from __future__ import annotations

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.importers.base import ConfigSources
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant


def _ignored_matchers(pattern: str) -> tuple[Matcher, ...]:
    kind = "glob" if any(c in pattern for c in "*?[") else "name"
    patterns = {pattern}
    for source, fused in {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }.items():
        if source in pattern:
            patterns.add(pattern.replace(source, fused))
    return tuple(Matcher(kind, value) for value in sorted(patterns))


class Fp8Importer:
    name = "fp8"

    def detect(self, src: ConfigSources) -> bool:
        cfg = src.hf_quant_config
        return bool(cfg) and cfg.get("quant_method") == "fp8"

    def build_plan(self, src: ConfigSources) -> QuantPlan:
        cfg = src.hf_quant_config or {}
        block = cfg.get("weight_block_size")
        dynamic = cfg.get("activation_scheme", "dynamic") != "static"
        if block is not None:
            if len(block) != 2:
                raise ValueError("fp8 weight_block_size must contain two integers")
            block_n, block_k = int(block[0]), int(block[1])
            weight = TensorQuant(
                QDType.FP8_E4M3,
                Granularity.BLOCK,
                block_shape=(block_n, block_k),
            )
            if not dynamic:
                raise ValueError("block FP8 requires dynamic activation quantization")
            act = TensorQuant(
                QDType.FP8_E4M3,
                Granularity.PER_CHANNEL,
                dynamic=True,
                group_size=block_k,
            )
        else:
            # note(chenghua): Serialized fused checkpoints need one common scale
            # because the production tensorwise GEMM accepts a scalar scale.
            weight = TensorQuant(QDType.FP8_E4M3, Granularity.PER_TENSOR)
            act = TensorQuant(
                QDType.FP8_E4M3,
                Granularity.PER_TENSOR,
                dynamic=dynamic,
            )

        default = QuantScheme(
            weight=weight,
            input=act,
            online=bool(cfg.get("online", False)),
            raw_config=dict(cfg),
            input_raw_config=dict(cfg),
        )

        ignored = cfg.get("ignored_layers") or cfg.get("modules_to_not_convert") or []
        rules = tuple(
            Rule(matcher, None)
            for name in ignored
            for matcher in _ignored_matchers(str(name))
        )
        return QuantPlan(rules=rules, default=default)


__all__ = ["Fp8Importer"]
