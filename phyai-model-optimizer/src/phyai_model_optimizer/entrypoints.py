"""User-facing entrypoints: ``oneshot`` (build/calibrate/quantize/save) and
``model_free_ptq`` (data-free, operates on a state dict without a model)."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn as nn

from phyai_model_optimizer.modifiers.base import Modifier, QuantResult
from phyai_model_optimizer.orchestrator import run_oneshot
from phyai_model_optimizer.pipelines.base import SequentialDriver, Target
from phyai_model_optimizer.quant_math import FP8Scheme

_PACK_FORMATS = ("compressed-tensors", "humming")


def _as_list(modifiers: Modifier | list[Modifier]) -> list[Modifier]:
    return [modifiers] if isinstance(modifiers, Modifier) else list(modifiers)


def _validate_modifier_pack_format(modifiers: list[Modifier], pack_format: str) -> None:
    if pack_format == "compressed-tensors":
        native_only = sorted(
            {
                modifier.weight_quant().dtype.value
                for modifier in modifiers
                if modifier.weight_quant().num_bits in (3, 6)
            }
        )
        if native_only:
            raise ValueError(
                f"{', '.join(native_only)} use Humming's continuous bit layout; "
                "use pack_format='humming'"
            )
    if pack_format == "compressed-tensors" and any(
        modifier.weight_quant().is_e5m2 for modifier in modifiers
    ):
        raise ValueError(
            "FP8 E5M2 is not distinguishable in the standard compressed-tensors "
            "float8 schema; use pack_format='humming'"
        )
    if pack_format != "humming":
        return
    fp4_dtypes = sorted(
        {
            modifier.weight_quant().dtype.value
            for modifier in modifiers
            if modifier.weight_quant().is_fp4
        }
    )
    if fp4_dtypes:
        raise ValueError(
            f"FP4 ({', '.join(fp4_dtypes)}) only supports "
            "pack_format='compressed-tensors'"
        )


def oneshot(
    model: nn.Module,
    modifiers: Modifier | list[Modifier],
    dataloader: Iterable | None = None,
    *,
    driver: SequentialDriver | None = None,
    pack_format: str = "compressed-tensors",
    save_dir: str | None = None,
    base_checkpoint: str | None = None,
) -> nn.Module:
    """Quantize ``model`` in one shot.

    The model must be built eager (``use_cuda_graph=False``) on the bf16 baseline
    (``use_quant_plan(None)``). Calibration modifiers (e.g. GPTQ) additionally need
    a ``dataloader`` and a model-specific :class:`SequentialDriver` — see
    ``pipelines.GenericSequentialDriver`` for plain block stacks, or
    ``compat.phyai_model.build_driver`` for pi0.5.

    ``base_checkpoint`` makes the output a complete, phyai-loadable checkpoint:
    non-quantized params are copied verbatim (HF names), only quantized layers
    replaced — required for phyai models whose on-disk keys are HF-named."""
    if pack_format not in _PACK_FORMATS:
        raise ValueError(
            f"pack_format must be one of {_PACK_FORMATS}, got {pack_format!r}"
        )
    mods = _as_list(modifiers)
    _validate_modifier_pack_format(mods, pack_format)

    if any(m.requires_calibration for m in mods):
        if dataloader is None:
            raise ValueError("calibration modifiers require a dataloader")
        if driver is None:
            raise ValueError(
                "calibration modifiers require a model-specific SequentialDriver; "
                "use pipelines.GenericSequentialDriver for a plain block stack, or "
                "compat.phyai_model.build_driver(model, num_timesteps=...) for pi0.5"
            )

    run_oneshot(
        model,
        mods,
        dataloader=dataloader,
        driver=driver,
        pack_format=pack_format,
        save_dir=save_dir,
        base_checkpoint=base_checkpoint,
    )
    return model


class _Holder(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


_GATE_SUFFIX = ".mlp.gate_proj"
_UP_SUFFIX = ".mlp.up_proj"


def _fp8_fused_sibling(name: str) -> str | None:
    if name.endswith(_GATE_SUFFIX):
        return f"{name[: -len(_GATE_SUFFIX)]}{_UP_SUFFIX}"
    if name.endswith(_UP_SUFFIX):
        return f"{name[: -len(_UP_SUFFIX)]}{_GATE_SUFFIX}"
    return None


def _slice_fp8_result(result: QuantResult, start: int, stop: int) -> QuantResult:
    q = result.q
    if not q.is_fp8:
        raise ValueError("_slice_fp8_result requires an FP8 result")
    if q.fp8_scheme is FP8Scheme.TENSORWISE:
        scale = result.scale
    else:
        block_n = (q.block_structure or (0, 0))[0]
        if start % block_n != 0 or stop % block_n != 0:
            raise ValueError(
                "fp8 block-128 fused legs must align to 128 output rows, "
                f"got row range [{start}, {stop})"
            )
        scale = result.scale[start // block_n : stop // block_n]
    return QuantResult(
        q=q,
        scale=scale,
        zero_point=None,
        fake_weight=result.fake_weight[start:stop],
    )


def _quantize_model_free_targets(
    targets: list[Target],
    progress_callback: Callable[[int, int, str], None] | None,
) -> None:
    by_name = {name: (holder, modifier) for name, holder, modifier in targets}
    processed: set[str] = set()
    completed = 0

    with torch.no_grad():
        for name, holder, modifier in targets:
            if name in processed:
                continue
            q = modifier.weight_quant()
            sibling_name = _fp8_fused_sibling(name) if q.is_fp8 else None
            if sibling_name is None:
                modifier.quantize_layer(holder, None)
                processed.add(name)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(targets), name)
                continue

            sibling = by_name.get(sibling_name)
            if sibling is None:
                raise ValueError(
                    f"FP8 target {name!r} requires its fused sibling {sibling_name!r}"
                )
            sibling_holder, sibling_modifier = sibling
            if sibling_modifier is not modifier:
                raise ValueError(
                    f"fused FP8 siblings {name!r} and {sibling_name!r} must use "
                    "the same modifier"
                )
            if holder.weight.shape[1] != sibling_holder.weight.shape[1]:
                raise ValueError(
                    f"fused FP8 siblings {name!r} and {sibling_name!r} must have "
                    "the same in_features"
                )
            if holder.weight.dtype != sibling_holder.weight.dtype:
                raise ValueError(
                    f"fused FP8 siblings {name!r} and {sibling_name!r} must have "
                    "the same dtype"
                )

            first_name, first_holder = name, holder
            second_name, second_holder = sibling_name, sibling_holder
            if name.endswith(_UP_SUFFIX):
                first_name, first_holder = sibling_name, sibling_holder
                second_name, second_holder = name, holder
            first_rows = int(first_holder.weight.shape[0])
            second_rows = int(second_holder.weight.shape[0])
            if q.fp8_scheme is FP8Scheme.BLOCK_128 and (
                first_rows % 128 != 0 or second_rows % 128 != 0
            ):
                raise ValueError(
                    "fp8 block-128 fused gate/up legs must each have out_features "
                    f"divisible by 128, got {first_rows} and {second_rows}"
                )

            fused = _Holder(
                torch.cat((first_holder.weight.data, second_holder.weight.data), dim=0)
            )
            fused_result = modifier.quantize_layer(fused, None)
            first_result = _slice_fp8_result(fused_result, 0, first_rows)
            second_result = _slice_fp8_result(
                fused_result, first_rows, first_rows + second_rows
            )
            for leg_name, leg_holder, leg_result in (
                (first_name, first_holder, first_result),
                (second_name, second_holder, second_result),
            ):
                leg_holder.weight.data.copy_(leg_result.fake_weight)
                leg_holder._ptq_result = leg_result
                processed.add(leg_name)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(targets), leg_name)


def model_free_ptq(
    source: str | os.PathLike[str] | Mapping[str, torch.Tensor] | nn.Module,
    modifiers: Modifier | list[Modifier],
    save_dir: str,
    *,
    pack_format: str = "compressed-tensors",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """Pure data-free PTQ. ``source`` is an ``nn.Module``, a ``state_dict``
    mapping, or a path to a ``.safetensors`` file, index, or directory. Indexed
    input shards retain their layout in the output. No model forward, so only
    ``requires_calibration == False`` modifiers are allowed. When provided,
    ``progress_callback`` receives ``(completed, total, module_name)`` after each
    matching weight is quantized."""
    if pack_format not in _PACK_FORMATS:
        raise ValueError(
            f"pack_format must be one of {_PACK_FORMATS}, got {pack_format!r}"
        )
    mods = _as_list(modifiers)
    _validate_modifier_pack_format(mods, pack_format)
    if any(m.requires_calibration for m in mods):
        raise ValueError("model_free_ptq only supports data-free modifiers (e.g. RTN)")

    if isinstance(source, nn.Module):
        run_oneshot(source, mods, pack_format=pack_format, save_dir=save_dir)
        return

    from phyai_model_optimizer.serialize import (
        _load_state_checkpoint,
        save_state_checkpoint,
    )

    clone_weights = isinstance(source, Mapping)
    checkpoint = _load_state_checkpoint(source)
    state = checkpoint.tensors
    targets: list[Target] = []

    for key, tensor in state.items():
        module_name = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_name is None or tensor.ndim != 2:
            continue
        holder = _Holder(tensor.detach().clone() if clone_weights else tensor)
        matched = None
        for modifier in mods:
            if modifier.matches(module_name, holder):
                matched = modifier
                break
        if matched is not None:
            targets.append((module_name, holder, matched))

    if not targets:
        raise RuntimeError(
            "model_free_ptq matched no 2-D weights; check modifier targets "
            f"({[m.targets() for m in mods]}) against the checkpoint's key names"
        )

    _quantize_model_free_targets(targets, progress_callback)

    save_state_checkpoint(checkpoint, targets, save_dir, pack_format)


__all__ = ["oneshot", "model_free_ptq", "QuantResult"]
