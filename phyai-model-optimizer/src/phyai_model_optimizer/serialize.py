"""Serialization backends selected by ``pack_format``.

Integer and FP4 results carry authoritative codes from their modifier. FP8 is
reconstructed from the fake-quantized weight on its exact scale grid.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from phyai_model_optimizer.compat import ct as ctc
from phyai_model_optimizer.pipelines.base import Target
from phyai_model_optimizer.quant_math import FP8Scheme, WeightQuant

_INT_FMT = "pack-quantized"
_FLOAT_FMT = "float-quantized"
_MIXED_FMT = "mixed-precision"
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_SINGLE = "model.safetensors"

# note(chenghua): files we regenerate, so skip them when copying aux files from the base.
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".safetensors.index.json",
    ".bin",
    ".bin.index.json",
    ".pt",
    ".pth",
)


@dataclass(frozen=True)
class _StateCheckpoint:
    tensors: "OrderedDict[str, torch.Tensor]"
    shard_map: dict[str, str] | None = None
    index: dict | None = None
    base_checkpoint: str | None = None


def _read_safetensors_index(path: str) -> tuple[dict, dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        raise ValueError(f"{path}: safetensors index must be a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{path}: missing or empty 'weight_map'")
    if not all(
        isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
    ):
        raise ValueError(f"{path}: 'weight_map' must map tensor names to shard names")
    return index, dict(weight_map)


def _resolve_shard(checkpoint_dir: str, shard: str, index_path: str) -> str:
    if os.path.isabs(shard):
        raise ValueError(f"{index_path}: shard paths must be relative, got {shard!r}")
    root = os.path.realpath(checkpoint_dir)
    path = os.path.realpath(os.path.join(root, shard))
    if os.path.commonpath((root, path)) != root:
        raise ValueError(
            f"{index_path}: shard path escapes checkpoint directory: {shard!r}"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{index_path} references missing shard {shard!r}")
    return path


def _load_state_checkpoint(
    source: str | os.PathLike[str] | Mapping[str, torch.Tensor],
) -> _StateCheckpoint:
    """Load a mapping, safetensors file, or indexed/multi-shard directory."""
    if isinstance(source, Mapping):
        tensors = OrderedDict()
        for key, tensor in source.items():
            if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "state_dict must map string keys to torch.Tensor values"
                )
            tensors[key] = tensor
        return _StateCheckpoint(tensors=tensors)

    source_path = os.path.abspath(os.fspath(source))
    index: dict | None = None
    indexed_map: dict[str, str] | None = None

    if os.path.isdir(source_path):
        checkpoint_dir = source_path
        index_path = os.path.join(checkpoint_dir, _SAFETENSORS_INDEX)
        if os.path.isfile(index_path):
            index, indexed_map = _read_safetensors_index(index_path)
            shard_names = list(dict.fromkeys(indexed_map.values()))
            paths = [
                (_resolve_shard(checkpoint_dir, name, index_path), name)
                for name in shard_names
            ]
        else:
            single = os.path.join(checkpoint_dir, _SAFETENSORS_SINGLE)
            if os.path.isfile(single):
                paths = [(single, _SAFETENSORS_SINGLE)]
            else:
                names = sorted(
                    name
                    for name in os.listdir(checkpoint_dir)
                    if name.endswith(".safetensors")
                    and os.path.isfile(os.path.join(checkpoint_dir, name))
                )
                if not names:
                    raise FileNotFoundError(
                        f"no safetensors found in {checkpoint_dir}: expected "
                        f"{_SAFETENSORS_INDEX!r}, {_SAFETENSORS_SINGLE!r}, or '*.safetensors'"
                    )
                paths = [(os.path.join(checkpoint_dir, name), name) for name in names]
    elif os.path.isfile(source_path):
        checkpoint_dir = os.path.dirname(source_path)
        if source_path.endswith(".safetensors.index.json"):
            index_path = source_path
            index, indexed_map = _read_safetensors_index(index_path)
            shard_names = list(dict.fromkeys(indexed_map.values()))
            paths = [
                (_resolve_shard(checkpoint_dir, name, index_path), name)
                for name in shard_names
            ]
        elif source_path.endswith(".safetensors"):
            paths = [(source_path, os.path.basename(source_path))]
        else:
            raise ValueError(
                f"checkpoint path must be a .safetensors file, index, or directory: {source_path}"
            )
    else:
        raise FileNotFoundError(f"checkpoint source does not exist: {source_path}")

    from safetensors.torch import load_file

    tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    actual_map: dict[str, str] = {}
    for path, shard_name in paths:
        shard_tensors = load_file(path)
        for key, tensor in shard_tensors.items():
            if key in tensors:
                raise ValueError(
                    f"tensor {key!r} occurs in more than one safetensors shard"
                )
            if indexed_map is not None:
                expected = indexed_map.get(key)
                if expected is not None and expected != shard_name:
                    raise ValueError(
                        f"index maps tensor {key!r} to {expected!r}, but it is stored in {shard_name!r}"
                    )
            tensors[key] = tensor
            actual_map[key] = shard_name

    if indexed_map is not None:
        missing = sorted(set(indexed_map) - set(tensors))
        if missing:
            preview = ", ".join(repr(key) for key in missing[:3])
            raise ValueError(
                f"safetensors index references missing tensor(s): {preview}"
            )

    preserve_shards = indexed_map is not None or len(paths) > 1
    return _StateCheckpoint(
        tensors=tensors,
        shard_map=actual_map if preserve_shards else None,
        index=index if preserve_shards else None,
        base_checkpoint=checkpoint_dir,
    )


def _copy_aux_files(base_checkpoint: str | None, output_dir: str) -> None:
    """Copy non-weight aux files from the base so the output is a complete checkpoint."""
    if base_checkpoint is None:
        return
    src = (
        base_checkpoint
        if os.path.isdir(base_checkpoint)
        else os.path.dirname(base_checkpoint)
    )
    if not src or not os.path.isdir(src):
        return
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if not os.path.isfile(p) or fn == "config.json":
            continue
        if any(fn.endswith(s) for s in _WEIGHT_SUFFIXES):
            continue
        dst = os.path.join(output_dir, fn)
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copy2(p, dst)


def _config_target(name: str, module: nn.Module) -> str:
    """The name phyai's QuantPlan resolves against: HF ``.prefix`` if present, else the module path."""
    return getattr(module, "prefix", None) or name


def _canonicalize_fused_targets(names: list[str]) -> list[str]:
    """Map matched gate/up source legs to their fused runtime prefix."""
    matched = set(names)
    result: set[str] = set()
    for name in names:
        if name.endswith(".mlp.gate_proj"):
            fused = f"{name[: -len('gate_proj')]}gate_up_proj"
            sibling = f"{name[: -len('gate_proj')]}up_proj"
            if sibling not in matched:
                raise ValueError(
                    f"fused target {name!r} is missing sibling {sibling!r}"
                )
            result.add(fused)
            continue
        if name.endswith(".mlp.up_proj"):
            fused = f"{name[: -len('up_proj')]}gate_up_proj"
            sibling = f"{name[: -len('up_proj')]}gate_proj"
            if sibling not in matched:
                raise ValueError(
                    f"fused target {name!r} is missing sibling {sibling!r}"
                )
            result.add(fused)
            continue
        result.add(name)
    return sorted(result)


def assemble_quant_config(
    targets: list[Target], *, canonicalize_fused_targets: bool = False
) -> tuple[dict, str]:
    """Group targets by owning modifier into CT ``config_groups``; returns ``(config_dict, format)``."""
    # note(chenghua): preserve modifier identity ordering.
    mods: list = []
    by_mod: dict[int, list[str]] = {}
    for name, module, modifier in targets:
        key = id(modifier)
        if key not in by_mod:
            by_mod[key] = []
            mods.append(modifier)
        by_mod[key].append(_config_target(name, module))

    schemes: dict[str, ctc.QuantizationScheme] = {}
    formats: set[str] = set()
    ignore: set[str] = set()
    for i, modifier in enumerate(mods):
        q = modifier.weight_quant()
        names = (
            _canonicalize_fused_targets(by_mod[id(modifier)])
            if canonicalize_fused_targets
            else sorted(by_mod[id(modifier)])
        )
        schemes[f"group_{i}"] = ctc.build_scheme(
            names, q, input_acts=modifier.input_args()
        )
        formats.add(ctc.compression_format(q))
        ignore.update(modifier.ignore())

    fmt = next(iter(formats)) if len(formats) == 1 else _MIXED_FMT
    config = ctc.build_config(schemes, ignore=sorted(ignore), fmt=fmt)
    return config.model_dump(), fmt


def _output_quant_config(
    targets: list[Target],
    pack_format: str,
    *,
    canonicalize_fused_targets: bool = False,
) -> dict:
    config, _fmt = assemble_quant_config(
        targets, canonicalize_fused_targets=canonicalize_fused_targets
    )
    if pack_format == "humming":
        config["pack_format"] = "humming"
        modifiers: list = []
        seen: set[int] = set()
        for _name, _module, modifier in targets:
            if id(modifier) not in seen:
                modifiers.append(modifier)
                seen.add(id(modifier))
        groups = config["config_groups"]
        for index, modifier in enumerate(modifiers):
            q = modifier.weight_quant()
            if q.is_float:
                groups[f"group_{index}"]["weights"]["humming_dtype"] = (
                    "float8e5m2" if q.is_e5m2 else "float8e4m3"
                )
    return config


def _validate_pack_format(targets: list[Target], pack_format: str) -> None:
    if pack_format not in ("compressed-tensors", "humming"):
        raise ValueError(
            "pack_format must be 'compressed-tensors' or 'humming', "
            f"got {pack_format!r}"
        )
    if pack_format == "compressed-tensors":
        native_only = sorted(
            {
                modifier.weight_quant().dtype.value
                for _name, _module, modifier in targets
                if modifier.weight_quant().num_bits in (3, 6)
            }
        )
        if native_only:
            raise ValueError(
                f"{', '.join(native_only)} use Humming's continuous bit layout; "
                "use pack_format='humming'"
            )
    if pack_format == "compressed-tensors" and any(
        modifier.weight_quant().is_e5m2 for _name, _module, modifier in targets
    ):
        raise ValueError(
            "FP8 E5M2 is not distinguishable in the standard compressed-tensors "
            "float8 schema; use pack_format='humming'"
        )
    if pack_format == "humming":
        fp4 = [
            modifier.weight_quant().dtype.value
            for _name, _module, modifier in targets
            if modifier.weight_quant().is_fp4
        ]
        if fp4:
            formats = ", ".join(sorted(set(fp4)))
            raise ValueError(
                f"FP4 ({formats}) only supports pack_format='compressed-tensors'"
            )


def _pack_int(
    codes: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor | None,
    q: WeightQuant,
    param_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Pack the modifier's chosen integer codes into canonical CT bytes, byte-identical
    to what CT's own ``quantize`` would emit."""
    from compressed_tensors.compressors import pack_to_int32

    out = {
        "weight_packed": pack_to_int32(codes.to(torch.int8), q.num_bits),
        "weight_scale": scale.to(param_dtype).contiguous(),
        "weight_shape": torch.tensor(list(codes.shape), dtype=torch.int64),
    }
    if not q.symmetric and zp is not None:
        out["weight_zero_point"] = pack_to_int32(
            zp.to(torch.int8), q.num_bits, packed_dim=0
        ).contiguous()
    return out


def _pack_fp8(
    weight_fake: torch.Tensor, scale: torch.Tensor, q: WeightQuant
) -> dict[str, torch.Tensor]:
    from compressed_tensors.quantization.lifecycle.forward import quantize

    args = ctc.weight_args(q)
    fp8_dtype = torch.float8_e5m2 if q.is_e5m2 else torch.float8_e4m3fn
    zp_t = torch.zeros_like(scale)
    wq = quantize(
        x=weight_fake.float(),
        scale=scale.float(),
        zero_point=zp_t,
        args=args,
        dtype=fp8_dtype,
    )
    return {
        "weight": wq.contiguous(),
        "weight_scale": scale.to(torch.float32).clone().contiguous(),
    }


def _pack_fp4(res, sl: slice) -> dict[str, torch.Tensor]:
    if res.packed_weight is None:
        raise RuntimeError("FP4 QuantResult missing packed_weight")

    out = {
        "weight_packed": res.packed_weight[sl].contiguous(),
        "weight_scale": res.scale[sl].contiguous(),
    }
    if res.q.is_nvfp4:
        if res.global_scale is None:
            raise RuntimeError("NVFP4 QuantResult missing global_scale")
        out["weight_global_scale"] = res.global_scale.clone().contiguous()
    elif res.global_scale is not None:
        raise RuntimeError("MXFP4 QuantResult must not contain global_scale")
    return out


def _pack_humming_codes(
    codes: torch.Tensor,
    num_bits: int,
    *,
    packed_dim: int = 1,
) -> torch.Tensor:
    if packed_dim not in (0, 1):
        raise ValueError(f"packed_dim must be 0 or 1, got {packed_dim}")
    values = codes.transpose(0, 1) if packed_dim == 0 else codes
    if values.shape[-1] % 32 != 0:
        raise ValueError(
            "Humming continuous bit packing requires the packed dimension to be "
            f"divisible by 32, got {values.shape[-1]}"
        )

    mask = (1 << num_bits) - 1
    unsigned = (values.to(torch.int64) + (1 << (num_bits - 1))) & mask
    blocks = unsigned.reshape(*unsigned.shape[:-1], -1, 32)
    packed = torch.zeros(
        *blocks.shape[:-1],
        num_bits,
        dtype=torch.int64,
        device=codes.device,
    )
    for index in range(32):
        bit_index = index * num_bits
        word = bit_index // 32
        offset = bit_index % 32
        value = blocks[..., index]
        packed[..., word] |= value << offset
        if offset + num_bits > 32:
            packed[..., word + 1] |= value >> (32 - offset)

    result = packed.to(torch.int32).flatten(-2)
    return result.transpose(0, 1).contiguous() if packed_dim == 0 else result


def _compress_module(module: nn.Module) -> dict[str, torch.Tensor]:
    res = getattr(module, "_ptq_result", None)
    if res is None:
        raise RuntimeError("module has no _ptq_result; was it quantized?")
    if res.q.is_fp4:
        return _pack_fp4(res, slice(None))
    if res.q.is_float:
        return _pack_fp8(module.weight.data, res.scale, res.q)
    if res.int_codes is None:
        raise RuntimeError("int QuantResult missing int_codes")
    return _pack_int(
        res.int_codes, res.scale, res.zero_point, res.q, module.weight.dtype
    )


def _write_config_json(
    output_dir: str, quant_config: dict, base_checkpoint: str | None = None
) -> None:
    path = os.path.join(output_dir, "config.json")
    base: dict = {}
    # note(chenghua): seed from the base arch config so the output is a complete, loadable checkpoint.
    src = None
    if os.path.exists(path):
        src = path
    elif base_checkpoint is not None:
        cand = (
            os.path.join(base_checkpoint, "config.json")
            if os.path.isdir(base_checkpoint)
            else os.path.join(os.path.dirname(base_checkpoint), "config.json")
        )
        if os.path.exists(cand):
            src = cand
    if src is not None:
        with open(src) as f:
            base = json.load(f)
    base["quantization_config"] = quant_config
    with open(path, "w") as f:
        json.dump(base, f, indent=2)


def _leg_spans(module: nn.Module, name: str) -> list[tuple[str, int]]:
    """Per-leg ``(hf_key_base, n_rows)`` in leg order.

    phyai names on-disk tensors by ``weight.hf_keys`` (per-leg for fused QKV), not the
    module path; plain ``nn.Linear`` falls back to the module name.
    """
    w = getattr(module, "weight", None)
    hf = getattr(w, "hf_keys", None) if w is not None else None
    lw = getattr(module, "logical_widths", None)
    if hf:
        bases = [k[: -len(".weight")] if k.endswith(".weight") else k for (k, _s) in hf]
        if lw and len(lw) == len(bases):
            return list(zip(bases, [int(x) for x in lw]))
        return [(bases[0], int(w.shape[0]))]
    return [(name, int(module.weight.shape[0]))]


def _base_tensors(
    model: nn.Module, base_checkpoint: str | None
) -> "OrderedDict[str, torch.Tensor]":
    """Starting tensor set: with ``base_checkpoint`` copy the original HF-named
    safetensors (non-quantized params keep exact keys byte-for-byte); else use
    ``model.state_dict()``."""
    if base_checkpoint is None:
        return OrderedDict((k, v.contiguous()) for k, v in model.state_dict().items())
    return _load_state_checkpoint(base_checkpoint).tensors


def _compress_slice(
    res, sl: slice, param_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    if res.q.is_fp4:
        return _pack_fp4(res, sl)
    if res.q.is_float:
        return _pack_fp8(res.fake_weight[sl], _slice_fp8_scale(res, sl), res.q)
    zp = res.zero_point[sl] if res.zero_point is not None else None
    if res.int_codes is None:
        raise RuntimeError("int QuantResult missing int_codes")
    codes = res.int_codes[sl]
    return _pack_int(codes, res.scale[sl], zp, res.q, param_dtype)


def _slice_bounds(sl: slice, rows: int) -> tuple[int, int]:
    start, stop, step = sl.indices(rows)
    if step != 1:
        raise ValueError("quantized weight slices must be contiguous")
    return start, stop


def _slice_fp8_scale(res, sl: slice) -> torch.Tensor:
    if not res.q.is_fp8:
        raise ValueError("_slice_fp8_scale requires an FP8 result")
    if res.q.fp8_scheme is FP8Scheme.TENSORWISE:
        return res.scale
    start, stop = _slice_bounds(sl, int(res.fake_weight.shape[0]))
    block_n = (res.q.block_structure or (0, 0))[0]
    if start % block_n != 0 or stop % block_n != 0:
        raise ValueError(
            "fp8 block-128 serialization requires output slices aligned to 128 "
            f"rows, got [{start}, {stop})"
        )
    return res.scale[start // block_n : stop // block_n]


def save_compressed_tensors(
    model: nn.Module,
    targets: list[Target],
    output_dir: str,
    base_checkpoint: str | None = None,
) -> None:
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    _copy_aux_files(base_checkpoint, output_dir)
    out = _base_tensors(model, base_checkpoint)

    for name, module, _mod in targets:
        res = getattr(module, "_ptq_result")
        pdt = module.weight.dtype
        offset = 0
        for base, n in _leg_spans(module, name):
            sl = slice(offset, offset + n)
            offset += n
            out.pop(f"{base}.weight", None)
            for local, tensor in _compress_slice(res, sl, pdt).items():
                out[f"{base}.{local}"] = tensor.contiguous()

    save_file(
        out, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"}
    )
    _write_config_json(
        output_dir,
        _output_quant_config(targets, "compressed-tensors"),
        base_checkpoint,
    )


def _pack_humming_slice(
    res, sl: slice, param_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Encode PTQ artifacts in Humming's native pre-transform storage.

    The compressed-tensors integer packer applies the same signed-to-unsigned
    offset and little-endian bit order as Humming. Contiguous FP8 bytes use that
    same order, so both formats can be emitted exactly on CPU without re-fitting
    scales or zero points.
    """
    if res.q.is_float:
        fake_weight = res.fake_weight[sl].float()
        scale = _slice_fp8_scale(res, sl)
        weight = _pack_fp8(fake_weight, scale, res.q)["weight"]
        if weight.shape[-1] % torch.int32.itemsize != 0:
            raise ValueError(
                "Humming FP8 packing requires in_features to be divisible by 4, "
                f"got {weight.shape[-1]}"
            )
        return {
            "weight": weight.view(torch.int32),
            "weight_scale": scale.to(param_dtype).contiguous(),
        }

    if res.int_codes is None:
        raise RuntimeError("int QuantResult missing int_codes")
    zero_point = res.zero_point[sl] if res.zero_point is not None else None
    num_bits = res.q.num_bits
    codes = res.int_codes[sl]
    if codes.shape[-1] * num_bits % 32 != 0:
        raise ValueError(
            "Humming integer packing requires in_features * num_bits to be "
            f"divisible by 32, got {codes.shape[-1]} * {num_bits}"
        )
    if zero_point is not None and codes.shape[-2] * num_bits % 32 != 0:
        raise ValueError(
            "Humming asymmetric packing requires out_features * num_bits to be "
            f"divisible by 32, got {codes.shape[-2]} * {num_bits}"
        )
    if num_bits in (3, 6):
        out = {
            "weight": _pack_humming_codes(codes, num_bits),
            "weight_scale": res.scale[sl].to(param_dtype).contiguous(),
        }
        if zero_point is not None:
            out["zero_point"] = _pack_humming_codes(
                zero_point,
                num_bits,
                packed_dim=0,
            )
        return out

    packed = _pack_int(codes, res.scale[sl], zero_point, res.q, param_dtype)
    out = {
        "weight": packed["weight_packed"],
        "weight_scale": packed["weight_scale"],
    }
    if "weight_zero_point" in packed:
        out["zero_point"] = packed["weight_zero_point"]
    return out


def save_humming(
    model: nn.Module,
    targets: list[Target],
    output_dir: str,
    base_checkpoint: str | None = None,
) -> None:
    """Humming-native packing for zero-conversion load in phyai.

    ``config.json`` still uses the CT schema so phyai's importer resolves the native
    Humming scheme.
    """
    from safetensors.torch import save_file

    _validate_pack_format(targets, "humming")

    os.makedirs(output_dir, exist_ok=True)
    _copy_aux_files(base_checkpoint, output_dir)
    out = _base_tensors(model, base_checkpoint)

    for name, module, _mod in targets:
        res = getattr(module, "_ptq_result")
        pdt = module.weight.dtype
        offset = 0
        for base, n in _leg_spans(module, name):
            sl = slice(offset, offset + n)
            offset += n
            out.pop(f"{base}.weight", None)
            tensors = _pack_humming_slice(res, sl, pdt)
            for local, t in tensors.items():
                out[f"{base}.{local}"] = t.detach().cpu().contiguous()

    save_file(
        out, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"}
    )
    _write_config_json(
        output_dir,
        _output_quant_config(targets, "humming"),
        base_checkpoint,
    )


def _replace_state_targets(
    state: "OrderedDict[str, torch.Tensor]",
    targets: list[Target],
    pack_format: str,
) -> tuple["OrderedDict[str, torch.Tensor]", dict[str, str]]:
    _validate_pack_format(targets, pack_format)

    replaced_keys = {
        f"{base}.weight"
        for name, module, _modifier in targets
        for base, _n_rows in _leg_spans(module, name)
    }
    out = OrderedDict(
        (key, tensor.detach().cpu().contiguous())
        for key, tensor in state.items()
        if key not in replaced_keys
    )
    generated_from: dict[str, str] = {}
    for name, module, _modifier in targets:
        res = getattr(module, "_ptq_result")
        param_dtype = module.weight.dtype
        offset = 0
        for base, n_rows in _leg_spans(module, name):
            source_key = f"{base}.weight"
            sl = slice(offset, offset + n_rows)
            offset += n_rows
            if pack_format == "compressed-tensors":
                tensors = _compress_slice(res, sl, param_dtype)
            else:
                tensors = _pack_humming_slice(res, sl, param_dtype)
            for local, tensor in tensors.items():
                key = f"{base}.{local}"
                out[key] = tensor.detach().cpu().contiguous()
                generated_from[key] = source_key
    return out, generated_from


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _save_sharded_state(
    state: "OrderedDict[str, torch.Tensor]",
    generated_from: dict[str, str],
    output_dir: str,
    shard_map: Mapping[str, str],
    index: dict | None,
) -> None:
    from safetensors.torch import save_file

    shards: "OrderedDict[str, OrderedDict[str, torch.Tensor]]" = OrderedDict()
    output_map: dict[str, str] = {}
    for key, tensor in state.items():
        source_key = generated_from.get(key, key)
        shard_name = shard_map.get(source_key)
        if shard_name is None:
            raise ValueError(f"no source shard recorded for tensor {source_key!r}")
        shards.setdefault(shard_name, OrderedDict())[key] = tensor
        output_map[key] = shard_name

    for shard_name, tensors in shards.items():
        path = os.path.join(output_dir, shard_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(tensors, path, metadata={"format": "pt"})

    output_index = dict(index or {})
    metadata = dict(output_index.get("metadata") or {})
    metadata["total_size"] = sum(_tensor_bytes(tensor) for tensor in state.values())
    output_index["metadata"] = metadata
    output_index["weight_map"] = output_map
    with open(os.path.join(output_dir, _SAFETENSORS_INDEX), "w", encoding="utf-8") as f:
        json.dump(output_index, f, indent=2)


def save_state_checkpoint(
    checkpoint: _StateCheckpoint,
    targets: list[Target],
    output_dir: str,
    pack_format: str,
) -> None:
    """Serialize quantized model-free holders while retaining every other tensor."""
    from safetensors.torch import save_file

    _validate_pack_format(targets, pack_format)
    # note(chenghua): Model-free checkpoints expose source legs, while phyai loads
    # the corresponding fused runtime module.
    quant_config = _output_quant_config(
        targets,
        pack_format,
        canonicalize_fused_targets=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    _copy_aux_files(checkpoint.base_checkpoint, output_dir)
    out, generated_from = _replace_state_targets(
        checkpoint.tensors, targets, pack_format
    )
    if checkpoint.shard_map is None:
        save_file(
            out,
            os.path.join(output_dir, _SAFETENSORS_SINGLE),
            metadata={"format": "pt"},
        )
    else:
        _save_sharded_state(
            out,
            generated_from,
            output_dir,
            checkpoint.shard_map,
            checkpoint.index,
        )
    _write_config_json(output_dir, quant_config, checkpoint.base_checkpoint)


def save(
    model: nn.Module,
    targets: list[Target],
    output_dir: str,
    pack_format: str,
    base_checkpoint: str | None = None,
) -> None:
    _validate_pack_format(targets, pack_format)
    if pack_format == "compressed-tensors":
        save_compressed_tensors(model, targets, output_dir, base_checkpoint)
    elif pack_format == "humming":
        save_humming(model, targets, output_dir, base_checkpoint)


__all__ = [
    "assemble_quant_config",
    "save",
    "save_compressed_tensors",
    "save_humming",
    "save_state_checkpoint",
]
