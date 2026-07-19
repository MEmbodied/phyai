from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
if str(PHYAI_SRC) not in sys.path:
    sys.path.insert(0, str(PHYAI_SRC))

import phyai.layers.linear as L  # noqa: E402
from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05DecoderLayerNative,
    WallOSS05NativeConfig,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _copy_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    params = dict(module.named_parameters())
    if name not in params:
        raise KeyError(f"{type(module).__name__} has no parameter {name!r}")
    param = params[name]
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: module={tuple(param.shape)} ckpt={tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def _load_layer0_weights(module: WallOSS05DecoderLayerNative, checkpoint: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    mapping = {
        "input_norm.norms.0.weight": ("model.layers.0.input_layernorms.0.weight", torch.float32),
        "input_norm.norms.1.weight": ("model.layers.0.input_layernorms.1.weight", torch.float32),
        "self_attn.projections.qkv_proj_experts.0.weight": ("model.layers.0.self_attn.qkv_proj_experts.0.weight", dtype),
        "self_attn.projections.qkv_proj_experts.0.bias": ("model.layers.0.self_attn.qkv_proj_experts.0.bias", dtype),
        "self_attn.projections.o_proj_experts.0.weight": ("model.layers.0.self_attn.o_proj_experts.0.weight", dtype),
        "self_attn.projections.qkv_proj_experts.1.weight": ("model.layers.0.self_attn.qkv_proj_experts.1.weight", dtype),
        "self_attn.projections.qkv_proj_experts.1.bias": ("model.layers.0.self_attn.qkv_proj_experts.1.bias", dtype),
        "self_attn.projections.o_proj_experts.1.weight": ("model.layers.0.self_attn.o_proj_experts.1.weight", dtype),
        "ffn.post_attention_norm.norms.0.weight": ("model.layers.0.post_attention_layernorms.0.weight", torch.float32),
        "ffn.post_attention_norm.norms.1.weight": ("model.layers.0.post_attention_layernorms.1.weight", torch.float32),
        "ffn.moe.experts.0.gate_up_proj.weight": ("model.layers.0.moe.experts.0.gate_up_proj.weight", dtype),
        "ffn.moe.experts.0.down_proj.weight": ("model.layers.0.moe.experts.0.down_proj.weight", dtype),
        "ffn.moe.experts.1.gate_up_proj.weight": ("model.layers.0.moe.experts.1.gate_up_proj.weight", dtype),
        "ffn.moe.experts.1.down_proj.weight": ("model.layers.0.moe.experts.1.down_proj.weight", dtype),
    }

    out: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for param_name, (ckpt_key, target_dtype) in mapping.items():
            tensor = sf.get_tensor(ckpt_key).to(dtype=target_dtype)
            _copy_param(module, param_name, tensor)
            out[param_name] = tensor.to(dtype=target_dtype)
            print(f"[loaded] {ckpt_key} -> {param_name} {tuple(tensor.shape)} {tensor.dtype}")
    return out


def _permute(tokens: torch.Tensor, expert_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_indices.dim() == 1:
        expert_indices = expert_indices.view(-1, 1)
    expand_factor = expert_indices.size(1)
    sorted_indices = torch.argsort(expert_indices.view(-1), stable=True)
    return tokens.index_select(0, sorted_indices // expand_factor), sorted_indices


def _unpermute(permuted_tokens: torch.Tensor, sorted_indices: torch.Tensor, probs: torch.Tensor | None) -> torch.Tensor:
    merge_factor = probs.size(1) if probs is not None else 1
    out = torch.zeros_like(permuted_tokens)
    out.index_copy_(0, sorted_indices.long(), permuted_tokens)
    out = out.reshape(-1, merge_factor, permuted_tokens.size(-1))
    if probs is not None:
        out = out * probs.unsqueeze(-1)
    return out.sum(dim=1)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = weight.to(device=x.device, dtype=torch.float32) * y
    return y.to(input_dtype)


def _norm_moe(x: torch.Tensor, cfg: WallOSS05NativeConfig, weights: dict[str, torch.Tensor], prefix: str, starts, ends) -> torch.Tensor:
    out = torch.zeros_like(x)
    for idx in [0, 1]:
        start, end = int(starts[idx].item()), int(ends[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        w = weights[f"{prefix}.norms.{idx}.weight"]
        out[start:end, :dim] = _rmsnorm(x[start:end, :dim], w, cfg.rms_norm_eps)
    return out


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _mrope(q, k, cos, sin, section):
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = torch.cat((cos.float(), cos.float()), dim=-1)
    sin = torch.cat((sin.float(), sin.float()), dim=-1)
    doubled = list(section) + list(section)
    cos_split = torch.cat([m[i % 3] for i, m in enumerate(cos.split(doubled, dim=-1))], dim=-1).unsqueeze(2)
    sin_split = torch.cat([m[i % 3] for i, m in enumerate(sin.split(doubled, dim=-1))], dim=-1).unsqueeze(2)
    return (
        (q.float() * cos_split + _rotate_half(q.float()) * sin_split).to(q_dtype),
        (k.float() * cos_split + _rotate_half(k.float()) * sin_split).to(k_dtype),
    )


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x.unsqueeze(3).expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


def _linear(x, w, b=None):
    return F.linear(x, w.to(device=x.device, dtype=x.dtype), None if b is None else b.to(device=x.device, dtype=x.dtype))


def _joint_attention_ref(hidden_states, token_types, starts, ends, row_id_map, cfg, weights, cos, sin, dtype):
    bsz, q_len = token_types.shape
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_dim = cfg.num_key_value_heads * head_dim

    q_buffer = torch.zeros(hidden_states.size(0), cfg.hidden_size, dtype=dtype, device=hidden_states.device)
    k_buffer = torch.zeros(hidden_states.size(0), kv_dim, dtype=dtype, device=hidden_states.device)
    v_buffer = torch.zeros(hidden_states.size(0), kv_dim, dtype=dtype, device=hidden_states.device)

    for idx in [0, 1]:
        start, end = int(starts[idx].item()), int(ends[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        x = hidden_states[start:end, :dim].to(dtype)
        qkv = _linear(
            x,
            weights[f"self_attn.projections.qkv_proj_experts.{idx}.weight"],
            weights[f"self_attn.projections.qkv_proj_experts.{idx}.bias"],
        )
        q, k, v = torch.split(qkv, [cfg.hidden_size, kv_dim, kv_dim], dim=-1)
        q_buffer[start:end] = q
        k_buffer[start:end] = k
        v_buffer[start:end] = v

    q = _unpermute(q_buffer, row_id_map, None).view(bsz, q_len, cfg.num_attention_heads, head_dim)
    k = _unpermute(k_buffer, row_id_map, None).view(bsz, q_len, cfg.num_key_value_heads, head_dim)
    v = _unpermute(v_buffer, row_id_map, None).view(bsz, q_len, cfg.num_key_value_heads, head_dim)

    q, k = _mrope(
        q.contiguous(),
        k.contiguous(),
        cos[..., : cos.size(3) // 2].contiguous().float(),
        sin[..., : sin.size(3) // 2].contiguous().float(),
        cfg.rope_scaling["mrope_section"],
    )

    k = _repeat_kv(k, cfg.num_attention_heads // cfg.num_key_value_heads)
    v = _repeat_kv(v, cfg.num_attention_heads // cfg.num_key_value_heads)
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    attn = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True if q_len > 1 else False,
    )
    attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, cfg.hidden_size).to(dtype)

    flat = attn.view(-1, cfg.hidden_size)
    permuted, _ = _permute(flat, token_types.reshape(-1))

    out = torch.zeros(permuted.size(0), cfg.hidden_size, dtype=dtype, device=hidden_states.device)
    for idx in [0, 1]:
        start, end = int(starts[idx].item()), int(ends[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        projected = _linear(permuted[start:end], weights[f"self_attn.projections.o_proj_experts.{idx}.weight"], None)
        out[start:end, :dim] = projected[:, :dim]
    return out


def _moe_ref(x, cfg, weights, starts, ends, dtype):
    out = torch.zeros_like(x)
    for idx in [0, 1]:
        start, end = int(starts[idx].item()), int(ends[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        intermediate = int(cfg.experts[idx]["intermediate_size"])
        y = x[start:end, :dim].to(dtype)
        gate_up = _linear(y, weights[f"ffn.moe.experts.{idx}.gate_up_proj.weight"], None)
        gate, up = gate_up.split([intermediate, intermediate], dim=-1)
        y = F.silu(gate) * up
        y = _linear(y, weights[f"ffn.moe.experts.{idx}.down_proj.weight"], None)
        out[start:end, :dim] = y[:, :dim]
    return out


def _decoder_ref(hidden_states, token_types, starts, ends, row_id_map, cfg, weights, cos, sin, dtype):
    residual = hidden_states
    x = _norm_moe(hidden_states, cfg, weights, "input_norm", starts, ends)
    x = _joint_attention_ref(x, token_types, starts, ends, row_id_map, cfg, weights, cos, sin, dtype)
    x = residual + x

    residual = x
    x = _norm_moe(x, cfg, weights, "ffn.post_attention_norm", starts, ends)
    x = _moe_ref(x, cfg, weights, starts, ends, dtype)
    return residual + x


def _make_cos_sin(cfg, bsz, seq, dtype, device):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).to(torch.float32) / head_dim))
    base = torch.arange(seq, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, seq)
    pos_ids = torch.stack([base, base + 1, base + 2], dim=0)
    freqs = (inv_freq[None, None, :, None].float().expand(3, bsz, -1, 1) @ pos_ids[:, :, None, :].float()).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _compare(name, native, ref, atol):
    diff = (native.float() - ref.float()).abs()
    print(f"\n--- {name} ---")
    print("shape:", tuple(native.shape), native.dtype, native.device)
    print("max_abs_diff:", float(diff.max()))
    print("mean_abs_diff:", float(diff.mean()))
    print("cosine:", float(F.cosine_similarity(native.flatten().float(), ref.flatten().float(), dim=0)))
    print("exact_equal:", bool(torch.equal(native, ref)))
    print("allclose:", bool(torch.allclose(native, ref, atol=atol, rtol=atol)))
    if float(diff.max()) > atol:
        raise SystemExit(f"FAILED: {name}")


def _run_one(cfg, checkpoint: Path, dtype: torch.dtype, device: torch.device):
    print(f"\n========== dtype={dtype} device={device} ==========")

    L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])

    bsz, seq = 2, 6
    token_types = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    flat_types = token_types.reshape(-1)
    counts = [(flat_types == i).sum().item() for i in range(2)]
    starts, ends = [], []
    cur = 0
    for c in counts:
        starts.append(cur)
        cur += c
        ends.append(cur)
    starts = torch.tensor(starts, dtype=torch.long, device=device)
    ends = torch.tensor(ends, dtype=torch.long, device=device)

    torch.manual_seed(40000 + (0 if dtype == torch.float32 else 1) + (0 if device.type == "cpu" else 100))
    unpermuted_hidden = torch.randn(flat_types.numel(), cfg.hidden_size, dtype=torch.float32, device=device).to(dtype)
    hidden_states, row_id_map = _permute(unpermuted_hidden, flat_types)

    cos, sin = _make_cos_sin(cfg, bsz, seq, dtype, device)

    module = WallOSS05DecoderLayerNative(
        cfg,
        layer_idx=0,
        params_dtype=dtype,
        device=device,
    ).to(device).eval()

    weights = _load_layer0_weights(module, checkpoint, dtype)

    with torch.no_grad():
        native = module(
            hidden_states.clone(),
            token_types=token_types,
            start_indices=starts,
            end_indices=ends,
            row_id_map=row_id_map,
            probs=None,
            orig_shape=(bsz, seq, cfg.hidden_size),
            position_embeddings=(cos, sin),
            attention_mask=None,
            adarms_conds=[None, None],
            projection_dtype=dtype,
        )
        ref = _decoder_ref(hidden_states.clone(), token_types, starts, ends, row_id_map, cfg, weights, cos, sin, dtype)

    _compare(f"decoder_layer_native_{dtype}_{device.type}", native, ref, 1e-5 if dtype == torch.float32 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    args = parser.parse_args()

    cfg = WallOSS05NativeConfig.from_checkpoint_and_train_config(
        _load_json(args.checkpoint / "config.json"),
        _load_yaml(args.train_config),
        norm_key=args.norm_key,
    )

    print("========== Config ==========")
    print("hidden_size:", cfg.hidden_size)
    print("num_attention_heads:", cfg.num_attention_heads)
    print("num_key_value_heads:", cfg.num_key_value_heads)
    print("rope_scaling:", cfg.rope_scaling)

    _run_one(cfg, args.checkpoint, torch.float32, torch.device("cpu"))

    if torch.cuda.is_available():
        _run_one(cfg, args.checkpoint, torch.bfloat16, torch.device("cuda:0"))

    print("\nPASS: native DecoderLayer layer-0 no-cache/mot_opt path matches independent reference.")


if __name__ == "__main__":
    main()
