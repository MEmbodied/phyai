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
    WallOSS05JointAttentionNative,
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


def _load_attn_weights(module: WallOSS05JointAttentionNative, checkpoint: Path, dtype: torch.dtype) -> None:
    mapping = {}
    for idx in [0, 1]:
        mapping[f"projections.qkv_proj_experts.{idx}.weight"] = f"model.layers.0.self_attn.qkv_proj_experts.{idx}.weight"
        mapping[f"projections.qkv_proj_experts.{idx}.bias"] = f"model.layers.0.self_attn.qkv_proj_experts.{idx}.bias"
        mapping[f"projections.o_proj_experts.{idx}.weight"] = f"model.layers.0.self_attn.o_proj_experts.{idx}.weight"

    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for param_name, ckpt_key in mapping.items():
            tensor = sf.get_tensor(ckpt_key).to(dtype=dtype)
            _copy_param(module, param_name, tensor)
            print(f"[loaded] {ckpt_key} -> {param_name} {tuple(tensor.shape)} {tensor.dtype}")


def _permute(tokens: torch.Tensor, expert_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_indices.dim() == 1:
        expert_indices = expert_indices.view(-1, 1)
    expand_factor = expert_indices.size(1)
    flatten_indices = expert_indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    return tokens.index_select(0, sorted_indices // expand_factor), sorted_indices


def _unpermute(permuted_tokens: torch.Tensor, sorted_indices: torch.Tensor, probs: torch.Tensor | None) -> torch.Tensor:
    if probs is not None:
        merge_factor = probs.size(1)
    else:
        merge_factor = 1
    unpermuted = torch.zeros_like(permuted_tokens)
    unpermuted.index_copy_(0, sorted_indices.long(), permuted_tokens)
    unpermuted = unpermuted.reshape(-1, merge_factor, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted = unpermuted * probs.unsqueeze(-1)
    return unpermuted.sum(dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _mrope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, section: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = torch.cat((cos.float(), cos.float()), dim=-1)
    sin = torch.cat((sin.float(), sin.float()), dim=-1)
    doubled = list(section) + list(section)
    cos_split = torch.cat([m[i % 3] for i, m in enumerate(cos.split(doubled, dim=-1))], dim=-1).unsqueeze(2)
    sin_split = torch.cat([m[i % 3] for i, m in enumerate(sin.split(doubled, dim=-1))], dim=-1).unsqueeze(2)
    q_out = q.float() * cos_split + _rotate_half(q.float()) * sin_split
    k_out = k.float() * cos_split + _rotate_half(k.float()) * sin_split
    return q_out.to(q_dtype), k_out.to(k_dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, slen, num_kv, head_dim = hidden_states.shape
    hidden_states = hidden_states.unsqueeze(3)
    hidden_states = hidden_states.expand(batch, slen, num_kv, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_kv * n_rep, head_dim)


def _attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: WallOSS05NativeConfig, dtype: torch.dtype) -> torch.Tensor:
    bsz, q_len, _, _ = q.shape
    k = _repeat_kv(k, cfg.num_attention_heads // cfg.num_key_value_heads)
    v = _repeat_kv(v, cfg.num_attention_heads // cfg.num_key_value_heads)

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True if q_len > 1 else False,
    )
    out = out.transpose(1, 2).contiguous()
    return out.view(bsz, q_len, -1).to(dtype)


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def _reference(
    hidden_states: torch.Tensor,
    token_types: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor | None,
    cfg: WallOSS05NativeConfig,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    bsz, q_len = token_types.shape
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_dim = cfg.num_key_value_heads * head_dim

    q_buffer = torch.zeros(hidden_states.size(0), cfg.hidden_size, dtype=dtype, device=hidden_states.device)
    k_buffer = torch.zeros(hidden_states.size(0), kv_dim, dtype=dtype, device=hidden_states.device)
    v_buffer = torch.zeros(hidden_states.size(0), kv_dim, dtype=dtype, device=hidden_states.device)

    for idx in [0, 1]:
        start = int(start_indices[idx].item())
        end = int(end_indices[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        x = hidden_states[start:end, :dim].to(dtype)
        qkv = _linear(x, weights[f"qkv_w_{idx}"], weights[f"qkv_b_{idx}"])
        q, k, v = torch.split(qkv, [cfg.hidden_size, kv_dim, kv_dim], dim=-1)
        q_buffer[start:end] = q
        k_buffer[start:end] = k
        v_buffer[start:end] = v

    q_un = _unpermute(q_buffer, row_id_map, probs)
    k_un = _unpermute(k_buffer, row_id_map, probs)
    v_un = _unpermute(v_buffer, row_id_map, probs)

    q = q_un.view(bsz, q_len, cfg.num_attention_heads, head_dim)
    k = k_un.view(bsz, q_len, cfg.num_key_value_heads, head_dim)
    v = v_un.view(bsz, q_len, cfg.num_key_value_heads, head_dim)

    section = cfg.rope_scaling["mrope_section"]
    q, k = _mrope(
        q.contiguous(),
        k.contiguous(),
        cos[..., : cos.size(3) // 2].contiguous().float(),
        sin[..., : sin.size(3) // 2].contiguous().float(),
        section,
    )

    attn = _attention_core(q, k, v, cfg, dtype)
    flat = attn.view(-1, cfg.hidden_size)
    permuted, _ = _permute(flat, token_types.reshape(-1))

    out = torch.zeros(permuted.size(0), cfg.hidden_size, dtype=dtype, device=hidden_states.device)
    for idx in [0, 1]:
        start = int(start_indices[idx].item())
        end = int(end_indices[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        projected = _linear(permuted[start:end], weights[f"o_w_{idx}"], None)
        out[start:end, :dim] = projected[:, :dim]
    return out


def _make_cos_sin(cfg: WallOSS05NativeConfig, bsz: int, seq: int, dtype: torch.dtype, device: torch.device):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).to(torch.float32)
            / head_dim
        )
    )
    base = torch.arange(seq, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, seq)
    position_ids = torch.stack([base, base + 1, base + 2], dim=0)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, bsz, -1, 1)
    pos = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ pos).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _compare(name: str, native: torch.Tensor, ref: torch.Tensor, atol: float):
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
    total = flat_types.numel()
    counts = [(flat_types == i).sum().item() for i in range(2)]
    starts, ends = [], []
    cur = 0
    for c in counts:
        starts.append(cur)
        cur += c
        ends.append(cur)

    # Expert-ordered hidden states as official mot_opt forward receives after permute.
    torch.manual_seed(30000 + (0 if dtype == torch.float32 else 1) + (0 if device.type == "cpu" else 100))
    unpermuted_hidden = torch.randn(total, cfg.hidden_size, dtype=torch.float32, device=device).to(dtype)
    hidden_states, row_id_map = _permute(unpermuted_hidden, flat_types)
    probs = None

    start_indices = torch.tensor(starts, dtype=torch.long, device=device)
    end_indices = torch.tensor(ends, dtype=torch.long, device=device)
    cos, sin = _make_cos_sin(cfg, bsz, seq, dtype, device)

    module = WallOSS05JointAttentionNative(
        cfg,
        layer_idx=0,
        params_dtype=dtype,
        device=device,
    ).to(device).eval()

    weights = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for idx in [0, 1]:
            weights[f"qkv_w_{idx}"] = sf.get_tensor(f"model.layers.0.self_attn.qkv_proj_experts.{idx}.weight").to(device=device, dtype=dtype)
            weights[f"qkv_b_{idx}"] = sf.get_tensor(f"model.layers.0.self_attn.qkv_proj_experts.{idx}.bias").to(device=device, dtype=dtype)
            weights[f"o_w_{idx}"] = sf.get_tensor(f"model.layers.0.self_attn.o_proj_experts.{idx}.weight").to(device=device, dtype=dtype)

    _load_attn_weights(module, checkpoint, dtype)

    with torch.no_grad():
        native = module(
            hidden_states.clone(),
            token_types=token_types,
            start_indices=start_indices,
            end_indices=end_indices,
            row_id_map=row_id_map,
            probs=probs,
            orig_shape=(bsz, seq, cfg.hidden_size),
            position_embeddings=(cos, sin),
            attention_mask=None,
            projection_dtype=dtype,
        )
        ref = _reference(
            hidden_states.clone(),
            token_types,
            start_indices,
            end_indices,
            row_id_map,
            probs,
            cfg,
            weights,
            cos,
            sin,
            dtype,
        )

    _compare(f"joint_attention_native_{dtype}_{device.type}", native, ref, 1e-5 if dtype == torch.float32 else 0.0)


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

    print("\nPASS: native JointAttention no-cache/mot_opt path matches independent reference.")


if __name__ == "__main__":
    main()
