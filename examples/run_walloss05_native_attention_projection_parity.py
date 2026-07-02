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
    WallOSS05JointAttentionProjectionNative,
    WallOSS05NativeConfig,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _copy_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    params = dict(module.named_parameters())
    if name not in params:
        raise KeyError(f"{type(module).__name__} has no parameter {name!r}; available={list(params)[:40]}")
    param = params[name]
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: module={tuple(param.shape)} ckpt={tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def _load_projection_weights(
    module: WallOSS05JointAttentionProjectionNative,
    checkpoint: Path,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    mapping = {}
    for idx in [0, 1]:
        mapping[f"qkv_w_{idx}"] = (
            f"qkv_proj_experts.{idx}.weight",
            f"model.layers.0.self_attn.qkv_proj_experts.{idx}.weight",
        )
        mapping[f"qkv_b_{idx}"] = (
            f"qkv_proj_experts.{idx}.bias",
            f"model.layers.0.self_attn.qkv_proj_experts.{idx}.bias",
        )
        mapping[f"o_w_{idx}"] = (
            f"o_proj_experts.{idx}.weight",
            f"model.layers.0.self_attn.o_proj_experts.{idx}.weight",
        )

    out: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for alias, (param_name, ckpt_key) in mapping.items():
            tensor = sf.get_tensor(ckpt_key).to(dtype=dtype)
            _copy_param(module, param_name, tensor)
            out[alias] = tensor
            print(f"[loaded] {ckpt_key} -> {param_name} {tuple(tensor.shape)} {tensor.dtype}")
    return out


def _reference_qkv(
    hidden_states: torch.Tensor,
    cfg: WallOSS05NativeConfig,
    weights: dict[str, torch.Tensor],
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_tokens, hidden = hidden_states.shape
    kv_dim = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)

    q_buffer = torch.zeros(total_tokens, cfg.hidden_size, dtype=hidden_states.dtype)
    k_buffer = torch.zeros(total_tokens, kv_dim, dtype=hidden_states.dtype)
    v_buffer = torch.zeros(total_tokens, kv_dim, dtype=hidden_states.dtype)

    for idx in [0, 1]:
        start = int(start_indices[idx].item())
        end = int(end_indices[idx].item())
        if start == end:
            continue

        dim_input = cfg.dim_inputs[idx]
        x = hidden_states[start:end, :dim_input]
        w = weights[f"qkv_w_{idx}"].to(dtype=x.dtype)
        b = weights[f"qkv_b_{idx}"].to(dtype=x.dtype)
        qkv = F.linear(x, w, b)
        q, k, v = torch.split(qkv, [cfg.hidden_size, kv_dim, kv_dim], dim=-1)

        q_buffer[start:end] = q
        k_buffer[start:end] = k
        v_buffer[start:end] = v

    return q_buffer, k_buffer, v_buffer


def _reference_o(
    attn_output: torch.Tensor,
    cfg: WallOSS05NativeConfig,
    weights: dict[str, torch.Tensor],
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
) -> torch.Tensor:
    total_tokens, hidden = attn_output.shape
    out = torch.zeros(total_tokens, hidden, dtype=attn_output.dtype)

    for idx in [0, 1]:
        start = int(start_indices[idx].item())
        end = int(end_indices[idx].item())
        if start == end:
            continue

        dim_input = cfg.dim_inputs[idx]
        x = attn_output[start:end]
        w = weights[f"o_w_{idx}"].to(dtype=x.dtype)
        projected = F.linear(x, w)
        out[start:end, :dim_input] = projected[:, :dim_input]

    return out


def _compare(name: str, native: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (native.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = float(F.cosine_similarity(native.flatten().float(), ref.flatten().float(), dim=0).item())
    exact_equal = bool(torch.equal(native, ref))
    allclose_1e_6 = bool(torch.allclose(native, ref, atol=1e-6, rtol=1e-6))
    print(f"\n--- {name} ---")
    print("shape:", tuple(native.shape), native.dtype)
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("cosine:", cosine)
    print("exact_equal:", exact_equal)
    print("allclose_1e_6:", allclose_1e_6)
    if max_abs > 1e-5 or cosine < 0.999999:
        raise SystemExit(f"FAILED: {name} projection parity failed")


def _run_one(dtype: torch.dtype, cfg: WallOSS05NativeConfig, checkpoint: Path) -> None:
    print(f"\n========== dtype={dtype} ==========")
    module = WallOSS05JointAttentionProjectionNative(
        cfg,
        layer_idx=0,
        params_dtype=dtype,
        device="cpu",
    )
    module.eval()
    weights = _load_projection_weights(module, checkpoint, dtype=dtype)

    torch.manual_seed(12000 + (0 if dtype == torch.float32 else 1))
    n0 = 5
    n1 = 4
    total = n0 + n1
    hidden_states = torch.randn(total, cfg.hidden_size, dtype=torch.float32).to(dtype)
    attn_output = torch.randn(total, cfg.hidden_size, dtype=torch.float32).to(dtype)
    start_indices = torch.tensor([0, n0], dtype=torch.long)
    end_indices = torch.tensor([n0, total], dtype=torch.long)

    with torch.no_grad():
        q_native, k_native, v_native = module.project_qkv_permuted(
            hidden_states.clone(),
            start_indices.clone(),
            end_indices.clone(),
        )
        o_native = module.project_output_permuted(
            attn_output.clone(),
            start_indices.clone(),
            end_indices.clone(),
        )

        q_ref, k_ref, v_ref = _reference_qkv(
            hidden_states.clone(),
            cfg,
            weights,
            start_indices,
            end_indices,
        )
        o_ref = _reference_o(
            attn_output.clone(),
            cfg,
            weights,
            start_indices,
            end_indices,
        )

    _compare("q_projection", q_native, q_ref)
    _compare("k_projection", k_native, k_ref)
    _compare("v_projection", v_native, v_ref)
    _compare("o_projection", o_native, o_ref)

    tail = o_native[n0:, cfg.dim_inputs[1]:]
    tail_max = float(tail.abs().max().item()) if tail.numel() else 0.0
    print("expert1 o-proj padded tail max abs:", tail_max)
    if tail_max != 0.0:
        raise SystemExit("FAILED: expert1 o-proj padded tail should remain zero")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    args = parser.parse_args()

    L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])

    ckpt_config = _load_json(args.checkpoint / "config.json")
    train_config = _load_yaml(args.train_config)
    cfg = WallOSS05NativeConfig.from_checkpoint_and_train_config(
        ckpt_config,
        train_config,
        norm_key=args.norm_key,
    )

    print("========== Config ==========")
    print("hidden_size:", cfg.hidden_size)
    print("num_attention_heads:", cfg.num_attention_heads)
    print("num_key_value_heads:", cfg.num_key_value_heads)
    print("head_dim:", cfg.hidden_size // cfg.num_attention_heads)
    print("dim_inputs:", cfg.dim_inputs)

    for dtype in [torch.float32, torch.bfloat16]:
        _run_one(dtype, cfg, args.checkpoint)

    print("\nPASS: native JointAttention projection-only path matches reference.")


if __name__ == "__main__":
    main()
