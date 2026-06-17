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
    WallOSS05DecoderFFNBlockNative,
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


def _official_qwen2_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normed = hidden_states * torch.rsqrt(variance + eps)
    normed = weight.to(torch.float32) * normed
    return normed.to(input_dtype)


def _mlp_forward(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    # Match the native module dtype. In the real checkpoint MoE weights are bf16,
    # while norm weights are fp32. For the bf16 parity path, linear weights must
    # be bf16 as well because torch F.linear does not mix bf16 inputs with fp32 weights.
    gate_up = gate_up.to(dtype=x.dtype, device=x.device)
    down = down.to(dtype=x.dtype, device=x.device)
    gate_up_out = F.linear(x, gate_up)
    gate_out, up_out = gate_up_out.split([intermediate_size, intermediate_size], dim=-1)
    act_out = F.silu(gate_out) * up_out
    return F.linear(act_out, down)


def _reference_ffn(
    hidden_states: torch.Tensor,
    cfg: WallOSS05NativeConfig,
    weights: dict[str, torch.Tensor],
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
) -> torch.Tensor:
    residual = hidden_states
    normed = torch.zeros_like(hidden_states)

    for idx in [0, 1]:
        start = int(start_indices[idx].item())
        end = int(end_indices[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        key = f"post_norm_{idx}"
        normed[start:end, :dim] = _official_qwen2_rmsnorm(
            hidden_states[start:end, :dim],
            weights[key],
            cfg.rms_norm_eps,
        ).to(hidden_states.dtype)

    moe_out = torch.zeros_like(hidden_states)
    for idx in [0, 1]:
        start = int(start_indices[idx].item())
        end = int(end_indices[idx].item())
        if start == end:
            continue
        dim = cfg.dim_inputs[idx]
        intermediate = int(cfg.experts[idx]["intermediate_size"])
        out = _mlp_forward(
            normed[start:end, :dim],
            weights[f"expert_{idx}_gate_up"],
            weights[f"expert_{idx}_down"],
            intermediate,
        )
        moe_out[start:end, :dim] = out[:, :dim].to(hidden_states.dtype)

    return residual + moe_out


def _load_weights(
    module: WallOSS05DecoderFFNBlockNative,
    checkpoint: Path,
    *,
    moe_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    mapping = {
        "post_norm_0": (
            "post_attention_norm.norms.0.weight",
            "model.layers.0.post_attention_layernorms.0.weight",
        ),
        "post_norm_1": (
            "post_attention_norm.norms.1.weight",
            "model.layers.0.post_attention_layernorms.1.weight",
        ),
        "expert_0_gate_up": (
            "moe.experts.0.gate_up_proj.weight",
            "model.layers.0.moe.experts.0.gate_up_proj.weight",
        ),
        "expert_0_down": (
            "moe.experts.0.down_proj.weight",
            "model.layers.0.moe.experts.0.down_proj.weight",
        ),
        "expert_1_gate_up": (
            "moe.experts.1.gate_up_proj.weight",
            "model.layers.0.moe.experts.1.gate_up_proj.weight",
        ),
        "expert_1_down": (
            "moe.experts.1.down_proj.weight",
            "model.layers.0.moe.experts.1.down_proj.weight",
        ),
    }

    out: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for alias, (param_name, ckpt_key) in mapping.items():
            raw = sf.get_tensor(ckpt_key)
            if alias.startswith("expert_"):
                tensor = raw.to(dtype=moe_dtype)
            else:
                tensor = raw.float()
            _copy_param(module, param_name, tensor)
            out[alias] = tensor
            print(f"[loaded] {ckpt_key} -> {param_name} {tuple(tensor.shape)} {tensor.dtype}")

    return out


def _run_one(dtype: torch.dtype, cfg: WallOSS05NativeConfig, checkpoint: Path) -> None:
    module = WallOSS05DecoderFFNBlockNative(
        cfg,
        layer_idx=0,
        params_dtype=dtype,
        device="cpu",
    )
    module.eval()
    weights = _load_weights(module, checkpoint, moe_dtype=dtype)

    torch.manual_seed(9090 + (0 if dtype == torch.float32 else 1))
    n0 = 4
    n1 = 3
    total = n0 + n1
    hidden_states = torch.randn(total, cfg.hidden_size, dtype=torch.float32).to(dtype)
    start_indices = torch.tensor([0, n0], dtype=torch.long)
    end_indices = torch.tensor([n0, total], dtype=torch.long)

    with torch.no_grad():
        native_out = module(
            hidden_states.clone(),
            start_indices.clone(),
            end_indices.clone(),
            adarms_conds=[None, None],
        )
        ref_out = _reference_ffn(
            hidden_states.clone(),
            cfg,
            weights,
            start_indices,
            end_indices,
        )

    diff = (native_out.float() - ref_out.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = float(F.cosine_similarity(native_out.flatten().float(), ref_out.flatten().float(), dim=0).item())
    exact_equal = bool(torch.equal(native_out, ref_out))
    allclose_1e_6 = bool(torch.allclose(native_out, ref_out, atol=1e-6, rtol=1e-6))

    print(f"\n===== dtype={dtype} =====")
    print("native_out shape:", tuple(native_out.shape), native_out.dtype)
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("cosine:", cosine)
    print("exact_equal:", exact_equal)
    print("allclose_1e_6:", allclose_1e_6)

    if max_abs > 1e-5 or cosine < 0.999999:
        raise SystemExit(f"FAILED: decoder FFN parity failed for dtype={dtype}")


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
    print("dim_inputs:", cfg.dim_inputs)
    print("experts:", cfg.experts)
    print("norm_moe:", cfg.norm_moe)
    print("mlp_moe:", cfg.mlp_moe)
    print("mot_opt:", cfg.mot_opt)

    for dtype in [torch.float32, torch.bfloat16]:
        _run_one(dtype, cfg, args.checkpoint)

    print("\nPASS: native decoder FFN/MoE subpath matches reference formula.")


if __name__ == "__main__":
    main()
