from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
if str(PHYAI_SRC) not in sys.path:
    sys.path.insert(0, str(PHYAI_SRC))

from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05AttentionCoreNative,
    WallOSS05NativeConfig,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _repeat_kv_reference(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states.unsqueeze(3)
    hidden_states = hidden_states.expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _prepare_mask_reference(
    attention_mask: torch.Tensor | None,
    *,
    bsz: int,
    q_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    causal_mask = attention_mask
    if attention_mask is not None:
        if len(attention_mask.shape) == 2:
            _, seq_len = attention_mask.shape
            causal_mask = attention_mask.view(bsz, 1, 1, seq_len).expand(
                bsz, 1, q_len, seq_len
            )
        elif len(attention_mask.shape) == 3:
            causal_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.shape) == 4:
            causal_mask = attention_mask
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        causal_mask = causal_mask.to(torch.bool)

    if q_len == 1:
        causal_mask = torch.ones(
            bsz,
            1,
            1,
            key_len,
            device=device,
            dtype=dtype,
        ).contiguous()
        causal_mask = causal_mask.to(torch.bool)

    return causal_mask


def _reference_attention_core(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cfg: WallOSS05NativeConfig,
    attention_mask: torch.Tensor | None,
    projection_dtype: torch.dtype,
) -> torch.Tensor:
    bsz, q_len, _, _ = query_states.shape

    key_states = _repeat_kv_reference(key_states, cfg.num_attention_heads // cfg.num_key_value_heads)
    value_states = _repeat_kv_reference(value_states, cfg.num_attention_heads // cfg.num_key_value_heads)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if query_states.dtype != projection_dtype:
        query_states = query_states.to(projection_dtype)
    if key_states.dtype != projection_dtype:
        key_states = key_states.to(projection_dtype)
    if value_states.dtype != projection_dtype:
        value_states = value_states.to(projection_dtype)

    causal_mask = _prepare_mask_reference(
        attention_mask,
        bsz=bsz,
        q_len=q_len,
        key_len=key_states.shape[2],
        device=query_states.device,
        dtype=query_states.dtype,
    )

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=cfg.attention_dropout,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    if attn_output.dtype != projection_dtype:
        attn_output = attn_output.to(projection_dtype)

    return attn_output


def _compare(name: str, native: torch.Tensor, ref: torch.Tensor, *, atol: float) -> None:
    diff = (native.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = float(F.cosine_similarity(native.flatten().float(), ref.flatten().float(), dim=0).item())
    exact_equal = bool(torch.equal(native, ref))
    allclose = bool(torch.allclose(native, ref, atol=atol, rtol=atol))

    print(f"\n--- {name} ---")
    print("shape:", tuple(native.shape), native.dtype, native.device)
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("cosine:", cosine)
    print("exact_equal:", exact_equal)
    print(f"allclose_{atol}:", allclose)

    if max_abs > atol or cosine < 0.999999:
        raise SystemExit(f"FAILED: {name} attention core parity failed")


def _make_mask(case: str, bsz: int, q_len: int, device: torch.device) -> torch.Tensor | None:
    if case == "none":
        return None
    if case == "2d":
        mask = torch.ones(bsz, q_len, dtype=torch.bool, device=device)
        if q_len > 2:
            mask[0, -1] = False
        return mask
    if case == "3d":
        mask = torch.ones(bsz, q_len, q_len, dtype=torch.bool, device=device)
        mask = torch.tril(mask)
        if q_len > 2:
            mask[0, :, -1] = False
        return mask
    if case == "4d":
        mask = torch.ones(bsz, 1, q_len, q_len, dtype=torch.bool, device=device)
        mask = torch.tril(mask)
        if q_len > 2:
            mask[0, :, :, -1] = False
        return mask
    raise ValueError(case)


def _run_one(cfg: WallOSS05NativeConfig, dtype: torch.dtype, device: torch.device, q_len: int, mask_case: str) -> None:
    print(f"\n========== dtype={dtype} device={device} q_len={q_len} mask={mask_case} ==========")

    bsz = 2
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    torch.manual_seed(21000 + q_len + (0 if dtype == torch.float32 else 100) + (0 if mask_case == "none" else len(mask_case)))
    q = torch.randn(bsz, q_len, cfg.num_attention_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    k = torch.randn(bsz, q_len, cfg.num_key_value_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    v = torch.randn(bsz, q_len, cfg.num_key_value_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    mask = _make_mask(mask_case, bsz, q_len, device)

    module = WallOSS05AttentionCoreNative(cfg).to(device).eval()

    with torch.no_grad():
        native_out = module(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=None if mask is None else mask.clone(),
            projection_dtype=dtype,
        )
        ref_out = _reference_attention_core(
            q.clone(),
            k.clone(),
            v.clone(),
            cfg,
            None if mask is None else mask.clone(),
            projection_dtype=dtype,
        )

    # Same PyTorch SDPA path should be exact for these tests.
    _compare(
        f"attention_core_q{q_len}_{mask_case}_{dtype}_{device.type}",
        native_out,
        ref_out,
        atol=1e-6 if dtype == torch.float32 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    args = parser.parse_args()

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
    print("attention_dropout:", cfg.attention_dropout)

    cpu_cases = [
        (torch.float32, torch.device("cpu"), 5, "none"),
        (torch.float32, torch.device("cpu"), 5, "2d"),
        (torch.float32, torch.device("cpu"), 5, "3d"),
        (torch.float32, torch.device("cpu"), 5, "4d"),
        (torch.float32, torch.device("cpu"), 1, "none"),
    ]

    for case in cpu_cases:
        _run_one(cfg, *case)

    if torch.cuda.is_available():
        cuda_cases = [
            (torch.float32, torch.device("cuda:0"), 5, "none"),
            (torch.float32, torch.device("cuda:0"), 5, "4d"),
            (torch.bfloat16, torch.device("cuda:0"), 5, "none"),
            (torch.bfloat16, torch.device("cuda:0"), 5, "4d"),
            (torch.bfloat16, torch.device("cuda:0"), 1, "none"),
        ]
        for case in cuda_cases:
            _run_one(cfg, *case)

    print("\nPASS: native WALL-OSS-0.5 attention core matches reference SDPA layout/mask behavior.")


if __name__ == "__main__":
    main()
