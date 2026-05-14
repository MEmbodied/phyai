"""Triton LayerNorm kernel for the SigLIP / BERT / ViT model families.

One public entry point:

* :func:`layernorm` — Standard ``y = (x - mean) / sqrt(var + eps) * gamma + beta``
  with optional ``beta``. Used by SigLIP's encoder ``layer_norm1`` /
  ``layer_norm2`` / ``post_layernorm`` and any vanilla ``nn.LayerNorm``.

Single-pass for ``D <= _SINGLE_BLOCK_MAX``, two-pass otherwise. Reductions
and the affine transform run in fp32 to match
:func:`torch.nn.functional.layer_norm`. Output is cast back to ``x.dtype``.

``gamma`` (weight) and ``beta`` (bias) may be in any floating dtype — the
kernel upcasts them to fp32. SigLIP HF checkpoints typically store both in
the activation dtype (bf16); flashinfer's CUDA layernorm requires fp32
gamma/beta, but this Triton kernel doesn't, which is convenient.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Largest single-block row width. Same budget as rms_norm.py — 8192 fp32
# lanes = 32 KiB per row, well within per-SM SRAM on Ampere/Hopper/Blackwell.
_SINGLE_BLOCK_MAX = 8192


# --------------------------------------------------------------------------- #
# Single-block kernel (one row per program; D fits a single tile)             #
# --------------------------------------------------------------------------- #


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    # Mean / variance with masked sums (so the padding lanes don't bias the
    # reduction). Cast n_cols to fp32 for the divisor.
    n = n_cols.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = tl.rsqrt(var + eps)
    x_hat = diff * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w
    if HAS_BIAS:
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y + b

    tl.store(
        out_ptr + row * out_row_stride + cols,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# --------------------------------------------------------------------------- #
# Two-pass kernel for very large hidden_size (D > _SINGLE_BLOCK_MAX)           #
# --------------------------------------------------------------------------- #


@triton.jit
def _layernorm_fwd_kernel_large(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    n = n_cols.to(tl.float32)

    # Pass 1: mean over the full row.
    s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        s += tl.where(mask, x, 0.0)
    mean = tl.sum(s, axis=0) / n

    # Pass 2: variance over the full row (with the now-known mean).
    sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        diff = tl.where(mask, x - mean, 0.0)
        sq += diff * diff
    var = tl.sum(sq, axis=0) / n
    rstd = tl.rsqrt(var + eps)

    # Pass 3: normalize, affine, store.
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x_hat = (x - mean) * rstd
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_hat * w
        if HAS_BIAS:
            b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y + b
        tl.store(
            out_ptr + row * out_row_stride + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


# --------------------------------------------------------------------------- #
# Python entry point                                                          #
# --------------------------------------------------------------------------- #


def _flatten_input(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    return x2d, orig_shape


def _check_inputs(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> None:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("phyai_kernel.triton.layernorm: tensors must live on CUDA")
    if weight.dim() != 1:
        raise RuntimeError(
            f"phyai_kernel.triton.layernorm: weight must be 1D, got {weight.dim()}D"
        )
    if x.shape[-1] != weight.shape[0]:
        raise RuntimeError(
            f"phyai_kernel.triton.layernorm: weight ({weight.shape[0]}) must match "
            f"last dim of input ({x.shape[-1]})"
        )
    if bias is not None:
        if not bias.is_cuda:
            raise RuntimeError("phyai_kernel.triton.layernorm: bias must live on CUDA")
        if bias.dim() != 1 or bias.shape[0] != weight.shape[0]:
            raise RuntimeError(
                f"phyai_kernel.triton.layernorm: bias shape {tuple(bias.shape)} must "
                f"match weight shape {tuple(weight.shape)}"
            )


def _launch_layernorm(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out2d: torch.Tensor,
    eps: float,
) -> None:
    n_rows, n_cols = x2d.shape
    if n_rows == 0:
        return

    has_bias = bias is not None
    # Triton kernels don't accept ``None``; pass weight as a stand-in pointer
    # and gate via the ``HAS_BIAS`` constexpr so the load is dropped.
    bias_ptr = bias if has_bias else weight

    if n_cols <= _SINGLE_BLOCK_MAX:
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
        _layernorm_fwd_kernel[(n_rows,)](
            x2d,
            weight,
            bias_ptr,
            out2d,
            x2d.stride(0),
            out2d.stride(0),
            n_cols,
            eps,
            HAS_BIAS=has_bias,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        block_size = 4096
        _layernorm_fwd_kernel_large[(n_rows,)](
            x2d,
            weight,
            bias_ptr,
            out2d,
            x2d.stride(0),
            out2d.stride(0),
            n_cols,
            eps,
            HAS_BIAS=has_bias,
            BLOCK_SIZE=block_size,
            num_warps=16,
        )


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard LayerNorm.

    ``out = ((x - mean) * rsqrt(var + eps) * weight + bias).to(x.dtype)``,
    with the reduction and affine done in fp32. ``bias`` is optional —
    pass ``None`` for nn.LayerNorm-with-bias-disabled style.
    """
    _check_inputs(x, weight, bias)
    x2d, orig_shape = _flatten_input(x)
    if out is None:
        out_t = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise RuntimeError(
                "phyai_kernel.triton.layernorm: `out` must match input shape and dtype"
            )
        out_t = out
    out2d = out_t.reshape(-1, orig_shape[-1])
    _launch_layernorm(x2d, weight, bias, out2d, eps)
    return out_t
