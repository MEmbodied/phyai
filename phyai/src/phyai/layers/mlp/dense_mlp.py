"""DenseMLP — generic 2-layer FFN with optional gating.

Covers two MLP topologies that occur side-by-side in modern VLA models
(SigLIP vision tower + Gemma text tower in PaliGemma is the canonical
example):

* **Gated** (``gated=True``) — SwiGLU / GeGLU style
  ``y = down(act(gate(x)) * up(x))``. Uses
  :class:`~phyai.layers.linear.MergedColumnParallelLinear` to fuse
  ``gate`` and ``up`` into one ``[gate, up]`` column-parallel matmul,
  then a flashinfer fused ``act_and_mul`` kernel, then a
  :class:`~phyai.layers.linear.RowParallelLinear` ``down``. The fused
  layout matches flashinfer's
  ``act(input[..., :H]) * input[..., H:]`` exactly: gate occupies the
  first half (shard_id 0) and up the second half (shard_id 1).

* **Plain** (``gated=False``) — ``y = fc2(act(fc1(x)))``. No fused
  kernel exists, so the activation runs through ``F.gelu`` (or
  ``F.gelu(approximate="tanh")``). This is BERT / CLIP / SigLIP /
  ViT-style. SiLU is rejected here because no real model uses
  non-gated SiLU.

Activation x gated matrix:

==========  ==============  ==========================  ===================================
``gated``   ``activation``  use case                    kernel
==========  ==============  ==========================  ===================================
``True``    ``"silu"``      Llama / Qwen SwiGLU         ``flashinfer.activation.silu_and_mul``
``True``    ``"gelu"``      erf GeGLU                   ``flashinfer.activation.gelu_and_mul``
``True``    ``"gelu_tanh"`` Gemma GeGLU                 ``flashinfer.activation.gelu_tanh_and_mul``
``False``   ``"gelu"``      BERT / CLIP MLP             ``F.gelu``
``False``   ``"gelu_tanh"`` SigLIP / Gemma2 MLP head    ``F.gelu(approximate="tanh")``
``False``   ``"silu"``      (no real model)             rejected — ``ValueError``
==========  ==============  ==========================  ===================================

Aliases ``gelu_pytorch_tanh`` and ``gelu_new`` both normalise to
``gelu_tanh``; underscore vs hyphen is also normalised.

Weight loading: each child linear attaches its own ``hf_keys`` and
``weight_loader`` at construction (see :mod:`phyai.weights`). HF naming
for the gated path defaults to ``gate_proj`` / ``up_proj`` /
``down_proj``; non-gated defaults to ``fc1`` / ``fc2``. The gated leg
names are overridable via the ``gated_hf_legs=`` constructor kwarg for
models that use non-standard names.

Limitations
-----------
* No fused activation+quantisation. flashinfer ships
  ``silu_and_mul_scaled_nvfp4_experts_quantize`` for MoE FP4 but
  plumbing that here would require a non-linear-op spec hook that does
  not exist yet.
* No pre-allocated output buffer for ``act_and_mul``. flashinfer
  allocates internally per call. A buffer-pool optimisation would be a
  separate, future change.
* ``enable_pdl`` is auto-detected by flashinfer; not exposed here.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


_GATED_ACTS = ("silu", "gelu", "gelu_tanh")
_PLAIN_ACTS = ("gelu", "gelu_tanh")
_TANH_ALIASES = ("gelu_tanh", "gelu_pytorch_tanh", "gelu_new")


def _canonicalise_activation(name: str) -> str:
    """Normalise underscore/hyphen and tanh aliases to a canonical form."""
    canon = name.lower().replace("-", "_")
    if canon in _TANH_ALIASES:
        return "gelu_tanh"
    return canon


def _torch_gated_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    canon = _canonicalise_activation(name)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        if canon == "silu":
            return F.silu(gate) * up
        if canon == "gelu":
            return F.gelu(gate) * up
        if canon == "gelu_tanh":
            return F.gelu(gate, approximate="tanh") * up
        raise ValueError(
            f"Unsupported gated activation {name!r}; expected one of {_GATED_ACTS!r}."
        )

    return _apply


def _resolve_gated_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Lazy-import flashinfer fused act-and-mul, with a torch fallback for smoke tests."""
    canon = _canonicalise_activation(name)
    if os.environ.get("PHYAI_FORCE_TORCH_GATED_ACT") == "1":
        return _torch_gated_act(canon)
    if canon == "silu":
        from flashinfer.activation import silu_and_mul

        return silu_and_mul
    if canon == "gelu":
        from flashinfer.activation import gelu_and_mul

        return gelu_and_mul
    if canon == "gelu_tanh":
        from flashinfer.activation import gelu_tanh_and_mul

        return gelu_tanh_and_mul
    raise ValueError(
        f"Unsupported gated activation {name!r}; expected one of {_GATED_ACTS!r}."
    )


def _resolve_plain_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    canon = _canonicalise_activation(name)
    if canon == "gelu":
        return F.gelu
    if canon == "gelu_tanh":
        return functools.partial(F.gelu, approximate="tanh")
    if canon == "silu":
        raise ValueError(
            "non-gated SiLU is not supported (no real model uses it; "
            "did you mean gated=True?)"
        )
    raise ValueError(
        f"Unsupported plain activation {name!r}; expected one of {_PLAIN_ACTS!r}."
    )


class DenseMLP(nn.Module):
    """Generic FFN block. See module docstring for the topology matrix.

    Parameters
    ----------
    hidden_size:
        Input / output channel width. Equal to model hidden size.
    intermediate_size:
        Width of the FFN's hidden activation. For SwiGLU/GeGLU this is
        the size of *each* of ``gate``, ``up``, and the input to
        ``down``.
    activation:
        ``"silu"`` | ``"gelu"`` | ``"gelu_tanh"``. Aliases
        ``gelu_pytorch_tanh`` / ``gelu_new`` map to ``gelu_tanh``.
    gated:
        If ``True`` (default), build the gated SwiGLU/GeGLU path. If
        ``False``, build the plain ``fc1->act->fc2`` path. Cannot combine
        with ``activation="silu"``.
    bias:
        Bias on every internal linear. Llama / Gemma FFNs use
        ``bias=False``; SigLIP / BERT-style use ``bias=True``.
    axis / sp_axis:
        Mesh axis for TP and (optionally) sequence-parallel entry. The
        entry layer (``gate_up_proj`` / ``fc1``) all-gathers along
        ``sp_axis`` if set; the exit layer always reduces along
        ``axis``.
    params_dtype:
        Dtype for parameter allocation. Defaults to torch default.
    spec_in / spec_out:
        Per-leg :class:`~phyai.layers.quant.WeightSpec`. Most configs
        use the same spec for both, but FP8 mixed-precision recipes
        commonly keep ``down_proj`` at bf16 (sensitive reduction). Two
        knobs, no wrapper class.
    mesh:
        Mesh name. Default ``"model"``.
    prefix:
        Dotted state-dict prefix for THIS module (not its parent).
        Children are constructed with ``prefix=f"{prefix}.gate_up_proj"``
        etc. Empty prefix means children skip ``hf_keys`` attachment;
        such an MLP can still run forward but will not load weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Literal["silu", "gelu", "gelu_tanh"] = "silu",
        gated: bool = True,
        bias: bool = False,
        axis: str = "tp",
        sp_axis: str | None = None,
        params_dtype: torch.dtype | None = None,
        spec_in: object | None = None,
        spec_out: object | None = None,
        gated_hf_legs: tuple[str, str] = ("gate_proj", "up_proj"),
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = _canonicalise_activation(activation)
        self.gated = gated
        self.bias_enabled = bias
        self.prefix = prefix

        if gated:
            self.gate_up_proj = MergedColumnParallelLinear(
                in_features=hidden_size,
                output_sizes=[intermediate_size, intermediate_size],
                axis=axis,
                sp_axis=sp_axis,
                gather_output=False,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_in,
                hf_legs=gated_hf_legs,
                mesh=mesh,
                prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                axis=axis,
                sp_axis=sp_axis,
                input_is_parallel=True,
                reduce_results=True,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_out,
                mesh=mesh,
                prefix=f"{prefix}.down_proj" if prefix else "down_proj",
            )
            # TODO(fp8/fp4 fused act-quant): wire a spec hook that fuses
            # silu_and_mul + per-token fp8/nvfp4 quant for the down_proj
            # input. Today the act and quant happen as two separate
            # kernels.
            self._act_and_mul = _resolve_gated_act(activation)
            self._act_fn = None
        else:
            self.fc1 = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=intermediate_size,
                axis=axis,
                sp_axis=sp_axis,
                gather_output=False,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_in,
                mesh=mesh,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
            )
            self.fc2 = RowParallelLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                axis=axis,
                sp_axis=sp_axis,
                input_is_parallel=True,
                reduce_results=True,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_out,
                mesh=mesh,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
            )
            self._act_and_mul = None
            self._act_fn = _resolve_plain_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            fused, _ = self.gate_up_proj(x)
            activated = self._act_and_mul(fused)
            out, _ = self.down_proj(activated)
            return out
        h, _ = self.fc1(x)
        h = self._act_fn(h)
        out, _ = self.fc2(h)
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"activation={self.activation!r}, gated={self.gated}, "
            f"bias={self.bias_enabled}"
        )


__all__ = ["DenseMLP"]
