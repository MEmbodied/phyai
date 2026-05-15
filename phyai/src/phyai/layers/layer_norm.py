"""RMSNorm + LayerNorm + AdaRMSNorm with selectable kernel backends.

Three related modules in this file:

* :class:`RMSNorm` — Llama / Qwen RMSNorm; :class:`GemmaRMSNorm` for the
  Gemma ``(1 + w)`` variant. Used by every text decoder phyai targets.
* :class:`LayerNorm` — standard mean/variance LayerNorm with optional
  bias. Used by SigLIP / BERT / ViT vision towers.
* :class:`AdaRMSNorm` — adaptive RMSNorm with a learned conditioning
  projection. Replaces the ``(1 + w)`` affine with ``(1 + scale)`` and
  ``+ shift`` from a per-token ``cond`` vector, and exposes a ``gate``
  output for the surrounding gated-residual. Used by pi0.5 / pi0.6 action
  experts (``use_adarms=True``).

Backend selection (constructor ``backend=``):

* :class:`RMSNorm` / :class:`GemmaRMSNorm` / :class:`LayerNorm`:
  ``"flashinfer"`` (default) or ``"phyai-kernel"``.
* :class:`AdaRMSNorm`: ``"phyai-kernel"`` (default, Triton on CUDA) or
  ``"torch"`` (eager fallback for CPU / MPS / non-CUDA). flashinfer has
  no AdaRMS kernel, so that backend is rejected.

Reductions and the affine multiply run in fp32 on every backend; output
is cast back to ``x.dtype``. RMSNorm's ``forward`` accepts an optional
``residual`` for the fused ``residual += x; rmsnorm(residual)`` path used
between attention and the MLP in most decoder blocks. LayerNorm has no
fused-add path today (SigLIP's encoder doesn't need one). AdaRMSNorm's
``forward(x, cond)`` returns a ``(out, gate)`` tuple.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from phyai.layers.placement import (
    CopyPlacement,
    Placement,
    split_prefix,
)

_VALID_BACKENDS: tuple[str, ...] = ("flashinfer", "phyai-kernel")


def _resolve_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown norm backend {name!r}; expected one of {_VALID_BACKENDS!r}."
        )
    return canonical


class RMSNorm(nn.Module):
    """Llama / Qwen RMSNorm.

    Computes ``y = (x * rsqrt(mean(x ** 2) + eps)) * weight``. The variance
    and the weight multiply both run in fp32; the result is cast back to
    ``x.dtype`` on the way out.

    Parameters
    ----------
    hidden_size:
        Size of the last dim of the input. Weight is ``(hidden_size,)``.
    eps:
        Added to the variance before ``rsqrt`` for numerical stability.
    backend:
        ``"flashinfer"`` (default) or ``"phyai-kernel"``. Underscore,
        hyphen, and case are normalized.
    dtype:
        Optional weight dtype. Defaults to the global default dtype.
        **flashinfer caveat**: the CUDA RMSNorm / GemmaRMSNorm /
        FusedAddRMSNorm kernels do *not* check weight dtype — they
        ``static_cast`` the weight pointer to the input ``c_type`` (fp16
        or bf16) inside the dispatch macro. Passing an fp32 weight when
        the input is bf16 silently produces garbage. So when
        ``backend="flashinfer"``, ``dtype`` must match the dtype of the
        tensor that will be normalized (typically ``torch.bfloat16``);
        do *not* leave it as the fp32 default. The ``"phyai-kernel"``
        Triton path accepts any floating dtype.

    The forward signature is ``forward(x, residual=None)``:

    * with ``residual`` left as ``None``, returns the normalized tensor;
    * with a ``residual`` tensor, returns ``(y, residual)``. Both buffers
      are written in place by the kernel and the same objects come back.

    On the no-residual path, higher-rank inputs are flattened to 2-D for
    the kernel and reshaped back. The residual path expects 2-D contiguous
    inputs (the kernels themselves don't know how to reshape).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        backend: str = "flashinfer",
        *,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.backend = _resolve_backend(backend)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.prefix = prefix
        self.weight = nn.Parameter(
            self._initial_weight(hidden_size, dtype), requires_grad=False
        )
        self._rmsnorm, self._fused_add_rmsnorm = self._load_kernels(self.backend)

    @staticmethod
    def _load_kernels(backend: str) -> tuple[Callable, Callable]:
        """Return ``(rmsnorm, fused_add_rmsnorm)`` for the chosen backend.

        The imports live inside each branch on purpose: picking one backend
        shouldn't drag in the other's package. Subclasses override this to
        swap in a different kernel pair, e.g. Gemma's ``(1 + w)`` variant.
        """
        if backend == "flashinfer":
            from flashinfer.norm import fused_add_rmsnorm, rmsnorm

            return rmsnorm, fused_add_rmsnorm
        from phyai_kernel import fused_add_rmsnorm, rmsnorm

        return rmsnorm, fused_add_rmsnorm

    @staticmethod
    def _initial_weight(hidden_size: int, dtype: torch.dtype | None) -> torch.Tensor:
        # The kernel multiplies by ``w``, so identity is ``w == 1``.
        return torch.ones(hidden_size, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            # Fused add then norm, in place. flashinfer mutates and returns
            # ``None``; phyai-kernel hands back the ``(x, residual)`` pair.
            # Collapse both into one return shape so callers don't care.
            ret = self._fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            if ret is None:
                return x, residual
            return ret

        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])
        out = self._rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(orig_shape)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, backend={self.backend!r}"
        )

    def placements(self) -> list[Placement]:
        parent, own = split_prefix(self.prefix)
        hf_base = f"{parent}.{own}" if parent else own
        return [
            CopyPlacement(
                hf_key=f"{hf_base}.weight",
                phyai_key=f"{self.prefix}.weight",
            )
        ]


class GemmaRMSNorm(RMSNorm):
    """Gemma-flavoured RMSNorm.

    Same wrapping as RMSNorm, just bound to the Gemma kernel pair: the
    multiplier is ``(1 + weight)`` and the weight starts at zero, so a
    freshly constructed module is the identity. Matches the HF Gemma /
    Gemma3 conventions.
    """

    @staticmethod
    def _load_kernels(backend: str) -> tuple[Callable, Callable]:
        if backend == "flashinfer":
            from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

            return gemma_rmsnorm, gemma_fused_add_rmsnorm
        from phyai_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm

        return gemma_rmsnorm, gemma_fused_add_rmsnorm

    @staticmethod
    def _initial_weight(hidden_size: int, dtype: torch.dtype | None) -> torch.Tensor:
        # The Gemma kernel multiplies by ``(1 + w)``, so identity is ``w == 0``.
        return torch.zeros(hidden_size, dtype=dtype)


class LayerNorm(nn.Module):
    """Standard LayerNorm with a selectable kernel backend.

    Computes ``y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias``,
    with mean / variance / affine all in fp32 and the output cast back to
    ``x.dtype``. This is the path SigLIP / BERT / ViT use; PaliGemma's
    SigLIP vision tower has three such norms (two per encoder layer plus
    a ``post_layernorm``).

    Parameters
    ----------
    hidden_size:
        Last dim of the input. ``weight`` and (when present) ``bias`` are
        ``(hidden_size,)``.
    eps:
        Numerical-stability epsilon. Default ``1e-5`` matches
        :class:`torch.nn.LayerNorm`. SigLIP HF configs use ``1e-6``.
    backend:
        ``"flashinfer"`` (default) or ``"phyai-kernel"``.
    bias:
        Whether to allocate a learnable ``beta``. Defaults to ``True``
        (SigLIP / BERT / ViT). flashinfer's kernel always reads ``beta``;
        when ``bias=False`` the wrapper feeds it a zero buffer so the
        kernel's add becomes a no-op.
    dtype:
        Optional weight / bias dtype. Defaults to the global default.
        flashinfer's CUDA kernel hard-checks ``gamma`` / ``beta`` in
        fp32 (``norm.cu`` aborts otherwise), so this wrapper overrides
        the caller's ``dtype`` to ``torch.float32`` when
        ``backend="flashinfer"`` — the parameters are allocated in fp32
        once at construction and the hot path can hand the buffers to
        the kernel directly, no per-forward cast. The Triton kernel
        accepts any floating dtype natively.
    prefix:
        Dotted state-dict prefix for placement loading.

    Forward
    -------
    ``forward(x) -> y`` where ``x`` is any 2-D-or-higher tensor with last
    dim ``hidden_size``. Higher-rank inputs are flattened to ``(N, D)``
    for the kernel and reshaped back.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        backend: str = "flashinfer",
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        self.backend = _resolve_backend(backend)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.has_bias = bias
        self.prefix = prefix

        # flashinfer's CUDA layernorm hard-requires fp32 gamma/beta. Pre-allocate
        # in fp32 once so the hot path skips the per-forward cast. phyai-kernel's
        # Triton path accepts any floating dtype, so honor the caller's ``dtype``.
        param_dtype = torch.float32 if self.backend == "flashinfer" else dtype

        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=param_dtype), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=param_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        self._layernorm = self._load_kernel(self.backend)
        # flashinfer always reads ``beta``. With ``bias=False`` we still need a
        # zero buffer to feed the kernel — kept in fp32 to match the contract.
        if self.backend == "flashinfer" and not bias:
            self.register_buffer(
                "_zero_beta",
                torch.zeros(hidden_size, dtype=torch.float32),
                persistent=False,
            )

    @staticmethod
    def _load_kernel(backend: str) -> Callable:
        if backend == "flashinfer":
            from flashinfer.norm import layernorm

            return layernorm
        from phyai_kernel import layernorm

        return layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])

        if self.backend == "flashinfer":
            # gamma / beta are pre-allocated in fp32 (see __init__) — hand the
            # buffers to the CUDA kernel directly, no per-forward cast.
            beta = self.bias.data if self.has_bias else self._zero_beta
            out = self._layernorm(x, self.weight.data, beta, self.variance_epsilon)
        else:
            # Triton kernel handles any dtype on weight/bias and accepts
            # ``bias=None`` directly — pass weight/bias.data through.
            bias_data = self.bias.data if self.has_bias else None
            out = self._layernorm(x, self.weight.data, bias_data, self.variance_epsilon)

        if needs_reshape:
            out = out.reshape(orig_shape)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, "
            f"bias={self.has_bias}, backend={self.backend!r}"
        )

    def placements(self) -> list[Placement]:
        parent, own = split_prefix(self.prefix)
        hf_base = f"{parent}.{own}" if parent else own
        out: list[Placement] = [
            CopyPlacement(
                hf_key=f"{hf_base}.weight",
                phyai_key=f"{self.prefix}.weight",
            )
        ]
        if self.has_bias:
            out.append(
                CopyPlacement(
                    hf_key=f"{hf_base}.bias",
                    phyai_key=f"{self.prefix}.bias",
                )
            )
        return out


# --------------------------------------------------------------------------- #
# AdaRMSNorm — adaptive RMSNorm with (scale, shift, gate) conditioning.
# --------------------------------------------------------------------------- #


_ADARMS_BACKENDS: tuple[str, ...] = ("phyai-kernel", "torch")


def _resolve_adarms_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical == "flashinfer":
        raise ValueError(
            "AdaRMSNorm has no flashinfer backend; pick 'phyai-kernel' "
            "(default, Triton CUDA) or 'torch' (eager fallback)."
        )
    if canonical not in _ADARMS_BACKENDS:
        raise ValueError(
            f"Unknown AdaRMSNorm backend {name!r}; expected one of "
            f"{_ADARMS_BACKENDS!r}."
        )
    return canonical


def _torch_adarmsnorm(
    x: torch.Tensor,
    modulation: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager torch reference path used by the ``"torch"`` backend.

    ``modulation`` must already be broadcast-shaped against ``x`` along
    the last dim (the caller's :class:`AdaRMSNorm.forward` handles the
    ``(B, 3D) -> (B, 1, 3D)`` unsqueeze for 3-D ``x``). Reductions and
    the affine run in fp32; gate is cast to ``x.dtype``.
    """
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = xf * (1.0 + scale.float()) + shift.float()
    return out.to(dtype), gate.to(dtype)


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with conditional ``(scale, shift, gate)`` modulation.

    Forward signature ``forward(x, cond) -> (out, gate)``::

        modulation = self.dense(cond)         # (..., 3 * D)
        normed     = x * rsqrt(mean(x^2)+eps) # fp32 reduction
        scale, shift, gate = chunk(modulation, 3, dim=-1)
        out  = (normed * (1 + scale) + shift).to(x.dtype)
        gate = gate.to(x.dtype)

    The ``(1 + weight)`` term of standard Gemma RMSNorm is *replaced* by
    ``(1 + scale)`` from the conditioning projection; there is no learned
    ``weight`` parameter on this class. ``self.dense.weight`` and
    ``self.dense.bias`` are zero-initialised so a freshly constructed
    AdaRMSNorm is the identity (``scale=0``, ``shift=0``, ``gate=0``).

    Used by the pi0.5 / pi0.6 action-expert decoder layers
    (``use_adarms=True``); the prefix-side decoder layers use plain
    :class:`GemmaRMSNorm`.

    Parameters
    ----------
    hidden_size:
        Last dim ``D`` of the input. Modulation projection produces
        ``3 * D`` channels.
    cond_dim:
        Width of the conditioning vector ``cond``. The dense projection is
        ``Linear(cond_dim, 3 * hidden_size, bias=True)``.
    eps:
        Numerical-stability epsilon for the variance reduction.
    backend:
        ``"phyai-kernel"`` (default — Triton on CUDA) or ``"torch"``
        (eager fp32 fallback for CPU / MPS / non-CUDA hosts and for
        ``torch.compile`` integration). flashinfer has no AdaRMS kernel
        and is rejected at construction time.
    dtype:
        Optional dtype for ``self.dense``'s parameters. Defaults to the
        global default dtype.
    prefix:
        Dotted state-dict prefix for placement loading.

    Forward
    -------
    * 3-D ``x`` ``(B, S, D)`` with 2-D ``cond`` ``(B, cond_dim)``: the
      modulation is unsqueezed to ``(B, 1, 3D)`` and broadcast across the
      sequence axis. ``gate`` comes back shaped ``(B, 1, D)`` so the
      caller's ``residual + out * gate`` broadcasts correctly.
    * Same-rank ``x`` and ``cond``: 1:1 per-row mapping. ``gate`` shape
      mirrors ``cond`` (``(N, D)``).

    The Triton kernel handles both shapes via a ``group_size`` derived
    from ``prod(x.shape[:-1]) / prod(modulation.shape[:-1])``; the torch
    backend just does the arithmetic broadcast.
    """

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        eps: float = 1e-6,
        backend: str = "phyai-kernel",
        *,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}.")
        self.backend = _resolve_adarms_backend(backend)
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        self.prefix = prefix

        # Zero-init dense for AdaRMS identity at construction time, matching
        # lerobot ``PiGemmaRMSNorm`` and the openpi action-expert reference.
        self.dense = nn.Linear(cond_dim, 3 * hidden_size, bias=True, dtype=dtype)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        self.dense.weight.requires_grad_(False)
        self.dense.bias.requires_grad_(False)

        self._adarms_kernel = self._load_kernel(self.backend)

    @staticmethod
    def _load_kernel(backend: str) -> Optional[Callable]:
        if backend == "phyai-kernel":
            from phyai_kernel import adarmsnorm

            return adarmsnorm
        # Torch backend uses :func:`_torch_adarmsnorm` inline.
        return None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"AdaRMSNorm: cond last dim {cond.shape[-1]} does not match "
                f"cond_dim={self.cond_dim}."
            )
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"AdaRMSNorm: x last dim {x.shape[-1]} does not match "
                f"hidden_size={self.hidden_size}."
            )

        modulation = self.dense(cond)
        # Match lerobot ``PiGemmaRMSNorm``: when ``x`` is 3-D ``(B, S, D)``
        # and ``modulation`` came from a 2-D ``cond`` ``(B, cond_dim)``,
        # broadcast the modulation across the sequence axis.
        if x.dim() == 3 and modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)

        if self.backend == "phyai-kernel":
            return self._adarms_kernel(x, modulation, self.variance_epsilon)
        return _torch_adarmsnorm(x, modulation, self.variance_epsilon)

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, cond_dim={self.cond_dim}, "
            f"eps={self.variance_epsilon}, backend={self.backend!r}"
        )

    def placements(self) -> list[Placement]:
        parent, own = split_prefix(self.prefix)
        hf_base = f"{parent}.{own}" if parent else own
        return [
            CopyPlacement(
                hf_key=f"{hf_base}.dense.weight",
                phyai_key=f"{self.prefix}.dense.weight",
            ),
            CopyPlacement(
                hf_key=f"{hf_base}.dense.bias",
                phyai_key=f"{self.prefix}.dense.bias",
            ),
        ]


__all__ = ["AdaRMSNorm", "GemmaRMSNorm", "LayerNorm", "RMSNorm"]
