"""RMSNorm with a selectable kernel backend.

The math comes from one of two backends, picked at construction time via
``backend=``:

* ``"flashinfer"`` (default): flashinfer.norm CUDA kernels.
* ``"phyai-kernel"``: phyai_kernel's Triton kernels.

Both compute Llama/Qwen-style RMSNorm with the variance reduction in fp32.
Passing a ``residual`` tensor folds ``residual += x; x = rmsnorm(residual)``
into a single in-place launch. That's the path used between attention and
the MLP in most decoder blocks.

GemmaRMSNorm is the same wrapper bound to the Gemma kernel pair: the
multiplier is ``(1 + w)`` and the weight starts at zero, so a freshly
constructed module is the identity.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from phyai.layers.loaders import ReplicatedLoader

_VALID_BACKENDS: tuple[str, ...] = ("flashinfer", "phyai-kernel")


def _resolve_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown RMSNorm backend {name!r}; expected one of {_VALID_BACKENDS!r}."
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
    ) -> None:
        super().__init__()
        self.backend = _resolve_backend(backend)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            self._initial_weight(hidden_size, dtype), requires_grad=False
        )
        # Replicated across TP ranks, like every other parameter that is
        # bit-identical on every rank.
        self.weight.loader = ReplicatedLoader()  # type: ignore[attr-defined]
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


__all__ = ["RMSNorm", "GemmaRMSNorm"]
