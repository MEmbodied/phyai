"""GPTQ — Hessian-based weight-only quantization (Frantar et al.).

Needs calibration (the Hessian comes from a :class:`HessianObserver` fed each
Linear's inputs), so it routes into the sequential pipeline. Grouped quant recomputes
group params at group boundaries from the error-compensated weight, matching
AutoGPTQ's non-static-group behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from phyai_model_optimizer.modifiers.base import Modifier, QuantResult
from phyai_model_optimizer.observers.base import Observer
from phyai_model_optimizer.observers.hessian import HessianObserver
from phyai_model_optimizer.quant_math import (
    QuantDType,
    WeightQuant,
    compute_scale_zp,
    quantize_to_codes,
)


def _quantize_columns(
    w: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor | None, q: WeightQuant
) -> torch.Tensor:
    """Dequantized column block ``w`` (N, c), in compressed-tensors' signed-int convention."""
    qmin, qmax = q.int_range()
    if zp is None:
        codes = torch.clamp(torch.round(w / scale), qmin, qmax)
        return codes * scale
    codes = torch.clamp(torch.round(w / scale) + zp, qmin, qmax)
    return (codes - zp) * scale


def gptq_solve(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    q: WeightQuant,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return ``(dequant_weight (N,K), scale (N,G), zero_point (N,G)|None)``."""
    W = weight.detach().float().clone()
    N, K = W.shape
    H = hessian.detach().float().clone()
    dev = W.device
    H = H.to(dev)

    group_size = q.group_size if q.group_size and q.group_size > 0 else K
    ngroups = K // group_size

    # note(chenghua): dead columns (never activated) carry no information, so pin them.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # note(chenghua): damping keeps the inverse well-conditioned; retry heavier if needed.
    diag = torch.arange(K, device=dev)
    base = percdamp * torch.mean(torch.diag(H)).clamp_min(1e-8)
    Hinv = None
    for mult in (1.0, 10.0, 100.0, 1000.0):
        Hd = H.clone()
        Hd[diag, diag] += base * mult
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except RuntimeError:
            continue
    if Hinv is None:
        raise RuntimeError("GPTQ: Hessian not positive-definite even after damping")

    Q = torch.zeros_like(W)
    scale = torch.zeros(N, ngroups, device=dev)
    zero_point = (
        None if q.symmetric or q.is_float else torch.zeros(N, ngroups, device=dev)
    )
    _group_cache: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}

    def group_params(col: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        # note(chenghua): freeze each group's (scale, zp) on first entry so every column
        # of a group shares one scale even when the group straddles a block boundary;
        # recomputing per column from the error-compensated W would desync the stored
        # scale from earlier columns' codes.
        g = col // group_size
        if g not in _group_cache:
            gs, ge = g * group_size, (g + 1) * group_size
            _group_cache[g] = compute_scale_zp(W[:, gs:ge], q)
        s, z = _group_cache[g]
        return s, z, g

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            w_col = W1[:, i : i + 1]
            d = Hinv1[i, i]
            s, z, g = group_params(col)
            scale[:, g : g + 1] = s
            if zero_point is not None:
                assert z is not None
                zero_point[:, g : g + 1] = z
            q_col = _quantize_columns(w_col, s, z, q)
            Q1[:, i : i + 1] = q_col
            err = (w_col - q_col) / d
            W1[:, i:] -= err * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i : i + 1] = err

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return Q.to(weight.dtype), scale, zero_point


class GPTQModifier(Modifier):
    def __init__(
        self,
        weight_dtype: QuantDType | str = QuantDType.INT4,
        *,
        symmetric: bool = True,
        group_size: int = 128,
        blocksize: int = 128,
        percdamp: float = 0.01,
        hessian_device: str | None = None,
        activation_dtype: QuantDType | str | None = None,
        targets: list[str] | None = None,
        ignore: list[str] | None = None,
    ) -> None:
        # TODO(chenghua): Add numerical and end-to-end calibration tests before enabling GPTQ.
        raise NotImplementedError(
            "GPTQModifier is not available because its implementation has not been tested"
        )

        dtype = QuantDType(weight_dtype)
        if not dtype.is_integer:
            raise ValueError(
                f"GPTQ only supports integer weight_dtype, got {dtype.value}"
            )
        resolved_activation = (
            None if activation_dtype is None else QuantDType(activation_dtype)
        )
        if resolved_activation is not None and (
            not resolved_activation.is_integer
            or not resolved_activation.supports_activation
        ):
            raise ValueError(
                "GPTQ only supports INT4/INT8 activation_dtype, got "
                f"{resolved_activation.value}"
            )
        super().__init__(
            targets=targets,
            ignore=ignore,
            activation_dtype=resolved_activation,
        )
        # note(chenghua): GPTQ is integer weight-only (activation, if any, is int).
        self._q = WeightQuant(
            dtype=dtype,
            symmetric=symmetric,
            group_size=group_size,
        )
        self.blocksize = blocksize
        self.percdamp = percdamp
        # note(chenghua): None accumulates the Hessian on the module's own device (no
        # per-forward host transfer on GPU); set "cpu" only if device memory is tight.
        self.hessian_device = hessian_device

    def weight_quant(self) -> WeightQuant:
        return self._q

    @property
    def requires_calibration(self) -> bool:
        return True

    def make_observer(self, module: nn.Module) -> Observer:
        device = self.hessian_device or module.weight.device
        return HessianObserver(in_features=int(module.weight.shape[1]), device=device)

    def quantize_layer(
        self, module: nn.Module, observer: Observer | None
    ) -> QuantResult:
        if not isinstance(observer, HessianObserver):
            raise ValueError("GPTQModifier.quantize_layer requires a HessianObserver")
        if observer.nsamples == 0:
            raise RuntimeError(
                "GPTQ observer saw no calibration tokens for this layer — check the "
                "calibration forward reached it (eager, use_cuda_graph=False)"
            )
        w = module.weight.data
        fake, scale, zp = gptq_solve(
            w,
            observer.hessian(),
            self._q,
            blocksize=self.blocksize,
            percdamp=self.percdamp,
        )
        codes = quantize_to_codes(fake, self._q, scale, zp)
        module.weight.data.copy_(fake)
        result = QuantResult(
            q=self._q, scale=scale, zero_point=zp, fake_weight=fake, int_codes=codes
        )
        setattr(module, "_ptq_result", result)
        return result


__all__ = ["GPTQModifier", "gptq_solve"]
