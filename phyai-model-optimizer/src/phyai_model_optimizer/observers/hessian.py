"""HessianObserver

Accumulates the GPTQ input Hessian ``H = sum_t x_t x_t^T``.

GPTQ needs the input second-moment ``H = X^T X`` (``K x K``) over all calibration
tokens (and, for the diffusion expert, all sampled timesteps — the caller keeps
calling ``observe``). The running-mean form below is numerically stable across
many batches.
"""

from __future__ import annotations

import torch

from phyai_model_optimizer.observers.base import Observer


class HessianObserver(Observer):
    def __init__(self, in_features: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.H = torch.zeros(
            self.in_features, self.in_features, dtype=torch.float32, device=device
        )

    def observe(self, x: torch.Tensor) -> None:
        # note(chenghua): accept the Linear's raw (possibly batched) input by flattening to (ntokens, K).
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"HessianObserver expected K={self.in_features}, got {x.shape[-1]}"
            )
        n = x.shape[0]
        if n == 0:
            return
        # note(chenghua): running mean of x x^T, matching reference GPTQ's scaling
        # (the factor cancels in the OBQ solve).
        total = self.nsamples + n
        self.H.mul_(self.nsamples / total)
        x = x.to(self.H.device)
        self.H.add_((2.0 / total) * (x.t() @ x))
        self.nsamples = total

    def hessian(self) -> torch.Tensor:
        return self.H

    def reset(self) -> None:
        super().reset()
        self.H.zero_()


__all__ = ["HessianObserver"]
