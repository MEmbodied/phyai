"""phyai model adapter for calibration — build models eager, enumerate blocks,
and drive sequential replay for the two-tower pi0.5 diffusion model.

Everything lazy-imports phyai so the toolkit stays usable without phyai installed.
Calibration has hard requirements the guards below enforce:

* Eager only — CUDA graphs bypass Python forward hooks, so observers would capture
  nothing (build with ``use_cuda_graph=False``).
* bf16 baseline — build under ``use_quant_plan(None)`` so Linears carry plain bf16
  weights that quantization can read and overwrite in place.

The expert tower is a diffusion denoiser whose activations vary by timestep, so its
calibration units are ``(sample, timestep)`` pairs and it must be quantized after
paligemma (it cross-attends the quantized paligemma KV).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def assert_calibration_ready(model: nn.Module) -> None:
    """Best-effort guardrails before hook-based calibration."""
    try:
        from phyai.engine_config import get_engine_config

        cfg = get_engine_config()
        if getattr(getattr(cfg, "runtime", None), "use_cuda_graph", False):
            raise RuntimeError(
                "calibration must run eager: set RuntimeConfig.use_cuda_graph=False "
                "(CUDA graphs bypass forward hooks -> observers capture nothing)"
            )
    except ImportError:
        pass  # note(chenghua): non-phyai model (e.g. a test stub) — nothing to check


def enumerate_target_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Ordered decoder blocks for pi0.5: paligemma_lm.layers then expert_stack.layers.

    paligemma first so the expert (which cross-attends the paligemma KV) is
    calibrated against already-quantized context.
    """
    blocks: list[tuple[str, nn.Module]] = []
    for stack_attr in ("paligemma_lm", "expert_stack"):
        stack = getattr(model, stack_attr, None)
        layers = getattr(stack, "layers", None) if stack is not None else None
        if layers is None:
            continue
        for i, layer in enumerate(layers):
            blocks.append((f"{stack_attr}.layers.{i}", layer))
    return blocks


class Pi05SequentialDriver:
    """SequentialDriver for pi0.5. Encodes tower order + diffusion timestep sweep.

    ``num_timesteps`` controls how many denoising timesteps each calibration
    sample is expanded into for the expert tower (weight scales are timestep-
    independent, but the GPTQ Hessian must cover the timestep distribution).
    """

    def __init__(self, model: nn.Module, *, num_timesteps: int = 4) -> None:
        assert_calibration_ready(model)
        self.model = model
        self.num_timesteps = num_timesteps
        self._blocks = enumerate_target_blocks(model)
        if not self._blocks:
            raise RuntimeError(
                "no paligemma_lm.layers / expert_stack.layers found; is this a pi0.5 model?"
            )

    def blocks(self) -> list[tuple[str, nn.Module]]:
        return self._blocks

    def seed(self, dataloader: Iterable) -> list[tuple[tuple, dict]]:
        raise NotImplementedError(
            "Pi05SequentialDriver.seed: run vision+embed per sample to the first "
            "paligemma block and capture (h, position_ids, rope, attn_ctx); expand "
            "into (sample, timestep) units at the expert-tower boundary. Complete "
            "against a live pi0.5 model on GPU (eager)."
        )

    def replay(self, block: nn.Module, args: tuple, kwargs: dict):
        # note(chenghua): block signature is model-specific — paligemma (h, position_ids,
        # rope, attn_ctx) vs expert (h, position_ids, cond, rope, attn_ctx, ...).
        return block(*args, **kwargs)

    def advance(self, args: tuple, kwargs: dict, output) -> tuple[tuple, dict]:
        # note(chenghua): substitute the hidden-state arg (h is positional[0]) but keep
        # the rest so cross-attention context (position_ids/rope/attn_ctx) is preserved.
        new_args = (output,) + tuple(args[1:])
        return new_args, kwargs


def build_driver(model: nn.Module, *, num_timesteps: int = 4) -> Pi05SequentialDriver:
    return Pi05SequentialDriver(model, num_timesteps=num_timesteps)


__all__ = [
    "assert_calibration_ready",
    "enumerate_target_blocks",
    "Pi05SequentialDriver",
    "build_driver",
]
