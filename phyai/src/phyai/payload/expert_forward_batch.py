"""Action-expert (suffix denoise) runner forward batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExpertForwardBatch:
    """Per-step inputs for one denoise step through an action expert.

    Only ``x_t`` and ``time_emb`` change across the
    ``num_inference_steps`` denoise steps of one inference; everything
    else the joint attention needs — the :class:`KVCachePool`, the
    planned flashinfer wrapper, the suffix position ids,
    ``cu_seqlens_q``, the ``paged_kv_*`` metadata buffers, and the
    constant suffix write indices — is owned by the action-expert
    runner. The runner captures the constants in ``setup()`` and
    refreshes per-inference metadata in ``plan_inference()``. The
    captured graph reads runner buffers directly, so any duplicate on
    the batch would be silently ignored.

    Fields
    ------
    x_t:
        ``(B, chunk_size, max_action_dim)`` denoise state at this step
        (in the engine's ``params_dtype``).
    time_emb:
        ``(B, expert_hidden)`` per-step conditioning vector (in
        ``params_dtype``) — already through whatever time MLP the
        consuming model uses to expand its scalar timestep into a
        hidden-width feature. The runner broadcasts this to
        ``(B * chunk_size, expert_hidden)`` and threads it through
        every layer's :class:`AdaRMSNorm`.
    """

    x_t: torch.Tensor
    time_emb: torch.Tensor


__all__ = ["ExpertForwardBatch"]
