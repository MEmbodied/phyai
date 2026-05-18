"""pi0.5 model runners: vision, LLM backbone, action expert.

Three runners decompose the pi0.5 inference path into independently
captureable units. Each runner takes the sub-modules it needs as
constructor arguments — there is no dependency on :class:`PI05Model`
at this layer, so a runner can be reused for any composition that
exposes the same parts.

* :class:`PI05VisionRunner` wraps :class:`PI05VisionTower` at fixed
  shape ``(3, 3, H, W)`` (three cameras per call) and produces image
  embeddings ``(3, num_patches, projection_dim)``.
* :class:`PI05LLMRunner` runs the prefix forward — paligemma's 18
  decoder layers — at fixed shape ``(B * n_per_sample, hidden_size)``,
  writing per-layer K/V into a :class:`KVCachePool`.
* :class:`PI05ExpertRunner` runs one Euler denoise step (action
  embedding + 18 expert layers + action projection) at fixed shape
  ``(B, chunk_size, max_action_dim)`` for the input ``x_t`` and
  ``(B,)`` for the timestep scalar. The runner reads cached prefix
  K/V and writes suffix K/V into the same cache pool.

Each runner builds a :class:`StaticCachedAttnCtx` once per forward
and threads it through ``stack(h, position_ids, [cond,] rope, ctx)``
— each layer's ``StaticCachedAttention`` (bound to its ``layer_idx``)
reads ``ctx.kv_pool`` / ``ctx.write_indices`` / ``ctx.attn_wrapper``
from the same ctx. The runner reads attention metadata
(``num_heads`` / ``num_kv_heads`` / ``head_dim`` / ``backend``) from
the first layer's attention module — every layer's metadata is
identical, the layer-local instance just disambiguates ``layer_id``.

The two attention backends a runner supports:

* ``"flashinfer-paged"`` — production path. The runner constructs a
  :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` with
  ``use_cuda_graph=True`` and pre-allocated ``paged_kv_*_buf`` /
  ``qo_indptr_buf`` tensors. ``plan()`` is called once per inference
  to copy new metadata into those buffers; the captured graph's
  ``run()`` reads through them automatically.
* ``"sdpa"`` — CPU / CI fallback. The captureable graph is skipped
  (the sdpa path includes a Python segment loop with ``.tolist()``,
  which breaks capture); the runner falls back to an eager forward
  call.

The pi0.5 block-prefix-LM mask (image+lang see image+lang only;
action sees image+lang+action) is realised by the two-runner split:
the LLM runner runs paligemma in isolation, then the expert runner
runs joint attention against the cached prefix K/V. Both runners
share a single :class:`RotaryEmbedding` (the joint attention space
requires it), so the constructor takes it as a direct argument.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    StaticCachedAttention,
    StaticCachedAttnCtx,
    get_global_fi_workspace,
)
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.models.pi05.modeling_pi05 import (
    ActionTimeHeads,
    PaliGemmaLanguageModel,
    PI05ExpertStack,
    PI05VisionTower,
)
from phyai.payload import (
    ExpertForwardBatch,
    LLMForwardBatch,
    VisionForwardBatch,
)
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import all_ranks_log


logger = logging.getLogger(__name__)


def _attn_metadata(stack_layers) -> StaticCachedAttention:
    """Return the first layer's :class:`StaticCachedAttention` instance.

    Used by the runners to read ``num_heads`` / ``num_kv_heads`` /
    ``head_dim`` / ``backend`` for ``wrapper.plan()`` and graph-mode
    selection. Every layer in a pi0.5 stack has the same attention
    config; only ``layer_id`` differs.
    """
    if len(stack_layers) == 0:
        raise ValueError("stack has no layers; cannot read attention metadata.")
    return stack_layers[0].attn


# ============================================================================ #
# Vision runner                                                                #
# ============================================================================ #


class PI05VisionRunner(ModelRunner):
    """SigLIP vision-tower runner with optional CUDA-graph capture.

    pi0.5 uses three cameras per inference; the runner is captured at
    fixed shape ``(3, 3, image_size, image_size)`` and replayed once per
    robot in the scheduler's batch (``B`` times when ``B > 1``).
    """

    def __init__(
        self,
        vision_tower: PI05VisionTower,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
    ) -> None:
        self.vision_tower = vision_tower
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.image_size = int(vision_tower.config.image_size)
        self.num_channels = int(vision_tower.config.num_channels)
        self.graph: CudaGraph | None = None

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05VisionRunner.setup")
        if not self.use_cuda_graph or self.device.type != "cuda":
            # TODO: raise error exception
            return
        all_ranks_log(
            logger,
            logging.INFO,
            "Entering PI05VisionRunner.setup: capturing vision-tower CUDA graph "
            "at fixed shape (3, %d, %d, %d).",
            self.num_channels,
            self.image_size,
            self.image_size,
        )
        example = {
            "pixel_values": torch.zeros(
                3,
                self.num_channels,
                self.image_size,
                self.image_size,
                dtype=self.params_dtype,
                device=self.device,
            ),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd, example)

    def _fwd(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_tower(pixel_values)

    def forward(self, batch: VisionForwardBatch) -> torch.Tensor:
        if self.graph is not None:
            return self.graph.replay({"pixel_values": batch.pixel_values})
        return self.vision_tower(batch.pixel_values)


# ============================================================================ #
# LLM backbone runner (prefix phase)                                           #
# ============================================================================ #


class PI05LLMRunner(ModelRunner):
    """PaliGemma backbone runner — runs paligemma's 18 layers over the
    per-sample-padded prefix and writes K/V to ``kv_pool``.

    Captured at fixed shape ``(B * n_per_sample, hidden_size)``. The
    prefix-self-attention :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper`
    is owned by the runner; its ``plan()`` is called once per inference
    via :meth:`plan_inference` to update the pre-allocated metadata
    buffers.

    Returns ``None`` from :meth:`forward` — the cache pool side-effect
    is the only output the scheduler consumes.
    """

    def __init__(
        self,
        paligemma_lm: PaliGemmaLanguageModel,
        rope: RotaryEmbedding,
        kv_pool: KVCachePool,
        *,
        batch_size: int,
        n_per_sample: int,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
        max_paged_kv_indices: int | None = None,
    ) -> None:
        self.paligemma_lm = paligemma_lm
        self.rope = rope
        self.kv_pool = kv_pool
        self.batch_size = int(batch_size)
        self.n_per_sample = int(n_per_sample)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        # Read attention metadata from the first layer; every layer's
        # config is identical, only layer_id differs.
        attn_proto = _attn_metadata(paligemma_lm.layers)
        self.num_heads = attn_proto.num_heads
        self.num_kv_heads = attn_proto.num_kv_heads
        self.head_dim = attn_proto.head_dim
        self.use_paged = attn_proto.backend == "flashinfer-paged"
        # cuda graph requires paged backend; sdpa fallback runs eager.
        self.use_cuda_graph = (
            bool(use_cuda_graph) and self.use_paged and self.device.type == "cuda"
        )
        self.max_paged_kv_indices = int(
            max_paged_kv_indices
            if max_paged_kv_indices is not None
            else self.batch_size * self.n_per_sample
        )
        self.hidden_size = int(paligemma_lm.config.hidden_size)

        # Static metadata buffers — runner-owned, refilled per inference.
        # On the paged path these double as the wrapper's own buffers.
        self.cu_q_buf = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indptr_buf = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indices_buf = torch.zeros(
            self.max_paged_kv_indices, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len_buf = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )

        self.wrapper: Any = None
        self.graph: CudaGraph | None = None

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05LLMRunner.setup")
        if self.use_paged:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05LLMRunner.setup: build flashinfer wrapper plan",
            )
            self._build_wrapper()
        if self.use_cuda_graph:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05LLMRunner.setup: building flashinfer paged-prefill "
                "wrapper and capturing prefix-forward CUDA graph at fixed shape "
                "(B*n_per_sample=%d, hidden_size=%d).",
                self.batch_size * self.n_per_sample,
                self.hidden_size,
            )
            self._capture_graph()

    def _build_wrapper(self) -> None:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        workspace = get_global_fi_workspace(self.device)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend="auto",
            use_cuda_graph=True,
            qo_indptr_buf=self.cu_q_buf,
            paged_kv_indptr_buf=self.paged_kv_indptr_buf,
            paged_kv_indices_buf=self.paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len_buf,
        )
        # Seed plan with values that match the buffer shapes so the
        # capture-time warmup has something valid to read.
        self._initial_plan()

    def _initial_plan(self) -> None:
        # Per-sample padded q layout — fixed across all inferences.
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * self.n_per_sample,
            self.n_per_sample,
            dtype=torch.int32,
            device=self.device,
        )
        # Use a plausible "all real" KV layout for the warmup plan: each
        # sample contributes ``min(n_per_sample, max_indices/B)`` real
        # tokens. Real values arrive at the first ``forward`` call.
        per_sample_real = min(
            self.n_per_sample, self.max_paged_kv_indices // self.batch_size
        )
        kv_indptr = torch.arange(
            0,
            (self.batch_size + 1) * per_sample_real,
            per_sample_real,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = torch.arange(
            self.batch_size * per_sample_real,
            dtype=torch.int32,
            device=self.device,
        )
        last_page = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        self.wrapper.plan(
            cu_q,
            kv_indptr,
            kv_indices,
            last_page,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            q_data_type=self.params_dtype,
            kv_data_type=self.params_dtype,
        )

    def _capture_graph(self) -> None:
        n = self.batch_size * self.n_per_sample
        example = {
            "hidden_states": torch.zeros(
                n, self.hidden_size, dtype=self.params_dtype, device=self.device
            ),
            "position_ids": torch.zeros(n, dtype=torch.int32, device=self.device),
            "write_indices": torch.zeros(n, dtype=torch.int64, device=self.device),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd, example)

    def _fwd(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        write_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run paligemma's 18 layers, writing K/V into ``self.kv_pool``.

        This is the prefix-phase forward. The block-prefix-LM mask is
        realised by simply having no action tokens here — the per-sample
        attention range covered by ``cu_seqlens_q`` and
        ``paged_kv_indptr`` is image+lang only.

        Returns the final hidden state ``(N_max, hidden_size)`` after
        the trailing norm. The scheduler discards it; the cache pool
        side-effect is what the next phase consumes.
        """
        ctx = StaticCachedAttnCtx(
            kv_pool=self.kv_pool,
            write_indices=write_indices,
            attn_wrapper=self.wrapper,
            cu_seqlens_q=self.cu_q_buf,
            paged_kv_indptr=self.paged_kv_indptr_buf,
            paged_kv_indices=self.paged_kv_indices_buf,
        )
        return self.paligemma_lm(hidden_states, position_ids, self.rope, ctx)

    # ------------------------------------------------------------------ #
    # Plan / forward                                                     #
    # ------------------------------------------------------------------ #

    def plan_inference(
        self,
        *,
        cu_seqlens_q: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        """Update the static metadata buffers for the next ``forward`` call.

        On the paged path the update goes through ``wrapper.plan()`` so
        flashinfer can refresh its internal split-k metadata; the sdpa
        fallback path simply ``copy_()``-s into the runner's buffers
        directly.
        """
        if self.use_paged:
            self.wrapper.plan(
                cu_seqlens_q.to(torch.int32),
                paged_kv_indptr.to(torch.int32),
                paged_kv_indices.to(torch.int32),
                paged_kv_last_page_len.to(torch.int32),
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=1,
                causal=False,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                q_data_type=self.params_dtype,
                kv_data_type=self.params_dtype,
            )
        else:
            self.cu_q_buf.copy_(cu_seqlens_q.to(torch.int32))
            self.paged_kv_indptr_buf.copy_(paged_kv_indptr.to(torch.int32))
            indices_i32 = paged_kv_indices.to(torch.int32)
            n = indices_i32.numel()
            self.paged_kv_indices_buf[:n].copy_(indices_i32)
            self.paged_kv_last_page_len_buf.copy_(
                paged_kv_last_page_len.to(torch.int32)
            )

    def forward(self, batch: LLMForwardBatch) -> None:
        """Run the captured (or eager) prefix forward over ``batch``.

        The runner owns the cache pool and the planned attention
        wrapper; the batch only carries the per-call variable inputs.
        """
        if self.graph is not None:
            self.graph.replay(
                {
                    "hidden_states": batch.hidden_states,
                    "position_ids": batch.position_ids,
                    "write_indices": batch.write_indices,
                }
            )
            return None
        # Eager fallback (sdpa or non-cuda-graph mode).
        self._fwd(
            hidden_states=batch.hidden_states,
            position_ids=batch.position_ids,
            write_indices=batch.write_indices,
        )
        return None


# ============================================================================ #
# Action expert runner (one Euler step)                                        #
# ============================================================================ #


class PI05ExpertRunner(ModelRunner):
    """One Euler denoise step: ``embed_action → 18 expert layers → project_action``.

    Captured at fixed shape ``(B, chunk_size, max_action_dim)`` for
    ``x_t`` and ``(B, expert_hidden)`` for the precomputed
    ``time_emb`` (already through the full time MLP — the scheduler
    builds a per-step lookup table once at :meth:`setup` and copies the
    right row in per Euler step). Within one inference all
    Euler steps share the same cache layout — :meth:`plan_inference`
    refreshes the wrapper buffers once and :meth:`forward` is replayed
    ``num_inference_steps`` times.

    The runner takes the expert stack and the action / time projection
    heads as constructor args; the shared :class:`RotaryEmbedding`
    is passed in by the caller (the scheduler) since it is also used
    by :class:`PI05LLMRunner` during the prefix phase.
    """

    def __init__(
        self,
        expert_stack: PI05ExpertStack,
        heads: ActionTimeHeads,
        rope: RotaryEmbedding,
        kv_pool: KVCachePool,
        *,
        batch_size: int,
        chunk_size: int,
        max_action_dim: int,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
        max_paged_kv_indices: int | None = None,
    ) -> None:
        self.expert_stack = expert_stack
        self.heads = heads
        self.rope = rope
        self.kv_pool = kv_pool
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.max_action_dim = int(max_action_dim)
        self.expert_hidden = int(heads.expert_hidden)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        attn_proto = _attn_metadata(expert_stack.layers)
        self.num_heads = attn_proto.num_heads
        self.num_kv_heads = attn_proto.num_kv_heads
        self.head_dim = attn_proto.head_dim
        self.use_paged = attn_proto.backend == "flashinfer-paged"
        self.use_cuda_graph = (
            bool(use_cuda_graph) and self.use_paged and self.device.type == "cuda"
        )
        self.max_paged_kv_indices = int(
            max_paged_kv_indices
            if max_paged_kv_indices is not None
            else self.batch_size * self.chunk_size * 32
        )

        # Static metadata buffers — same dual-purpose pattern as the LLM runner.
        self.cu_q_buf = torch.arange(
            0,
            (self.batch_size + 1) * self.chunk_size,
            self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.paged_kv_indptr_buf = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indices_buf = torch.zeros(
            self.max_paged_kv_indices, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len_buf = torch.ones(
            self.batch_size, dtype=torch.int32, device=self.device
        )

        # pos_ids_suffix is per-inference (depends on real_lens) but
        # not per-Euler-step. Static buffer; runner refreshes once per
        # inference.
        self.pos_ids_suffix_buf = torch.zeros(
            self.batch_size * self.chunk_size, dtype=torch.int32, device=self.device
        )
        # write_indices_suffix is constant across inferences (the suffix
        # slab base never moves once the scheduler is set up). The
        # scheduler hands it to the runner via :meth:`set_write_indices`
        # at startup.
        self.write_indices_suffix_buf = torch.zeros(
            self.batch_size * self.chunk_size, dtype=torch.int64, device=self.device
        )

        self.wrapper: Any = None
        self.graph: CudaGraph | None = None

    def set_write_indices(self, write_indices_suffix: torch.Tensor) -> None:
        """Bind the suffix-slab slot indices once at scheduler setup."""
        if write_indices_suffix.shape != self.write_indices_suffix_buf.shape:
            raise ValueError(
                f"write_indices_suffix shape {tuple(write_indices_suffix.shape)} "
                f"!= {tuple(self.write_indices_suffix_buf.shape)}."
            )
        self.write_indices_suffix_buf.copy_(write_indices_suffix.to(torch.int64))

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05ExpertRunner.setup")
        if self.use_paged:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05ExpertRunner.setup: build flashinfer wrapper plan",
            )
            self._build_wrapper()
        if self.use_cuda_graph:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05ExpertRunner.setup: building flashinfer paged-prefill "
                "wrapper and capturing expert-forward CUDA graph at fixed shape "
                "(B=%d, chunk_size=%d, max_action_dim=%d).",
                self.batch_size,
                self.chunk_size,
                self.max_action_dim,
            )
            self._capture_graph()

    def _build_wrapper(self) -> None:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        workspace = get_global_fi_workspace(self.device)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend="auto",
            use_cuda_graph=True,
            qo_indptr_buf=self.cu_q_buf,
            paged_kv_indptr_buf=self.paged_kv_indptr_buf,
            paged_kv_indices_buf=self.paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len_buf,
        )
        self._initial_plan()

    def _initial_plan(self) -> None:
        # cu_q is fixed [0, chunk, 2*chunk, ...]; the buffer already holds it.
        # Seed kv layout with a "small" plausible joint length so plan() succeeds.
        per_sample_kv = min(
            self.chunk_size * 4, self.max_paged_kv_indices // self.batch_size
        )
        kv_indptr = torch.arange(
            0,
            (self.batch_size + 1) * per_sample_kv,
            per_sample_kv,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = torch.arange(
            self.batch_size * per_sample_kv,
            dtype=torch.int32,
            device=self.device,
        )
        last_page = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        self.wrapper.plan(
            self.cu_q_buf,
            kv_indptr,
            kv_indices,
            last_page,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            q_data_type=self.params_dtype,
            kv_data_type=self.params_dtype,
        )

    def _capture_graph(self) -> None:
        example = {
            "x_t": torch.zeros(
                self.batch_size,
                self.chunk_size,
                self.max_action_dim,
                dtype=self.params_dtype,
                device=self.device,
            ),
            "time_emb": torch.zeros(
                self.batch_size,
                self.expert_hidden,
                dtype=self.params_dtype,
                device=self.device,
            ),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd, example)

    def _fwd(
        self,
        *,
        x_t: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler denoise step.

        Pipeline (all captureable)::

            x_t      → action_in_proj → reshape  ─┐
                                                  ├→ expert 18 layers → action_out_proj → v_t
            time_emb → repeat_interleave (cond) ──┘

        ``time_emb`` arrives already through the full time MLP — the
        scheduler precomputes a ``(num_inference_steps, expert_hidden)``
        lookup table at :meth:`setup` time and copies the row for the
        current step into a static ``(B, expert_hidden)`` buffer. This
        keeps the sinusoidal embed and the two time-MLP matmuls out of
        the captured graph entirely; their values are the same across
        inferences (only the schedule index changes), so the precompute
        is free in steady state.

        The expert layers run joint attention against
        ``[cached prefix K/V, fresh suffix K/V]``. Cached prefix lives
        at the prefix-slab slots in ``self.kv_pool``; the runner's
        ``write_indices_suffix_buf`` aims fresh writes at the suffix
        slab. ``paged_kv_indices`` reads them in interleaved per-sample
        order.
        """
        # action_in_proj: (B, chunk, action_dim) → (B, chunk, expert_hidden)
        action_emb = self.heads.embed_action(x_t)
        suffix_h = action_emb.reshape(self.batch_size * self.chunk_size, -1)

        # Broadcast the per-sample time embedding across this sample's
        # chunk_size action tokens.
        cond_per_token = time_emb.repeat_interleave(self.chunk_size, dim=0)

        ctx = StaticCachedAttnCtx(
            kv_pool=self.kv_pool,
            write_indices=self.write_indices_suffix_buf,
            attn_wrapper=self.wrapper,
            cu_seqlens_q=self.cu_q_buf,
            paged_kv_indptr=self.paged_kv_indptr_buf,
            paged_kv_indices=self.paged_kv_indices_buf,
        )
        suffix_out = self.expert_stack(
            suffix_h,
            self.pos_ids_suffix_buf,
            cond_per_token,
            self.rope,
            ctx,
        )
        suffix_out_3d = suffix_out.view(self.batch_size, self.chunk_size, -1)
        return self.heads.project_action(suffix_out_3d)

    # ------------------------------------------------------------------ #
    # Plan / forward                                                     #
    # ------------------------------------------------------------------ #

    def plan_inference(
        self,
        *,
        pos_ids_suffix: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        """Refresh the joint-attention metadata for one inference (10 Euler steps).

        Called once per inference (before the first :meth:`forward`).
        Updates ``pos_ids_suffix`` and the wrapper buffers; ``cu_q`` and
        ``write_indices_suffix`` are constants and do not move.
        """
        self.pos_ids_suffix_buf.copy_(pos_ids_suffix.to(torch.int32))
        if self.use_paged:
            self.wrapper.plan(
                self.cu_q_buf,
                paged_kv_indptr.to(torch.int32),
                paged_kv_indices.to(torch.int32),
                paged_kv_last_page_len.to(torch.int32),
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=1,
                causal=False,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                q_data_type=self.params_dtype,
                kv_data_type=self.params_dtype,
            )
        else:
            self.paged_kv_indptr_buf.copy_(paged_kv_indptr.to(torch.int32))
            indices_i32 = paged_kv_indices.to(torch.int32)
            n = indices_i32.numel()
            self.paged_kv_indices_buf[:n].copy_(indices_i32)
            self.paged_kv_last_page_len_buf.copy_(
                paged_kv_last_page_len.to(torch.int32)
            )

    def forward(self, batch: ExpertForwardBatch) -> torch.Tensor:
        """Run one Euler step. Returns the velocity ``v_t`` ``(B, chunk, action_dim)``.

        Only ``batch.x_t`` and ``batch.time_emb`` are consumed per step;
        all attention metadata must already be staged on the runner via
        :meth:`plan_inference` (and :meth:`set_write_indices` at setup).
        """
        if self.graph is not None:
            return self.graph.replay({"x_t": batch.x_t, "time_emb": batch.time_emb})
        return self._fwd(x_t=batch.x_t, time_emb=batch.time_emb)


__all__ = [
    "PI05ExpertRunner",
    "PI05LLMRunner",
    "PI05VisionRunner",
]
