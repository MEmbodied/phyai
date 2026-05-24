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

Backend ownership
-----------------
Paged-attention backends are constructed once per runner via the
per-stack registry factory ``factory(runner=self)`` — the backend
reads ``runner.batch_size`` / ``runner.max_paged_kv_indices`` / etc.
and allocates its own static buffers in
:meth:`init_cuda_graph_state`. Layers do NOT own the backend; the
runner threads it through every layer's :meth:`forward` via the
flavor-specific ctx (:class:`ARAttnCtx` for paligemma,
:class:`DiffusionAttnCtx` for the expert).

The two attention backends a runner supports:

* ``"flashinfer"`` — production GPU path. The backend builds a
  paged-prefill wrapper with ``use_cuda_graph=True`` and
  pre-allocated index buffers; :meth:`replay_metadata` re-plans them
  in place per inference so the captured ``run`` reads through the
  updated values.
* ``"eager"`` — CPU / CI fallback. The contiguous-slab matmul path;
  not graph-captureable; the runner falls back to per-step
  :meth:`init_forward_metadata`.

The pi0.5 block-prefix-LM mask (image+lang see image+lang only;
action sees image+lang+action) is realised by the two-runner split:
the LLM runner runs paligemma in isolation, then the expert runner
runs joint attention against the cached prefix K/V. Both runners
share a single :class:`RotaryEmbedding` (the joint attention space
requires it), so the constructor takes it as a direct argument.
"""

from __future__ import annotations

import logging

import torch

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    ARAttention,
    ARAttentionBackend,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
    AttnLayout,
    AttnMode,
    DiffusionAttention,
    DiffusionAttentionBackend,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
    get_ar_backend_factory,
    get_diffusion_backend_factory,
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


def _ar_attn_proto(stack_layers) -> ARAttention:
    """Return the first layer's :class:`ARAttention` instance.

    Used by the LLM runner to read ``num_heads`` / ``num_kv_heads`` /
    ``head_dim`` / ``backend`` for the backend factory and capture-shape
    seed. Every layer in a pi0.5 paligemma stack has the same attention
    config; only ``layer_id`` differs.
    """
    if len(stack_layers) == 0:
        raise ValueError("stack has no layers; cannot read attention metadata.")
    return stack_layers[0].attn


def _diffusion_attn_proto(stack_layers) -> DiffusionAttention:
    """Return the first layer's :class:`DiffusionAttention` instance.

    Used by the expert runner; same role as :func:`_ar_attn_proto` but
    typed for the diffusion / action-expert stack.
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

    Captured at fixed shape ``(B * n_per_sample, hidden_size)``. Owns
    a single :class:`ARAttentionBackend` instance built via
    :func:`get_ar_backend_factory`; the backend allocates its own static
    buffers in :meth:`init_cuda_graph_state` and
    re-plans them in place per inference via
    :meth:`replay_metadata` (graph) or
    :meth:`init_forward_metadata` (eager fallback).

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
        self.attn_proto: ARAttention = _ar_attn_proto(paligemma_lm.layers)
        self.num_heads = self.attn_proto.num_heads
        self.num_kv_heads = self.attn_proto.num_kv_heads
        self.head_dim = self.attn_proto.head_dim
        self.hidden_size = int(paligemma_lm.config.hidden_size)
        self.max_paged_kv_indices = int(
            max_paged_kv_indices
            if max_paged_kv_indices is not None
            else self.batch_size * self.n_per_sample
        )

        # Build the runner-scoped backend instance. Backend reads runner
        # state for buffer sizing in init_cuda_graph_state.
        factory = get_ar_backend_factory(self.attn_proto.backend)
        self.attn_backend: ARAttentionBackend = factory(self)
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.attn_backend.supports_capture()
            and self.device.type == "cuda"
        )

        self._capture_plan: ARAttnPlanHandle | None = None
        self.graph: CudaGraph | None = None

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05LLMRunner.setup")
        # Always allocate static buffers + build wrapper — graph mode
        # needs them, and eager mode benefits from stable addresses.
        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size * self.n_per_sample,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        # Capture-time seed plan + graph capture.
        if self.use_cuda_graph:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05LLMRunner.setup: building capture-warmup plan and "
                "capturing prefix-forward CUDA graph at fixed shape "
                "(B*n_per_sample=%d, hidden_size=%d).",
                self.batch_size * self.n_per_sample,
                self.hidden_size,
            )
            self._capture_plan = self.attn_backend.init_capture_metadata(
                self._capture_seed_metadata()
            )
            self._capture_graph()

    def _capture_seed_metadata(self) -> ARAttnMetadata:
        # Per-sample padded q layout — fixed across all inferences.
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * self.n_per_sample,
            self.n_per_sample,
            dtype=torch.int32,
            device=self.device,
        )
        # A plausible "all real" KV layout for the warmup plan: each
        # sample contributes ``min(n_per_sample, max_indices/B)`` real
        # tokens. Real values arrive at the first ``plan_inference`` call.
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
        write_indices = torch.zeros(
            self.batch_size * self.n_per_sample,
            dtype=torch.int64,
            device=self.device,
        )
        return ARAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=self.batch_size,
            num_query_tokens=self.batch_size * self.n_per_sample,
            cu_seqlens_q=cu_q,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            write_indices=write_indices,
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

    # ------------------------------------------------------------------ #
    # Forward path                                                       #
    # ------------------------------------------------------------------ #

    def _fwd(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        write_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run paligemma's 18 layers, writing K/V into ``self.kv_pool``."""
        ctx = ARAttnCtx(
            backend=self.attn_backend,
            plan=self._capture_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=write_indices,
        )
        return self.paligemma_lm(hidden_states, position_ids, self.rope, ctx)

    def plan_inference(self, meta: ARAttnMetadata) -> None:
        """Stage attention metadata for the next ``forward`` call.

        Graph mode: re-plan the captured backend buffers in place via
        :meth:`ARAttentionBackend.replay_metadata`. Eager mode: build a
        fresh plan via :meth:`ARAttentionBackend.init_forward_metadata`.
        """
        if self.use_cuda_graph:
            self.attn_backend.replay_metadata(self._capture_plan, meta)
        else:
            self._capture_plan = self.attn_backend.init_forward_metadata(meta)

    def forward(self, batch: LLMForwardBatch) -> None:
        if self.graph is not None:
            self.graph.replay(
                {
                    "hidden_states": batch.hidden_states,
                    "position_ids": batch.position_ids,
                    "write_indices": batch.write_indices,
                }
            )
            return None
        # Eager fallback (eager backend or non-cuda-graph mode).
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
    """One Euler denoise step: ``embed_action -> 18 expert layers -> project_action``.

    Captured at fixed shape ``(B, chunk_size, max_action_dim)`` for
    ``x_t`` and ``(B, expert_hidden)`` for the precomputed
    ``time_emb`` (already through the full time MLP — the scheduler
    builds a per-step lookup table once at :meth:`setup` and copies the
    right row in per Euler step). Within one inference all
    Euler steps share the same cache layout — :meth:`plan_inference`
    refreshes the wrapper buffers once and :meth:`forward` is replayed
    ``num_inference_steps`` times.

    Two runner-owned static tensors on top of the backend's own static
    state:

    * ``pos_ids_suffix_buf`` — refreshed per inference via
      :meth:`plan_inference`.
    * ``write_indices_suffix_buf`` — constant across inferences;
      bound once at scheduler setup via :meth:`set_write_indices`.

    Both are baked into the captured graph by Python identity (read
    off ``self`` inside :meth:`_fwd`), so they stay outside the
    backend's :meth:`replay_metadata` contract.
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
        self.attn_proto: DiffusionAttention = _diffusion_attn_proto(expert_stack.layers)
        self.num_heads = self.attn_proto.num_heads
        self.num_kv_heads = self.attn_proto.num_kv_heads
        self.head_dim = self.attn_proto.head_dim
        self.max_paged_kv_indices = int(
            max_paged_kv_indices
            if max_paged_kv_indices is not None
            else self.batch_size * self.chunk_size * 32
        )

        # pos_ids_suffix is per-inference (depends on real_lens) but
        # not per-Euler-step. Static buffer; runner refreshes once per
        # inference.
        self.pos_ids_suffix_buf = torch.zeros(
            self.batch_size * self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        # write_indices_suffix is constant across inferences (the suffix
        # slab base never moves once the scheduler is set up). The
        # scheduler hands it to the runner via :meth:`set_write_indices`
        # at startup.
        self.write_indices_suffix_buf = torch.zeros(
            self.batch_size * self.chunk_size,
            dtype=torch.int64,
            device=self.device,
        )

        factory = get_diffusion_backend_factory(self.attn_proto.backend)
        self.attn_backend: DiffusionAttentionBackend = factory(self)
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.attn_backend.supports_capture()
            and self.device.type == "cuda"
        )

        self._capture_plan: DiffusionAttnPlanHandle | None = None
        self.graph: CudaGraph | None = None

    def set_write_indices(self, write_indices_suffix: torch.Tensor) -> None:
        """Bind the suffix-slab slot indices once at scheduler setup."""
        if write_indices_suffix.shape != self.write_indices_suffix_buf.shape:
            raise ValueError(
                f"write_indices_suffix shape {tuple(write_indices_suffix.shape)} "
                f"!= {tuple(self.write_indices_suffix_buf.shape)}."
            )
        self.write_indices_suffix_buf.copy_(write_indices_suffix.to(torch.int64))

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05ExpertRunner.setup")
        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size * self.chunk_size,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        if self.use_cuda_graph:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05ExpertRunner.setup: capturing expert-forward CUDA "
                "graph at fixed shape (B=%d, chunk_size=%d, max_action_dim=%d).",
                self.batch_size,
                self.chunk_size,
                self.max_action_dim,
            )
            self._capture_plan = self.attn_backend.init_capture_metadata(
                self._capture_seed_metadata()
            )
            self._capture_graph()

    def _capture_seed_metadata(self) -> DiffusionAttnMetadata:
        # cu_q is fixed [0, chunk, 2*chunk, ...] across all inferences.
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * self.chunk_size,
            self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        # Seed kv layout with a "small" plausible joint length so plan() succeeds.
        per_sample_kv = min(
            self.chunk_size * 4,
            self.max_paged_kv_indices // self.batch_size,
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
        return DiffusionAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=self.batch_size,
            num_query_tokens=self.batch_size * self.chunk_size,
            cu_seqlens_q=cu_q,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            write_indices=self.write_indices_suffix_buf,
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

    # ------------------------------------------------------------------ #
    # Forward path                                                       #
    # ------------------------------------------------------------------ #

    def _fwd(
        self,
        *,
        x_t: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler denoise step (see class docstring)."""
        action_emb = self.heads.embed_action(x_t)
        suffix_h = action_emb.reshape(self.batch_size * self.chunk_size, -1)
        cond_per_token = time_emb.repeat_interleave(self.chunk_size, dim=0)
        ctx = DiffusionAttnCtx(
            backend=self.attn_backend,
            plan=self._capture_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=self.write_indices_suffix_buf,
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

    def plan_inference(self, meta: DiffusionAttnMetadata) -> None:
        """Refresh metadata for one inference (10 Euler steps share it).

        ``pos_ids_suffix`` is updated from ``meta.position_ids``;
        ``cu_q`` and ``write_indices_suffix`` are constants and stay on
        the runner.
        """
        if meta.position_ids is None:
            raise ValueError(
                "PI05ExpertRunner.plan_inference requires meta.position_ids."
            )
        self.pos_ids_suffix_buf.copy_(meta.position_ids.to(torch.int32))
        if self.use_cuda_graph:
            self.attn_backend.replay_metadata(self._capture_plan, meta)
        else:
            self._capture_plan = self.attn_backend.init_forward_metadata(meta)

    def forward(self, batch: ExpertForwardBatch) -> torch.Tensor:
        if self.graph is not None:
            return self.graph.replay({"x_t": batch.x_t, "time_emb": batch.time_emb})
        return self._fwd(x_t=batch.x_t, time_emb=batch.time_emb)


__all__ = [
    "PI05ExpertRunner",
    "PI05LLMRunner",
    "PI05VisionRunner",
]
