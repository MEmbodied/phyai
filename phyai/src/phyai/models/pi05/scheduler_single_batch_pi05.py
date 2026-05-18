"""pi0.5 single-batch end-to-end inference scheduler.

The scheduler is the user-facing entry point for pi0.5 inference. It
owns:

* The :class:`KVCachePool` and the prefix / suffix
  :class:`StaticCache` slabs over it.
* Three runners — :class:`PI05VisionRunner`,
  :class:`PI05LLMRunner`, :class:`PI05ExpertRunner` — each with its
  own captured CUDA graph (when the engine backend supports paged
  attention).

A single ``step()`` call runs the full inference: vision tower per
camera-stack, prefix forward writing K/V into the pool, then a 10-step
Euler loop reading those K/V plus the freshly-computed suffix K/V.

"Single batch" here means one inference at a time with a batch size
(``batch_size``) fixed at scheduler construction. There is no continuous
batching or preemption — that lives in a separate scheduler later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.cache import KVCachePool, StaticCache
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.model_runner_pi05 import (
    PI05ExpertRunner,
    PI05LLMRunner,
    PI05VisionRunner,
)
from phyai.models.pi05.modeling_pi05 import PI05Model
from phyai.models.pi05.batch_layout_pi05 import (
    build_joint_last_page_len,
    build_joint_paged_kv_indices,
    build_prefix_last_page_len,
    build_prefix_padded_pos_ids,
    build_prefix_padded_write_indices,
    build_prefix_paged_kv_indices,
    build_suffix_pos_ids,
    build_suffix_write_indices,
    pack_prefix_per_sample_padded,
)
from phyai.payload import (
    ExpertForwardBatch,
    LLMForwardBatch,
    VisionForwardBatch,
)
from phyai.runtime.schedule import Scheduler


@dataclass
class PI05Request:
    """One pi0.5 inference request.

    ``pixel_values`` carries every robot's three cameras stacked along
    the camera axis: shape ``(B, 3, 3, H, W)`` — ``B`` robots × 3 cameras
    × 3 channels × ``image_size``×``image_size``.

    ``input_ids`` is the padded language-token tensor (``(B,
    tokenizer_max_length)`` int64); ``lang_lens`` carries the
    un-padded real lengths.

    ``noise`` is optional; if ``None`` the scheduler samples a fresh
    Gaussian.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    noise: torch.Tensor | None = None


class PI05SingleBatchScheduler(Scheduler):
    """Single-batch end-to-end pi0.5 inference orchestrator."""

    def __init__(
        self,
        model: PI05Model,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        cfg: PI05Config = model.config
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.params_dtype = model.params_dtype
        self.cfg = cfg
        self.batch_size = int(batch_size)
        if self.batch_size != 1:
            raise ValueError(
                f"batch_size must be 1 in PI05SingleBatchScheduler, got {batch_size}."
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.model = model

        # Layout knobs.
        self.image_token_count = cfg.vision.num_patches * 3  # 3 cameras per robot
        self.n_per_sample = self.image_token_count + cfg.tokenizer_max_length

        # KVCachePool: sentinel slot 0 + prefix slab + suffix slab.
        prefix_capacity = self.batch_size * self.n_per_sample
        suffix_capacity = self.batch_size * cfg.chunk_size
        # +1 slot for the sentinel so padding tokens have somewhere
        # harmless to land.
        self.sentinel_slot = 0
        self.prefix_base = 1
        self.suffix_base = self.prefix_base + prefix_capacity
        total_slots = 1 + prefix_capacity + suffix_capacity

        self.kv_pool = KVCachePool(
            num_layers=cfg.num_layers,
            num_slots=total_slots,
            num_kv_heads=cfg.text.num_key_value_heads,
            head_dim=cfg.text.head_dim,
            dtype=self.params_dtype,
            device=self.device,
        )
        self.prefix_static = StaticCache(
            self.kv_pool, base_offset=self.prefix_base, capacity=prefix_capacity
        )
        self.suffix_static = StaticCache(
            self.kv_pool, base_offset=self.suffix_base, capacity=suffix_capacity
        )

        # Runners (constructed but not yet warmed up; setup() does it).
        # Each runner takes the individual sub-modules it needs — there is
        # no PI05Model dependency at the runner layer.
        self.vision_runner = PI05VisionRunner(
            model.vision,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )
        # max paged-kv-indices for prefix self-attn: B * n_per_sample
        # (worst case all real tokens). For joint attn during the expert
        # phase: B * (n_per_sample + chunk_size).
        self.llm_runner = PI05LLMRunner(
            model.paligemma_lm,
            model.rope,
            self.kv_pool,
            batch_size=self.batch_size,
            n_per_sample=self.n_per_sample,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
            max_paged_kv_indices=self.batch_size * self.n_per_sample,
        )
        self.expert_runner = PI05ExpertRunner(
            model.expert_stack,
            model.heads,
            model.rope,
            self.kv_pool,
            batch_size=self.batch_size,
            chunk_size=cfg.chunk_size,
            max_action_dim=cfg.max_action_dim,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
            max_paged_kv_indices=self.batch_size * (self.n_per_sample + cfg.chunk_size),
        )

        # ``(num_inference_steps, expert_hidden)`` lookup table —
        # populated by :meth:`setup` after ``load_pretrained``. Each
        # row is the full time-MLP output for one Euler step's ``t``;
        # broadcast to ``(B, expert_hidden)`` via ``expand`` and fed
        # straight to the expert runner's captured input buffer.
        self.time_emb_table: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Warm up runners and capture every CUDA graph.

        Must be called **after** ``load_pretrained`` — :attr:`time_emb_table`
        is computed here from the (now-real) ``time_mlp_in/out`` weights,
        so calling this on freshly-allocated weights would bake garbage
        values into the table.
        """
        self.vision_runner.setup()
        self.llm_runner.setup()
        self.expert_runner.setup()

        # Suffix write indices are constant — bind them once.
        write_indices_suffix = build_suffix_write_indices(
            self.batch_size,
            self.cfg.chunk_size,
            suffix_slot_base=self.suffix_base,
            device=self.device,
        )
        self.expert_runner.set_write_indices(write_indices_suffix)

        # Precompute the time-embedding table for the linear flow-matching
        # schedule ``t = 1.0 + step * (-1/N)``. ``embed_time`` is the full
        # MLP (sinusoidal → Linear → SiLU → Linear → SiLU); its output
        # depends only on the time scalar and the time-MLP weights, both
        # fixed across inferences, so a one-time precompute eliminates
        # the matmuls from every Euler step's captured graph.
        N = self.cfg.num_inference_steps
        ts = 1.0 + torch.arange(N, dtype=torch.float32, device=self.device) * (-1.0 / N)
        with torch.no_grad():
            self.time_emb_table = self.model.heads.embed_time(ts).contiguous()

    # ------------------------------------------------------------------ #
    # Step (one inference)                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, request: PI05Request) -> torch.Tensor:
        """Run one inference; return the action chunk ``(B, chunk, action_dim)``."""
        cfg = self.cfg
        device = self.device
        dtype = self.params_dtype
        self._validate(request)

        real_lens = (
            request.lang_lens.to(device=device, dtype=torch.int32)
            + self.image_token_count
        )
        n_real_total = int(real_lens.sum())

        # 1. Vision pass — single replay of the captured (3, 3, H, W) graph.
        # B is constrained to 1, so one replay is the whole vision phase.
        # ``vision_out`` aliases the vision graph's static output buffer;
        # downstream consumers (pack / LLM runner) only read it before
        # any other vision forward fires, so no copy-out is needed.
        pixel_values = request.pixel_values.to(device=device, dtype=dtype)
        vision_out = self.vision_runner.forward(
            VisionForwardBatch(pixel_values=pixel_values[0])
        )
        # (3, num_patches, projection_dim) → (1, image_token_count, hidden_size)
        image_embs = vision_out.flatten(0, 1).unsqueeze(0)

        # 2. Lang embed (eager) + per-sample padded packing.
        lang_embs = self.model.paligemma_lm.embed_lang(
            request.input_ids.to(device=device, dtype=torch.int64)
        )
        if lang_embs.dtype != dtype:
            lang_embs = lang_embs.to(dtype)
        packed = pack_prefix_per_sample_padded(
            image_embs,
            lang_embs,
            request.lang_lens.to(device=device, dtype=torch.int64),
            n_per_sample=self.n_per_sample,
        )

        # 3. Reset caches and plan the prefix layout.
        self.prefix_static.reset()
        self.suffix_static.reset()
        # The static cache's role is bookkeeping; the actual write goes
        # to slots inferred from real_lens / prefix_slot_base. We
        # explicitly bump the cursor to ``n_real_total`` to record the
        # allocation.
        _ = self.prefix_static.allocate(n_real_total)
        _ = self.suffix_static.allocate(self.batch_size * cfg.chunk_size)

        cu_q_prefix = torch.arange(
            0,
            (self.batch_size + 1) * self.n_per_sample,
            self.n_per_sample,
            dtype=torch.int32,
            device=device,
        )
        paged_kv_indptr_prefix = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=device
        )
        paged_kv_indptr_prefix[1:] = torch.cumsum(real_lens, 0)
        paged_kv_indices_prefix = build_prefix_paged_kv_indices(
            n_real_total, prefix_slot_base=self.prefix_base, device=device
        )
        paged_kv_last_prefix = build_prefix_last_page_len(real_lens)
        write_indices = build_prefix_padded_write_indices(
            real_lens,
            n_per_sample=self.n_per_sample,
            prefix_slot_base=self.prefix_base,
            sentinel_slot=self.sentinel_slot,
        )
        position_ids = build_prefix_padded_pos_ids(
            self.batch_size, self.n_per_sample, device=device
        )

        # Plan the LLM runner's wrapper / sdpa metadata buffers.
        self.llm_runner.plan_inference(
            cu_seqlens_q=cu_q_prefix,
            paged_kv_indptr=paged_kv_indptr_prefix,
            paged_kv_indices=paged_kv_indices_prefix,
            paged_kv_last_page_len=paged_kv_last_prefix,
        )

        # 4. Prefix forward — populates the cache pool. ``packed`` is a
        # fresh contiguous tensor; the runner's ``replay`` will copy it
        # into the captured input buffer, so no scheduler-side staging.
        # Attention metadata was already staged on the runner via
        # ``plan_inference`` above; the batch carries only the per-call
        # variable inputs.
        llm_batch = LLMForwardBatch(
            hidden_states=packed,
            position_ids=position_ids,
            write_indices=write_indices,
        )
        self.llm_runner.forward(llm_batch)

        # 5. Plan the expert runner's joint-attention metadata.
        pos_ids_suffix = build_suffix_pos_ids(real_lens, cfg.chunk_size)
        paged_kv_indptr_full = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=device
        )
        paged_kv_indptr_full[1:] = torch.cumsum(real_lens + cfg.chunk_size, 0)
        paged_kv_indices_full = build_joint_paged_kv_indices(
            real_lens,
            cfg.chunk_size,
            prefix_slot_base=self.prefix_base,
            suffix_slot_base=self.suffix_base,
        )
        paged_kv_last_full = build_joint_last_page_len(real_lens, cfg.chunk_size)
        self.expert_runner.plan_inference(
            pos_ids_suffix=pos_ids_suffix,
            paged_kv_indptr=paged_kv_indptr_full,
            paged_kv_indices=paged_kv_indices_full,
            paged_kv_last_page_len=paged_kv_last_full,
        )

        # 6. Sample noise (or use the user's) and run 10-step Euler.
        if request.noise is None:
            x_t = torch.randn(
                self.batch_size,
                cfg.chunk_size,
                cfg.max_action_dim,
                dtype=dtype,
                device=device,
            )
        else:
            x_t = request.noise.to(device=device, dtype=dtype).contiguous()

        if self.time_emb_table is None:
            raise RuntimeError(
                "PI05SingleBatchScheduler.step() called before setup(); "
                "the time-embedding lookup table has not been built."
            )

        dt = -1.0 / cfg.num_inference_steps
        for step in range(cfg.num_inference_steps):
            # Look up the precomputed AdaRMS condition for this step's t
            # and broadcast across all B samples (all batch elements share
            # the same t in this scheduler). The expand view is fed
            # straight to the expert runner — its ``replay`` materializes
            # the broadcast into the captured input buffer.
            time_emb = self.time_emb_table[step : step + 1].expand(self.batch_size, -1)
            expert_batch = ExpertForwardBatch(x_t=x_t, time_emb=time_emb)
            v_t = self.expert_runner.forward(expert_batch)
            # Capture-time output buffer is reused; clone is implicit
            # via the addition (allocates a fresh tensor each step,
            # which we then bind to ``x_t`` for the next iteration).
            x_t = x_t + dt * v_t.to(x_t.dtype)
        return x_t

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #

    def _validate(self, req: PI05Request) -> None:
        cfg = self.cfg
        if req.pixel_values.shape[0] != self.batch_size:
            raise ValueError(
                f"pixel_values batch dim {req.pixel_values.shape[0]} != "
                f"scheduler batch_size {self.batch_size}."
            )
        if req.pixel_values.shape[1] != 3:
            raise ValueError(
                f"pixel_values must have 3 cameras, got {req.pixel_values.shape[1]}."
            )
        if req.pixel_values.shape[-1] != cfg.vision.image_size:
            raise ValueError(
                f"pixel_values H/W {req.pixel_values.shape[-2:]} != "
                f"image_size {cfg.vision.image_size}."
            )
        if req.input_ids.shape != (self.batch_size, cfg.tokenizer_max_length):
            raise ValueError(
                f"input_ids shape {tuple(req.input_ids.shape)} != "
                f"(B, tokenizer_max_length)=({self.batch_size}, "
                f"{cfg.tokenizer_max_length})."
            )
        if req.lang_lens.shape != (self.batch_size,):
            raise ValueError(
                f"lang_lens shape {tuple(req.lang_lens.shape)} != "
                f"(B,)=({self.batch_size},)."
            )
        if req.noise is not None and req.noise.shape != (
            self.batch_size,
            cfg.chunk_size,
            cfg.max_action_dim,
        ):
            raise ValueError(
                f"noise shape {tuple(req.noise.shape)} != "
                f"(B, chunk_size, max_action_dim)="
                f"({self.batch_size}, {cfg.chunk_size}, {cfg.max_action_dim})."
            )


__all__ = ["PI05Request", "PI05SingleBatchScheduler"]
