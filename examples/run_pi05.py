"""Run pi0.5 inference end-to-end through the phyai engine plugin path.

This example exercises the engine + plugin contract for the multi-batch
``PI05WS1Scheduler``:

    EngineArgs(plugin_args=PI05Args(max_batch_size=...))  ->  Engine(...)
        ->  engine.step(PI05Request(B in [1, max_batch_size]))   ->  actions

It runs three phases:

1. ``max_batch_size=1`` latency sweep — regression check that the
   single-batch path (the only path that existed before the multi-batch
   rewrite) still runs end-to-end and reports per-step latency.

2. ``max_batch_size=4`` latency sweep with ``actual_B=4`` — the new
   multi-batch path at full saturation.

3. Equivalence check on a fresh ``max_batch_size=4`` engine. With
   ``actual_B=1`` (one real robot, three padded), the first output row
   must match the ``actual_B=4`` run where row 0 holds the same inputs
   replicated 4×. This is the load-bearing correctness check for the
   sentinel-routed padding: if a padded sample's K/V leaks into row 0's
   attention, the equivalence breaks.

The inputs here are *dummy* — random pixel values and a one-token
"prompt" — so the action numbers themselves aren't meaningful. The
example verifies wiring (no NaN, no shape errors) and the multi-batch
invariant.

Run::

    LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat/lib uv run python examples/run_pi05.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request


PI05_BASE_WEIGHTS = Path(
    "/mnt/bos-multimodal/wangchenghua/hf_models/pi05_base/model.safetensors"
)


def make_dummy_request(
    *,
    batch_size: int,
    image_size: int,
    tokenizer_max_length: int,
    chunk_size: int,
    max_action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> PI05Request:
    """Build a deterministic placeholder :class:`PI05Request`.

    Three random images per robot, a single-token "prompt" padded to
    ``tokenizer_max_length`` with zeros, and explicit ``noise`` so the
    Euler loop is reproducible across the equivalence runs (otherwise
    the scheduler would draw fresh noise sized for ``max_batch_size``,
    consuming a different amount of RNG state in each run).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    pixel_values = torch.rand(
        batch_size,
        3,
        3,
        image_size,
        image_size,
        dtype=dtype,
        device=device,
        generator=g,
    )
    input_ids = torch.zeros(
        batch_size, tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, 0] = 2  # any non-pad token id
    lang_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
    noise = torch.randn(
        batch_size,
        chunk_size,
        max_action_dim,
        dtype=dtype,
        device=device,
        generator=g,
    )
    return PI05Request(
        pixel_values=pixel_values,
        input_ids=input_ids,
        lang_lens=lang_lens,
        noise=noise,
    )


def time_step(
    engine: Engine, request: PI05Request, *, n_warmup: int, n_timed: int
) -> dict[str, float]:
    """Warm + n_timed timed ``engine.step`` calls; return latency stats in ms."""
    for _ in range(n_warmup):
        _ = engine.step(request)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def report(
    label: str,
    actions: torch.Tensor,
    stats: dict[str, float],
    *,
    n_warmup: int,
    n_timed: int,
) -> None:
    print(f"[{label}]")
    print(f"  action chunk shape : {tuple(actions.shape)}")
    print(f"  action chunk dtype : {actions.dtype}")
    print(f"  action chunk device: {actions.device}")
    print(
        f"  step latency       : mean={stats['mean']:.2f} ms  "
        f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
        f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
        f"(n_warmup={n_warmup}, n_timed={n_timed})"
    )
    print(f"  first action row   : {actions[0, 0].float().tolist()}")
    print(f"  has NaN            : {bool(torch.isnan(actions).any().item())}")


def make_engine(max_batch_size: int) -> Engine:
    return Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                weights_paths=[PI05_BASE_WEIGHTS],
                max_batch_size=max_batch_size,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=True),
            ),
        )
    )


def main() -> None:
    if not PI05_BASE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"pi05_base weights not found at {PI05_BASE_WEIGHTS}; edit the "
            f"PI05_BASE_WEIGHTS constant or copy the safetensors file."
        )

    plugin_cfg = PI05Config()  # default geometry == pi05_base
    device = torch.device("cuda")
    dtype = torch.bfloat16

    n_warmup = 3
    n_timed = 30

    # ----------------------------------------------------------------- #
    # Phase 1: max_batch_size=1 regression.                             #
    # ----------------------------------------------------------------- #
    engine = make_engine(max_batch_size=1)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        request_b1 = make_dummy_request(
            batch_size=1,
            image_size=plugin_cfg.vision.image_size,
            tokenizer_max_length=plugin_cfg.tokenizer_max_length,
            chunk_size=plugin_cfg.chunk_size,
            max_action_dim=plugin_cfg.max_action_dim,
            device=device,
            dtype=dtype,
            seed=0,
        )
        actions_b1 = engine.step(request_b1)
        stats_b1 = time_step(engine, request_b1, n_warmup=n_warmup, n_timed=n_timed)
        report(
            "max_batch_size=1, actual_B=1",
            actions_b1,
            stats_b1,
            n_warmup=n_warmup,
            n_timed=n_timed,
        )
    finally:
        engine.close()

    # ----------------------------------------------------------------- #
    # Phase 2: max_batch_size=4, actual_B=4 saturated multi-batch.      #
    # ----------------------------------------------------------------- #
    engine = make_engine(max_batch_size=4)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        request_b4 = make_dummy_request(
            batch_size=4,
            image_size=plugin_cfg.vision.image_size,
            tokenizer_max_length=plugin_cfg.tokenizer_max_length,
            chunk_size=plugin_cfg.chunk_size,
            max_action_dim=plugin_cfg.max_action_dim,
            device=device,
            dtype=dtype,
            seed=1,
        )
        actions_b4 = engine.step(request_b4)
        stats_b4 = time_step(engine, request_b4, n_warmup=n_warmup, n_timed=n_timed)
        report(
            "max_batch_size=4, actual_B=4",
            actions_b4,
            stats_b4,
            n_warmup=n_warmup,
            n_timed=n_timed,
        )

        # ------------------------------------------------------------- #
        # Phase 3: equivalence — actual_B=1 vs actual_B=4 (row 0 same). #
        # ------------------------------------------------------------- #
        # Build a single-robot request, then a 4-robot request whose
        # row 0 holds the same inputs (rows 1-3 are different garbage).
        # Sentinel routing means: in the B=1 run, padded rows 1-3 are
        # routed to slot 0 and don't write any K/V that row 0 reads.
        # In the B=4 run, rows 1-3 write real K/V to their own slots
        # but row 0's paged_kv_indices_full only points at row 0's
        # slots, so no cross-contamination. Outputs of row 0 must
        # match (modulo bf16 noise).
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        eq_b1 = make_dummy_request(
            batch_size=1,
            image_size=plugin_cfg.vision.image_size,
            tokenizer_max_length=plugin_cfg.tokenizer_max_length,
            chunk_size=plugin_cfg.chunk_size,
            max_action_dim=plugin_cfg.max_action_dim,
            device=device,
            dtype=dtype,
            seed=42,
        )
        eq_b4_pixel = torch.empty(
            4,
            3,
            3,
            plugin_cfg.vision.image_size,
            plugin_cfg.vision.image_size,
            dtype=dtype,
            device=device,
        )
        eq_b4_pixel[0] = eq_b1.pixel_values[0]
        eq_b4_pixel[1:] = make_dummy_request(
            batch_size=3,
            image_size=plugin_cfg.vision.image_size,
            tokenizer_max_length=plugin_cfg.tokenizer_max_length,
            chunk_size=plugin_cfg.chunk_size,
            max_action_dim=plugin_cfg.max_action_dim,
            device=device,
            dtype=dtype,
            seed=99,
        ).pixel_values

        eq_b4_input_ids = torch.zeros(
            4, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
        )
        eq_b4_input_ids[:, 0] = 2
        eq_b4_lang_lens = torch.ones(4, dtype=torch.int64, device=device)

        eq_b4_noise = torch.empty(
            4,
            plugin_cfg.chunk_size,
            plugin_cfg.max_action_dim,
            dtype=dtype,
            device=device,
        )
        eq_b4_noise[0] = eq_b1.noise[0]
        # Garbage noise for the other rows; doesn't influence row 0.
        eq_b4_noise[1:] = torch.randn_like(eq_b4_noise[1:])

        eq_b4 = PI05Request(
            pixel_values=eq_b4_pixel,
            input_ids=eq_b4_input_ids,
            lang_lens=eq_b4_lang_lens,
            noise=eq_b4_noise,
        )

        out_eq_b1 = engine.step(eq_b1)  # actual_B=1 on max_batch_size=4
        out_eq_b4 = engine.step(eq_b4)  # actual_B=4

        diff = (out_eq_b4[0].float() - out_eq_b1[0].float()).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        print("[equivalence: actual_B=1 vs actual_B=4 row 0]")
        print(
            f"  shapes             : B=1 {tuple(out_eq_b1.shape)}  "
            f"B=4 {tuple(out_eq_b4.shape)}"
        )
        print(f"  max |diff|         : {max_abs:.4e}")
        print(f"  mean |diff|        : {mean_abs:.4e}")
        # bf16 has ~7-bit mantissa; tolerance set generously for the
        # 10-step Euler loop's accumulated rounding.
        atol = 1e-2
        ok = max_abs <= atol
        print(f"  result             : {'PASS' if ok else 'FAIL'}  (atol={atol})")
        if not ok:
            raise SystemExit(
                f"Multi-batch equivalence FAILED: max|diff|={max_abs:.4e} > {atol}. "
                "Padded sample K/V is leaking into row 0's attention."
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
