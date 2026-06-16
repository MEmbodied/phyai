"""Run WALL-OSS-FLOW inference end-to-end through the PhyAI engine plugin path.

This example spins up the ``walloss`` plugin behind ``Engine``, feeds a
dummy validate request that mirrors the existing wall-x FLOW fake inference
path, and reports logits shape plus per-step latency.

The dummy inputs are not semantically meaningful. The script is intended to
verify Engine/plugin wiring and provide a small reproducible smoke benchmark.

Run::

    python examples/run_walloss_flow.py --checkpoint /path/to/wall-oss-flow

On A40/sm86 hosts, use a PhyAI version that includes the upstream sm86
Linear-kernel registration fix, such as nightly ``0.1.1.dev20260601`` or newer.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.walloss.configuration_walloss import WallOSSFlowConfig
from phyai.models.walloss.main_walloss import WallOSSArgs
from phyai.models.walloss.scheduler_ws1_walloss import WallOSSFlowRequest


def make_dummy_request(
    *,
    batch_size: int,
    seq_len: int,
    plugin_cfg: WallOSSFlowConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> WallOSSFlowRequest:
    """Build a placeholder WALL-OSS-FLOW validate request."""
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1).contiguous()

    return WallOSSFlowRequest(
        input_ids=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        moe_token_types=torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        position_ids=position_ids,
        proprioception=torch.zeros(
            batch_size, 1, plugin_cfg.proprio_dim, dtype=dtype, device=device
        ),
        agent_pos_mask=torch.ones(
            batch_size, 1, plugin_cfg.proprio_dim, dtype=dtype, device=device
        ),
        dof_mask=torch.ones(
            batch_size,
            plugin_cfg.action_horizon,
            plugin_cfg.action_dim,
            dtype=dtype,
            device=device,
        ),
        dataset_names=plugin_cfg.dataset_name,
    )


def benchmark(
    engine: Engine,
    request: WallOSSFlowRequest,
    *,
    n_warmup: int,
    n_timed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Warm up and time ``engine.step`` calls; return the last logits and stats."""
    outputs = None
    for _ in range(n_warmup):
        outputs = engine.step(request)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    if outputs is None:
        raise RuntimeError("benchmark did not run any engine.step call.")

    logits = outputs.logits
    return logits, {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the WALL-OSS-FLOW checkpoint folder.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-timed", type=int, default=3)
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Enable CUDA graph in RuntimeConfig. Disabled by default for smoke tests.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise NotADirectoryError(f"--checkpoint must be a directory: {args.checkpoint}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.seq_len <= 0:
        raise ValueError(f"--seq-len must be positive, got {args.seq_len}.")
    if args.n_timed <= 0:
        raise ValueError(f"--n-timed must be positive, got {args.n_timed}.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    plugin_cfg = WallOSSFlowConfig.from_checkpoint(args.checkpoint)

    engine = Engine(
        EngineArgs(
            plugin="walloss",
            plugin_args=WallOSSArgs(
                checkpoint_dir=args.checkpoint,
                max_batch_size=args.batch_size,
                action_horizon=plugin_cfg.action_horizon,
                dataset_name=plugin_cfg.dataset_name,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=args.use_cuda_graph),
            ),
        )
    )

    try:
        request = make_dummy_request(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            plugin_cfg=plugin_cfg,
            device=device,
            dtype=dtype,
        )
        logits, stats = benchmark(
            engine, request, n_warmup=args.n_warmup, n_timed=args.n_timed
        )

        print(f"config            : {plugin_cfg}")
        print(f"logits shape      : {tuple(logits.shape)}")
        print(f"logits dtype      : {logits.dtype}")
        print(f"logits device     : {logits.device}")
        print(f"has_nan           : {torch.isnan(logits).any().item()}")
        print(f"has_inf           : {torch.isinf(logits).any().item()}")
        print(
            f"step latency      : mean={stats['mean']:.2f} ms  "
            f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
            f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
            f"(n_warmup={args.n_warmup}, n_timed={args.n_timed})"
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
