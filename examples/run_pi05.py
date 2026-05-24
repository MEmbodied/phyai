"""Run pi0.5 inference end-to-end through the phyai engine plugin path.

Spins up the pi0.5 plugin behind ``Engine``, feeds dummy inputs (random
pixels and a single-token "prompt") for ``--batch-size`` robots, and
prints per-step latency. The action numbers themselves are meaningless
because the inputs are random — the script exists to verify the engine
wiring and to time it.

Run::

    uv run python examples/run_pi05.py --checkpoint /path/to/pi05_base/

The argument is a HuggingFace-style checkpoint **folder**: it must
contain ``config.json`` and either ``model.safetensors`` or
``model.safetensors.index.json`` plus its shards.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
from phyai.utils import load_config


def make_dummy_request(
    *,
    batch_size: int,
    plugin_cfg: PI05Config,
    device: torch.device,
    dtype: torch.dtype,
) -> PI05Request:
    """Build a placeholder ``PI05Request``: random pixels + one-token prompt."""
    pixel_values = torch.rand(
        batch_size,
        3,
        3,
        plugin_cfg.vision.image_size,
        plugin_cfg.vision.image_size,
        dtype=dtype,
        device=device,
    )
    input_ids = torch.zeros(
        batch_size, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, 0] = 2  # any non-pad token id
    lang_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
    return PI05Request(
        pixel_values=pixel_values,
        input_ids=input_ids,
        lang_lens=lang_lens,
    )


def benchmark(
    engine: Engine,
    request: PI05Request,
    *,
    n_warmup: int,
    n_timed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Warm + ``n_timed`` ``engine.step`` calls; return last action + ms stats."""
    actions: torch.Tensor | None = None
    for _ in range(n_warmup):
        actions = engine.step(request)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actions = engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    assert actions is not None
    return actions, {
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
        help=(
            "Path to the pi05_base checkpoint folder. Must contain "
            "config.json and either model.safetensors or "
            "model.safetensors.index.json with its shards."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of robots per request (also sets the engine's max_batch_size).",
    )
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=30)
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got: {args.checkpoint}"
        )

    plugin_cfg = load_config(args.checkpoint, PI05Config)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    engine = Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                checkpoint_dir=args.checkpoint,
                max_batch_size=args.batch_size,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=True),
            ),
        )
    )
    try:
        request = make_dummy_request(
            batch_size=args.batch_size,
            plugin_cfg=plugin_cfg,
            device=device,
            dtype=dtype,
        )
        actions, stats = benchmark(
            engine, request, n_warmup=args.n_warmup, n_timed=args.n_timed
        )

        print(f"action chunk shape : {tuple(actions.shape)}")
        print(f"action chunk dtype : {actions.dtype}")
        print(f"action chunk device: {actions.device}")
        print(
            f"step latency       : mean={stats['mean']:.2f} ms  "
            f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
            f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
            f"(n_warmup={args.n_warmup}, n_timed={args.n_timed})"
        )
        print(f"first action row   : {actions[0, 0].float().tolist()}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
