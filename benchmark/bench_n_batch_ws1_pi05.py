"""End-to-end pi0.5 ws1 (single-card) latency benchmark, swept over batch sizes.

Builds the pi0.5 engine via the standard plugin path once per batch
size, feeds a deterministic dummy request, and
hands it to the generic :class:`NBatchBenchRunner` from
:mod:`bench_n_batch` for timing + optional Nsight Systems / Perfetto
profile capture.

The numbers are end-to-end ``Engine.step`` latency — they include the
vision tower replays, the LLM prefix forward, all ten Euler steps,
and the per-step scheduler glue. The action chunks themselves are
garbage (inputs are random); the script exists for performance
measurement, not correctness.

Run::

    uv run python \\
        benchmark/bench_n_batch_ws1_pi05.py \\
        --checkpoint /path/to/pi05_base \\
        --batch-sizes 1 2 4 --n-warmup 5 --n-timed 30 \\
        --result-file ./pi05_ws1_results.jsonl

Profile a tight window with the torch profiler (Perfetto-loadable)::

    ... --profile-backend torch --profile-output-dir ./prof \\
        --profile-start-step 5 --profile-num-steps 3

Profile under Nsight Systems::

    nsys profile --capture-range=cudaProfilerApi \\
        --capture-range-end=stop -o ./prof/pi05_ws1 \\
        uv run python benchmark/bench_n_batch_ws1_pi05.py \\
        --checkpoint /path/to/pi05_base --batch-sizes 4 \\
        --profile-backend nsys --profile-start-step 5 --profile-num-steps 3

``torch`` and ``nsys`` are exclusive — pick one per run. NVTX ranges
emitted by ``nsys`` mode are only captured by an enclosing
``nsys profile``; nothing happens if you select ``--profile-backend
nsys`` without that wrapper.

The pi0.5 scheduler is already instrumented with named event scopes
(``pi05.vision_loop`` / ``pi05.lang_pack`` / ``pi05.llm_prefix_plan``
/ ``pi05.llm_prefix_fwd`` / ``pi05.expert_plan`` / ``pi05.expert_loop``
with per-Euler-step ``pi05.expert_step``), and the bench runner wraps
each timed step in ``bench.step`` — every profile backend sees these
as named ranges with no extra wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import bench_n_batch as bnb
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
from phyai.utils import load_config
from phyai.utils.profile import (
    add_profile_cli_args,
    install_profiler,
    profile_config_from_args,
)


_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def make_dummy_request(
    *,
    batch_size: int,
    num_images: int,
    plugin_cfg: PI05Config,
    device: torch.device,
    dtype: torch.dtype,
    lang_len: int = 1,
    seed: int = 0,
) -> PI05Request:
    """Deterministic random request for ``batch_size`` robots."""
    if not 1 <= lang_len <= plugin_cfg.tokenizer_max_length:
        raise ValueError(
            f"lang_len must be in [1, {plugin_cfg.tokenizer_max_length}], got {lang_len}"
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    pixel_values = torch.rand(
        batch_size,
        num_images,
        3,
        plugin_cfg.vision.image_size,
        plugin_cfg.vision.image_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    input_ids = torch.zeros(
        batch_size, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, :lang_len] = 2  # any non-pad token id
    lang_lens = torch.full((batch_size,), lang_len, dtype=torch.int64, device=device)
    return PI05Request(
        pixel_values=pixel_values,
        input_ids=input_ids,
        lang_lens=lang_lens,
    )


def make_setup_fn(
    *,
    checkpoint: Path,
    dtype: torch.dtype,
    device_target: str,
    use_cuda_graph: bool,
    num_images: int = 3,
    vision_params_dtype: torch.dtype | None = None,
    actual_batch_size: int | None = None,
    lang_len: int = 1,
    seed: int = 0,
):
    """Build the per-batch-size ``setup_fn`` closure for :class:`NBatchBenchRunner`.

    Reads the plugin config once (it's the same across all batch
    sizes); the engine itself is rebuilt per batch size so each gets
    a fresh ``max_batch_size``-sized scheduler / KV pool.
    """
    plugin_cfg = load_config(checkpoint, PI05Config)
    device = torch.device(device_target)
    inputs_image_shape = [
        [plugin_cfg.vision.image_size, plugin_cfg.vision.image_size, 3]
        for _ in range(num_images)
    ]

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        actual_B = batch_size if actual_batch_size is None else actual_batch_size
        if not 1 <= actual_B <= batch_size:
            raise ValueError(
                f"actual_batch_size must be in [1, max_batch_size={batch_size}], "
                f"got {actual_B}"
            )
        engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=checkpoint,
                    max_batch_size=batch_size,
                    vision_params_dtype=vision_params_dtype,
                    inputs_image_shape=inputs_image_shape,
                ),
                config=EngineConfig(
                    device=DeviceConfig(target=device_target, params_dtype=dtype),
                    runtime=RuntimeConfig(use_cuda_graph=use_cuda_graph),
                ),
            )
        )
        request = make_dummy_request(
            batch_size=actual_B,
            num_images=num_images,
            plugin_cfg=plugin_cfg,
            device=device,
            dtype=dtype,
            lang_len=lang_len,
            seed=seed,
        )
        return bnb.BenchSpec(
            name="ws1_pi05",
            step_callable=lambda: engine.step(request),
            teardown_callable=engine.close,
            sample_count=actual_B,
        )

    return setup_fn


def make_extras_fn(
    *,
    dtype_name: str,
    device_target: str,
    use_cuda_graph: bool,
    plugin_cfg: PI05Config,
    actual_batch_size: int | None,
    lang_len: int,
    seed: int,
    quantization: dict[str, Any],
):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        actual_B = batch_size if actual_batch_size is None else actual_batch_size
        lang_buckets = sorted(
            {b for b in (16, 48, 112) if b < plugin_cfg.tokenizer_max_length}
            | {plugin_cfg.tokenizer_max_length}
        )
        lang_bucket = next(
            (bucket for bucket in lang_buckets if bucket >= lang_len),
            plugin_cfg.tokenizer_max_length,
        )
        return {
            "model": "pi05",
            "scheduler": "ws1",
            "dtype": dtype_name,
            "device": device_target,
            "use_cuda_graph": use_cuda_graph,
            "max_batch_size": batch_size,
            "actual_batch_size": actual_B,
            "lang_len": lang_len,
            "lang_bucket": lang_bucket,
            "chunk_size": plugin_cfg.chunk_size,
            "seed": seed,
            "quantization": quantization,
        }

    return extras_fn


def checkpoint_quantization_metadata(checkpoint: Path) -> dict[str, Any]:
    """Return portable quantization metadata without recording host paths."""
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    groups = []
    for name, group in sorted((quant.get("config_groups") or {}).items()):
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        groups.append(
            {
                "name": name,
                "targets": len(group.get("targets") or []),
                "weight_num_bits": weight.get("num_bits"),
                "weight_type": weight.get("type"),
                "weight_humming_dtype": weight.get("humming_dtype"),
                "weight_strategy": weight.get("strategy"),
                "weight_block_structure": weight.get("block_structure"),
                "activation_num_bits": activation.get("num_bits"),
                "activation_type": activation.get("type"),
                "activation_strategy": activation.get("strategy"),
                "activation_group_size": activation.get("group_size"),
                "activation_dynamic": activation.get("dynamic"),
            }
        )
    return {
        "quant_method": quant.get("quant_method"),
        "format": quant.get("format"),
        "pack_format": quant.get("pack_format"),
        "status": quant.get("quantization_status"),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--dtype",
        choices=sorted(_DTYPES),
        default="bf16",
        help="Engine params_dtype (default bf16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Engine device target (default "cuda"; pass "cpu" for CPU-only debug).',
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph capture (engine still runs, just no replay).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of cameras per robot (default 3, the pi05_base contract).",
    )
    parser.add_argument(
        "--actual-batch-size",
        type=int,
        default=None,
        help="Request batch size; defaults to each swept max batch size.",
    )
    parser.add_argument(
        "--lang-len",
        type=int,
        default=1,
        help="Real prompt length before scheduler bucket padding (default 1).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vision-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help=(
            "Vision tower compute precision. 'float32' runs SigLIP + projector "
            "in fp32 (openpi/lerobot parity) while the rest stays at --dtype."
        ),
    )

    bnb.add_bench_cli_args(parser)
    add_profile_cli_args(parser)

    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got: {args.checkpoint}"
        )

    dtype = _DTYPES[args.dtype]
    use_cuda_graph = not args.no_cuda_graph and args.device == "cuda"
    plugin_cfg = load_config(args.checkpoint, PI05Config)
    quantization = checkpoint_quantization_metadata(args.checkpoint)

    # Install whatever profiler the CLI requested. NoOp is the default
    # when --profile-backend is "none" (or rank is excluded).
    profile_cfg = profile_config_from_args(args)
    install_profiler(profile_cfg)

    setup_fn = make_setup_fn(
        checkpoint=args.checkpoint,
        dtype=dtype,
        device_target=args.device,
        use_cuda_graph=use_cuda_graph,
        num_images=args.num_images,
        vision_params_dtype=(torch.float32 if args.vision_dtype == "float32" else None),
        actual_batch_size=args.actual_batch_size,
        lang_len=args.lang_len,
        seed=args.seed,
    )
    extras_fn = make_extras_fn(
        dtype_name=args.dtype,
        device_target=args.device,
        use_cuda_graph=use_cuda_graph,
        plugin_cfg=plugin_cfg,
        actual_batch_size=args.actual_batch_size,
        lang_len=args.lang_len,
        seed=args.seed,
        quantization=quantization,
    )

    runner = bnb.NBatchBenchRunner(
        setup_fn=setup_fn,
        extras_fn=extras_fn,
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
