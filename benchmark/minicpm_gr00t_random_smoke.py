"""Full-size MiniCPM-GR00T latency baseline with random or real weights."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

from phyai.engine import Engine, EngineArgs
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm, RMSNorm
from phyai.models.minicpm_gr00t.main_minicpm_gr00t import MiniCPMGR00TArgs
from phyai.models.minicpm_gr00t.modeling_minicpm_gr00t import (
    MiniCPMGR00TQwenRMSNormGated,
)
from phyai.models.minicpm_gr00t.scheduler_ws1_minicpm_gr00t import (
    MiniCPMGR00TRequest,
)


DEFAULT_PROCESSOR_PATH = Path(__file__).resolve().parent / "minicpm-v46-processor"
DEFAULT_PROMPT_TEMPLATE = (
    "The robot is LIBERO Franka, a simulated single-arm Franka manipulator. "
    "Its action control method is absolute single-arm end-effector pose in the "
    "unified 80D layout with gripper closed command, and its action FPS is 20 Hz. "
    "Task: {instruction}"
)
ACTION_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def initialize_random_weights(engine: Engine, seed: int = 123) -> tuple[int, int]:
    """Initialize every checkpoint-backed parameter without writing a checkpoint."""
    model = engine.entry.model
    if model is None:
        raise RuntimeError("MiniCPM-GR00T model was not created.")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    parameter_count = 0
    parameter_bytes = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter_count += parameter.numel()
            parameter_bytes += parameter.numel() * parameter.element_size()
            if name.endswith("A_log") or name.endswith("dt_bias"):
                parameter.zero_()
            elif parameter.ndim == 1:
                parameter.zero_()
            else:
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
        for module in model.modules():
            if isinstance(module, GemmaRMSNorm):
                module.weight.zero_()
            elif isinstance(module, (LayerNorm, RMSNorm)):
                module.weight.fill_(1.0)
            elif isinstance(module, MiniCPMGR00TQwenRMSNormGated):
                module.weight.fill_(1.0)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()
    return parameter_count, parameter_bytes


def model_storage(engine: Engine) -> tuple[int, int]:
    model = engine.entry.model
    if model is None:
        raise RuntimeError("MiniCPM-GR00T model was not created.")
    parameters = list(model.parameters())
    return (
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    )


def build_synthetic_request(seed: int = 123, seq_len: int = 214) -> MiniCPMGR00TRequest:
    """Build the vision layout for three processor-produced 224x224 images."""
    image_tokens = 192
    if seq_len < image_tokens:
        raise ValueError(f"seq_len must be at least {image_tokens}, got {seq_len}.")
    input_ids = torch.full((1, seq_len), 1, dtype=torch.int64)
    input_ids[:, :image_tokens] = 248056
    pixel_values = torch.zeros((1, 3, 14, 43_008), dtype=torch.float32)
    target_sizes = torch.tensor(((32, 32), (32, 32), (32, 32)), dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed)
    return MiniCPMGR00TRequest(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=pixel_values,
        target_sizes=target_sizes,
        state=torch.zeros((1, 1, 80), dtype=torch.float32),
        noise=torch.randn((1, 30, 80), generator=generator),
    )


def build_processor_request(
    processor_path: Path,
    *,
    image_size: int,
    instruction: str,
    seed: int,
) -> MiniCPMGR00TRequest:
    if not processor_path.is_dir():
        raise FileNotFoundError(
            f"Processor directory not found: {processor_path}. "
            "Pass --synthetic-input or download the official processor files."
        )
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    blank = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    images = [Image.fromarray(blank), Image.fromarray(blank)]
    content = [{"type": "image", "image": image} for image in images]
    content.append(
        {
            "type": "text",
            "text": DEFAULT_PROMPT_TEMPLATE.format(instruction=instruction),
        }
    )
    processed = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": False},
    )
    generator = torch.Generator().manual_seed(seed)
    return MiniCPMGR00TRequest(
        input_ids=processed["input_ids"],
        attention_mask=processed["attention_mask"],
        pixel_values=processed["pixel_values"],
        target_sizes=processed["target_sizes"],
        state=torch.zeros((1, 1, 80), dtype=torch.float32),
        noise=torch.randn((1, 30, 80), generator=generator),
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def format_stats(stats: dict[str, float]) -> str:
    order = ("mean", "p50", "p90", "p99", "stdev", "min", "max")
    return " ".join(f"{name}={stats[name]:.3f}" for name in order)


def benchmark(
    engine: Engine,
    request: MiniCPMGR00TRequest,
    *,
    n_warmup: int,
    n_timed: int,
) -> tuple[torch.Tensor, float, dict[str, float], dict[str, float]]:
    """Return cold wall time plus steady-state CUDA-event and wall statistics."""
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    actions = engine.step(request)
    torch.cuda.synchronize()
    cold_wall_ms = (time.perf_counter() - cold_start) * 1000.0

    for _ in range(n_warmup):
        actions = engine.step(request)
    torch.cuda.synchronize()

    gpu_times_ms: list[float] = []
    wall_times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        actions = engine.step(request)
        end.record()
        end.synchronize()
        wall_times_ms.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_times_ms.append(start.elapsed_time(end))

    return actions, cold_wall_ms, summarize(gpu_times_ms), summarize(wall_times_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional real MiniCPM-GR00T policy checkpoint; default uses random weights.",
    )
    parser.add_argument(
        "--processor-path",
        type=Path,
        default=DEFAULT_PROCESSOR_PATH,
        help="Official MiniCPM-V 4.6 processor/tokenizer directory.",
    )
    parser.add_argument(
        "--synthetic-input",
        action="store_true",
        help="Bypass AutoProcessor but retain the default 214-token/two-image shapes.",
    )
    parser.add_argument("--synthetic-seq-len", type=int, default=214)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--instruction", default="open the middle drawer of the cabinet"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-timed", type=int, default=100)
    parser.add_argument("--gdn-backend", choices=("fla", "flashinfer"), default="fla")
    parser.add_argument(
        "--action-dtype",
        choices=tuple(ACTION_DTYPES),
        default="bfloat16",
        help="Action parameter/activation dtype.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.n_warmup < 0:
        parser.error("--n-warmup must be non-negative")
    if args.n_timed <= 0:
        parser.error("--n-timed must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PHYAI_NORM_BACKEND", "phyai-kernel")
    action_params_dtype = ACTION_DTYPES[args.action_dtype]

    preprocess_start = time.perf_counter()
    if args.synthetic_input:
        request = build_synthetic_request(args.seed, args.synthetic_seq_len)
        input_mode = "synthetic"
    else:
        request = build_processor_request(
            args.processor_path,
            image_size=args.image_size,
            instruction=args.instruction,
            seed=args.seed,
        )
        input_mode = "official_processor"
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    engine = Engine(
        EngineArgs(
            plugin="minicpm_gr00t",
            plugin_args=MiniCPMGR00TArgs(
                checkpoint=args.checkpoint,
                gdn_backend=args.gdn_backend,
                action_params_dtype=action_params_dtype,
            ),
        )
    )
    try:
        if args.checkpoint is None:
            parameter_count, parameter_bytes = initialize_random_weights(
                engine, args.seed
            )
            weight_mode = "random"
        else:
            parameter_count, parameter_bytes = model_storage(engine)
            weight_mode = "checkpoint"

        actions, cold_wall_ms, gpu_stats, wall_stats = benchmark(
            engine,
            request,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )
        image_tokens = int((request.input_ids == 248056).sum())
        print(
            f"weights={weight_mode} parameters={parameter_count:,} "
            f"storage={parameter_bytes / 2**30:.3f}GiB"
        )
        print(
            f"input={input_mode} seq_len={request.input_ids.shape[1]} "
            f"image_tokens={image_tokens} pixel_values={tuple(request.pixel_values.shape)} "
            f"target_sizes={request.target_sizes.tolist()} preprocess_once={preprocess_ms:.3f}ms"
        )
        print(
            f"backend norm={os.environ['PHYAI_NORM_BACKEND']} "
            f"gdn={args.gdn_backend} action_dtype={args.action_dtype} "
            f"warmup={args.n_warmup} timed={args.n_timed}"
        )
        print(f"cold_start_wall={cold_wall_ms:.3f}ms (includes JIT)")
        print(f"latency_gpu_ms {format_stats(gpu_stats)}")
        print(f"latency_wall_ms {format_stats(wall_stats)}")
        print(
            f"actions shape={tuple(actions.shape)} dtype={actions.dtype} "
            f"finite={bool(torch.isfinite(actions).all())} "
            f"action_min={actions.min().item():.6f} "
            f"action_max={actions.max().item():.6f} "
            f"action_mean={actions.mean().item():.6f} "
            f"action_std={actions.std().item():.6f}"
        )

        if args.output_json is not None:
            report = {
                "weight_mode": weight_mode,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "input_mode": input_mode,
                "seq_len": int(request.input_ids.shape[1]),
                "image_tokens": image_tokens,
                "pixel_values_shape": list(request.pixel_values.shape),
                "target_sizes": request.target_sizes.tolist(),
                "parameters": parameter_count,
                "parameter_bytes": parameter_bytes,
                "norm_backend": os.environ["PHYAI_NORM_BACKEND"],
                "gdn_backend": args.gdn_backend,
                "action_dtype": args.action_dtype,
                "action_params_dtype": str(action_params_dtype).removeprefix("torch."),
                "seed": args.seed,
                "n_warmup": args.n_warmup,
                "n_timed": args.n_timed,
                "preprocess_once_ms": preprocess_ms,
                "cold_start_wall_ms": cold_wall_ms,
                "latency_gpu_ms": gpu_stats,
                "latency_wall_ms": wall_stats,
                "actions_finite": bool(torch.isfinite(actions).all()),
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"report={args.output_json}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
