"""End-to-end Cosmos3 action/policy demo"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _save_video(video: torch.Tensor, path: str, fps: float) -> None:
    """Save ``[1, 3, T, H, W]`` or ``[3, T, H, W]`` in [0,1] to mp4."""
    if video.ndim == 5:
        video = video[0]
    frames = (video.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).cpu()
    if path.endswith(".pt"):
        torch.save(frames, path)
        return
    try:
        import imageio

        imageio.mimsave(
            path,
            frames.numpy(),
            fps=int(round(fps)),
            quality=8,
            macro_block_size=1,
            output_params=["-f", "mp4"],
        )
    except Exception as exc:
        fallback = path.rsplit(".", 1)[0] + ".pt"
        torch.save(frames, fallback)
        print(
            f"[warn] no mp4 writer ({exc}); saved to {fallback}. "
            f"Install `imageio imageio-ffmpeg` for mp4."
        )


def _save_action(action: torch.Tensor, path: str) -> None:
    """Save ``[1, chunk, dim]`` action tensor as JSON."""
    data = {
        "shape": list(action.shape),
        "dtype": str(action.dtype).replace("torch.", ""),
        "data": action.squeeze(0).tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_action_from_file(path: str, chunk_index: int = 0) -> torch.Tensor:
    """Load action from the Cosmos3 example JSON format."""
    with open(path) as f:
        data = json.load(f)
    if "action_chunks" in data:
        chunk = data["action_chunks"][chunk_index]
        return torch.tensor(chunk, dtype=torch.float32)
    if "data" in data:
        shape = data.get("shape", None)
        t = torch.tensor(data["data"], dtype=torch.float32)
        if shape:
            t = t.reshape(shape)
        return t
    raise ValueError(f"Unrecognized action file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Cosmos3-Nano checkpoint dir"
    )
    parser.add_argument(
        "--image", required=True, help="Observation image (first frame)"
    )
    parser.add_argument("--prompt", default="robot manipulates objects")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--mode",
        choices=("policy", "forward_dynamics", "inverse_dynamics"),
        default="policy",
    )
    parser.add_argument("--domain-name", default="agibotworld")
    parser.add_argument(
        "--action-file",
        default=None,
        help="JSON file with action chunks (required for forward_dynamics)",
    )
    parser.add_argument("--action-chunk-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--raw-action-dim", type=int, default=29)
    parser.add_argument("--out", default=".cache/cosmos3_policy_out")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.cosmos3 import Cosmos3ActionRequest, pixel_to_latent_shape
    from phyai.models.cosmos3.main_cosmos3_policy import Cosmos3PolicyArgs
    from phyai_utils_tools.models.cosmos3 import Cosmos3PolicyProcessor

    device = "cuda"
    dtype = torch.bfloat16

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[engine] creating cosmos3_policy engine ...")
    engine = Engine(
        EngineArgs(
            plugin="cosmos3_policy",
            plugin_args=Cosmos3PolicyArgs(
                checkpoint_dir=args.checkpoint,
                flow_shift=args.flow_shift,
                decode_video=True,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )

    try:
        print("[processor] preprocessing ...")
        processor = Cosmos3PolicyProcessor(
            tokenizer_name_or_path=f"{args.checkpoint}/text_tokenizer",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            mode=args.mode,
            domain_name=args.domain_name,
            action_chunk_size=args.action_chunk_size,
            raw_action_dim=args.raw_action_dim,
            negative_prompt=args.negative_prompt,
            device=device,
            params_dtype=dtype,
        )

        raw_input: dict = {
            "images": args.image,
            "task": args.prompt,
        }
        if args.mode == "forward_dynamics":
            if args.action_file is None:
                raise ValueError("--action-file is required for forward_dynamics mode.")
            raw_input["cond_action"] = _load_action_from_file(
                args.action_file, args.action_chunk_index
            )

        processed = processor.preprocess(raw_input)

        video_shape = pixel_to_latent_shape(
            processed.video_shape[0], processed.video_shape[1], processed.video_shape[2]
        )
        request = Cosmos3ActionRequest(
            text_ids=processed.text_ids.to(device),
            text_mask=processed.text_mask.to(device),
            neg_text_ids=processed.neg_text_ids.to(device),
            neg_text_mask=processed.neg_text_mask.to(device),
            video_shape=video_shape,
            mode=processed.mode,
            domain_id=processed.domain_id,
            action_chunk=processed.action_chunk,
            raw_action_dim=processed.raw_action_dim,
            cond_video_pixels=processed.pixel_values.to(device=device, dtype=dtype),
            cond_action=(
                processed.cond_action.to(device=device, dtype=dtype)
                if processed.cond_action is not None
                else None
            ),
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )

        print(
            f"[run] mode={args.mode} domain={args.domain_name} "
            f"latent={video_shape} steps={args.steps} "
            f"action_chunk={args.action_chunk_size}x{args.raw_action_dim}"
        )
        result = engine.step(request)

        output = processor.postprocess(result)
        action = output["action"]
        print(
            f"[done] action shape={tuple(action.shape)} "
            f"range=[{action.min():.4f}, {action.max():.4f}]"
        )

        action_path = f"{args.out}_action.json"
        _save_action(action, action_path)
        print(f"[saved] action -> {action_path}")

        if "pixels" in output:
            video_path = f"{args.out}.mp4"
            _save_video(output["pixels"], video_path, args.fps)
            print(f"[saved] video -> {video_path}")

    finally:
        engine.close()


if __name__ == "__main__":
    main()
