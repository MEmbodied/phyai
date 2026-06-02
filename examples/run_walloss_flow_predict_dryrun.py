"""Run WALL-OSS-FLOW policy predict dry-run through the PhyAI engine plugin path.

This example keeps the real processor / prepare_batch logic outside the core
runner and scheduler. It builds a dummy observation, converts it with wall-x's
official prepare_batch helper, then sends a WallOSSFlowPredictRequest through
Engine(plugin="walloss").

Run::

    PYTHONPATH=/phyai_workspace/src/wall-x:$PYTHONPATH \
    python examples/run_walloss_flow_predict_dryrun.py \
      --checkpoint /data/share/x-square-robot/wall-oss-flow \
      --wall-x-root /phyai_workspace/src/wall-x
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.walloss.configuration_walloss import WallOSSFlowConfig
from phyai.models.walloss.main_walloss import WallOSSArgs
from phyai.models.walloss.scheduler_ws1_walloss import WallOSSFlowPredictRequest


def _load_prepare_batch(wall_x_root: Path):
    """Load wall_x/serving/policy/utils.py without importing wall_x.serving.__init__."""
    utils_path = wall_x_root / "wall_x" / "serving" / "policy" / "utils.py"
    spec = importlib.util.spec_from_file_location("wall_x_policy_utils_local", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load prepare_batch from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_batch


def extract_predict_action(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for key in ("predict_action", "actions", "action", "pred_action"):
            if key in output:
                return output[key]

    if hasattr(output, "predict_action"):
        return output.predict_action

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                return item

    raise RuntimeError(f"cannot extract predict_action from output type={type(output)}")


def make_predict_request(
    *,
    checkpoint: Path,
    wall_x_root: Path,
    device: torch.device,
) -> WallOSSFlowPredictRequest:
    if str(wall_x_root) not in sys.path:
        sys.path.insert(0, str(wall_x_root))

    from wall_x.model.action_head import Normalizer
    from wall_x.utils.constant import action_statistic_dof

    prepare_batch = _load_prepare_batch(wall_x_root)

    cfg = WallOSSFlowConfig.from_checkpoint(checkpoint)
    raw_cfg = json.loads((checkpoint / "config.json").read_text())

    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    normalizer_propri = Normalizer(action_statistic_dof, raw_cfg["agent_pos_config"])

    camera_key = ["camera_0", "camera_1"]
    obs = {
        "camera_0": np.zeros((256, 256, 3), dtype=np.uint8),
        "camera_1": np.full((256, 256, 3), 127, dtype=np.uint8),
        "prompt": "Pick up the object and move it to the target position.",
        "state": np.zeros((cfg.proprio_dim,), dtype=np.float32),
        "dataset_names": cfg.dataset_name,
    }

    batch = prepare_batch(
        obs=obs,
        processor=processor,
        normalizer_propri=normalizer_propri,
        camera_key=camera_key,
        agent_pos_dim=cfg.proprio_dim,
        action_dim=cfg.action_dim,
        pred_horizon=cfg.action_horizon,
        fixed_action_dim=cfg.action_dim,
        max_length=2048,
        image_factor=28,
        min_pixels=56 * 56,
        max_pixels=256 * 28 * 28,
        predict_mode="diffusion",
        device=str(device),
    )
    batch = dict(batch)

    return WallOSSFlowPredictRequest(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        moe_token_types=batch["moe_token_types"],
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        proprioception=batch["proprioception"],
        agent_pos_mask=batch["agent_pos_mask"],
        dof_mask=batch["dof_mask"],
        dataset_names=batch["dataset_names"],
        predict_mode="diffusion",
        action_dim=cfg.action_dim,
        action_horizon=cfg.action_horizon,
    )


def benchmark(
    engine: Engine,
    request: WallOSSFlowPredictRequest,
    *,
    n_warmup: int,
    n_timed: int,
):
    output = None

    for _ in range(n_warmup):
        output = engine.step(request)

    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    if output is None:
        raise RuntimeError("benchmark did not run any engine.step call.")

    return output, {
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wall-x-root", type=Path, default=Path("/phyai_workspace/src/wall-x"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-warmup", type=int, default=0)
    parser.add_argument("--n-timed", type=int, default=1)
    args = parser.parse_args()

    if str(args.wall_x_root) not in sys.path:
        sys.path.insert(0, str(args.wall_x_root))

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    plugin_cfg = WallOSSFlowConfig.from_checkpoint(args.checkpoint)

    request = make_predict_request(
        checkpoint=args.checkpoint,
        wall_x_root=args.wall_x_root,
        device=device,
    )

    engine = Engine(
        EngineArgs(
            plugin="walloss",
            plugin_args=WallOSSArgs(
                checkpoint_dir=args.checkpoint,
                max_batch_size=1,
                action_horizon=plugin_cfg.action_horizon,
                dataset_name=plugin_cfg.dataset_name,
                precision_policy="selected_bf16",
            ),
            config=EngineConfig(
                device=DeviceConfig(target=args.device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )

    try:
        output, stats = benchmark(
            engine,
            request,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )
        predict_action = extract_predict_action(output)

        print(f"config                 : {plugin_cfg}")
        print(f"predict_action shape   : {tuple(predict_action.shape)}")
        print(f"predict_action dtype   : {predict_action.dtype}")
        print(f"predict_action device  : {predict_action.device}")
        print(f"has_nan                : {torch.isnan(predict_action).any().item()}")
        print(f"has_inf                : {torch.isinf(predict_action).any().item()}")
        print(f"predict_action min     : {float(predict_action.min().item())}")
        print(f"predict_action max     : {float(predict_action.max().item())}")
        print(f"predict_action mean    : {float(predict_action.float().mean().item())}")
        print(
            f"step latency           : mean={stats['mean']:.2f} ms  "
            f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
            f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
            f"(n_warmup={args.n_warmup}, n_timed={args.n_timed})"
        )

        expected_shape = (1, plugin_cfg.action_horizon, plugin_cfg.action_dim)
        if tuple(predict_action.shape) != expected_shape:
            raise AssertionError(
                f"predict_action shape {tuple(predict_action.shape)} != {expected_shape}"
            )
        if predict_action.dtype != torch.float32:
            raise AssertionError(f"predict_action dtype is not float32: {predict_action.dtype}")
        if predict_action.device.type != "cuda":
            raise AssertionError(f"predict_action is not on CUDA: {predict_action.device}")
        if not torch.isfinite(predict_action).all().item():
            raise AssertionError("predict_action contains NaN or Inf")

        print("WALLOSS ENGINE POLICY PREDICT DRY RUN PASSED")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
