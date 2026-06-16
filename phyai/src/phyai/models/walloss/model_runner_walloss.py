"""Model runner wrapper for WALL-OSS-FLOW.

First version policy:
- use the official wall-x model loader;
- do not rewrite Qwen2.5-VL / MoE / action head;
- keep the runner small so it can later be replaced by a deeper PhyAI-native runner.

Two execution paths are intentionally separated:
- ``forward`` keeps the original fake-validate logits path;
- ``predict`` supports policy-level FLOW action prediction.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import torch

from phyai.runtime.model_runner import ModelRunner


PrecisionPolicy = Literal["full_bf16", "selected_bf16", "float32"]


class WallOSSFlowRunner(ModelRunner):
    """Thin wrapper around wall-x's Qwen2_5_VLMoEForAction."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        precision_policy: PrecisionPolicy = "full_bf16",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.precision_policy = precision_policy
        self.model: torch.nn.Module | None = None

        if self.precision_policy not in {"full_bf16", "selected_bf16", "float32"}:
            raise ValueError(
                "precision_policy must be one of "
                f"{{'full_bf16', 'selected_bf16', 'float32'}}, got {precision_policy!r}."
            )

    def setup(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint_dir.is_dir():
            raise NotADirectoryError(f"checkpoint_dir is not a directory: {self.checkpoint_dir}")
        if not (self.checkpoint_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"missing model.safetensors under checkpoint_dir: {self.checkpoint_dir}"
            )

        # Lazy import: importing phyai.engine should not require wall-x native
        # extensions until the walloss plugin is actually set up.
        from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
            Qwen2_5_VLMoEForAction,
        )

        # Engine may set torch default dtype to the target params dtype while
        # constructing plugins. Keep the wall-x loader on its official path:
        # load weights under float32 default dtype first, then explicitly cast
        # according to the requested precision policy.
        old_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            model = Qwen2_5_VLMoEForAction.from_pretrained(str(self.checkpoint_dir))
        finally:
            torch.set_default_dtype(old_default_dtype)

        model = model.to(self.device)
        model = self._apply_precision_policy(model)
        model.eval()
        self.model = model

    def _apply_precision_policy(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.precision_policy == "selected_bf16":
            if self.dtype != torch.bfloat16:
                raise ValueError(
                    "precision_policy='selected_bf16' requires dtype=torch.bfloat16, "
                    f"got {self.dtype}."
                )
            if not hasattr(model, "to_bfloat16_for_selected_params"):
                raise AttributeError(
                    "wall-x model does not expose to_bfloat16_for_selected_params()."
                )
            return model.to_bfloat16_for_selected_params()

        if self.precision_policy == "float32":
            return model.float()

        # Default first-version validate path: cast the whole model to runtime dtype.
        if self.dtype == torch.bfloat16:
            return model.bfloat16()
        if self.dtype == torch.float16:
            return model.half()
        if self.dtype == torch.float32:
            return model.float()
        return model.to(dtype=self.dtype)

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("WallOSSFlowRunner called before setup().")
        return self.model

    def _load_raw_config(self) -> dict[str, Any]:
        cfg_path = self.checkpoint_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing wall-oss config.json: {cfg_path}")
        return json.loads(cfg_path.read_text())

    def _has_normalizers(self) -> bool:
        if self.model is None:
            return False
        action_preprocessor = getattr(self.model, "action_preprocessor", None)
        if action_preprocessor is None:
            return False
        return (
            getattr(action_preprocessor, "normalizer_action", None) is not None
            and getattr(action_preprocessor, "normalizer_propri", None) is not None
        )

    def set_normalizers(
        self,
        normalizer_action: torch.nn.Module,
        normalizer_propri: torch.nn.Module,
    ) -> None:
        """Set action/proprio normalizers on the model and move them to runner device."""
        model = self._require_model()

        normalizer_action = copy.deepcopy(normalizer_action).to(self.device)
        normalizer_propri = copy.deepcopy(normalizer_propri).to(self.device)

        if hasattr(model, "set_normalizer"):
            model.set_normalizer(normalizer_action, normalizer_propri)
        elif hasattr(model, "action_preprocessor"):
            model.action_preprocessor.set_normalizer(normalizer_action, normalizer_propri)
        else:
            raise AttributeError("wall-x model has neither set_normalizer nor action_preprocessor.")

    def setup_default_normalizers(self) -> None:
        """Construct default wall-x normalizers from checkpoint config and constants.

        FLOW checkpoint currently does not ship normalizer_action.pth /
        normalizer_propri.pth in our environment. This mirrors the official dry-run
        fallback based on wall_x.utils.constant.action_statistic_dof, then moves
        the normalizers to the runner device.
        """
        if self._has_normalizers():
            return

        from wall_x.model.action_head import Normalizer
        from wall_x.utils.constant import action_statistic_dof

        raw_cfg = self._load_raw_config()

        # Engine may keep torch default dtype as the runtime params dtype while
        # constructing plugin components. wall-x normalizer statistics are
        # official float32 constants, so construct them under float32 default
        # dtype before moving them to the target device.
        old_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            normalizer_action = Normalizer(action_statistic_dof, raw_cfg["dof_config"])
            normalizer_propri = Normalizer(action_statistic_dof, raw_cfg["agent_pos_config"])
        finally:
            torch.set_default_dtype(old_default_dtype)

        self.set_normalizers(normalizer_action, normalizer_propri)

    @torch.no_grad()
    def forward(self, batch: dict[str, Any]) -> Any:
        """Run the original validate/logits path."""
        model = self._require_model()
        return model(**batch, mode="validate")

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, Any],
        *,
        predict_mode: str = "diffusion",
        action_dim: int,
        action_horizon: int,
    ) -> Any:
        """Run policy-level FLOW action prediction."""
        model = self._require_model()
        self.setup_default_normalizers()
        return model(
            **batch,
            mode="predict",
            predict_mode=predict_mode,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
