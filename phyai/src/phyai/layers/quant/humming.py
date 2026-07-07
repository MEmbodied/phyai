"""HummingWeightSpec — a physical ``WeightSpec`` backed by the humming library."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from phyai.layers.quant.base import AllocationRequest

# Scale strategies that carry a K-direction group size in the spec_id.
_GROUPED_SCALE = ("group", "block")


@dataclass
class HummingWeightSpec:
    """Physical spec that defers packed layout + GEMM to humming.

    ``w_dtype`` / ``a_dtype`` / ``scale_type`` use humming's canonical strings
    (``"int4"``, ``"float8e4m3"``, ``"bfloat16"``; ``"channel"`` / ``"group"`` /
    ``"block"`` / ``"tensor"``). ``a_dtype`` of a 16-bit float means weight-only
    (activation stays bf16/fp16, no runtime activation quant).
    """

    w_dtype: str
    a_dtype: str = "bfloat16"
    scale_type: str = "channel"
    group_size: int = 0
    group_size_n: int = 0
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    scale_dtype: str | None = None
    # Raw per-layer HF quant config, threaded from the importer (option B1).
    # When set, humming parses it directly (awq/gptq/fp8/compressed-tensors/...).
    raw_config: dict | None = None
    pad_n_to_multiple: int = 256
    pad_k_to_multiple: int = 128
    weight_dtype: torch.dtype = torch.int32
    # Filled at process_after_loading; the JSON compute-config humming wants.
    _compute_config: str = field(default="", repr=False)

    @property
    def spec_id(self) -> str:
        parts = [f"humming_w{self.w_dtype}_a{self.a_dtype}_{self.scale_type}"]
        if self.scale_type in _GROUPED_SCALE and self.group_size:
            group = f"g{self.group_size}"
            if self.scale_type == "block" and self.group_size_n:
                group += f"x{self.group_size_n}"
            parts.append(group)
        if self.has_zero_point:
            parts.append("fpzp" if self.is_fp_zero_point else "zp")
        return "_".join(parts)

    # ------------------------------------------------------------------
    # humming schema construction (imports humming; CUDA host only)
    # ------------------------------------------------------------------

    def _weight_schema(self):
        from phyai.utils.humming import BaseWeightSchema, HummingWeightSchema

        if self.raw_config is not None:
            return BaseWeightSchema.from_config(dict(self.raw_config))
        kwargs: dict = {
            "b_dtype": self.w_dtype,
            "weight_scale_group_size": self.group_size,
            "weight_scale_group_size_n": self.group_size_n,
            "has_zero_point": self.has_zero_point,
            "is_fp_zero_point": self.is_fp_zero_point,
            "weight_scale_type": self.scale_type,
        }
        if self.scale_dtype is not None:
            kwargs["bs_dtype"] = self.scale_dtype
        return HummingWeightSchema(**kwargs)

    def _input_schema(self):
        from phyai.utils.humming import HummingInputSchema

        # 16-bit float activation == weight-only; humming defaults a_dtype to
        # the compute f16 dtype when the input schema is None.
        if self.a_dtype in (None, "bfloat16", "float16"):
            return None
        return HummingInputSchema(a_dtype=self.a_dtype)

    # ------------------------------------------------------------------
    # WeightSpec protocol
    # ------------------------------------------------------------------

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        from phyai.utils.humming import require_humming

        require_humming()
        if len(request.weight_shape) != 2:
            raise ValueError(
                f"HummingWeightSpec.allocate expects a 2-D weight_shape (N, K), "
                f"got {request.weight_shape!r}"
            )
        out_per_rank, in_per_rank = request.weight_shape
        weight_schema = self._weight_schema()
        attrs = weight_schema.get_tensors_attrs(
            shape_n=out_per_rank,
            shape_k=in_per_rank,
            param_dtype=request.params_dtype,
            has_bias=False,
            stack_size=len(request.logical_widths),
        )
        for name, meta in attrs.items():
            param = nn.Parameter(
                torch.empty(
                    tuple(meta["shape"]),
                    dtype=meta["dtype"],
                    device=request.device,
                ),
                requires_grad=False,
            )
            # extra_attrs (input_dim/output_dim/packed_dim/packed_factor/scale_type)
            # is what the TP scale-sharding rework (Phase 2) consumes.
            param._humming_attrs = dict(meta.get("extra_attrs", {}))
            setattr(layer, name, param)

        # Kernel scratch buffer; real device only.
        layer.register_buffer(
            "locks", torch.zeros(1024, dtype=torch.int32, device=request.device)
        )
        layer.logical_widths = list(request.logical_widths)
        # Stash the logical (N, K) and dtype for process_after_loading; after
        # transform_humming_layer the stored weight is padded/opaque, so we
        # must never re-derive N/K from layer.weight.shape.
        layer._humming_in_per_rank = in_per_rank
        layer._humming_out_per_rank = out_per_rank
        layer._humming_params_dtype = request.params_dtype
        layer._humming_spec = self

    def process_after_loading(self, layer: nn.Module) -> None:
        from phyai.utils.humming import (
            HummingMethod,
            HummingWeightSchema,
            require_humming,
        )

        require_humming()
        weight_schema = self._weight_schema()
        input_schema = self._input_schema()

        if not isinstance(weight_schema, HummingWeightSchema):
            # Importer schemas (awq/gptq/fp8/compressed-tensors/modelopt/mxfp4)
            # need convert_humming to turn foreign loaded tensors into native
            # humming tensors.
            raise NotImplementedError(
                "HummingWeightSpec: importer-schema conversion (raw_config with a "
                "non-humming quant_method) lands in Phase 3; pass native humming "
                "identity fields for now."
            )

        shape_n = sum(layer.logical_widths)
        HummingMethod.prepare_layer_meta(
            layer,
            shape_n=shape_n,
            shape_k=layer._humming_in_per_rank,
            weight_schema=weight_schema,
            input_schema=input_schema,
            pad_n_to_multiple=self.pad_n_to_multiple,
            pad_k_to_multiple=self.pad_k_to_multiple,
            has_bias=getattr(layer, "bias", None) is not None,
            torch_dtype=layer._humming_params_dtype,
        )
        HummingMethod.transform_humming_layer(layer)
        self._compute_config = json.dumps(
            {
                "gemm_type": "dense",
                "use_f16_accum": False,
                "use_batch_invariant": False,
            }
        )


__all__ = ["HummingWeightSpec"]
