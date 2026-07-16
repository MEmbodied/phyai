"""HummingWeightSpec — a physical ``WeightSpec`` backed by the humming library."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from phyai.layers.quant.base import AllocationRequest

# Scale strategies that carry a K-direction group size in the spec_id.
_GROUPED_SCALE = ("group", "block")
_RUNTIME_TENSOR_NAMES = {
    "weight_packed": "weight",
    "weight_zero_point": "zero_point",
}


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
    input_group_size: int = 0
    input_dynamic: bool = True
    # Raw per-layer HF quant config, threaded from the importer (option B1).
    # When set, humming parses it directly (awq/gptq/fp8/compressed-tensors/...).
    raw_config: dict | None = None
    input_raw_config: dict | None = None
    online: bool = False
    native_storage: bool = False
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

    def _humming_weight_schema(self):
        from phyai.utils.humming import HummingWeightSchema

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

    def _weight_schema(self):
        from phyai.utils.humming import BaseWeightSchema

        if self.raw_config is not None:
            return BaseWeightSchema.from_config(dict(self.raw_config))
        return self._humming_weight_schema()

    def _input_schema(self):
        from phyai.utils.humming import BaseInputSchema, HummingInputSchema

        if self.input_raw_config is not None:
            return BaseInputSchema.from_config(dict(self.input_raw_config))
        if self.a_dtype in (None, "bfloat16", "float16"):
            return None
        if not self.input_dynamic:
            if self.a_dtype == "float8e4m3":
                return BaseInputSchema.from_config(
                    {
                        "quant_method": "fp8",
                        "activation_scheme": "static",
                    }
                )
            raise ValueError(
                "static Humming activations require a source input schema; "
                f"got a_dtype={self.a_dtype!r}"
            )
        return HummingInputSchema(
            a_dtype=self.a_dtype,
            input_scale_group_size=self.input_group_size,
        )

    @staticmethod
    def _runtime_tensor_name(schema_name: str) -> str:
        return _RUNTIME_TENSOR_NAMES.get(schema_name, schema_name)

    @staticmethod
    def _quantize_online_weight(
        source: torch.Tensor,
        schema,
        param_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        from phyai_kernel import fp8_quantize_weight_per_block

        from humming import dtypes
        from humming.config import WeightScaleType
        from humming.schema.humming import HummingWeightSchema
        from humming.utils.weight import quantize_weight

        if schema.weight_scale_type is WeightScaleType.BLOCK:
            if (
                str(schema.b_dtype) != "float8e4m3"
                or schema.weight_scale_group_size_n != 128
                or schema.weight_scale_group_size != 128
                or schema.has_zero_point
            ):
                raise ValueError(
                    "online Humming block quantization supports symmetric "
                    "FP8 E4M3 weights with block_shape=(128, 128)"
                )
            quant, scale = fp8_quantize_weight_per_block(source, (128, 128))
            tensors = {
                "weight": quant.view(torch.int32),
                "weight_scale": scale,
            }
            schema.validate_tensors(
                tensors,
                source.shape[-2],
                source.shape[-1],
                param_dtype,
                source.shape[0] if source.ndim == 3 else None,
            )
            return tensors

        if schema.weight_scale_type is not WeightScaleType.TENSOR:
            return HummingWeightSchema.quant_tensor(source, schema, param_dtype)

        outputs = quantize_weight(
            weight=source,
            dtype=schema.b_dtype,
            scale_dtype=dtypes.DataType.from_torch_dtype(param_dtype),
            group_size=schema.weight_scale_group_size,
            has_zero_point=schema.has_zero_point,
            has_global_scale=True,
            is_fp_zero_point=schema.is_fp_zero_point,
            pack=True,
        )
        tensors = {
            name: tensor.reshape(1) if name == "global_scale" else tensor
            for name, tensor in zip(
                ("weight", "weight_scale", "zero_point", "global_scale"),
                outputs,
                strict=True,
            )
            if tensor is not None and tensor.numel() > 0
        }
        schema.validate_tensors(
            tensors,
            source.shape[-2],
            source.shape[-1],
            param_dtype,
            source.shape[0] if source.ndim == 3 else None,
        )
        return tensors

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
        weight_attrs = weight_schema.get_tensors_attrs(
            shape_n=out_per_rank,
            shape_k=in_per_rank,
            param_dtype=request.params_dtype,
            has_bias=False,
            stack_size=len(request.logical_widths),
        )
        storage_names: dict[str, str] = {}
        for schema_name, meta in weight_attrs.items():
            name = self._runtime_tensor_name(schema_name)
            if name in storage_names:
                raise RuntimeError(
                    f"humming schema maps both {storage_names[name]!r} and "
                    f"{schema_name!r} to layer parameter {name!r}"
                )
            param = nn.Parameter(
                torch.empty(
                    tuple(meta["shape"]),
                    dtype=meta["dtype"],
                    device=request.device,
                ),
                requires_grad=False,
            )
            extra_attrs = dict(meta.get("extra_attrs", {}))
            if extra_attrs.get("scale_type") == "tensor" and param.numel() == len(
                request.logical_widths
            ):
                extra_attrs["stacked_scalar"] = True
            param._quant_attrs = extra_attrs
            param._quant_source_name = schema_name
            if name == "weight":
                param._accepted_source_dtypes = (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    param.dtype,
                )
            setattr(layer, name, param)
            storage_names[name] = schema_name

        input_storage_names: dict[str, str] = {}
        input_schema = self._input_schema()
        if input_schema is not None:
            input_attrs = input_schema.get_tensors_attrs(
                shape_k=in_per_rank,
                param_dtype=request.params_dtype,
                stack_size=len(request.logical_widths),
            )
            for schema_name, meta in input_attrs.items():
                name = self._runtime_tensor_name(schema_name)
                if hasattr(layer, name):
                    raise RuntimeError(
                        f"humming input tensor {schema_name!r} collides with "
                        f"weight tensor {name!r}"
                    )
                param = nn.Parameter(
                    torch.empty(
                        tuple(meta["shape"]),
                        dtype=meta["dtype"],
                        device=request.device,
                    ),
                    requires_grad=False,
                )
                extra_attrs = dict(meta.get("extra_attrs", {}))
                if param.numel() == len(request.logical_widths):
                    extra_attrs["stacked_scalar"] = True
                param._quant_attrs = extra_attrs
                param._quant_source_name = schema_name
                setattr(layer, name, param)
                input_storage_names[name] = schema_name

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
        layer._quant_storage_names = storage_names
        layer._quant_input_storage_names = input_storage_names
        layer._humming_pending_weight = None
        layer._humming_logical_weight_shape = (out_per_rank, in_per_rank)

    def process_after_loading(self, layer: nn.Module) -> None:
        from phyai.utils.humming import (
            HummingInputSchema,
            HummingMethod,
            HummingWeightSchema,
            require_humming,
        )

        if getattr(layer, "_humming_processed", False):
            return
        require_humming()
        weight_schema = self._weight_schema()
        input_schema = self._input_schema()

        pending = getattr(layer, "_humming_pending_weight", None)
        if pending is not None:
            target_schema = self._humming_weight_schema()
            source = pending.detach().to(device=layer.weight.device).contiguous()
            if not source.is_cuda:
                raise RuntimeError("online Humming weight quantization requires CUDA")
            if not torch.isfinite(source).all():
                raise ValueError("online Humming source weight must be finite")
            tensors = self._quantize_online_weight(
                source,
                target_schema,
                layer._humming_params_dtype,
            )
            for runtime_name in tuple(layer._quant_storage_names):
                if runtime_name not in tensors and hasattr(layer, runtime_name):
                    delattr(layer, runtime_name)
            for name, tensor in tensors.items():
                setattr(layer, name, nn.Parameter(tensor, requires_grad=False))
            layer._quant_storage_names = {name: name for name in tensors}
            layer._humming_pending_weight = None
            weight_schema = target_schema
        elif not isinstance(weight_schema, HummingWeightSchema):
            tensors = {
                schema_name: getattr(layer, runtime_name).detach()
                for runtime_name, schema_name in layer._quant_storage_names.items()
            }
            weight_schema, tensors = weight_schema.convert_humming(
                tensors,
                shape_n_stacks=list(layer.logical_widths),
                shape_k_stacks=[layer._humming_in_per_rank],
                param_dtype=layer._humming_params_dtype,
            )
            for runtime_name in layer._quant_storage_names:
                if runtime_name not in tensors and hasattr(layer, runtime_name):
                    delattr(layer, runtime_name)
            for name, tensor in tensors.items():
                setattr(layer, name, nn.Parameter(tensor, requires_grad=False))
            layer._quant_storage_names = {name: name for name in tensors}

        for name in ("weight_scale", "global_scale"):
            tensor = getattr(layer, name, None)
            if tensor is None:
                continue
            values = tensor.detach().float()
            invalid_zero = (values == 0).any()
            invalid_sign = name == "global_scale" and (values < 0).any()
            if not torch.isfinite(values).all() or invalid_zero or invalid_sign:
                requirement = (
                    "strictly positive" if name == "global_scale" else "nonzero"
                )
                raise ValueError(f"Humming {name} must be finite and {requirement}")

        static_input_scale: torch.Tensor | None = None
        if input_schema is not None and not isinstance(
            input_schema, HummingInputSchema
        ):
            input_tensors = {
                schema_name: getattr(layer, runtime_name).detach()
                for runtime_name, schema_name in layer._quant_input_storage_names.items()
            }
            for zero_name in ("input_zero_point",):
                zero = input_tensors.get(zero_name)
                if zero is not None and torch.count_nonzero(zero):
                    raise ValueError(
                        "Humming static activation quantization does not support "
                        "a nonzero input zero point"
                    )
            for scale_name in ("input_scale", "input_global_scale"):
                scale = input_tensors.get(scale_name)
                if scale is None:
                    continue
                scale = scale.detach().float().reshape(-1)
                if not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError(
                        f"Humming {scale_name} must be finite and strictly positive"
                    )
                static_input_scale = scale.max().reshape(1).contiguous()
                break

            sm_version = (
                torch.cuda.get_device_capability(layer.weight.device)
                if layer.weight.is_cuda
                else None
            )
            input_schema, _ = input_schema.convert_humming(
                input_tensors,
                shape_n_stacks=list(layer.logical_widths),
                shape_k_stacks=[layer._humming_in_per_rank],
                param_dtype=layer._humming_params_dtype,
                sm_version=sm_version,
            )
            for runtime_name in tuple(layer._quant_input_storage_names):
                if hasattr(layer, runtime_name):
                    delattr(layer, runtime_name)
            layer._quant_input_storage_names = {}

        shape_n = sum(layer.logical_widths)
        is_row_parallel = hasattr(layer, "input_is_parallel")
        row_adds_bias = not is_row_parallel or getattr(layer, "tp_rank", 0) == 0
        fuse_bias = bool(
            getattr(layer, "bias", None) is not None
            and not getattr(layer, "skip_bias_add", False)
            and row_adds_bias
        )
        HummingMethod.prepare_layer_meta(
            layer,
            shape_n=shape_n,
            shape_k=layer._humming_in_per_rank,
            weight_schema=weight_schema,
            input_schema=input_schema,
            pad_n_to_multiple=self.pad_n_to_multiple,
            pad_k_to_multiple=self.pad_k_to_multiple,
            has_bias=fuse_bias,
            torch_dtype=layer._humming_params_dtype,
        )
        HummingMethod.transform_humming_layer(layer)
        layer._humming_bias_fused = fuse_bias
        layer.register_buffer(
            "_humming_static_input_scale",
            static_input_scale,
            persistent=False,
        )
        self._compute_config = json.dumps(
            {
                "gemm_type": "dense",
                "use_f16_accum": False,
                "use_batch_invariant": False,
            }
        )
        layer._humming_processed = True

    def load_weight(
        self,
        layer: nn.Module,
        loaded: torch.Tensor,
        shard_id: "int | str | None",
        default_loader: Callable[[nn.Parameter, torch.Tensor, object], None],
    ) -> None:
        if loaded.dtype in (torch.float16, torch.bfloat16, torch.float32):
            online = self.online or (
                self.raw_config is None and not self.native_storage
            )
            if not online:
                raise TypeError(
                    "serialized Humming weight source has high-precision dtype "
                    f"{loaded.dtype}; declare an online QuantScheme instead"
                )
            pending = getattr(layer, "_humming_pending_weight", None)
            if pending is None:
                pending = torch.empty(
                    layer._humming_logical_weight_shape,
                    dtype=loaded.dtype,
                    device=loaded.device,
                )
                layer._humming_pending_weight = pending
            elif pending.dtype != loaded.dtype:
                raise TypeError(
                    "all fused high-precision Humming source weights must share a dtype"
                )
            default_loader(pending, loaded, shard_id)
            return
        if loaded.dtype != layer.weight.dtype:
            raise TypeError(
                f"Humming serialized weight expects {layer.weight.dtype}, "
                f"got {loaded.dtype}"
            )
        default_loader(layer.weight, loaded, shard_id)


__all__ = ["HummingWeightSpec"]
