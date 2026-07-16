"""FlashInfer linear kernels.

FP8 support is deliberately capability-gated. Tensorwise FP8 uses the
low-latency TRTLLM ``mm_fp8`` path only for small M on SM100/103 and the
scalar-scale ``bmm_fp8`` path otherwise. Block-128 FP8 uses the Hopper
blockscale runner on SM90, CUTLASS on SM100/103/120/121, and cuTile on
SM110. There is no implicit Torch FP8 fallback: an unsupported shape raises
at dispatch time.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import torch
from phyai_kernel import nvfp4_scale_output

from phyai.engine_config import get_engine_config
from phyai.layers.linear.backend import KernelProbe
from phyai.layers.linear.registry import register_linear_kernel
from phyai.layers.quant import ActivationView, Granularity
from phyai.layers.quant.nvfp4 import flashinfer_nvfp4_e4m3_max
from phyai.parallel.state import Mode, current_mode


try:
    import flashinfer  # noqa: F401
    import flashinfer.gemm as _fi_gemm
except Exception:  # pragma: no cover — depends on install
    flashinfer = None  # type: ignore[assignment]
    _fi_gemm = None  # type: ignore[assignment]
    _HAS_FLASHINFER = False
else:
    _HAS_FLASHINFER = True

if _HAS_FLASHINFER:
    try:
        from flashinfer import (
            prepare_low_latency_gemm_weights as _fi_prepare_low_latency,
        )
    except (AttributeError, ImportError):  # pragma: no cover - version dependent
        _fi_prepare_low_latency = None  # type: ignore[assignment]

    try:
        from flashinfer.quantization import SfLayout as _FiSfLayout
        from flashinfer.quantization import nvfp4_quantize as _fi_nvfp4_quantize
    except (AttributeError, ImportError):  # pragma: no cover - version dependent
        _FiSfLayout = None  # type: ignore[assignment]
        _fi_nvfp4_quantize = None  # type: ignore[assignment]

    try:
        from flashinfer.gemm import is_cuda_tile_available as _fi_is_cutile_available
    except (AttributeError, ImportError):  # pragma: no cover - version dependent
        _fi_is_cutile_available = None  # type: ignore[assignment]

    try:
        from flashinfer.gemm.gemm_base import (
            get_fp8_blockscale_gemm_runner_sm90 as _fi_get_fp8_blockscale_runner_sm90,
        )
    except (AttributeError, ImportError):  # pragma: no cover - version dependent
        _fi_get_fp8_blockscale_runner_sm90 = None  # type: ignore[assignment]

    _fi_autotune = getattr(flashinfer, "autotune", None)
    try:
        from flashinfer.utils import _get_cache_buf as _fi_get_cache_buf
    except (AttributeError, ImportError):  # pragma: no cover - version dependent
        _fi_get_cache_buf = None  # type: ignore[assignment]
else:
    _fi_prepare_low_latency = None  # type: ignore[assignment]
    _FiSfLayout = None  # type: ignore[assignment]
    _fi_nvfp4_quantize = None  # type: ignore[assignment]
    _fi_is_cutile_available = None  # type: ignore[assignment]
    _fi_get_fp8_blockscale_runner_sm90 = None  # type: ignore[assignment]
    _fi_autotune = None  # type: ignore[assignment]
    _fi_get_cache_buf = None  # type: ignore[assignment]


# TinyGEMM is a latency kernel for tiny row counts (FlashInfer documents an
# ideal range of 1-8). Letting ``auto`` profile it at vision/LLM row counts can
# take minutes for one shape. Large-M problems still tune exact cuDNN tactics.
_AUTO_TINYGEMM_MAX_M = 8
_BF16_AUTOTUNE_WORKSPACE_BYTES = 64 * 1024 * 1024
_FP8_WORKSPACE_BYTES = 64 * 1024 * 1024
_FP8_MM_MAX_M = 16
_FP8_MM_SMS = frozenset({100, 103})
_FP8_BMM_CUBLAS_SMS = frozenset({89, 90})
_FP8_BMM_CUTLASS_SMS = frozenset({100, 103, 110, 120, 121})
_FP8_BLOCK_SM90_SMS = frozenset({90})
_FP8_BLOCK_CUTLASS_SMS = frozenset({100, 103, 120, 121})
_FP8_BLOCK_CUTILE_SMS = frozenset({110})
_NVFP4_SMS = frozenset({100, 103, 110, 120, 121})
_NVFP4_MAX = 6.0
_CUTILE_AVAILABLE: bool | None = None


def _cutile_available() -> bool:
    global _CUTILE_AVAILABLE
    if _CUTILE_AVAILABLE is not None:
        return _CUTILE_AVAILABLE
    if _fi_is_cutile_available is None:
        return False
    if current_mode() is Mode.GRAPH_CAPTURING:
        return False
    try:
        _CUTILE_AVAILABLE = bool(_fi_is_cutile_available())
    except Exception:
        _CUTILE_AVAILABLE = False
    return _CUTILE_AVAILABLE


def _runtime_sm(x: torch.Tensor) -> int:
    if x.device.type != "cuda":
        return 0
    major, minor = torch.cuda.get_device_capability(x.device)
    return major * 10 + minor


def _scalar_scale(scale: torch.Tensor, *, name: str) -> torch.Tensor:
    if scale.numel() != 1:
        raise RuntimeError(
            f"FlashInfer scalar-scale GEMM requires one {name} scale, "
            f"got shape {tuple(scale.shape)}"
        )
    return scale.reshape(1).to(dtype=torch.float32).contiguous()


def _nvfp4_per_token_base_scale_inv() -> float:
    # note(chenghua): keep this a Python scalar to avoid FlashInfer's tensor
    # ``.item()`` sync. The helper also tracks its optional 4-over-6 mode.
    return 1.0 / (flashinfer_nvfp4_e4m3_max() * _NVFP4_MAX)


def _gemm_api_available(name: str) -> bool:
    return _fi_gemm is not None and callable(getattr(_fi_gemm, name, None))


def flashinfer_supported_specs_for_sm(sm: int) -> set[str]:
    """Return logical specs supported by the installed FlashInfer runtime."""
    if not _HAS_FLASHINFER:
        return set()

    specs: set[str] = set()
    if sm >= 80 and _gemm_api_available("mm_bf16"):
        specs.add("bf16")

    if sm in (_FP8_BMM_CUBLAS_SMS | _FP8_BMM_CUTLASS_SMS):
        if _gemm_api_available("bmm_fp8"):
            specs.add("fp8_per_tensor")

    block_available = False
    if sm in _FP8_BLOCK_SM90_SMS:
        block_available = callable(_fi_get_fp8_blockscale_runner_sm90)
    elif sm in _FP8_BLOCK_CUTLASS_SMS:
        block_available = _gemm_api_available("gemm_fp8_nt_groupwise")
    elif sm in _FP8_BLOCK_CUTILE_SMS:
        block_available = (
            _gemm_api_available("gemm_fp8_nt_groupwise") and _cutile_available()
        )
    if block_available:
        specs.add("fp8_block_128_128")

    if (
        sm in _NVFP4_SMS
        and _gemm_api_available("mm_fp4")
        and _fi_nvfp4_quantize is not None
        and _FiSfLayout is not None
    ):
        specs.add("nvfp4_block_16_128x4")
    return specs


@register_linear_kernel(
    prefer_for={
        ("bf16", "prefill"),
        ("fp8_per_tensor", "decode"),
        ("fp8_per_tensor", "prefill"),
        ("fp8_block_128_128", "prefill"),
        ("fp8_block_128_128", "decode"),
        ("nvfp4_block_16_128x4", "prefill"),
        ("nvfp4_block_16_128x4", "decode"),
    },
)
class FlashInferKernel:
    """bf16 + block-fp8 + NVFP4 via flashinfer.gemm.

    For block-fp8 we assume DeepSeek-V3 style weight layout:
    ``layer.weight`` is ``(N, K)`` fp8_e4m3fn, ``layer.weight_scale`` is
    ``(N // bn, K // bk)`` fp32. The SM90 runner consumes BF16 ``x`` and
    quantizes internally; Blackwell paths consume rowwise-quantized FP8 ``x``
    with a ``(M, K // bk)`` scale tensor from
    :meth:`spec.quantize_activation`.

    For NVFP4, ``layer.weight`` is packed ``(N, K // 2)`` uint8,
    ``layer.weight_scale`` uses FlashInfer's 128x4 layout, and
    ``layer.weight_global_scale`` is the per-tensor descale factor. Activation
    quantization is online and per-token; a Triton epilogue applies the row
    decode scale and optional bias after dense ``mm_fp4``.

    ``prefer_for`` is attached at decoration time and consulted by
    :class:`phyai.layers.linear.registry.LinearKernelRegistry` —
    everything else falls through to registration order.
    """

    name = "flashinfer"

    def __init__(self) -> None:
        self._nvfp4_per_token_scale_inv: float | None = None
        self._bf16_tuned_shapes: set[tuple[object, ...]] = set()
        self._bf16_tune_lock = Lock()
        self._fp8_ready_shapes: set[tuple[object, ...]] = set()
        self._fp8_tune_lock = Lock()
        self._nvfp4_ready_shapes: set[tuple[object, ...]] = set()
        self._nvfp4_warmup_lock = Lock()
        self._low_latency_permutation_cache: dict[Any, torch.Tensor] = {}
        self._sm90_block_runner: Any | None = None
        self._sm90_block_workspaces: dict[tuple[object, ...], torch.Tensor] = {}

    def supports_capture(self) -> bool:
        # note(chenghua): Eager warmup resolves JIT, handles, workspaces, and
        # backend heuristics before capture records each backend's launch sequence.
        return True

    def can_handle(self, probe: KernelProbe) -> bool:
        supported_specs = flashinfer_supported_specs_for_sm(probe.sm)
        if probe.spec_id not in supported_specs:
            return False
        if probe.spec_id == "bf16":
            return (
                probe.in_dtype == torch.bfloat16 and probe.out_dtype == torch.bfloat16
            )
        if probe.spec_id == "fp8_per_tensor":
            if probe.in_dtype != torch.bfloat16 or probe.out_dtype != torch.bfloat16:
                return False
            if probe.M is None:
                return False
            if probe.N % 16 != 0 or probe.K % 16 != 0:
                return False
            return True
        if probe.spec_id == "fp8_block_128_128":
            if probe.in_dtype != torch.bfloat16 or probe.out_dtype != torch.bfloat16:
                return False
            if probe.N % 128 != 0 or probe.K % 128 != 0:
                return False
            return probe.M is not None
        if probe.spec_id == "nvfp4_block_16_128x4":
            return (
                probe.in_dtype == torch.bfloat16
                and probe.out_dtype == torch.bfloat16
                and probe.N % 32 == 0
                and probe.K % 32 == 0
            )
        return False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = layer.spec
        if spec.spec_id == "bf16":
            return self._bf16(layer, x, bias)
        if spec.spec_id == "fp8_per_tensor":
            return self._tensorwise_fp8(layer, x, bias)
        if spec.spec_id.startswith("fp8_block_"):
            return self._block_fp8(layer, x, bias)
        if spec.spec_id == "nvfp4_block_16_128x4":
            return self._nvfp4(layer, x, bias)
        raise RuntimeError(f"FlashInferKernel got unhandled spec_id={spec.spec_id!r}")

    def apply_prequantized(
        self,
        layer: torch.nn.Module,
        activation: ActivationView,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if not layer.spec.spec_id.startswith("fp8_block_"):
            raise RuntimeError(
                "FlashInfer prequantized activation path requires block FP8 weights"
            )
        prefix = activation.x.shape[:-1]
        y = self._block_fp8_from_activation(
            layer,
            activation,
            bias,
            out_dtype=layer.params_dtype,
        )
        return y.reshape(*prefix, -1)

    # ------------------------------------------------------------------
    # bf16: mm_bf16(a (M,K) row, b (K,N) col, bias (N,))
    # ------------------------------------------------------------------

    def _bf16(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        assert _fi_gemm is not None
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        runtime = get_engine_config().runtime
        if runtime.flashinfer_autotune and x_2d.is_cuda:
            if _fi_get_cache_buf is None:
                raise RuntimeError(
                    "flashinfer_autotune requires FlashInfer's GEMM "
                    "workspace cache API."
                )
            # cuDNN tactics may grow FlashInfer's default 32 MiB buffer by a
            # few bytes.  Reserve once before any graph is captured so the
            # workspace shape stays in the autotuner key and its storage
            # pointer remains stable across all PI0.5 graph captures.
            _fi_get_cache_buf(
                "mm_bf16_workspace",
                _BF16_AUTOTUNE_WORKSPACE_BYTES,
                x_2d.device,
            )
        configured_backend = runtime.flashinfer_bf16_backend
        backend = (
            "cudnn"
            if configured_backend == "auto" and x_2d.shape[0] > _AUTO_TINYGEMM_MAX_M
            else configured_backend
        )
        tune_key = (
            backend,
            x_2d.device.type,
            x_2d.device.index,
            tuple(x_2d.shape),
            tuple(x_2d.stride()),
            tuple(layer.weight.shape),
            tuple(layer.weight.stride()),
            x_2d.dtype,
            bias is not None,
        )

        def run() -> torch.Tensor:
            return _fi_gemm.mm_bf16(
                x_2d,
                layer.weight.t(),
                bias=bias,
                out_dtype=x.dtype,
                backend=backend,
            )

        def run_tuned(*, tune_mode: bool) -> torch.Tensor:
            assert _fi_autotune is not None
            with _fi_autotune(
                tune_mode,
                tuning_buckets=(int(x_2d.shape[0]),),
            ):
                return run()

        if not runtime.flashinfer_autotune:
            y = run()
        elif current_mode() is Mode.GRAPH_CAPTURING:
            if tune_key not in self._bf16_tuned_shapes:
                raise RuntimeError(
                    "FlashInfer BF16 GEMM reached CUDA-graph capture before its "
                    f"shape was tuned: M={x_2d.shape[0]}, N={layer.weight.shape[0]}, "
                    f"K={x_2d.shape[1]}, backend={backend!r}. Run at least one "
                    "eager warmup iteration before capture."
                )
            y = run_tuned(tune_mode=False)
        elif tune_key in self._bf16_tuned_shapes:
            y = run_tuned(tune_mode=False)
        else:
            if _fi_autotune is None:
                raise RuntimeError(
                    "flashinfer_autotune requires a FlashInfer release "
                    "that exports flashinfer.autotune."
                )
            with self._bf16_tune_lock:
                if tune_key in self._bf16_tuned_shapes:
                    y = run_tuned(tune_mode=False)
                else:
                    y = run_tuned(tune_mode=True)
                    self._bf16_tuned_shapes.add(tune_key)
        return y.reshape(*x.shape[:-1], -1)

    # ------------------------------------------------------------------
    # tensorwise FP8: TRTLLM low latency for small M, BMM otherwise
    # ------------------------------------------------------------------

    @staticmethod
    def _fp8_shape_key(
        kind: str,
        x: torch.Tensor,
        layer: torch.nn.Module,
        *,
        backend: str | None = None,
    ) -> tuple[object, ...]:
        x_2d = x.reshape(-1, x.shape[-1])
        return (
            kind,
            backend,
            x.device.type,
            x.device.index,
            tuple(x_2d.shape),
            tuple(x_2d.stride()),
            tuple(layer.weight.shape),
            tuple(layer.weight.stride()),
            x_2d.dtype,
            layer.weight.dtype,
        )

    def _ensure_fp8_workspace(self, name: str, device: torch.device) -> None:
        if _fi_get_cache_buf is not None and device.type == "cuda":
            _fi_get_cache_buf(name, _FP8_WORKSPACE_BYTES, device)

    def _run_fp8(
        self,
        key: tuple[object, ...],
        fn,
        *,
        workspace_name: str | None,
        device: torch.device,
    ) -> torch.Tensor:
        runtime = get_engine_config().runtime
        ready = key in self._fp8_ready_shapes
        if current_mode() is Mode.GRAPH_CAPTURING and not ready:
            raise RuntimeError(
                "FlashInfer FP8 reached CUDA-graph capture before eager "
                f"warmup for shape key={key!r}"
            )
        if not ready and workspace_name is not None:
            self._ensure_fp8_workspace(workspace_name, device)

        use_autotune = bool(runtime.flashinfer_autotune)
        if use_autotune and _fi_autotune is None:
            raise RuntimeError(
                "flashinfer_autotune requires FlashInfer's autotune API for FP8"
            )

        def invoke(tune_mode: bool | None) -> torch.Tensor:
            if tune_mode is None:
                return fn()
            assert _fi_autotune is not None
            with _fi_autotune(tune_mode):
                return fn()

        if ready:
            return invoke(False if use_autotune else None)

        with self._fp8_tune_lock:
            if key in self._fp8_ready_shapes:
                return invoke(False if use_autotune else None)
            result = invoke(True if use_autotune else None)
            self._fp8_ready_shapes.add(key)
            return result

    def _prepared_low_latency_weight(
        self,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        if _fi_prepare_low_latency is None:
            raise RuntimeError("FlashInfer mm_fp8 weight preparation is unavailable")
        weight = layer.weight
        source = (
            weight.data_ptr(),
            getattr(weight, "_version", None),
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.device,
        )
        cached_source = getattr(layer, "_flashinfer_fp8_weight_source", None)
        cached_weight = getattr(layer, "_flashinfer_fp8_prepared_weight", None)
        if cached_source == source and cached_weight is not None:
            return cached_weight
        if current_mode() is Mode.GRAPH_CAPTURING:
            raise RuntimeError(
                "FlashInfer mm_fp8 weight preparation must finish before "
                "CUDA-graph capture"
            )
        prepared = _fi_prepare_low_latency(
            weight,
            self._low_latency_permutation_cache,
        )
        # note(chenghua): Keep the transformed layout alive without registering
        # it as a second parameter or broadcast weight.
        layer._flashinfer_fp8_prepared_weight = prepared
        layer._flashinfer_fp8_weight_source = source
        return prepared

    def _tensorwise_fp8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        spec = layer.spec
        act = spec.quantize_activation(x_2d, layer)
        if act.x_scale is None:
            raise RuntimeError(
                "tensorwise FP8 activation quantization returned no scale"
            )
        a_scale = _scalar_scale(act.x_scale, name="activation")
        b_scale = _scalar_scale(layer.weight_scale, name="weight")
        alpha = (a_scale * b_scale).reshape(1).contiguous()
        sm = _runtime_sm(x)
        M = x_2d.shape[0]
        use_low_latency = (
            sm in _FP8_MM_SMS
            and M <= _FP8_MM_MAX_M
            and layer.weight.shape[0] % 128 == 0
            and K % 128 == 0
            and _fi_prepare_low_latency is not None
            and _gemm_api_available("mm_fp8")
        )
        if use_low_latency:
            prepared = self._prepared_low_latency_weight(layer)
            backend = "trtllm_low_latency"
            key = self._fp8_shape_key("fp8_mm", act.x, layer, backend=backend)

            def run_mm() -> torch.Tensor:
                assert _fi_gemm is not None
                return _fi_gemm.mm_fp8(
                    act.x.reshape(M, K),
                    prepared,
                    alpha,
                    out_dtype=x.dtype,
                    backend=backend,
                )

            y = self._run_fp8(
                key,
                run_mm,
                workspace_name=None,
                device=x.device,
            )
        elif sm in (_FP8_BMM_CUBLAS_SMS | _FP8_BMM_CUTLASS_SMS):
            backend = "cublas" if sm in _FP8_BMM_CUBLAS_SMS else "cutlass"
            key = self._fp8_shape_key("fp8_bmm", act.x, layer, backend=backend)

            def run_bmm() -> torch.Tensor:
                assert _fi_gemm is not None
                a = act.x.reshape(1, M, K)
                b = layer.weight.t().unsqueeze(0)
                y_batched = _fi_gemm.bmm_fp8(
                    a,
                    b,
                    a_scale,
                    b_scale,
                    x.dtype,
                    backend=backend,
                )
                return y_batched.squeeze(0)

            y = self._run_fp8(
                key,
                run_bmm,
                workspace_name="bmm_fp8_workspace",
                device=x.device,
            )
        else:
            raise RuntimeError(f"no tensorwise FP8 FlashInfer path for sm={sm}, M={M}")
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], -1)

    # ------------------------------------------------------------------
    # note(chenghua): Hopper quantizes BF16 input inside its runner, while
    # Blackwell groupwise kernels require pre-quantized FP8 activations.
    # ------------------------------------------------------------------

    def _block_fp8_sm90(
        self,
        layer: torch.nn.Module,
        x_2d: torch.Tensor,
        x_scale: torch.Tensor | None,
        *,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if _fi_get_fp8_blockscale_runner_sm90 is None:
            raise RuntimeError("FlashInfer SM90 blockscale runner is unavailable")

        M, K = x_2d.shape
        N = layer.weight.shape[0]
        backend = (
            "sm90_blockscale_fp8_input"
            if x_scale is not None
            else "sm90_blockscale_bf16_input"
        )
        key = self._fp8_shape_key(
            "fp8_blockscale",
            x_2d,
            layer,
            backend=backend,
        )
        if current_mode() is Mode.GRAPH_CAPTURING and key not in self._fp8_ready_shapes:
            raise RuntimeError(
                "FlashInfer SM90 block FP8 reached CUDA-graph capture before "
                f"eager runner/workspace warmup for shape key={key!r}"
            )

        with self._fp8_tune_lock:
            runner = self._sm90_block_runner
            if runner is None:
                runner = _fi_get_fp8_blockscale_runner_sm90()
                self._sm90_block_runner = runner

            workspace = self._sm90_block_workspaces.get(key)
            if workspace is None:
                if current_mode() is Mode.GRAPH_CAPTURING:
                    raise RuntimeError(
                        "FlashInfer SM90 block FP8 workspace must be allocated "
                        "during eager warmup"
                    )
                workspace_size = int(runner.get_workspace_size(M, N, K))
                workspace = torch.empty(
                    max(workspace_size, 1),
                    dtype=torch.uint8,
                    device=x_2d.device,
                )
                self._sm90_block_workspaces[key] = workspace

            runner.configure_workspace(workspace)
            out = torch.empty((M, N), dtype=out_dtype, device=x_2d.device)
            runner.run_gemm(
                x_2d,
                layer.weight,
                out,
                x_scale,
                layer.weight_scale,
            )
            self._fp8_ready_shapes.add(key)
        return out

    def _block_fp8_from_activation(
        self,
        layer: torch.nn.Module,
        activation: ActivationView,
        bias: torch.Tensor | None,
        *,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if activation.granularity is not Granularity.BLOCK:
            raise RuntimeError(
                "prequantized block FP8 Linear requires block-granularity activation"
            )
        act_x = activation.x
        act_scale = activation.x_scale
        if act_x.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"block FP8 activation must be E4M3, got {act_x.dtype}")
        if act_scale is None or act_scale.dtype is not torch.float32:
            raise TypeError("block FP8 activation requires FP32 scales")
        if not act_x.is_contiguous():
            raise ValueError("prequantized block FP8 activation must be contiguous")

        K = act_x.shape[-1]
        x_2d = act_x.reshape(-1, K)
        M = x_2d.shape[0]
        if K != layer.weight.shape[1]:
            raise RuntimeError(
                f"block FP8 activation K={K} does not match weight K={layer.weight.shape[1]}"
            )
        if act_scale.shape != (M, K // 128):
            raise RuntimeError(
                f"unexpected block FP8 activation scale shape {tuple(act_scale.shape)}; "
                f"expected {(M, K // 128)}"
            )

        sm = _runtime_sm(x_2d)
        if sm in _FP8_BLOCK_SM90_SMS:
            expected_scale_stride = (1, (M + 3) // 4 * 4)
            if act_scale.stride() != expected_scale_stride:
                raise ValueError(
                    "FlashInfer SM90 prequantized block FP8 requires activation "
                    f"scale stride {expected_scale_stride}, got {act_scale.stride()}"
                )
            y = self._block_fp8_sm90(
                layer,
                x_2d,
                act_scale,
                out_dtype=out_dtype,
            )
            if bias is not None:
                y = y + bias
            return y

        assert _fi_gemm is not None
        if not act_scale.is_contiguous():
            raise ValueError(
                "FlashInfer Blackwell block FP8 requires row-major activation scales"
            )
        padded_m = (M + 3) // 4 * 4
        if padded_m != M:
            # note(chenghua): cuTile/CUTLASS require M%4. These allocations are
            # captured and replayed with stable storage.
            act_x_padded = torch.empty(
                (padded_m, K), dtype=act_x.dtype, device=act_x.device
            )
            act_x_padded.zero_()
            act_x_padded[:M].copy_(act_x)
            act_scale_padded = torch.ones(
                (padded_m, K // 128), dtype=act_scale.dtype, device=act_scale.device
            )
            act_scale_padded[:M].copy_(act_scale)
            act_x, act_scale = act_x_padded, act_scale_padded

        if sm in _FP8_BLOCK_CUTILE_SMS:
            backend = "cutile"
        elif sm in _FP8_BLOCK_CUTLASS_SMS:
            backend = "cutlass"
        else:
            raise RuntimeError(f"no block FP8 FlashInfer path for sm={sm}")
        key = self._fp8_shape_key(
            "fp8_groupwise",
            act_x,
            layer,
            backend=backend,
        )

        def run_groupwise() -> torch.Tensor:
            return _fi_gemm.gemm_fp8_nt_groupwise(
                act_x,
                layer.weight,
                a_scale=act_scale,
                b_scale=layer.weight_scale,
                scale_major_mode="K",
                mma_sm=2 if padded_m >= 256 else 1,
                scale_granularity_mnk=(1, 128, 128),
                out_dtype=out_dtype,
                backend=backend,
            )

        y = self._run_fp8(
            key,
            run_groupwise,
            workspace_name="gemm_fp8_nt_groupwise_workspace",
            device=act_x.device,
        )[:M]
        if bias is not None:
            y = y + bias
        return y

    def _block_fp8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = layer.spec
        if spec.block_shape != (128, 128):
            raise RuntimeError(
                "FlashInfer block FP8 currently supports block_shape=(128, 128)"
            )
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        sm = _runtime_sm(x)
        if sm in _FP8_BLOCK_SM90_SMS:
            y = self._block_fp8_sm90(
                layer,
                x_2d,
                None,
                out_dtype=x.dtype,
            )
            if bias is not None:
                y = y + bias
            return y.reshape(*x.shape[:-1], -1)

        act = spec.quantize_activation(x_2d, layer)
        y = self._block_fp8_from_activation(
            layer,
            act,
            bias,
            out_dtype=x.dtype,
        )
        return y.reshape(*x.shape[:-1], -1)

    # ------------------------------------------------------------------
    # nvfp4: mm_fp4 with 128x4 scale-factor layout
    # ------------------------------------------------------------------

    def _nvfp4(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise RuntimeError(
                "online dense NVFP4 requires BF16 input/output; applying the "
                "per-token decode scale after an FP16 GEMM can overflow"
            )
        assert _fi_gemm is not None
        assert _fi_nvfp4_quantize is not None
        assert _FiSfLayout is not None
        if self._nvfp4_per_token_scale_inv is None:
            self._nvfp4_per_token_scale_inv = _nvfp4_per_token_base_scale_inv()
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        key = (
            "nvfp4",
            x.device.type,
            x.device.index,
            tuple(x_2d.shape),
            tuple(x_2d.stride()),
            tuple(layer.weight.shape),
            tuple(layer.weight.stride()),
            x.dtype,
            bias is not None,
        )
        if (
            current_mode() is Mode.GRAPH_CAPTURING
            and key not in self._nvfp4_ready_shapes
        ):
            raise RuntimeError(
                "FlashInfer NVFP4 reached CUDA-graph capture before eager "
                f"warmup for shape key={key!r}"
            )

        def run() -> torch.Tensor:
            act_x, act_scale, per_token_scale = _fi_nvfp4_quantize(
                x_2d,
                self._nvfp4_per_token_scale_inv,
                sfLayout=_FiSfLayout.layout_128x4,
                do_shuffle=False,
                enable_pdl=None,
                backend="cuda",
                per_token_activation=True,
            )
            output = _fi_gemm.mm_fp4(
                act_x,
                layer.weight.t(),
                act_scale,
                layer.weight_scale.t().view(torch.uint8),
                _scalar_scale(layer.weight_global_scale, name="NVFP4 weight global"),
                x.dtype,
                None,
                block_size=16,
                use_nvfp4=True,
                backend="auto",
            )
            # note(chenghua): mm_fp4 accepts only scalar alpha, so the dynamic
            # row decode and bias share one epilogue launch.
            nvfp4_scale_output(output, per_token_scale, bias)
            return output

        if key in self._nvfp4_ready_shapes:
            y = run()
        else:
            with self._nvfp4_warmup_lock:
                if key in self._nvfp4_ready_shapes:
                    y = run()
                else:
                    y = run()
                    self._nvfp4_ready_shapes.add(key)
        return y.reshape(*x.shape[:-1], -1)
