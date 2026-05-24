"""FlashInferKernel — bf16 (cublasLt/cuDNN/TGV) + block-fp8 groupwise GEMM.

flashinfer's ``mm_fp8`` ≠ generic fp8 GEMM: it targets ``trtllm_low_latency``
with pre-processed weights and a single alpha scalar. Per-tensor and
per-channel fp8 therefore stay on :class:`TorchKernel` for now; this kernel
covers the paths where flashinfer is unambiguously the best choice:

* bf16 GEMM on sm≥89 (cuBLASLt / cuDNN / TGV, autoselected by flashinfer);
* block-FP8 (DeepSeek-V3 style) on sm≥100 via ``gemm_fp8_nt_groupwise``.

If flashinfer is not installed, :meth:`can_handle` returns ``False`` for
every probe and the fallback :class:`TorchKernel` picks up the work.
"""

from __future__ import annotations

import torch

from phyai.layers.linear.backend import Granularity, KernelProbe
from phyai.layers.linear.registry import register_linear_kernel


try:
    import flashinfer  # noqa: F401
    import flashinfer.gemm as _fi_gemm

    _HAS_FLASHINFER = True
except Exception:  # pragma: no cover — depends on install
    _fi_gemm = None  # type: ignore[assignment]
    _HAS_FLASHINFER = False


@register_linear_kernel(
    prefer_for={
        ("bf16", "prefill"),
        ("fp8_block_128_128", "prefill"),
        ("fp8_block_128_128", "decode"),
    },
)
class FlashInferKernel:
    """bf16 + block-fp8 via flashinfer.gemm.

    For block-fp8 we assume DeepSeek-V3 style weight layout:
    ``layer.weight`` is ``(N, K)`` fp8_e4m3fn, ``layer.weight_scale`` is
    ``(N // bn, K // bk)`` fp32, and ``x`` gets rowwise-quantised to fp8
    with a ``(M, K // bk)`` scale tensor by :meth:`spec.quantize_activation`.

    ``prefer_for`` is attached at decoration time and consulted by
    :class:`phyai.layers.linear.registry.LinearKernelRegistry` —
    everything else falls through to registration order.
    """

    name = "flashinfer"

    def supports_capture(self) -> bool:
        # First-call concerns (JIT, cudnn handle init, backend heuristic)
        # all happen during ``CudaGraph.capture``'s side-stream warmup
        # iterations and are gone by the time we enter the capture region.
        # The captured kernel is a single cuDNN / cuBLAS / cutlass / tgv
        # matmul launch — the Python wrapper's per-call overhead also
        # disappears inside the graph (it only runs at capture time).
        # On sm_100+ this is the only way to land on the cutlass / tgv
        # paths from inside captured runners; on sm_90 it's neutral.
        return True

    def can_handle(self, probe: KernelProbe) -> bool:
        if not _HAS_FLASHINFER:
            return False
        if probe.spec_id == "bf16":
            # cuBLASLt/cuDNN paths cover sm80+; flashinfer's own heuristic
            # picks the right backend at call time.
            return probe.sm >= 80 and probe.in_dtype == torch.bfloat16
        if probe.spec_id.startswith("fp8_block_"):
            # gemm_fp8_nt_groupwise is sm100+ only today.
            return probe.sm >= 100
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
        if spec.spec_id.startswith("fp8_block_"):
            return self._block_fp8(layer, x, bias)
        raise RuntimeError(f"FlashInferKernel got unhandled spec_id={spec.spec_id!r}")

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
        # weight is (N, K) row-major; ``.t()`` is the (K, N) column-major view.
        y = _fi_gemm.mm_bf16(
            x_2d,
            layer.weight.t(),
            bias=bias,
            out_dtype=x.dtype,
        )
        return y.reshape(*x.shape[:-1], -1)

    # ------------------------------------------------------------------
    # block-fp8: gemm_fp8_nt_groupwise
    # ------------------------------------------------------------------

    def _block_fp8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        assert _fi_gemm is not None
        spec = layer.spec
        assert spec.block_shape is not None
        bn, bk = spec.block_shape
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        # Per-token rowwise fp8 activation; spec handles the scale shape.
        act = spec.quantize_activation(x_2d, layer)
        # groupwise GEMM: a (m, k) row-major, b (n, k) col-major.
        y = _fi_gemm.gemm_fp8_nt_groupwise(
            act.x,
            layer.weight,
            a_scale=act.x_scale.reshape(-1),
            b_scale=layer.weight_scale,
            scale_granularity_mnk=(1, bn, bk),
            out_dtype=x.dtype,
        )
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], -1)
