"""FlashInfer-backed :class:`GreenCtxBackend` (default).

Wraps :func:`flashinfer.green_ctx.split_device_green_ctx` and
:func:`flashinfer.green_ctx.split_device_green_ctx_by_sm_count`. Provides
``create_single`` by way of ``split_by_count(num_groups=1, ...)`` and a
best-effort ``destroy`` that calls ``cuStreamDestroy + cuGreenCtxDestroy``
on the stream's bound green ctx.

flashinfer's ``split_device_*`` leaks driver memory on every call — none
of ``cuStreamDestroy``, ``cuGreenCtxDestroy``, ``torch.cuda.empty_cache``,
or process-side ``gc`` recover it. This is an upstream issue (likely in
flashinfer or the CUDA driver layer). The defensive posture in phyai is
to keep vGPU objects long-lived; this ``destroy`` keeps things from
getting strictly worse but does not fix the root cause.
"""

from __future__ import annotations

import torch

from phyai.vgpu._spec import ShardSpec
from phyai.vgpu.backend import register_vgpu_backend
from phyai.vgpu.exceptions import VGPURuntimeError


@register_vgpu_backend("flashinfer")
class FlashInferBackend:
    """Multi-shard, disjoint-split backend (default)."""

    name = "flashinfer"

    @staticmethod
    def _wrap(streams, resources) -> list[ShardSpec]:
        out: list[ShardSpec] = []
        n = len(streams)
        for i, (s, r) in enumerate(zip(streams, resources)):
            out.append(
                ShardSpec(
                    stream=s,
                    sm_count=int(r.sm.smCount),
                    is_remainder=(i == n - 1),
                    _backend_handle=r,
                    _backend_name="flashinfer",
                )
            )
        return out

    def split_by_count(
        self,
        device: torch.device,
        num_groups: int,
        min_count: int,
    ) -> list[ShardSpec]:
        from flashinfer.green_ctx import split_device_green_ctx

        try:
            streams, resources = split_device_green_ctx(
                device,
                num_groups,
                min_count,
            )
        except RuntimeError as e:
            raise VGPURuntimeError(str(e)) from e
        return self._wrap(streams, resources)

    def split_by_sm_counts(
        self,
        device: torch.device,
        sm_counts: list[int],
    ) -> list[ShardSpec]:
        from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

        try:
            streams, resources = split_device_green_ctx_by_sm_count(
                device,
                sm_counts,
            )
        except RuntimeError as e:
            raise VGPURuntimeError(str(e)) from e
        return self._wrap(streams, resources)

    def create_single(
        self,
        device: torch.device,
        num_sms: int,
    ) -> ShardSpec:
        # split_by_count(1, num_sms) returns [chosen, remainder]; drop
        # remainder via best-effort destroy and return a non-remainder spec.
        specs = self.split_by_count(device, 1, num_sms)
        for rem in specs[1:]:
            self.destroy(rem)
        head = specs[0]
        return ShardSpec(
            stream=head.stream,
            sm_count=head.sm_count,
            is_remainder=False,
            _backend_handle=head._backend_handle,
            _backend_name=head._backend_name,
        )

    def destroy(self, spec: ShardSpec) -> None:
        """Best-effort: destroy the stream and its bound green ctx.

        Recovers the green-ctx handle via ``cuStreamGetGreenCtx`` (the
        ``CUdevResource`` flashinfer hands us doesn't carry the green ctx
        itself, only the SM resource description). Even with this cleanup
        the per-call driver leak persists; see module docstring.
        """
        try:
            import cuda.bindings.driver as cu
        except ImportError:
            return
        stream_handle = int(spec.stream.cuda_stream)
        try:
            status, gh = cu.cuStreamGetGreenCtx(stream_handle)
        except Exception:
            status, gh = None, 0
        if stream_handle:
            try:
                cu.cuStreamDestroy(stream_handle)
            except Exception:
                pass
        if status is not None and status == cu.CUresult.CUDA_SUCCESS and int(gh):
            try:
                cu.cuGreenCtxDestroy(gh)
            except Exception:
                pass
