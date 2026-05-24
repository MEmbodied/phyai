"""CUDA device-capability helpers shared across phyai.

:func:`device_capability` returns the raw ``(major, minor)`` tuple that
callers like :func:`phyai.vgpu.topology.round_up_sm_count` expect, and
raises if CUDA is unavailable. :func:`sm_arch` returns the packed integer
form (``major * 10 + minor``) used for kernel dispatch keys, with a
graceful ``0`` fallback so init paths stay safe on developer laptops or
in forked subprocesses. :func:`init_cuda` / :func:`init_cublas` are the
discrete bootstrap entry points the engine and tests call to pin device
+ default dtype and tune cuBLAS/cuDNN — each is independently callable
so callers can opt into pieces without committing to the full engine
orchestration. :func:`print_topology` dumps a per-device summary plus a
peer-access matrix for the local node; :func:`print_distributed_topology`
extends that to a multi-node :mod:`torch.distributed` group with per-host
IB HCAs and GPU↔NIC affinity from ``nvidia-smi topo -m``.
"""

from __future__ import annotations

import sys
from typing import TextIO

import torch


def device_capability(
    device: "torch.device | str | int | None" = None,
) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def current_device() -> torch.device:
    """Return the current device for tensor allocation.

    Picks the active CUDA device when CUDA is available, otherwise
    ``cpu``. ``torch.cuda.current_device()`` returns an int rank, but
    callers typically want a ``torch.device`` they can pass to
    ``.to(...)`` or ``torch.empty(..., device=...)``; this wraps the
    rank into a ``torch.device("cuda", rank)``.

    Use this in place of hard-coded ``"cuda"`` / ``"cpu"`` strings so
    a process started under ``CUDA_VISIBLE_DEVICES=...`` lands on the
    intended device, and CPU-only dev / CI environments degrade
    gracefully.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def sm_arch(
    device: "torch.device | str | int | None" = None,
) -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except (RuntimeError, AssertionError):
        # CUDA may be visible but unusable (e.g. forked subprocess of a
        # parent that already initialized CUDA).
        return 0
    return major * 10 + minor


def init_cuda(
    device: "torch.device | str",
    params_dtype: torch.dtype,
) -> torch.dtype:
    """Pin the CUDA current device and the process default dtype.

    Returns the previously-set ``torch.get_default_dtype()`` so
    :meth:`Engine.close` can restore it. CPU device is a no-op for the
    device half (``torch.cuda.set_device`` is skipped) but the dtype
    pin still happens — fp32 / fp64 weights need it just as much.
    """
    saved = torch.get_default_dtype()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index if dev.index is not None else 0)
    torch.set_default_dtype(params_dtype)
    return saved


def init_cublas() -> None:
    """Tune cuBLAS / cuDNN for inference workloads.

    Currently a no-op placeholder — the discrete entry point is in
    place so future tuning (allow_tf32 toggles, ``CUBLAS_WORKSPACE_CONFIG``,
    cuDNN benchmark mode) lands in one well-named function rather than
    drifting into ``Engine.__init__``.
    """
    return None


def print_topology(*, file: TextIO | None = None) -> None:
    out = file if file is not None else sys.stdout

    if not torch.cuda.is_available():
        print("CUDA: unavailable", file=out)
        return

    n = torch.cuda.device_count()
    cur = torch.cuda.current_device()
    print(f"CUDA: {n} device(s), current=cuda:{cur}", file=out)

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        major, minor = device_capability(i)
        mem_gib = props.total_memory / (1 << 30)
        print(
            f"  cuda:{i}  {props.name}  sm_{major}{minor}  "
            f"SMs={props.multi_processor_count}  mem={mem_gib:.1f} GiB",
            file=out,
        )

    if n >= 2:
        print("peer access (P=can access, .=cannot):", file=out)
        print("       " + " ".join(f"{j:>3}" for j in range(n)), file=out)
        for i in range(n):
            cells = []
            for j in range(n):
                if i == j:
                    cells.append("  -")
                else:
                    ok = torch.cuda.can_device_access_peer(i, j)
                    cells.append("  P" if ok else "  .")
            print(f"  {i:>3}: " + "".join(cells), file=out)


def print_distributed_topology(*, file: TextIO | None = None) -> None:
    import glob
    import os
    import socket
    import subprocess

    import torch.distributed as dist

    out = file if file is not None else sys.stdout

    if not dist.is_available() or not dist.is_initialized():
        print(
            "torch.distributed not initialized; "
            "call dist.init_process_group(...) first",
            file=out,
        )
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
    gpu_uuid = ""
    if dev_idx >= 0:
        gpu_uuid = str(getattr(torch.cuda.get_device_properties(dev_idx), "uuid", ""))

    ib_hcas = sorted(os.path.basename(p) for p in glob.glob("/sys/class/infiniband/*"))

    nvsmi_topo = ""
    try:
        nvsmi_topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.rstrip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    info = {
        "hostname": socket.gethostname(),
        "local_rank_env": os.environ.get("LOCAL_RANK"),
        "dev_idx": dev_idx,
        "gpu_uuid": gpu_uuid,
        "ib_hcas": ib_hcas,
        "nvsmi_topo": nvsmi_topo,
    }

    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, info)

    if rank != 0:
        return

    by_host: dict[str, list[tuple[int, dict]]] = {}
    for r, item in enumerate(gathered):
        assert item is not None
        by_host.setdefault(item["hostname"], []).append((r, item))

    print(
        f"distributed: world_size={world_size}, hosts={len(by_host)}",
        file=out,
    )

    for host, ranks_on_host in by_host.items():
        print(f"\n[{host}]", file=out)
        for r, item in ranks_on_host:
            lr = item["local_rank_env"] or "?"
            uuid = item["gpu_uuid"] or "?"
            print(
                f"  rank {r:>3} (LOCAL_RANK={lr}): cuda:{item['dev_idx']}  {uuid}",
                file=out,
            )
        # NIC info is host-level, so report once per host (first rank).
        rep = ranks_on_host[0][1]
        if rep["ib_hcas"]:
            print(f"  IB HCAs: {', '.join(rep['ib_hcas'])}", file=out)
        if rep["nvsmi_topo"]:
            print("  nvidia-smi topo -m:", file=out)
            for line in rep["nvsmi_topo"].splitlines():
                print(f"    {line}", file=out)
