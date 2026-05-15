"""Benchmark phyai_kernel.adarmsnorm against torch eager and ``torch.compile``.

Run::

    python benchmark/bench_adarmsnorm.py
    python benchmark/bench_adarmsnorm.py --dtype bf16 --shapes "B,S,D"
    python benchmark/bench_adarmsnorm.py --pattern per_token

Two broadcast patterns are exercised:

* ``per_batch`` (default, the pi0.5 action-expert pattern):
  ``x.shape=(B, S, D)``, ``modulation.shape=(B, 1, 3*D)``. Each
  modulation row is broadcast across ``S`` activations.
* ``per_token``: ``x.shape=(B*S, D)``, ``modulation.shape=(B*S, 3*D)``.
  1:1 mapping. Useful for sanity-checking the no-broadcast path.

Latencies are reported in microseconds (median across many iters) and a
TB/s number based on the read+write traffic each variant performs.
``torch.compile`` is the most realistic alternative a user would reach
for if they didn't have the Triton kernel.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from typing import Callable, List, Optional, Tuple

import torch
import triton

import phyai_kernel

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_PATTERNS = ("per_batch", "per_token")


# --------------------------------------------------------------------------- #
# Eager reference                                                              #
# --------------------------------------------------------------------------- #


def _ref_adarmsnorm(
    x: torch.Tensor, modulation: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirrors lerobot ``PiGemmaRMSNorm`` math."""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = xf * (1.0 + scale.float()) + shift.float()
    return out.to(dtype), gate.to(dtype)


def _build_compiled_ref(mode: str = "reduce-overhead") -> Callable:
    return torch.compile(_ref_adarmsnorm, mode=mode, dynamic=False)


# --------------------------------------------------------------------------- #
# Variant closures                                                             #
# --------------------------------------------------------------------------- #


def _build_phyai_fn() -> Callable:
    return lambda x, mod, eps: phyai_kernel.adarmsnorm(x, mod, eps)


def _build_eager_fn() -> Callable:
    return lambda x, mod, eps: _ref_adarmsnorm(x, mod, eps)


def _build_compiled_fn(compile_mode: str) -> Callable:
    fn = _build_compiled_ref(compile_mode)
    return lambda x, mod, eps: fn(x, mod, eps)


# --------------------------------------------------------------------------- #
# Bench driver                                                                 #
# --------------------------------------------------------------------------- #


def _bytes_per_call(pattern: str, B: int, S: int, D: int, dtype: torch.dtype) -> int:
    """Approximate HBM traffic per kernel call.

    Counts ``x`` read + ``out`` write in the activation dtype, plus the
    modulation read (and gate write) sized per the broadcast pattern.
    Modulation projection ``dense(cond)`` is intentionally excluded — the
    benchmark times only the AdaRMS kernel's footprint.
    """
    bw = torch.tensor([], dtype=dtype).element_size()
    n_x = B * S * D
    n_mod_rows = B if pattern == "per_batch" else B * S
    n_mod = n_mod_rows * 3 * D
    n_gate = n_mod_rows * D
    # x in + out + modulation in + gate out
    return (n_x * 2 + n_mod + n_gate) * bw


def _bench_one(
    fn: Callable,
    args: Tuple,
    warmup: int = 25,
    iters: int = 100,
) -> float:
    """Median latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(lambda: fn(*args), warmup=10, rep=iters)
    return ms * 1000.0


def _make_inputs(
    B: int, S: int, D: int, *, dtype: torch.dtype, pattern: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(B, S, D, device="cuda", dtype=dtype) * 0.5
    if pattern == "per_batch":
        modulation = torch.randn(B, 1, 3 * D, device="cuda", dtype=dtype) * 0.1
    else:
        modulation = torch.randn(B, S, 3 * D, device="cuda", dtype=dtype) * 0.1
    return x, modulation


def _parse_shapes(spec: str) -> List[Tuple[int, int, int]]:
    """Parse ``"B,S,D;B,S,D;..."`` into a list of triples."""
    out: List[Tuple[int, int, int]] = []
    for triple in spec.split(";"):
        triple = triple.strip()
        if not triple:
            continue
        parts = [int(p) for p in triple.split(",")]
        if len(parts) != 3:
            raise ValueError(f"bad shape spec {triple!r}; expected 'B,S,D'")
        out.append((parts[0], parts[1], parts[2]))
    return out


# pi0.5 action-expert: width=1024, chunk=50, ~768 image + 200 lang + 50 action
# = ~1018 prefix tokens. We bracket realistic shapes and add a few stress sizes.
_DEFAULT_SHAPES = [
    (1, 50, 1024),  # pi0.5 inference, suffix-only denoise step
    (4, 50, 1024),  # batched inference
    (1, 1018, 1024),  # full pi0.5 sequence (prefix+suffix), expert-side
    (8, 1018, 1024),  # larger batch
    (1, 200, 2048),  # gemma_2b prefix LM at lang-only resolution
    (1, 50, 2048),  # 2b expert action tokens (hypothetical gemma_2b expert)
    (1, 1024, 4096),  # awkward-large stress
]


def main() -> int:
    parser = argparse.ArgumentParser("phyai-kernel AdaRMSNorm benchmark")
    parser.add_argument(
        "--dtype",
        choices=tuple(_DTYPES),
        default="bf16",
        help="activation dtype (default bf16)",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help='"B,S,D;B,S,D;..." (default = pi0.5-realistic preset)',
    )
    parser.add_argument(
        "--pattern",
        choices=_PATTERNS,
        default="per_batch",
        help="modulation broadcast pattern (default per_batch — pi0.5 action expert)",
    )
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="torch.compile mode (default reduce-overhead)",
    )
    parser.add_argument(
        "--skip_compiled",
        action="store_true",
        help="skip the torch.compile column (useful when compile is broken)",
    )
    args = parser.parse_args()

    dtype = _DTYPES[args.dtype]
    shapes = _parse_shapes(args.shapes) if args.shapes else _DEFAULT_SHAPES
    pattern = args.pattern

    print(
        f"dtype={args.dtype}  pattern={pattern}  compile_mode={args.compile_mode}\n"
        f"-----------------------------------------------------------------"
    )
    header = (
        f"{'B':>4} {'S':>5} {'D':>6} | "
        f"{'phyai (µs)':>11} {'eager (µs)':>11} {'compile (µs)':>13} | "
        f"{'phyai TB/s':>10} | "
        f"{'speedup vs eager':>17} {'speedup vs compile':>19}"
    )
    print(header)
    print("-" * len(header))

    phyai_fn = _build_phyai_fn()
    eager_fn = _build_eager_fn()
    compiled_fn = None if args.skip_compiled else _build_compiled_fn(args.compile_mode)

    for B, S, D in shapes:
        x, mod = _make_inputs(B, S, D, dtype=dtype, pattern=pattern)
        eps = 1e-6

        t_phyai = _bench_one(phyai_fn, (x, mod, eps))
        t_eager = _bench_one(eager_fn, (x, mod, eps))
        if compiled_fn is not None:
            t_compiled = _bench_one(compiled_fn, (x, mod, eps))
        else:
            t_compiled = float("nan")

        bytes_total = _bytes_per_call(pattern, B, S, D, dtype)
        tbps_phyai = bytes_total / (t_phyai * 1e6)
        speedup_eager = t_eager / t_phyai if t_phyai > 0 else float("nan")
        speedup_compiled = (
            t_compiled / t_phyai
            if (t_phyai > 0 and compiled_fn is not None)
            else float("nan")
        )

        print(
            f"{B:>4d} {S:>5d} {D:>6d} | "
            f"{t_phyai:>10.2f}  {t_eager:>10.2f}  {t_compiled:>12.2f} | "
            f"{tbps_phyai:>9.2f}  | "
            f"{speedup_eager:>16.2f}x {speedup_compiled:>18.2f}x"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
