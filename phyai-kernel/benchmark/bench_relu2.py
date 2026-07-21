"""Benchmark fused ReLU-squared against PyTorch eager.

The default shapes cover Cosmos3-Edge text and policy generation MLPs::

    python benchmark/bench_relu2.py
"""

from __future__ import annotations

import argparse

import torch
import triton

import phyai_kernel


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _latency_us(fn) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=100, rep=500) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="53,1966,3542")
    parser.add_argument("--hidden", type=int, default=9216)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = _DTYPES[args.dtype]
    token_counts = [int(value) for value in args.tokens.split(",") if value]
    slower_shapes: list[tuple[int, int]] = []
    print("tokens hidden torch_us triton_us speedup")
    for tokens in token_counts:
        input = torch.randn(tokens, args.hidden, device="cuda", dtype=dtype)
        expected = torch.relu(input).square()
        actual = phyai_kernel.relu2(input)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        torch_us = _latency_us(lambda: torch.relu(input).square())
        triton_us = _latency_us(lambda: phyai_kernel.relu2(input))
        speedup = torch_us / triton_us
        print(
            f"{tokens:6d} {args.hidden:6d} {torch_us:8.2f} "
            f"{triton_us:9.2f} {speedup:7.3f}x"
        )
        if speedup <= 1.0:
            slower_shapes.append((tokens, args.hidden))

    if slower_shapes:
        print(f"fused relu2 did not beat PyTorch for: {slower_shapes}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
