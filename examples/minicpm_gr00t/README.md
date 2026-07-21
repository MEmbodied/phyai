# MiniCPM-GR00T local deployment

This guide runs the MiniCPM-V 4.6 + GR00T unified-80D policy through the
PhyAI Engine. It covers local inference only; PhyAI does not expose a standalone
network service in this example.

## Requirements

- Linux with a CUDA-capable NVIDIA GPU
- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A compatible MiniCPM-GR00T checkpoint (`.pth` or safetensors)
- The MiniCPM-V 4.6 processor/tokenizer directory

The original `.pth` checkpoint is loaded directly and does not need conversion.

## Install

From the repository root:

```bash
uv sync
```

`uv sync` creates `.venv` and installs the workspace packages and their locked
dependencies. Use `uv run` for the commands below so they run in that
environment.

## Run one inference

```bash
export CKPT=/path/to/rank_0_jobid_552915_iter_40000.pth
export PROCESSOR=/path/to/MiniCPM-V-4.6

PHYAI_USE_CUDA_GRAPH=1 \
uv run python examples/minicpm_gr00t/run_minicpm_gr00t.py \
  --checkpoint "${CKPT}" \
  --vlm-path "${PROCESSOR}" \
  --instruction "open the middle drawer of the cabinet" \
  --seed 123
```

With no `--image` arguments, the example uses two deterministic blank 224×224
RGB frames. For real inputs, pass exactly two images in base-camera, wrist-camera
order:

```bash
PHYAI_USE_CUDA_GRAPH=1 \
uv run python examples/minicpm_gr00t/run_minicpm_gr00t.py \
  --checkpoint "${CKPT}" \
  --vlm-path "${PROCESSOR}" \
  --image /path/to/base_camera.png \
  --image /path/to/wrist_camera.png \
  --instruction "open the middle drawer of the cabinet" \
  --seed 123 \
  --save-actions /tmp/minicpm_gr00t_actions.pt
```

A successful run reports an action tensor with shape `(1, 30, 80)`, dtype
`torch.float32`, and `finite=True`. The first run can take longer because kernels
and the CUDA Graph are initialized.

## Benchmark

Use warmup iterations and exclude them from the timed sample:

```bash
PHYAI_USE_CUDA_GRAPH=1 \
uv run python benchmark/minicpm_gr00t_random_smoke.py \
  --checkpoint "${CKPT}" \
  --processor-path "${PROCESSOR}" \
  --n-warmup 10 \
  --n-timed 100 \
  --output-json /tmp/phyai_minicpm_gr00t_benchmark.json
```

Run benchmarks on an otherwise idle GPU. Compare results only when checkpoint,
input shape, warmup count, timed count, software environment, and timing scope
are aligned.

## Verify

```bash
uv run pytest -q \
  phyai-utils-tools/tests/test_minicpm_gr00t_processor.py

uv run ruff check \
  examples/minicpm_gr00t \
  benchmark/minicpm_gr00t_random_smoke.py \
  phyai-utils-tools/src/phyai_utils_tools/models/minicpm_gr00t \
  phyai-utils-tools/tests/test_minicpm_gr00t_processor.py
```

Do not treat fixed-input numerical checks as task-level accuracy. Closed-loop
policy quality must be evaluated separately on a benchmark such as LIBERO.
