---
name: phyai-kernel-opt
description: Implement, port, profile, tune, and integrate GPU kernels for phyai and phyai-kernel with numerical validation, architecture-specific backend selection, autotuning, CUDA Graph coverage, Nsight Compute evidence, reproducible benchmarks, and model-level verification. Use when asked to write or fuse a kernel, diagnose why a CUDA or Triton kernel is slow, compare kernel backends, tune tactics, improve latency, throughput, bandwidth, or TFLOPS, or optimize workloads for NVIDIA server GPUs and Jetson devices.
---

# phyai kernel optimization

Treat kernel work as an end-to-end engineering task. A fast isolated kernel is not complete until it preserves the production contract, integrates into the real dispatch path, survives CUDA Graph capture, and improves the target workload.

## Prepare

1. Read the repository `CLAUDE.md` and relevant `.memory` entries before inspecting code.
2. Define the production contract before changing an implementation:
   - operator formula and accumulation precision;
   - input, output, scale, and workspace dtypes;
   - logical shapes, physical layouts, strides, padding, alignment, and scale semantics;
   - representative and boundary shapes from the real caller;
   - eager and CUDA Graph requirements;
   - target GPU architecture and the user's latency, throughput, memory, or power objective.
3. Run `nvidia-smi` before every CUDA program. Use an idle GPU for benchmarks. If all suitable GPUs are busy, report that condition rather than presenting contaminated measurements.
4. On Jetson, record the active power mode, clocks, thermal state, and memory pressure. Do not change `nvpmodel`, clocks, or power limits without user approval.
5. Create `.tmp/` if needed. Reuse existing source checkouts and inspect their current commit before fetching. Clone a missing repository only when it is needed for the task:
   - `https://github.com/flashinfer-ai/flashinfer.git`
   - `https://github.com/tile-ai/tilelang.git`
   - `https://github.com/NVIDIA/cutlass.git`
   - `https://github.com/NVIDIA/TensorRT-LLM.git`
   - `https://github.com/NVIDIA/TensorRT-Edge-LLM.git`
   - `https://github.com/NVIDIA/TileGym.git`
   - `https://github.com/triton-lang/triton.git`
6. Prefer the installed library source when reproducing the behavior of the installed runtime. Use the latest local checkout for design research, but do not assume its API or layout matches the pinned dependency.
7. Create one run directory for the task. Follow the sibling [`ncu-report-skill`](../ncu-report-skill/SKILL.md) layout under `profile/<run_name>/`. Keep harnesses, reports, analysis data, iteration history, and the final report in that run directory.

## Your responsibility

Own the full optimization loop:

- Locate the real producer, kernel, consumer, dispatch rule, and graph-capture boundary.
- Establish a trusted numerical oracle and explicit tolerances before tuning.
- Verify logical values and physical contracts. For quantized paths, check scale direction, scale dtype, packing, padding, strides, and the consumer's raw-pointer interpretation.
- Measure the current production path and at least one independent reference implementation.
- Profile before choosing a bottleneck. Distinguish compute, bandwidth, memory latency, launch overhead, occupancy, synchronization, and tail imbalance with evidence.
- Change one performance idea at a time, rerun correctness, and measure under the same conditions.
- Preserve existing package APIs and dispatch patterns unless the current abstraction cannot represent the required contract.
- Add focused tests, a reproducible benchmark, an autotune surface, CUDA Graph coverage, and model or pipeline validation proportional to the integration risk.
- Record rejected iterations as well as successful ones. A rejected tactic is useful evidence and prevents repeated work.
- Separate measured results from source-based inference. Do not claim performance or correctness on hardware that was not tested.
- Keep generated profiles and hardware-specific artifacts out of public source paths unless the user asks to commit them.
- Update `.memory/` with files changed, commands, results, decisions, remaining risks, and artifact locations.

Do not stop after a microbenchmark when the user asked to accelerate a model. Conversely, do not run a full model first when a small harness can disprove correctness or identify the bottleneck faster.

## Implementation priority

Choose the narrowest proven implementation that fits the target:

1. For Jetson Thor, Orin NX, and AGX Orin, inspect TensorRT-Edge-LLM first. Check the exact device architecture, supported datatypes, memory limits, and power mode rather than treating all Jetson devices alike.
2. For server GPUs, inspect the pinned FlashInfer implementation and TensorRT-LLM first. Consult SGLang and vLLM when studying integration, layout selection, padding, graph capture, and dispatch policy.
3. Use CUTLASS as the low-level reference for GEMM, attention, TMA, tensor-core layouts, and architecture constraints.
4. On Blackwell, try an existing TileGym, CuTe DSL, or cuTile implementation when it covers the operator and target SM. Use CUTLASS when those paths cannot express the operation, do not meet the performance target, or are unsupported on the device.
5. For elementwise, reduction, quantization, epilogue, and producer-consumer fusion, try Triton first. Move to CUDA or CUTLASS only after identifying a concrete Triton limitation such as code generation, register pressure, unsupported instructions, TMA/layout needs, or launch behavior.
6. Use TileLang when the repository already uses it or its scheduling model has a clear advantage for the operator. Do not select it only to add another implementation.
7. Reuse proven library kernels for GEMM, attention, convolution, sorting, and other established primitives. Hand-roll the core primitive only when existing kernels cannot satisfy the contract or measured workload.

Never transplant a kernel based only on matching logical shapes. Match dtype codes, scale convention, physical storage, strides, padding, alignment, accumulator type, output epilogue, and architecture gates.

## Required tools

Use the tools that answer the current performance question. Preserve their raw artifacts.

### Timing and workload traces

- Use CUDA events for device latency. Place synchronization outside the timed region.
- Use repeated launches or CUDA Graph replays inside one event interval for kernels near event-resolution limits.
- Measure eager steady state and CUDA Graph steady state separately when production supports both.
- Record cold JIT, compilation, or autotuning time separately. Never mix it into steady-state kernel latency.
- Use PyTorch profiler or Nsight Systems to count launches, expose CPU gaps, synchronization, allocator work, graph boundaries, and overlap.
- Use the repository's model profiler for stage latency and end-to-end impact when integrating into a model.

### Nsight Compute

Read and follow [`ncu-report-skill/SKILL.md`](../ncu-report-skill/SKILL.md) for every nontrivial CUDA optimization. At minimum:

1. Build a focused harness with representative production shapes and `-lineinfo`, or use the real binary when a standalone harness would change behavior.
2. Collect a full report and a source-counter report in the run directory.
3. Parse reports with the `ncu_report` Python API or the sibling skill helpers. Do not infer conclusions only from terminal summary text.
4. Inspect occupancy, achieved throughput, memory traffic, cache behavior, warp stalls, tensor-core utilization, per-SM balance, and PM-sampling tail behavior.
5. Compare the baseline and final kernel with the same metric set.

If NCU cannot run because of permissions, unsupported hardware, or missing tools, record the exact limitation and use available traces, profiler counters, disassembly, and roofline data. Do not invent NCU conclusions.

### Safety and compiler inspection

- Use `compute-sanitizer` when pointer arithmetic, masking, asynchronous copies, shared-memory staging, or custom CUDA code creates an out-of-bounds or race risk.
- Inspect generated Triton IR/PTX, `cuobjdump`, or `nvdisasm` only when the hypothesis concerns vectorization, spills, instruction selection, tensor-core use, or compiler scheduling.
- Check allocated registers, shared memory, local-memory spills, and compiled launch metadata for each serious tactic.
- Verify that benchmark outputs are consumed and that compilation cannot eliminate the measured work.

## Optimization loop

Run the following loop until the target is met or evidence shows that further kernel work has little useful headroom.

### 0. Freeze the experiment

Record:

- repository commit and working-tree state;
- GPU name, SM version, driver, CUDA, Torch, Triton, and FlashInfer versions;
- exact shapes, dtypes, layouts, strides, alignment, input distribution, and random seed;
- power mode and clocks when they can affect results;
- warmup count, timed count, timing mode, graph mode, and workspace policy;
- formula used for FLOPs and bytes.

Do not put personal paths or machine identity in source code. Store machine-specific metadata only in ignored benchmark artifacts or the task report.

### 1. Establish correctness and baselines

Build three levels when applicable:

1. A semantic oracle, usually PyTorch or a higher-precision implementation.
2. The current production implementation with its real producer and consumer layout.
3. A strong independent kernel such as FlashInfer, TensorRT-LLM, CUTLASS, Triton, cuBLASLt, or an architecture-specific reference.

Use the same inputs and output contract. Include required transpose, padding, quantization, epilogue, and workspace costs if production pays them. Also report kernel-only latency when it helps locate the cost.

### 2. Profile and classify the bottleneck

Choose the diagnosis from measurements:

- Compute bound: high math-pipeline utilization, high tensor-core activity, and low memory headroom.
- Bandwidth bound: DRAM or L2 throughput near the sustainable limit with predictable byte traffic.
- Memory-latency bound: long-scoreboard stalls with low bandwidth and insufficient independent work.
- Launch bound: small kernels, many launches, visible CPU gaps, or a large benefit from graph replay or fusion.
- Occupancy or register bound: high register or shared-memory use, spills, few active warps, and improved latency from a smaller tile or fewer stages.
- Synchronization bound: barrier or wait stalls, over-staged pipelines, or serialization between producer and consumer.
- Tail or imbalance bound: per-SM active-cycle variance, variable-length work, or PM-sampling utilization that collapses near the end.

Name the metrics that support the classification. Avoid labels such as "memory bound" without numbers.

### 3. State one hypothesis

Write one sentence that predicts both the code change and the observable result. For example:

> Fuse activation and group quantization to remove one full intermediate write/read and one launch; expect lower DRAM bytes and lower pipeline latency without changing GEMM time.

Reject hypotheses that do not identify a measurable mechanism.

### 4. Implement one logical tactic

Prefer changes in this order:

1. Remove unnecessary work, conversion, padding, allocation, synchronization, or launches.
2. Select a better algorithm or backend for the workload regime.
3. Fuse adjacent producer, reduction, epilogue, or quantization work when the intermediate has one consumer.
4. Fix physical layout so the producer writes the consumer's native format directly.
5. Improve global-memory coalescing, vector width, cache reuse, and transaction count.
6. Improve tiling, warp assignment, persistent scheduling, split-K policy, staging, and occupancy.
7. Use tensor cores, TMA, warp specialization, architecture-specific instructions, or a lower-level implementation when profiling justifies the complexity.

Keep unrelated refactors out of the iteration so the result remains attributable.

### 5. Validate before timing

Run the smallest sufficient correctness suite after every tactic:

- random and adversarial values, including zeros and extreme representable values;
- minimum, representative, maximum, non-power-of-two, and alignment-boundary shapes;
- contiguous and supported strided layouts;
- exact scale storage and stride checks for quantized consumers;
- eager execution and CUDA Graph replay;
- distributed or tensor-parallel semantics when the layer owns communication or bias behavior.

Use bitwise comparison when matching a serialized format, external quantizer, graph replay, or exact backend path. Otherwise set `atol` and `rtol` from the numerical contract and report cosine or relative L2 when they are meaningful.

### 6. Measure with the frozen protocol

Measure baseline and candidate with identical inputs, allocations, workspace, stream, graph mode, and sample count. Alternate or pair A/B samples when drift is plausible. Report the distribution, not only the fastest observation.

For each workload shape, capture at least:

- median, mean, minimum, p20, p80, and sample count;
- speedup and absolute time saved;
- TFLOPS for compute-heavy kernels, with the FLOP formula stated;
- effective GB/s or TB/s for movement-heavy kernels, with the byte formula stated;
- launch count and integrated pipeline latency for fused paths.

### 7. Mark the iteration

Append one record to `profile/<run_name>/analysis/iterations.jsonl` for every attempted tactic, including rejected ones. Use this schema:

```json
{
  "iteration": 3,
  "label": "fused-group-quant",
  "hypothesis": "remove intermediate activation traffic and one launch",
  "change": "fuse GeGLU and group-128 FP8 quantization",
  "shapes": ["50x8192", "784x32768"],
  "correct": true,
  "median_ms": 0.0234,
  "tflops": null,
  "speedup_vs_baseline": 1.84,
  "ncu_evidence": "lower DRAM bytes; no new spills",
  "decision": "keep"
}
```

Add a short diff identifier or source hash when multiple discarded implementations existed outside commits.

### 8. Keep, revise, or reject

Keep a tactic only when it is correct and improves the target workload without an unacceptable regression on required shapes. Treat changes below measurement noise as ties. Revert only the rejected changes you introduced; preserve unrelated user work.

Re-profile after a meaningful speedup because the bottleneck may move. Stop when one of these conditions holds:

- the requested target is met;
- the kernel approaches the measured roofline or a strong vendor baseline;
- remaining latency lies outside the kernel;
- further gains require a portability, maintenance, accuracy, memory, or compile-time tradeoff the user has not accepted.

## Autotuning requirements

Leave an autotune interface for every kernel family with multiple plausible tactics.

- Triton kernels should expose a bounded `@triton.autotune` configuration set or an equivalent repository helper.
- CUDA, CUTLASS, CuTe DSL, cuTile, and library-backed paths should expose explicit tactic enumeration and a cacheable selector when more than one implementation can win.
- Include every trait that can change the winner in the key: architecture, dtype, layout, alignment, group size, relevant exact dimensions or buckets, and semantic flags.
- Prune invalid tactics before timing and validate candidate outputs before accepting a winner.
- Warm and tune before CUDA Graph capture. Never benchmark tactics or allocate persistent workspaces while a graph is being captured.
- Cache winners when tuning cost matters. Keep cache compatibility tied to the kernel, dependency, architecture, and tactic-key versions.
- Provide a deterministic default and a way to force a tactic for debugging and reproducibility.
- Keep the candidate set small enough that first-run compilation and tuning remain reasonable.

Do not add fake autotuning when only one valid implementation exists. Preserve an API seam that can accept more tactics later, and state why the current set has one member.

## How to measure

Use a benchmark that mirrors the real call site while isolating the question under test.

### Experimental rules

- Warm until JIT compilation, allocator growth, library-handle creation, workspace setup, and autotune finish.
- Preallocate production-owned outputs and workspaces outside the timed region unless allocation is part of the production contract.
- Use the same stream and synchronization points as production.
- For sub-10-us kernels, time multiple launches per sample and divide by the replay count.
- Use enough samples to make p20 and p80 meaningful. Increase the count when variance overlaps the expected gain.
- Keep inputs resident on the device. Do not include host transfer unless the application includes it.
- Check thermal and clock drift before and after long Jetson runs.
- Report both kernel-only and producer-to-consumer pipeline latency when fusion or layout changes move work across kernel boundaries.
- Run at least one integrated layer or model benchmark after a kernel win. The integrated result decides whether the optimization succeeded.

### Metric formulas

State the operation-count convention. For dense GEMM, the common convention is `2*M*N*K` FLOPs. Add activation, reduction, or epilogue operations only if they are counted consistently for every candidate.

```text
TFLOPS = FLOPs / latency_seconds / 1e12
effective_GBps = logical_or_measured_bytes / latency_seconds / 1e9
speedup = baseline_median / candidate_median
time_saved_ms = baseline_median_ms - candidate_median_ms
```

For fused kernels, label byte counts as logical, estimated, or measured. Prefer NCU-measured traffic when comparing cache-sensitive implementations.

### Required result files

Store these under the run directory:

- `analysis/environment.json`: software, hardware, clock, and experiment metadata;
- `analysis/baseline.json`: baseline distributions and correctness;
- `analysis/iterations.jsonl`: one point per optimization attempt;
- `analysis/final.json`: final per-shape and integrated results;
- `analysis/progression.svg`: latency and TFLOPS progression;
- `REPORT.md`: evidence, decisions, commands, and remaining risks.

Generate `progression.svg` from `iterations.jsonl`, not from manually copied values. Plot iteration on the x-axis. Use two aligned panels or clearly labeled axes: latency should show lower is better, while TFLOPS should show higher is better. Mark rejected points distinctly and include the baseline. When TFLOPS does not describe the operator, plot effective bandwidth or pipeline latency instead and explain the substitution.

## How to report

Lead with the measured outcome and its scope. Use this order:

1. State the target GPU, workload shapes, timing mode, baseline, final latency, speedup, and absolute time saved.
2. State correctness results and tolerances, including graph replay and physical-layout checks.
3. Show a compact per-shape table with latency distribution and TFLOPS or bandwidth.
4. Explain the original bottleneck with named trace or NCU metrics.
5. Explain the final implementation and why it changes those metrics.
6. Report layer, pipeline, or model impact. Separate kernel-only gains from end-to-end gains.
7. List rejected tactics and the evidence for rejecting them.
8. State hardware coverage gaps, unsupported shapes, compile-time cost, memory tradeoffs, and any untested architecture path.
9. Link `REPORT.md`, raw NCU reports, JSON results, traces, and `progression.svg`.

Never report "faster" without the workload and measurement method. Never extrapolate a result from one architecture to another as measured fact. If the candidate wins only for a regime, encode that regime in dispatch rather than presenting one global speedup.

## Baselines

Use at least two baselines when available:

- PyTorch for semantic correctness and a portable performance floor.
- The current phyai production path, including the costs it actually pays.
- FlashInfer for supported attention, normalization, activation, quantization, GEMM, and fused operations.
- Triton for portable custom elementwise, reduction, quantization, and fusion kernels.
- TensorRT-LLM, TensorRT-Edge-LLM, CUTLASS, cuBLASLt, CuTe DSL, cuTile, or TileGym for architecture-specific comparisons.

Do not compare kernels with different contracts. Normalize dtype, precision, padding, layout, scale semantics, epilogue, output allocation, and graph mode, or report the difference explicitly.

## Completion checklist

Do not call the task complete until all applicable items pass:

- The production contract and representative shapes are documented.
- A trusted oracle and current production baseline exist.
- Correctness covers edge shapes, physical layouts, graph replay, and relevant parallel semantics.
- A reproducible benchmark reports distributions rather than one best sample.
- NCU or an explicitly documented fallback supports the bottleneck diagnosis.
- Every optimization attempt has a point in `iterations.jsonl`.
- The final result includes `progression.svg` with latency and TFLOPS, bandwidth, or pipeline progression.
- The kernel exposes bounded autotuning or a documented single-tactic extension point.
- Tuning and workspace initialization happen before graph capture.
- The integrated layer, pipeline, or model shows the expected benefit.
- Tests, formatting, and repository checks pass.
- Generated source contains no personal path or machine identity.
- No CUDA jobs started by the task remain running.
- `.memory/` records the work, validation, artifacts, and hardware gaps.
