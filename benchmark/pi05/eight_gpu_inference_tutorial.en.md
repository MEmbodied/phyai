# PhyAI pi0.5 Eight-GPU Inference Tutorial

Reproduce **PhyAI `pi05_wn` 8-GPU DP inference + concurrent LIBERO demos**.  
Chinese: [`八卡推理从零开始教程.md`](./八卡推理从零开始教程.md)

---

## 1. What you get

```text
8-GPU PhyAI WebSocket server (port 8000)
  ↑
32 LIBERO shards (4 suites × 8 tasks)
  ↓
Per-shard JSON (success / timing) + optional wait videos
```

Recommended settings:

| Item | Value |
| --- | --- |
| GPUs / batch | 8 GPUs, `MAX_BATCH_SIZE=32` (B=4 per GPU) |
| Chunk | `CHUNK_SIZE=10`, `SEND_ACTION_CHUNKS=1` (true chunk=10) |
| Batching wait | `MAX_WAIT_TIME=0.02` |
| Recording (optional) | `continuous` + 20fps |

Architecture: all LIBERO clients talk to rank0; after the batch fills (or timeout), DP scatter → each GPU runs its slice → gather and reply. All 8 ranks sync in one step. CUDA graphs use a fixed padded shape, so a partial batch is not proportionally faster.

Chunk modes:

| Mode | Flag | Behavior |
| --- | --- | --- |
| True chunk=10 | `SEND_ACTION_CHUNKS=1` | Return 10 actions once; client runs them locally, then requests again |
| Pseudo chunk=1 | `SEND_ACTION_CHUNKS=0` | Model still produces 10; server returns 1 and buffers the rest; request every step |

---

## 2. Setup

Needs: 8 free GPUs, Docker, `tmux`.

```bash
export WORKSPACE="$HOME/phyai_workplace"   # change me: must contain phyai / vla-evaluation-harness / phyai_models
export PHYAI_ROOT="$WORKSPACE/phyai"
export VLA_ROOT="$WORKSPACE/vla-evaluation-harness"
export MODEL_ROOT="$WORKSPACE/phyai_models"
export DEMO_ROOT="$PHYAI_ROOT/libero_wn_demo"
```

Checks:

```bash
ls "$MODEL_ROOT/pi05_libero_phyai_converted" "$MODEL_ROOT/paligemma-3b-pt-224"
ls "$PHYAI_ROOT/.venv/bin/torchrun" "$VLA_ROOT/.venv/bin/vla-eval"
```

Images:

```bash
sg docker -c 'docker pull nvcr.io/nvidia/pytorch:25.12-py3'
sg docker -c 'docker pull ghcr.io/allenai/vla-evaluation-harness/libero:latest'
```

Tokenizer must be offline: place `$MODEL_ROOT/paligemma-3b-pt-224` first (`HF_HUB_OFFLINE=1` is set by the script).  
Always pass `PHYAI_ROOT` / `VLA_ROOT` / `MODEL_ROOT` explicitly; do not copy another machine’s absolute paths.

---

## 3. Shortest path (batch32 + true chunk10)

### A. Confirm idle

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
ss -ltnp | grep 8000 || echo '8000 free'
```

Cleanup leftovers if needed:

```bash
tmux kill-session -t phyai_pi05_wn_demo 2>/dev/null || true
sg docker -c 'docker rm -f phyai_pi05_wn_demo' 2>/dev/null || true
```

### B. Start server

```bash
sg docker -c "
SESSION=phyai_pi05_wn_demo CONTAINER=phyai_pi05_wn_demo \
PHYAI_ROOT=$PHYAI_ROOT VLA_ROOT=$VLA_ROOT MODEL_ROOT=$MODEL_ROOT \
MAX_BATCH_SIZE=32 MAX_WAIT_TIME=0.02 CHUNK_SIZE=10 SEND_ACTION_CHUNKS=1 \
MASTER_PORT=29619 \
bash $DEMO_ROOT/start_pi05_wn_server.sh
"
```

After ~1–3 minutes:

```bash
curl -sS http://127.0.0.1:8000/config
# expect max_batch_size=32
tmux attach -t phyai_pi05_wn_demo   # detach: Ctrl-b d
```

### C. Run 32 shards

```bash
sg docker -c "bash $DEMO_ROOT/run_libero_32_clientchunk10_maxwait002.sh"
```

Outputs under `$VLA_ROOT/results/clientchunk10_maxwait002/`: suite `*.json` and `wait_videos/*.mp4` (pixels are only in the videos).

### D. Cleanup

```bash
tmux kill-session -t phyai_pi05_wn_demo
sg docker -c 'docker rm -f phyai_pi05_wn_demo' 2>/dev/null || true
```

---

## 4. Variants

**Pseudo chunk=1 (batch32)**

```bash
# same server command with SEND_ACTION_CHUNKS=0 and new SESSION/CONTAINER/MASTER_PORT
sg docker -c "bash $DEMO_ROOT/run_libero_32_serverchunk1_maxwait002.sh"
```

**batch16 (B=2 per GPU, 16 shards)**

```bash
# MAX_BATCH_SIZE=16, SEND_ACTION_CHUNKS=0 or 1
sg docker -c "bash $DEMO_ROOT/run_libero_batch16_serverchunk1_maxwait002.sh"
```

Start batch16 and batch32 as separate servers; do not switch inside one `torchrun`.

**Custom experiment**: copy `$DEMO_ROOT/clientchunk10_maxwait002/`, edit yaml `output_dir` / suite, then point the client script’s `DEMO_ROOT` and `RESULTS_ROOT` at your dirs.

**Pure inference latency (no LIBERO)**:

```bash
sg docker -c "bash $PHYAI_ROOT/benchmark/run_pi05_wn_latency_dp8_docker.sh"
```

Reference: batch16 ~60ms, batch32 ~97ms (pure `Engine.step`, not end-to-end).

---

## 5. Recording and results

For end-to-end stalls use `continuous`, not `step`:

```yaml
docker:
  env:
    - VLA_EVAL_WAIT_VIDEO_DIR=/workspace/results/wait_videos
    - VLA_EVAL_WAIT_VIDEO_MODE=continuous
    - VLA_EVAL_WAIT_VIDEO_FPS=20
    - VLA_EVAL_WAIT_VIDEO_MODEL=<prefix>
```

`continuous` freezes while waiting on the model; freeze length ≈ real wait. `step` removes waits. Changing export fps does not shorten real stalls.

Useful JSON fields: `metrics.success`, `avg_model_wait_ms`, `model_buffer_hits`, `model_inference_calls`, `wait_video_mode`.  
End-to-end wait ≈ queue + predict_batch + ws, often much larger than pure GPU bench.

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Stuck in setup | Offline tokenizer; change `MASTER_PORT`; remove same-name container; free GPUs |
| `:8000/config` fails | Wait for graph capture; `tmux capture-pane -t <SESSION> -p -S -80` |
| Shard cannot connect | Host networking; URL=`ws://127.0.0.1:8000`; `NO_PROXY='*'` |
| Partial results | Check `$VLA_ROOT/results/<exp>/*_logs/`; `taskunknown` ≈ task0 |
| FlashInfer/CUDA clash | Use Docker; avoid bare-metal host runs |

---

## 7. File index (under `$DEMO_ROOT`)

| File | Purpose |
| --- | --- |
| `start_pi05_wn_server.sh` | 8-GPU server |
| `run_libero_32_clientchunk10_maxwait002.sh` | batch32 true chunk10 |
| `run_libero_32_serverchunk1_maxwait002.sh` | batch32 pseudo chunk=1 |
| `run_libero_batch16_serverchunk1_maxwait002.sh` | batch16 pseudo chunk=1 |
| `clientchunk10_maxwait002/` etc. | continuous demo configs |
| `experiment_setup.md` | Historical experiment notes |
