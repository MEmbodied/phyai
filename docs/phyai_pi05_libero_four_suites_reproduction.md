# PhyAI pi0.5 跑 LIBERO 四组件完整复现文档

本文档说明如何在一台新机器上使用 PhyAI 推理 pi0.5，并通过 vla-evaluation-harness 跑完整 LIBERO 四组件 benchmark。文档不依赖当前机器的本地路径，所有路径都用环境变量表示。

四组件指：

```text
libero_spatial -> configs/benchmarks/libero/spatial.yaml
libero_object  -> configs/benchmarks/libero/object.yaml
libero_goal    -> configs/benchmarks/libero/goal.yaml
libero_10      -> configs/benchmarks/libero/10.yaml
```

完整口径：

```text
模式：sync
chunk_size：10
每个组件：10 tasks x 50 episodes = 500 episodes
总量：4 suites x 500 episodes = 2000 episodes
模型：PhyAI pi0.5 LIBERO converted checkpoint
仿真：vla-evaluation-harness LIBERO Docker 容器
结果：每个 suite 一个 JSON，记录 success、steps、timing、chunk size
```

## 1. 机器与资源要求

建议机器：

```text
GPU：至少 1 张 CUDA GPU，显存建议 >= 48GB
系统：Linux
容器：Docker + NVIDIA Container Toolkit
Python 环境管理：uv
辅助工具：tmux, nvidia-smi, ss
```

必须准备的模型资源：

```text
PhyAI converted checkpoint：pi05_libero_phyai_converted
PaLI-Gemma tokenizer / processor：paligemma-3b-pt-224
```

`paligemma-3b-pt-224` 是 gated 资源，推荐从已有机器同步，不建议在复现时现场下载。

## 2. 推荐环境变量

根据目标机器实际路径设置：

```bash
export PHYAI_ROOT=$HOME/phyai
export VLA_ROOT=$HOME/vla-evaluation-harness
export MODEL_ROOT=$HOME/phyai_models
export PHYAI_CONTAINER=phyai_libero_eval

export PHYAI_CKPT_HOST=$MODEL_ROOT/pi05_libero_phyai_converted
export TOKENIZER_HOST=$MODEL_ROOT/paligemma-3b-pt-224

export PHYAI_CKPT_IN_CONTAINER=/data/share/pi05_libero_phyai_converted
export TOKENIZER_IN_CONTAINER=/data/share/paligemma-3b-pt-224

export LIBERO_IMAGE=ghcr.io/allenai/vla-evaluation-harness/libero:latest
```

确认资源存在：

```bash
test -d "$PHYAI_CKPT_HOST"
test -d "$TOKENIZER_HOST"
```

如果模型在另一台机器上，示例同步命令如下：

```bash
rsync -azP \
  /path/to/pi05_libero_phyai_converted \
  /path/to/paligemma-3b-pt-224 \
  user@target-host:$MODEL_ROOT/
```

## 3. 获取代码并创建环境

clone PhyAI：

```bash
git clone https://github.com/MEmbodied/phyai.git "$PHYAI_ROOT"
cd "$PHYAI_ROOT"
uv sync
```

clone vla-evaluation-harness：

```bash
git clone https://github.com/allenai/vla-evaluation-harness.git "$VLA_ROOT"
cd "$VLA_ROOT"
uv sync
./.venv/bin/vla-eval --help >/tmp/vla_eval_help.log
```

如果目标机器不能联网，可以在能联网机器上 clone 后用 `rsync` 同步两个仓库；同步后仍建议在目标机器上执行 `uv sync`，让本机 Python、CUDA、依赖和 editable path 正确。

## 4. 准备 LIBERO Docker 镜像

vla-harness 跑 LIBERO 时会启动 LIBERO benchmark 容器。x86_64 机器可以直接使用官方镜像：

```bash
docker pull "$LIBERO_IMAGE"
```

如果目标机器是 ARM64，例如 Thor，而官方镜像只有 amd64，需要在目标机器本地构建 ARM64 镜像：

```bash
cd "$VLA_ROOT"
export DOCKER_DEFAULT_PLATFORM=linux/arm64
docker/build.sh libero

docker image inspect "$LIBERO_IMAGE" \
  --format '{{.Architecture}} {{.Os}}'
```

期望输出：

```text
arm64 linux
```

如果是 x86_64，期望输出一般是：

```text
amd64 linux
```

## 5. 创建 PhyAI Docker 容器

建议在 Docker 中运行 PhyAI server，容器挂载 PhyAI 代码、vla-harness 代码和模型目录：

```bash
docker run -dit --gpus all \
  -v "$PHYAI_ROOT":/phyai_workspace \
  -v "$VLA_ROOT":/vla-evaluation-harness \
  -v "$MODEL_ROOT":/data/share \
  -w /phyai_workspace \
  --cap-add=SYS_ADMIN \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --name "$PHYAI_CONTAINER" \
  nvcr.io/nvidia/pytorch:25.12-py3 bash
```

容器内同步 PhyAI 环境：

```bash
docker exec "$PHYAI_CONTAINER" bash -lc '
cd /phyai_workspace
python3 -m pip install -U uv
uv sync
'
```

如果 `uv sync` 生成的 editable path 与容器路径不一致，可加兼容 symlink。只有在 import 报错指向旧宿主路径时才需要执行；`HOST_USER` 填目标机器上的用户名：

```bash
export HOST_USER=$(id -un)
export COMPAT_PARENT=/compat_mount

docker exec "$PHYAI_CONTAINER" bash -lc "
mkdir -p $COMPAT_PARENT/$HOST_USER
ln -sfn /phyai_workspace $COMPAT_PARENT/$HOST_USER/phyai
"
```

确认容器内 import：

```bash
docker exec "$PHYAI_CONTAINER" bash -lc '
cd /phyai_workspace
export PYTHONPATH=/phyai_workspace/phyai/src:/phyai_workspace/phyai-kernel:/phyai_workspace/phyai-utils-tools/src:/vla-evaluation-harness/src
/phyai_workspace/.venv/bin/python - <<PY
import phyai
import vla_eval
print("phyai", phyai.__file__)
print("vla_eval", vla_eval.__file__)
PY
'
```

## 6. 运行前检查

检查 GPU、端口和 Docker：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
ss -ltnp | grep -E ':8000|:8001' || true
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

如 GPU 上已有重负载，完整 benchmark 的耗时会不稳定，建议换空闲 GPU 或等待资源释放。

## 7. 启动 PhyAI pi0.5 server

取 PhyAI 容器 IP：

```bash
export PHYAI_CONTAINER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PHYAI_CONTAINER")
export PHYAI_SERVER_URL=ws://$PHYAI_CONTAINER_IP:8000
echo "$PHYAI_SERVER_URL"
```

启动 server：

```bash
mkdir -p "$VLA_ROOT/results"

tmux new-session -d -s phyai_pi05_libero_server "
docker exec $PHYAI_CONTAINER bash -lc '
cd /vla-evaluation-harness
export PYTHONPATH=/phyai_workspace/phyai/src:/phyai_workspace/phyai-kernel:/phyai_workspace/phyai-utils-tools/src:/vla-evaluation-harness/src
export PHYAI_TOKENIZER_PATH=$TOKENIZER_IN_CONTAINER
export PHYAI_CAMERA_MODE=two_camera
/phyai_workspace/.venv/bin/python -m vla_eval.model_servers.phyai \
  --checkpoint_path $PHYAI_CKPT_IN_CONTAINER \
  --device cuda:0 \
  --params_dtype bfloat16 \
  --attn_backend flashinfer \
  --norm_backend phyai-kernel \
  --linear_backend flashinfer \
  --flashinfer_workspace_bytes 536870912 \
  --chunk_size 10 \
  --host 0.0.0.0 \
  --port 8000
' 2>&1 | tee $VLA_ROOT/results/phyai_pi05_libero_server.log
"
```

关键配置含义：

| 配置 | 值 | 说明 |
| --- | --- | --- |
| `--checkpoint_path` | `/data/share/pi05_libero_phyai_converted` | PhyAI converted pi0.5 LIBERO 权重 |
| `PHYAI_TOKENIZER_PATH` | `/data/share/paligemma-3b-pt-224` | tokenizer / processor 文件 |
| `PHYAI_CAMERA_MODE` | `two_camera` | LIBERO 发送 agentview 与 wrist 两路图像 |
| `--params_dtype` | `bfloat16` | 参数 dtype |
| `--attn_backend` | `flashinfer` | attention 后端 |
| `--norm_backend` | `phyai-kernel` | norm 后端 |
| `--linear_backend` | `flashinfer` | linear 后端 |
| `--flashinfer_workspace_bytes` | `536870912` | 512MiB workspace |
| `--chunk_size` | `10` | 每次推理产出 10 个 action |
| CUDA graph | 默认开启 | 不要传 `--no-use_cuda_graph` |

等待 ready：

```bash
tail -f "$VLA_ROOT/results/phyai_pi05_libero_server.log"
```

必须看到：

```text
capturing vision-tower CUDA graph
capturing 4 prefix-forward CUDA graph(s)
capturing the full 10-step Euler loop as one CUDA graph
Starting server on ws://0.0.0.0:8000
```

## 8. 创建 smoke 配置并验证

先跑一个最小 smoke，确认模型、LIBERO Docker、WebSocket、timing 字段都可用。

```bash
cat > "$VLA_ROOT/configs/benchmarks/libero/smoke_test_phyai_local.yaml" <<YAML
server:
  url: "$PHYAI_SERVER_URL"

docker:
  image: $LIBERO_IMAGE

output_dir: "$VLA_ROOT/results/phyai_pi05_libero_smoke"

benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    name: "phyai_pi05_libero_smoke"
    episodes_per_task: 1
    max_tasks: 1
    mode: "sync"
    params:
      suite: libero_spatial
      seed: 7
      num_steps_wait: 10
YAML

cd "$VLA_ROOT"
env NO_PROXY='*' no_proxy='*' ./.venv/bin/vla-eval run \
  --config configs/benchmarks/libero/smoke_test_phyai_local.yaml \
  --server-url "$PHYAI_SERVER_URL" \
  --dev \
  --yes
```

检查 smoke 结果：

```bash
cd "$VLA_ROOT"
json=$(ls -t results/phyai_pi05_libero_smoke/*.json | head -1)
./.venv/bin/python scripts/summarize_timing.py "$json"
```

必须看到：

```text
raw_chunk_size_max=10 served_chunk_size_max=10
```

如果没有 timing 或 chunk 字段，通常是 `vla-eval run` 没有加 `--dev`，导致容器没有挂载宿主的 vla-harness `src`。

## 9. 顺序跑 LIBERO 四组件

创建长跑脚本。脚本会为每个 suite 生成一个运行时 YAML，显式写入本次 `RUN_ID` 的 `output_dir`，避免结果散落在全局 `results/` 目录里。

```bash
cat > "$VLA_ROOT/run_phyai_pi05_libero_four_suites.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

: "${PHYAI_SERVER_URL:?must set PHYAI_SERVER_URL}"
: "${PHYAI_CKPT_IN_CONTAINER:?must set PHYAI_CKPT_IN_CONTAINER}"
: "${LIBERO_IMAGE:=ghcr.io/allenai/vla-evaluation-harness/libero:latest}"

cd "$(dirname "$0")"

RUN_ID="phyai_pi05_libero_four_$(date +%Y%m%d_%H%M%S)"
OUT="results/${RUN_ID}"
mkdir -p "$OUT/configs"

{
  echo "RUN_ID=${RUN_ID}"
  echo "START=$(date -Is)"
  echo "MODEL=phyai_pi05"
  echo "SERVER_URL=${PHYAI_SERVER_URL}"
  echo "CHECKPOINT=${PHYAI_CKPT_IN_CONTAINER}"
  echo "PHYAI_CAMERA_MODE=two_camera"
  echo "MODE=sync"
  echo "CHUNK_SIZE=10"
  echo "SERVER_CONFIG=use_cuda_graph=True attn=flashinfer norm=phyai-kernel linear=flashinfer workspace=536870912 params_dtype=bfloat16"
} | tee "$OUT/run_summary.log"

make_cfg() {
  local suite_name="$1"
  local suite="$2"
  local cfg="$3"
  cat > "$cfg" <<YAML
server:
  url: "${PHYAI_SERVER_URL}"

docker:
  image: ${LIBERO_IMAGE}

output_dir: "${OUT}"

benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    subname: ${suite}
    episodes_per_task: 50
    mode: "sync"
    params:
      suite: ${suite}
      seed: 7
      num_steps_wait: 10
YAML
}

for item in \
  spatial:libero_spatial \
  object:libero_object \
  goal:libero_goal \
  libero10:libero_10
do
  name="${item%%:*}"
  suite="${item#*:}"
  cfg="$OUT/configs/${name}.yaml"
  log="$OUT/phyai_${name}.log"
  make_cfg "$name" "$suite" "$cfg"

  echo "SUITE_START model=phyai suite=${name} cfg=${cfg} server=${PHYAI_SERVER_URL} time=$(date -Is)" | tee -a "$OUT/run_summary.log"

  {
    echo "MODEL=phyai"
    echo "SUITE=${name}"
    echo "LIBERO_SUITE=${suite}"
    echo "CONFIG=${cfg}"
    echo "START_ISO=$(date -Is)"
    /usr/bin/time -p env NO_PROXY='*' no_proxy='*' ./.venv/bin/vla-eval run \
      --config "$cfg" \
      --server-url "$PHYAI_SERVER_URL" \
      --dev \
      --yes
    echo "END_ISO=$(date -Is)"
  } 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}

  echo "SUITE_END model=phyai suite=${name} status=${status} log=${log} time=$(date -Is)" | tee -a "$OUT/run_summary.log"

  result_json=$(ls -t "$OUT/${suite}_sync_"*.json "$OUT"/*"${suite}"*_sync_*.json 2>/dev/null | head -1 || true)
  if [ -n "$result_json" ]; then
    ./.venv/bin/python scripts/summarize_timing.py "$result_json" | sed "s/^/TIMING phyai_${name} /" | tee -a "$OUT/run_summary.log"
  else
    echo "WARN no result json found for ${name}" | tee -a "$OUT/run_summary.log"
  fi

  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

echo "ALL_DONE $(date -Is)" | tee -a "$OUT/run_summary.log"
SH
chmod +x "$VLA_ROOT/run_phyai_pi05_libero_four_suites.sh"
```

启动长跑：

```bash
cd "$VLA_ROOT"
tmux new-session -d -s phyai_pi05_libero_four \
  "PHYAI_SERVER_URL=$PHYAI_SERVER_URL PHYAI_CKPT_IN_CONTAINER=$PHYAI_CKPT_IN_CONTAINER LIBERO_IMAGE=$LIBERO_IMAGE ./run_phyai_pi05_libero_four_suites.sh"
```

查看进度：

```bash
tmux ls
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
sed -n '1,220p' "$latest/run_summary.log"
tail -80 "$latest"/phyai_spatial.log
```

## 10. 解析成功率与 timing

长跑完成后：

```bash
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
cat "$latest/run_summary.log"
```

用 `summarize_timing.py` 重新汇总全部 JSON：

```bash
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
./.venv/bin/python scripts/summarize_timing.py "$latest"/*.json
```

统计成功率：

```bash
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
./.venv/bin/python - <<'PY' "$latest"/*.json
import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    p = Path(arg)
    data = json.loads(p.read_text())
    eps = [ep for task in data.get("tasks", []) for ep in task.get("episodes", [])]
    succ = sum(1 for ep in eps if ep.get("metrics", {}).get("success"))
    total = len(eps)
    rate = succ / total * 100.0 if total else 0.0
    steps = sum(int(ep.get("steps", 0)) for ep in eps)
    print(f"{p.name}: success={succ}/{total} rate={rate:.1f}% steps={steps} mean_success={data.get('mean_success')}")
PY
```

必须记录的字段：

```text
RUN_ID
结果目录
checkpoint
server_url
suite
success / total
success rate
steps
/usr/bin/time -p real
model_wait_sec
model_inference_sec
env_step_sec
obs_sec
avg_model_inference_ms
model_inference_calls
model_buffer_hits
raw_chunk_size_max
served_chunk_size_max
```

`raw_chunk_size_max=10` 且 `served_chunk_size_max=10` 是四组件复现的关键校验项。

## 11. 预期参考结果

不同机器、GPU、驱动、负载会影响耗时；成功率也可能有少量随机波动。此前同口径参考结果如下：

| 组件 | 成功率 | 成功数 | steps | 总耗时 | 模型纯推理时间 | env step 时间 | benchmark 等待 action 时间 | 平均单次模型推理 | 推理调用次数 | buffer hit | chunk 验证 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `libero_spatial` | 97.8% | 489/500 | 52,926 | 4,198.73s | 200.56s | 1,239.61s | 2,269.16s | 36.33ms | 5,520 | 47,406 | raw=10, served=10 |
| `libero_object` | 99.8% | 499/500 | 68,712 | 5,199.23s | 256.92s | 1,169.84s | 3,453.50s | 36.18ms | 7,102 | 61,610 | raw=10, served=10 |
| `libero_goal` | 98.0% | 490/500 | 56,292 | 3,981.09s | 210.86s | 1,045.35s | 2,365.71s | 36.10ms | 5,841 | 50,451 | raw=10, served=10 |
| `libero_10` | 94.2% | 471/500 | 134,962 | 9,150.13s | 496.56s | 2,227.75s | 6,237.24s | 36.21ms | 13,713 | 121,249 | raw=10, served=10 |

参考值不是验收硬阈值。复现时优先确认：

```text
四组件都完成 500 episodes
chunk 验证 raw=10 served=10
server 日志确认 CUDA graph capture
结果 JSON 包含 timing 字段
```

## 12. 常见问题

### 12.1 LIBERO Docker 镜像架构不匹配

如果目标机器是 ARM64，而官方镜像只有 amd64，需要本地构建：

```bash
cd "$VLA_ROOT"
export DOCKER_DEFAULT_PLATFORM=linux/arm64
docker/build.sh libero
```

### 12.2 容器连不上 PhyAI server

PhyAI server 如果跑在 bridge Docker 容器里，宿主侧 `vla-eval` 需要连容器 IP：

```bash
export PHYAI_CONTAINER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PHYAI_CONTAINER")
export PHYAI_SERVER_URL=ws://$PHYAI_CONTAINER_IP:8000
```

### 12.3 JSON 没有 timing 字段

运行 benchmark 必须加 `--dev`。这会把宿主 `$VLA_ROOT/src` 挂载进 LIBERO 容器，确保使用带 timing 的 runner。

### 12.4 Paligemma tokenizer 缺失

`paligemma-3b-pt-224` 是 gated 资源。建议从已有机器同步到 `$MODEL_ROOT/paligemma-3b-pt-224`，不要依赖复现机器现场下载。

### 12.5 PhyAI server 没有 CUDA graph capture 日志

检查启动命令是否误传了 `--no-use_cuda_graph`，或是否走了错误 server adapter。正确日志必须包含：

```text
capturing vision-tower CUDA graph
capturing 4 prefix-forward CUDA graph(s)
capturing the full 10-step Euler loop as one CUDA graph
```

### 12.6 释放资源

```bash
tmux kill-session -t phyai_pi05_libero_four || true
tmux kill-session -t phyai_pi05_libero_server || true
docker stop "$PHYAI_CONTAINER" || true
ss -ltnp | grep ':8000' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

