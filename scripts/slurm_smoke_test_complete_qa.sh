#!/usr/bin/env bash
#SBATCH -J smoke-test-qa
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_test_qa_%j.out
#SBATCH --error=logs/smoke_test_qa_%j.err

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"

mkdir -p logs

# Conda environment
if [[ -f /home/kz345/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/kz345/miniconda3/etc/profile.d/conda.sh
  conda activate atmbench
else
  echo "ERROR: conda not found" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Force node-local disk for temp files & compilation caches.
# 必须无条件覆盖继承来的 TMPDIR (sbatch --export=ALL 会把提交环境的 /tmp/... 传进来)，
# 否则 Triton/Inductor 会把 kernel 写进只有 ~16G 的 /tmp tmpfs，写满后 vLLM 引擎崩溃 → 500 错误。
export TMPDIR="/local/${USER}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PIP_CACHE_DIR="/local/${USER}/pip_cache"
export TRITON_CACHE_DIR="/local/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/local/${USER}/inductor_cache"
export XDG_CACHE_HOME="/local/${USER}/xdg_cache"
export VLLM_CACHE_ROOT="/local/${USER}/vllm_cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

# Avoid Triton/CUDA build issues
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export AS=/usr/bin/as
export LD=/usr/bin/ld
export AR=/usr/bin/ar
export PATH=/usr/bin:/bin:${PATH}

# ── Parameters ───────────────────────────────────────────────────────────────

QA_FILE="${QA_FILE:-data/atm-bench/atm-bench-hard.json}"
CONFIG_FILE="${CONFIG_FILE:-auto}"  # 自动根据数据集选择配置
OUTPUT_DIR="${OUTPUT_DIR:-output/smoke_test_hard}"
USE_RERANKER="${USE_RERANKER:-true}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-10}"
VL_EMBEDDING_MODEL="${VL_EMBEDDING_MODEL:-openai/clip-vit-large-patch14}"
VL_BATCH_SIZE="${VL_BATCH_SIZE:-16}"
RERANKER_BATCH_SIZE="${RERANKER_BATCH_SIZE:-1}"
RERANKER_MAX_LENGTH="${RERANKER_MAX_LENGTH:-512}"
EVIDENCE_TEXT_MAX_CHARS="${EVIDENCE_TEXT_MAX_CHARS:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
INSERT_RAW_IMAGES="${INSERT_RAW_IMAGES:-true}"

# vLLM 配置
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.8}"
# 插入原始图像后多模态 prompt 可达 ~12K tokens，8192 会触发 400 (context 超长)。
# Qwen3-VL 支持长上下文，设 32768 容纳 raw-image prompt 并留出输出空间。
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "  🧪 SMOKE TEST: COMPLETE QA PIPELINE (硬题集)"
echo "================================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "  Configuration:"
echo "    QA File: ${QA_FILE}"
echo "    Config File: ${CONFIG_FILE}"
echo "    Output Dir: ${OUTPUT_DIR}"
echo "    Use Reranker: ${USE_RERANKER}"
echo "    vLLM Model: ${VLLM_MODEL}"
echo "    vLLM Port: ${VLLM_PORT}"
echo "    Retrieval Top K: ${RETRIEVAL_TOP_K}"
echo "    VL Embedding: ${VL_EMBEDDING_MODEL}"
echo "    Force Rebuild: ${FORCE_REBUILD}"
echo "    Evidence Max Chars: ${EVIDENCE_TEXT_MAX_CHARS}"
echo "    Max Tokens: ${MAX_TOKENS}"
echo "    Insert Raw Images: ${INSERT_RAW_IMAGES}"
echo ""
echo "================================================================================"
echo ""

# ── Check dependencies ───────────────────────────────────────────────────────

echo "[Setup] Checking dependencies..."

python -c "import torch; print('✅ torch:', torch.__version__)" || {
  echo "❌ ERROR: torch not found"
  exit 1
}

python -c "import vllm; print('✅ vllm:', vllm.__version__)" || {
  echo "⚠️ vllm not installed, installing..."
  pip install -q vllm
}

echo ""

# ── Start vLLM server ────────────────────────────────────────────────────────

echo "[vLLM] Starting vLLM server..."
echo "  Model: ${VLLM_MODEL}"
echo "  Port: ${VLLM_PORT}"
echo "  GPU Memory: ${VLLM_GPU_MEM}"

# Kill any existing process on this port to avoid "Address already in use"
echo "  Clearing port ${VLLM_PORT}..."
fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true
sleep 1

python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${VLLM_GPU_MEM}" \
  --port "${VLLM_PORT}" \
  --trust-remote-code \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --enforce-eager \
  > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"

# Wait for vLLM to start (Qwen3-VL 权重从 RDS 加载约需 3 分钟，enforce-eager 跳过编译，留足 10 分钟余量)
echo "  Waiting for vLLM to start... (权重加载约 3-5 分钟)"
sleep 15
for i in {1..300}; do
  if curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
    echo "  ✅ vLLM server ready (总耗时: $((15 + i*2)) 秒)"
    break
  fi
  # 如果 vLLM 进程已经死了，立刻报错，不用等满超时
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "  ❌ vLLM 进程已退出 (PID ${VLLM_PID} 不存在)"
    echo "  检查日志: tail -50 logs/vllm_${SLURM_JOB_ID}.log"
    tail -50 logs/vllm_${SLURM_JOB_ID}.log
    exit 1
  fi
  if [[ $i -eq 300 ]]; then
    echo "  ❌ vLLM server failed to start (超时 615 秒)"
    echo "  检查日志: tail -50 logs/vllm_${SLURM_JOB_ID}.log"
    kill ${VLLM_PID} 2>/dev/null || true
    tail -50 logs/vllm_${SLURM_JOB_ID}.log
    exit 1
  fi
  if [[ $((i % 30)) -eq 0 ]]; then
    echo "  等待中... $((i*2)) 秒"
  fi
  sleep 2
done

echo ""

# ── Run answer generation ────────────────────────────────────────────────────

echo "[Generation] Starting answer generation..."
echo ""

RERANKER_FLAG=""
if [[ "${USE_RERANKER}" == "true" ]]; then
  RERANKER_FLAG="--use-reranker --reranker-model ${RERANKER_MODEL}"
fi

FORCE_REBUILD_FLAG=""
if [[ "${FORCE_REBUILD}" == "true" || "${FORCE_REBUILD}" == "1" ]]; then
  FORCE_REBUILD_FLAG="--force-rebuild"
fi

RAW_IMAGE_FLAG=""
if [[ "${INSERT_RAW_IMAGES}" == "true" || "${INSERT_RAW_IMAGES}" == "1" ]]; then
  RAW_IMAGE_FLAG="--insert-raw-images"
fi

python scripts/eval_with_best_config_complete.py \
  --qa-file "${QA_FILE}" \
  --config-file "${CONFIG_FILE}" \
  ${RERANKER_FLAG} \
  --device cuda \
  --provider vllm \
  --model "${VLLM_MODEL}" \
  --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
  --output-dir "${OUTPUT_DIR}" \
  --retrieval-top-k "${RETRIEVAL_TOP_K}" \
  --vl-embedding-model "${VL_EMBEDDING_MODEL}" \
  --vl-batch-size "${VL_BATCH_SIZE}" \
  --evidence-text-max-chars "${EVIDENCE_TEXT_MAX_CHARS}" \
  --max-tokens "${MAX_TOKENS}" \
  --reranker-batch-size "${RERANKER_BATCH_SIZE}" \
  --reranker-max-length "${RERANKER_MAX_LENGTH}" \
  ${FORCE_REBUILD_FLAG} \
  ${RAW_IMAGE_FLAG} \
  --max-workers 2 \
  --num-frames 8 \
  --max-total-frames 32

GENERATION_RESULT=$?

# ── Cleanup ──────────────────────────────────────────────────────────────────

echo ""
echo "[Cleanup] Stopping vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
sleep 2

# ── Post-processing ──────────────────────────────────────────────────────────

echo ""
echo "================================================================================"
echo "  SMOKE TEST COMPLETE"
echo "================================================================================"

if [[ ${GENERATION_RESULT} -eq 0 ]] && [[ -f "${OUTPUT_DIR}/mmrag_answers.jsonl" ]]; then
  ANSWER_COUNT=$(wc -l < "${OUTPUT_DIR}/mmrag_answers.jsonl")
  echo ""
  echo "✅ Smoke test successful!"
  echo ""
  echo "📊 Results:"
  echo "   Total answers: ${ANSWER_COUNT}"
  echo "   Output: ${OUTPUT_DIR}/"
  echo ""

  # Show sample answer + 成功/失败统计
  echo "📝 Sample answer & stats:"
  OUTPUT_DIR="${OUTPUT_DIR}" python3 << 'EOF'
import json, os
out_dir = os.environ["OUTPUT_DIR"]
ok = failed = 0
sample = None
with open(f"{out_dir}/mmrag_answers.jsonl") as f:
    for line in f:
        a = json.loads(line)
        if a.get("status") == "ok":
            ok += 1
            if sample is None:
                sample = a
        else:
            failed += 1
print(f"   ✅ ok: {ok}   ❌ failed: {failed}")
if sample is not None:
    print(f"   Sample ID: {sample.get('id')}")
    print(f"   Sample Answer: {str(sample.get('answer',''))[:120]}...")
    if 'total_tokens' in sample:
        print(f"   Tokens: {sample['total_tokens']}")
else:
    print("   ⚠️ 没有任何成功的答案，请检查 vLLM endpoint 配置")
EOF

  echo ""
  echo "🎯 Next step: Run full pipeline on standard set"
  echo "   sbatch scripts/slurm_complete_qa.sh"

else
  echo "❌ Smoke test failed"
  exit 1
fi

echo ""
echo "================================================================================"
echo "  Job finished at $(date)"
echo "================================================================================"
