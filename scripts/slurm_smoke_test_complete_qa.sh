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

# Prefer local disk for temporary files
export TMPDIR="${TMPDIR:-/local/${USER}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/local/${USER}/pip_cache}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

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

# vLLM 配置
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.8}"

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

python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${VLLM_GPU_MEM}" \
  --port "${VLLM_PORT}" \
  --trust-remote-code \
  --max-model-len 8192 \
  > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"

# Wait for vLLM to start (增加等待时间，模型加载需要 20-30 秒)
echo "  Waiting for vLLM to start... (可能需要 30-60 秒)"
sleep 15
for i in {1..60}; do
  if curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
    echo "  ✅ vLLM server ready"
    break
  fi
  if [[ $i -eq 60 ]]; then
    echo "  ❌ vLLM server failed to start (超时 120 秒)"
    echo "  检查日志: tail -50 logs/vllm_${SLURM_JOB_ID}.log"
    kill ${VLLM_PID} 2>/dev/null || true
    tail -50 logs/vllm_${SLURM_JOB_ID}.log
    exit 1
  fi
  if [[ $((i % 10)) -eq 0 ]]; then
    echo "  等待中... $i/60 秒"
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

python scripts/eval_with_best_config_complete.py \
  --qa-file "${QA_FILE}" \
  --config-file "${CONFIG_FILE}" \
  ${RERANKER_FLAG} \
  --device cuda \
  --provider vllm \
  --model "${VLLM_MODEL}" \
  --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
  --output-dir "${OUTPUT_DIR}" \
  --max-workers 2 \
  --num-frames 4 \
  --max-total-frames 8

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

  # Show sample answer
  echo "📝 Sample answer:"
  python3 << 'EOF'
import json
with open(f"{OUTPUT_DIR}/mmrag_answers.jsonl") as f:
    answer = json.loads(f.readline())
    print(f"   ID: {answer.get('id')}")
    print(f"   Status: {answer.get('status')}")
    answer_text = answer.get('answer', '')
    print(f"   Answer: {answer_text[:100]}...")
    if 'total_tokens' in answer:
        print(f"   Tokens: {answer['total_tokens']}")
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
