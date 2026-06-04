#!/usr/bin/env bash
#SBATCH -J complete-qa
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/complete_qa_%j.out
#SBATCH --error=logs/complete_qa_%j.err

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

QA_FILE="${QA_FILE:-data/atm-bench/atm-bench.json}"
CONFIG_FILE="${CONFIG_FILE:-auto}"  # 自动根据数据集选择配置
OUTPUT_DIR="${OUTPUT_DIR:-output/answers_complete_qa_standard}"
USE_RERANKER="${USE_RERANKER:-true}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"

# vLLM 配置
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.8}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "  COMPLETE QA PIPELINE (标准集)"
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
echo "  Questions: $(grep -c '\"id\"' ${QA_FILE} || echo 'unknown')"
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
  --max-workers 4 \
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
echo "  PIPELINE COMPLETE"
echo "================================================================================"

if [[ ${GENERATION_RESULT} -eq 0 ]] && [[ -f "${OUTPUT_DIR}/mmrag_answers.jsonl" ]]; then
  ANSWER_COUNT=$(wc -l < "${OUTPUT_DIR}/mmrag_answers.jsonl")
  echo ""
  echo "✅ Pipeline successful!"
  echo ""
  echo "📊 Results:"
  echo "   Total answers: ${ANSWER_COUNT}"
  echo "   Output: ${OUTPUT_DIR}/"
  echo ""

  # Calculate statistics
  python3 << 'EOF'
import json
total_tokens = 0
total_prompt = 0
total_completion = 0
success_count = 0

with open(f"{OUTPUT_DIR}/mmrag_answers.jsonl") as f:
    for line in f:
        answer = json.loads(line)
        if answer.get('status') == 'ok':
            success_count += 1
            total_tokens += answer.get('total_tokens', 0)
            total_prompt += answer.get('prompt_tokens', 0)
            total_completion += answer.get('completion_tokens', 0)

print(f"📈 Token Statistics:")
print(f"   Successful answers: {success_count}")
if total_tokens > 0:
    print(f"   Total tokens: {total_tokens}")
    print(f"   Prompt tokens: {total_prompt}")
    print(f"   Completion tokens: {total_completion}")
    print(f"   Avg tokens per Q: {total_tokens / success_count:.0f}")
EOF

  echo ""
  echo "🎯 Next step: Evaluate with LLM Judge"
  echo "   python memqa/utils/evaluator/evaluate_qa.py \\"
  echo "     --ground-truth ${QA_FILE} \\"
  echo "     --predictions ${OUTPUT_DIR}/mmrag_answers.jsonl \\"
  echo "     --output-dir ${OUTPUT_DIR}/eval \\"
  echo "     --metrics llm em atm"

else
  echo "❌ Pipeline failed"
  if [[ -f "logs/vllm_${SLURM_JOB_ID}.log" ]]; then
    echo ""
    echo "vLLM logs:"
    tail -20 "logs/vllm_${SLURM_JOB_ID}.log"
  fi
  exit 1
fi

echo ""
echo "================================================================================"
echo "  Job finished at $(date)"
echo "================================================================================"
