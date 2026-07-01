#!/usr/bin/env bash
#SBATCH -J eval-best-rerank
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_best_rerank_%j.out
#SBATCH --error=logs/eval_best_rerank_%j.err

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

# Dataset selection (standard or hard)
DATASET_TYPE="${DATASET_TYPE:-standard}"

# QA file and matching optimized routing config
if [[ "${DATASET_TYPE}" == "hard" ]]; then
  QA_FILE="data/atm-bench/atm-bench-hard.json"
  CONFIG_DEFAULT="config/best_routing_config_hard.json"
  OUTPUT_SUFFIX="hard"
else
  QA_FILE="data/atm-bench/atm-bench.json"
  CONFIG_DEFAULT="config/best_routing_config.json"
  OUTPUT_SUFFIX="standard"
fi

# Configuration
CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DEFAULT}}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
USE_RERANKER="${USE_RERANKER:-true}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-output/answers_best_config_reranker_${OUTPUT_SUFFIX}}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "  GENERATE ANSWERS WITH BEST CONFIG + RERANKER"
echo "================================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "  Configuration:"
echo "    Dataset type: ${DATASET_TYPE}"
echo "    QA file: ${QA_FILE}"
echo "    Config file: ${CONFIG_FILE}"
echo "    Use reranker: ${USE_RERANKER}"
echo "    Reranker model: ${RERANKER_MODEL}"
echo "    Output directory: ${OUTPUT_DIR}"
echo ""
echo "================================================================================"
echo ""

# ── Dependency checks ────────────────────────────────────────────────────────

echo "[Setup] Checking dependencies..."

python -c "import torch; print('✅ torch:', torch.__version__)" || {
  echo "❌ ERROR: torch not found in environment"
  echo "    Try: conda activate atmbench && pip install torch"
  exit 1
}

python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || {
  echo "⚠️ transformers not installed, installing..."
  pip install -q transformers
}

echo ""

# ── Run evaluation ───────────────────────────────────────────────────────────

echo "[Evaluation] Starting answer generation with best config + reranker..."
echo ""

if [[ "${USE_RERANKER}" == "true" ]]; then
  python scripts/eval_with_best_config.py \
    --qa-file "${QA_FILE}" \
    --config-file "${CONFIG_FILE}" \
    --use-reranker \
    --reranker-model "${RERANKER_MODEL}" \
    --device cuda \
    --output-dir "${OUTPUT_DIR}"
else
  python scripts/eval_with_best_config.py \
    --qa-file "${QA_FILE}" \
    --config-file "${CONFIG_FILE}" \
    --device cuda \
    --output-dir "${OUTPUT_DIR}"
fi

# ── Post-processing ──────────────────────────────────────────────────────────

echo ""
echo "================================================================================"
echo "  GENERATION COMPLETE"
echo "================================================================================"

# Check if answers were generated
if [[ -f "${OUTPUT_DIR}/answers.jsonl" ]]; then
  ANSWER_COUNT=$(wc -l < "${OUTPUT_DIR}/answers.jsonl")
  echo ""
  echo "✅ Answers generated successfully!"
  echo ""
  echo "📂 Output location:"
  echo "   ${OUTPUT_DIR}/"
  echo ""
  echo "📊 Results:"
  echo "   Total answers: ${ANSWER_COUNT}"
  echo "   Answers file: ${OUTPUT_DIR}/answers.jsonl"
  echo "   Config info: ${OUTPUT_DIR}/config_info.json"
  echo ""
  echo "🎯 Next step: Evaluate with LLM Judge"
  echo "   python memqa/utils/evaluator/evaluate_qa.py \\"
  echo "     --ground-truth ${QA_FILE} \\"
  echo "     --predictions ${OUTPUT_DIR}/answers.jsonl \\"
  echo "     --output-dir ${OUTPUT_DIR}/eval \\"
  echo "     --metrics llm em atm \\"
  echo "     --judge-model gpt-5-mini"
else
  echo "❌ Answer generation failed - answers.jsonl not found"
  exit 1
fi

echo ""
echo "================================================================================"
echo "  Job finished at $(date)"
echo "================================================================================"
