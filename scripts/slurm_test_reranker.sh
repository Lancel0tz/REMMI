#!/usr/bin/env bash
#SBATCH -J test-reranker
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/test_reranker_%j.out
#SBATCH --error=logs/test_reranker_%j.err

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"

mkdir -p logs output/reranker_test

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

# ── Parameters ────────────────────────────────────────────────────────────

QA_FILE="${QA_FILE:-data/atm-bench/atm-bench.json}"
RERANK_TOP_K="${RERANK_TOP_K:-50}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-output/reranker_test}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "  RERANKER EFFECT TEST"
echo "================================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo ""
echo "  Configuration:"
echo "    QA File: ${QA_FILE}"
echo "    Reranker Model: ${RERANKER_MODEL}"
echo "    Rerank Top-K: ${RERANK_TOP_K}"
echo "    Device: ${DEVICE}"
echo "    Output: ${OUTPUT_DIR}/"
echo ""
echo "================================================================================"
echo ""

# ── Check dependencies ───────────────────────────────────────────────────────

echo "[Setup] Checking dependencies..."

python -c "import torch; print('✅ torch:', torch.__version__)" || {
  echo "❌ ERROR: torch not found"
  exit 1
}

python -c "import transformers; print('✅ transformers:', transformers.__version__)" || {
  echo "⚠️ transformers not installed, installing..."
  pip install -q transformers
}

echo ""

# ── Run test ─────────────────────────────────────────────────────────────────

echo "[Test] Starting Reranker effect test..."
echo ""

python scripts/test_reranker_effect.py \
  --qa-file "${QA_FILE}" \
  --device "${DEVICE}" \
  --rerank-top-k "${RERANK_TOP_K}" \
  --reranker-model "${RERANKER_MODEL}" \
  --output-dir "${OUTPUT_DIR}"

# ── Post-processing ──────────────────────────────────────────────────────────

echo ""
echo "================================================================================"
echo "  TEST COMPLETE"
echo "================================================================================"

# Check results
if [[ -f "${OUTPUT_DIR}/reranker_comparison.json" ]]; then
  echo ""
  echo "✅ Results Summary:"
  python3 -c "
import json
with open('${OUTPUT_DIR}/reranker_comparison.json') as f:
    data = json.load(f)
    metrics = data['metrics']
    print(f\"  Retrieval Only R@10:  {metrics['retrieval_only']['r10']:.4f}\")
    print(f\"  With Reranker R@10:   {metrics['with_reranker']['r10']:.4f}\")
    print(f\"  Improvement:          {metrics['improvement']['r10']:+.4f} ({metrics['improvement']['r10']*100:+.2f}%)\")
"

  echo ""
  echo "📂 Output file:"
  echo "   - ${OUTPUT_DIR}/reranker_comparison.json"

else
  echo "❌ Test failed - results file not found"
  exit 1
fi

echo ""
echo "================================================================================"
echo "  Job finished at $(date)"
echo "================================================================================"
