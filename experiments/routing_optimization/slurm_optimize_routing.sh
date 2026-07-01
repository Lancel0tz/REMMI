#!/usr/bin/env bash
#SBATCH -J optimize-routing
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/optimize_routing_%j.out
#SBATCH --error=logs/optimize_routing_%j.err

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

# ── Optimization Parameters ──────────────────────────────────────────────────

# Number of optimization trials (default: 50)
N_TRIALS="${N_TRIALS:-50}"

# Dataset to optimize on (standard or hard)
DATASET_TYPE="${DATASET_TYPE:-standard}"

# Output directory
OUTPUT_BASE="${OUTPUT_BASE:-output/optimization_results}"
OUTPUT_DIR="${OUTPUT_BASE}/${DATASET_TYPE}_set_trials${N_TRIALS}"

# QA file path
if [[ "${DATASET_TYPE}" == "hard" ]]; then
  QA_FILE="data/atm-bench/atm-bench-hard.json"
else
  QA_FILE="data/atm-bench/atm-bench.json"
fi

# ── Header ───────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "  ROUTING STRATEGY OPTIMIZATION (R@10 as PRIMARY METRIC)"
echo "================================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "  Configuration:"
echo "    Dataset type: ${DATASET_TYPE}"
echo "    QA file: ${QA_FILE}"
echo "    Number of trials: ${N_TRIALS}"
echo "    Output directory: ${OUTPUT_DIR}"
echo "    Primary metric: R@10 (Recall@10)"
echo ""
echo "================================================================================"
echo ""

# ── Check dependencies ───────────────────────────────────────────────────────

echo "[Setup] Checking dependencies..."

# Check and install packages using conda-activated Python
python -c "import optuna; print('✅ optuna:', optuna.__version__)" 2>/dev/null || {
  echo "⚠️ optuna not installed, installing..."
  pip install -q optuna
}

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

# ── Run optimization ─────────────────────────────────────────────────────────

echo "[Optimization] Starting Bayesian optimization pipeline..."
echo ""

python scripts/QA_Agent/MMRAG/run_optimization_pipeline.py \
  --qa-file "${QA_FILE}" \
  --n-trials "${N_TRIALS}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-type "${DATASET_TYPE}" \
  --seed 42

# ── Post-processing ──────────────────────────────────────────────────────────

echo ""
echo "================================================================================"
echo "  OPTIMIZATION COMPLETE"
echo "================================================================================"

# Check results
if [[ -f "${OUTPUT_DIR}/optimization_summary.json" ]]; then
  echo ""
  echo "✅ Results Summary:"
  python3 -c "
import json
with open('${OUTPUT_DIR}/optimization_summary.json') as f:
    summary = json.load(f)
    print(f\"  Best Trial: #{summary['best_trial_id']}\")
    print(f\"  Best R@10: {summary['best_r10']:.4f}\")
    print(f\"  Total trials: {summary['n_trials']}\")
"

  echo ""
  echo "📂 Output files:"
  echo "   - ${OUTPUT_DIR}/optimization_summary.json (best configuration)"
  echo "   - ${OUTPUT_DIR}/trials_history.json (all trials)"
  echo "   - ${OUTPUT_DIR}/trial_*/ (individual trial details)"

else
  echo "❌ Optimization failed - summary file not found"
  exit 1
fi

echo ""
echo "================================================================================"
echo "  Job finished at $(date)"
echo "================================================================================"
