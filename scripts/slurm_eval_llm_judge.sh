#!/usr/bin/env bash
#SBATCH -J eval-llm-judge
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_llm_judge_%j.out
#SBATCH --error=logs/eval_llm_judge_%j.err

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

# ── Parameters ───────────────────────────────────────────────────────────────

# Predictions file
PREDICTIONS_FILE="${PREDICTIONS_FILE:-output/answers_best_config_reranker_standard/answers.jsonl}"

# Ground truth file
GROUND_TRUTH="${GROUND_TRUTH:-data/atm-bench/atm-bench.json}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-output/answers_best_config_reranker_standard/eval}"

# Judge settings
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
JUDGE_REASONING="${JUDGE_REASONING:-minimal}"
METRICS="${METRICS:-llm em atm}"
MAX_WORKERS="${MAX_WORKERS:-4}"
REQUEST_DELAY="${REQUEST_DELAY:-2.0}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "  LLM JUDGE EVALUATION"
echo "================================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo ""
echo "  Configuration:"
echo "    Predictions: ${PREDICTIONS_FILE}"
echo "    Ground Truth: ${GROUND_TRUTH}"
echo "    Output: ${OUTPUT_DIR}"
echo ""
echo "  Judge Settings:"
echo "    Provider: ${JUDGE_PROVIDER}"
echo "    Model: ${JUDGE_MODEL}"
echo "    Reasoning Effort: ${JUDGE_REASONING}"
echo "    Metrics: ${METRICS}"
echo "    Max Workers: ${MAX_WORKERS}"
echo "    Request Delay: ${REQUEST_DELAY}s"
echo ""
echo "================================================================================"
echo ""

# ── Validation ───────────────────────────────────────────────────────────────

echo "[Setup] Validating inputs..."

if [[ ! -f "${PREDICTIONS_FILE}" ]]; then
  echo "❌ ERROR: Predictions file not found: ${PREDICTIONS_FILE}"
  exit 1
fi

if [[ ! -f "${GROUND_TRUTH}" ]]; then
  echo "❌ ERROR: Ground truth file not found: ${GROUND_TRUTH}"
  exit 1
fi

PREDICTION_COUNT=$(wc -l < "${PREDICTIONS_FILE}")
echo "   ✓ Found predictions file: ${PREDICTION_COUNT} entries"
echo "   ✓ Ground truth file exists"
echo ""

# ── Run evaluation ───────────────────────────────────────────────────────────

echo "[Evaluation] Starting LLM Judge evaluation..."
echo ""

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "${GROUND_TRUTH}" \
  --predictions "${PREDICTIONS_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics ${METRICS} \
  --judge-provider "${JUDGE_PROVIDER}" \
  --judge-model "${JUDGE_MODEL}" \
  --judge-reasoning-effort "${JUDGE_REASONING}" \
  --max-workers "${MAX_WORKERS}" \
  --request-delay "${REQUEST_DELAY}"

# ── Post-processing ──────────────────────────────────────────────────────────

echo ""
echo "================================================================================"
echo "  EVALUATION COMPLETE"
echo "================================================================================"

# Check results
if [[ -f "${OUTPUT_DIR}/llm_judge_summary.json" ]]; then
  echo ""
  echo "✅ Evaluation successful!"
  echo ""
  echo "📊 LLM Judge Results:"
  python3 << 'EOF'
import json
import sys
try:
    with open(f"{OUTPUT_DIR}/llm_judge_summary.json") as f:
        data = json.load(f)
        print(f"  Accuracy:    {data['accuracy']:.2%}")
        print(f"  Avg Score:   {data['avg_score']:.4f}")

    print("\n📋 By Question Type:")
    with open(f"{OUTPUT_DIR}/llm_judge_summary.json") as f:
        data = json.load(f)
        for qtype, metrics in data.get('per_qtype', {}).items():
            acc = metrics['accuracy']
            score = metrics['avg_score']
            print(f"  {qtype:20} - Acc: {acc:7.2%}, Score: {score:.4f}")
except Exception as e:
    print(f"  Error reading results: {e}", file=sys.stderr)
EOF

  echo ""
  echo "📂 Output files:"
  echo "   - ${OUTPUT_DIR}/llm_judge_summary.json"
  if [[ -f "${OUTPUT_DIR}/deterministic_accuracy_summary.json" ]]; then
    echo "   - ${OUTPUT_DIR}/deterministic_accuracy_summary.json (EM)"
  fi
  if [[ -f "${OUTPUT_DIR}/atm_gpt-5-mini_summary.json" ]]; then
    echo "   - ${OUTPUT_DIR}/atm_gpt-5-mini_summary.json (ATM)"
  fi
  echo "   - ${OUTPUT_DIR}/detailed_results.jsonl (per-question details)"
  echo "   - ${OUTPUT_DIR}/eval.log"

else
  echo "❌ Evaluation failed - results file not found"
  exit 1
fi

echo ""
echo "================================================================================"
echo "  Job finished at $(date)"
echo "================================================================================"
