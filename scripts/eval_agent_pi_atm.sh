#!/usr/bin/env bash
# ATM + EM eval for agent answers — runs on LOGIN node (no GPU, gpt-5-mini judge).
# Usage: bash scripts/eval_agent_pi_atm.sh <answers.jsonl> [ground_truth.json]
set -euo pipefail

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"
source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate atmbench
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ANSWERS="${1:?usage: eval_agent_pi_atm.sh <answers.jsonl> [ground_truth.json]}"
GT="${2:-data/atm-bench/atm-bench-hard.json}"
OUT="$(dirname "${ANSWERS}")/eval"

echo "Predictions: ${ANSWERS} ($(wc -l < "${ANSWERS}") rows)"
echo "Ground truth: ${GT}"
echo "Output: ${OUT}"
echo ""

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "${GT}" \
  --predictions  "${ANSWERS}" \
  --output-dir   "${OUT}" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 8 \
  --request-delay 0.5

echo ""
echo "=== ATM summary ==="
cat "${OUT}/atm_gpt-5-mini_summary.json" 2>/dev/null | python3 -m json.tool || true
