#!/usr/bin/env bash
# 3-judge an answers.jsonl: 3 independent gpt-5-mini passes into distinct dirs,
# print each pass + mean. Single-judge n=31 has ~±3-5 noise; 3-judge settles it.
# Usage: bash scripts/judge3.sh <answers.jsonl> <tag>
set -uo pipefail
REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"; cd "${REPO_ROOT}"
source /home/kz345/miniconda3/etc/profile.d/conda.sh; conda activate atmbench
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ANS="${1:?answers.jsonl}"; TAG="${2:?tag}"; GT="data/atm-bench/atm-bench-hard.json"
accs=()
for j in 1 2 3; do
  OUT="$(dirname "${ANS}")/eval_j${j}"; rm -rf "${OUT}"; mkdir -p "${OUT}"
  python memqa/utils/evaluator/evaluate_qa.py --ground-truth "${GT}" --predictions "${ANS}" \
    --output-dir "${OUT}" --metrics em atm --judge-provider openai --judge-model gpt-5-mini \
    --judge-reasoning-effort minimal --max-workers 8 --request-delay 0.5 > "${OUT}/log.txt" 2>&1
  a=$(python3 -c "import json;print(json.load(open('${OUT}/atm_gpt-5-mini_summary.json'))['accuracy'])" 2>/dev/null || echo "NaN")
  accs+=("$a"); echo "  [${TAG}] judge ${j}: acc=${a}"
done
python3 -c "
xs=[float(x) for x in '${accs[*]}'.split() if x!='NaN']
if xs:
    m=sum(xs)/len(xs)
    print('  [${TAG}] 3-judge: %s  mean=%.4f (%.1f%%)'%('/'.join('%.3f'%x for x in xs), m, m*100))
"
