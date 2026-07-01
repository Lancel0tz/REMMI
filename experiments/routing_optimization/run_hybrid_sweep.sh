#!/usr/bin/env bash
# Hybrid retrieval weight sensitivity sweep.
# Runs multiple (meta, sparse, dense) weight configs on selected splits,
# then prints a summary table for comparison.
#
# Usage:
#   bash scripts/QA_Agent/MMRAG/run_hybrid_sweep.sh          # both splits
#   SWEEP_SPLITS=hard bash scripts/QA_Agent/MMRAG/run_hybrid_sweep.sh  # hard only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

OUTPUT_BASE="output/QA_Agent/MMRAG/hybrid_sweep"
mkdir -p "${OUTPUT_BASE}"

# Weight configs: meta sparse dense filter_mode fusion
CONFIGS=(
  # --- Dense-only (ablation baseline) ---
  "0.0  0.0  1.0  soft  rrf"
  # --- BM25-only ---
  "0.0  1.0  0.0  soft  rrf"
  # --- Dense-heavy ---
  "0.1  0.2  0.7  soft  rrf"
  "0.1  0.1  0.8  soft  rrf"
  "0.05 0.15 0.8  soft  rrf"
  # --- Balanced ---
  "0.2  0.3  0.5  soft  rrf"
  "0.3  0.4  0.3  soft  rrf"
  "0.2  0.4  0.4  soft  rrf"
  # --- Sparse-heavy ---
  "0.1  0.6  0.3  soft  rrf"
  "0.2  0.5  0.3  soft  rrf"
  # --- Hard filter mode ---
  "0.2  0.3  0.5  hard  rrf"
  "0.1  0.2  0.7  hard  rrf"
  # --- weighted_sum fusion ---
  "0.2  0.3  0.5  soft  weighted_sum"
  "0.1  0.2  0.7  soft  weighted_sum"
)

# Which splits to run: "hard", "full", or "hard full" (default: both)
SWEEP_SPLITS="${SWEEP_SPLITS:-hard full}"

qa_file_for_split() {
  case "$1" in
    hard) echo "./data/atm-bench/atm-bench-hard.json" ;;
    full) echo "./data/atm-bench/atm-bench.json" ;;
    *) echo "Unknown split: $1" >&2; exit 1 ;;
  esac
}

RESULTS_CSV="${OUTPUT_BASE}/sweep_results.csv"
echo "split,fusion,filter_mode,w_meta,w_sparse,w_dense,R@1,R@5,R@10,R@25,R@50,R@100,R@200" > "${RESULTS_CSV}"

# Count total
total_configs=0
for split in ${SWEEP_SPLITS}; do
  total_configs=$(( total_configs + ${#CONFIGS[@]} ))
done
run_idx=0

for split in ${SWEEP_SPLITS}; do
  QA_FILE="$(qa_file_for_split "${split}")"

  for config in "${CONFIGS[@]}"; do
    read -r W_META W_SPARSE W_DENSE FILTER FUSION <<< "${config}"
    run_idx=$((run_idx + 1))

    METHOD_NAME="sweep_${split}_${FUSION}_m${W_META}_s${W_SPARSE}_d${W_DENSE}_${FILTER}"

    echo ""
    echo "=========================================="
    echo "[${run_idx}/${total_configs}] ${METHOD_NAME}"
    echo "=========================================="

    export SPLIT_NAME="${split}" QA_FILE
    export HYBRID_FUSION="${FUSION}"
    export HYBRID_WEIGHT_META="${W_META}"
    export HYBRID_WEIGHT_SPARSE="${W_SPARSE}"
    export HYBRID_WEIGHT_DENSE="${W_DENSE}"
    export HYBRID_FILTER_MODE="${FILTER}"
    export METHOD_NAME
    export OUTPUT_BASE

    bash "${SCRIPT_DIR}/run_retrieval_only_cpu_hybrid.sh" 2>&1 \
      | grep -v "^Batches:\|^Retrieval:\|^MMRag QA:\|^Loading weights:" || true

    # Extract recall values
    SUM="${OUTPUT_BASE}/${METHOD_NAME}/retrieval_recall_summary.json"
    if [[ -f "${SUM}" ]]; then
      python3 -c "
import json, sys
s = json.load(open('${SUM}'))['recall']
ks = ['R@1','R@5','R@10','R@25','R@50','R@100','R@200']
vals = ','.join(f'{s.get(k,0):.6f}' for k in ks)
print(f'${split},${FUSION},${FILTER},${W_META},${W_SPARSE},${W_DENSE},{vals}')
" >> "${RESULTS_CSV}"
    fi
  done
done

echo ""
echo "============================================"
echo "SWEEP COMPLETE — Summary Table"
echo "============================================"
echo ""

python3 - "${RESULTS_CSV}" <<'PYEND'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("No results.")
    sys.exit(0)

splits = sorted(set(r['split'] for r in rows))

for split in splits:
    split_rows = [r for r in rows if r['split'] == split]
    print(f"\n=== {split.upper()} split ===")
    hdr = f"{'fusion':<14} {'filter':<6} {'w_m':>5} {'w_s':>5} {'w_d':>5} | {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@25':>6} {'R@50':>6} {'R@100':>6} {'R@200':>6}"
    print(hdr)
    print("-" * len(hdr))

    # Sort by R@10 descending
    split_rows.sort(key=lambda r: -float(r['R@10']))

    for r in split_rows:
        vals = [f"{float(r[k])*100:6.2f}" for k in ['R@1','R@5','R@10','R@25','R@50','R@100','R@200']]
        print(f"{r['fusion']:<14} {r['filter_mode']:<6} {r['w_meta']:>5} {r['w_sparse']:>5} {r['w_dense']:>5} | {' '.join(vals)}")

print(f"\nFull CSV: {sys.argv[1]}")
PYEND
