#!/usr/bin/env bash
# Cheapest end-to-end smoke test: 5 QAs from atm-bench-hard.json,
# Oracle retrieval (gold evidence IDs), GPT-5-mini answerer in
# `batch_results` mode (text captions, no raw images), GPT-5-mini judge.
#
# Cost: typically well under $1. Time: 1–3 minutes.
# Purpose: confirm the whole pipeline (data → answerer → judge → metrics)
# wires up correctly on a Mac with only an OpenAI key. The QS number
# from this run is NOT directly comparable to paper Table 3 because
# (a) only 5 questions, (b) gpt-5-mini < gpt-5, (c) batch_results vs raw.
#
# Run from repo root:
#   bash scripts/QA_Agent/Oracle/run_oracle_smoke.sh
#
# Override knobs:
#   N=20                        # how many QAs to sample
#   MODEL=gpt-5                 # answerer model
#   JUDGE=gpt-5-mini            # judge model

set -euo pipefail

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key 2>/dev/null || true)}"
if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "OPENAI_API_KEY not set and api_keys/.openai_key not found." >&2
  exit 1
fi
export OPENAI_API_KEY

N="${N:-5}"
MODEL="${MODEL:-gpt-5-mini}"
JUDGE="${JUDGE:-gpt-5-mini}"
RUN_TAG="smoke_${MODEL//[^a-zA-Z0-9]/_}_n${N}"

WORK_DIR="output/QA_Agent/Oracle/${RUN_TAG}"
SAMPLE_QA="${WORK_DIR}/atm-bench-hard.sample.json"
PREDICTIONS="${WORK_DIR}/oracle_${RUN_TAG}.jsonl"
EVAL_DIR="${WORK_DIR}/eval"
mkdir -p "${WORK_DIR}"

# 1. Sample N questions deterministically (no shuffling — first N for
#    reproducibility; if you want stratified sampling use a custom script).
python - <<PY
import json, sys
from pathlib import Path
src = Path("./data/atm-bench/atm-bench-hard.json")
dst = Path("${SAMPLE_QA}")
data = json.loads(src.read_text())
items = data["qas"] if isinstance(data, dict) else data
sample = items[: ${N}]
dst.write_text(json.dumps(sample, indent=2))
print(f"[smoke] wrote {len(sample)} QA items -> {dst}")
PY

# 2. Oracle answerer in batch_results (text-only) mode.
echo "[smoke] running Oracle answerer (${MODEL}) on ${N} QAs..."
python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "${SAMPLE_QA}" \
  --media-source batch_results \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider openai \
  --model "${MODEL}" \
  --reasoning-effort minimal \
  --max-workers 4 \
  --timeout 60 \
  --output-file "${PREDICTIONS}"

# 3. Evaluate with the judge.
echo "[smoke] running judge (${JUDGE})..."
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "${SAMPLE_QA}" \
  --predictions "${PREDICTIONS}" \
  --output-dir "${EVAL_DIR}" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model "${JUDGE}" \
  --judge-reasoning-effort minimal \
  --max-workers 2

echo
echo "[smoke] done. Predictions: ${PREDICTIONS}"
echo "[smoke] eval: ${EVAL_DIR}/"
ls -la "${EVAL_DIR}/" 2>/dev/null || true
