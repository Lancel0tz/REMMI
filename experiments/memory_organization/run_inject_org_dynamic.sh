#!/usr/bin/env bash
#
# org_dynamic (canonical INJECT pipeline) with an OpenAI-API answerer.
# Retrieval: frozen config/best_routing_config_hard.json (auto-selected) +
# bge-reranker-base, top-10 evidence injected, single answerer call per Q.
#
# For the canonical Qwen3-VL answerer (vLLM + CUDA, insert_raw_images=true)
# use scripts/slurm_complete_qa_hard.sh on the GPU server instead.
#
# Usage (repo root; GPU box: DEVICE=cuda):
#   MODEL=gpt-5-mini  bash experiments/memory_organization/run_inject_org_dynamic.sh
#   MODEL=gpt-5.5     bash experiments/memory_organization/run_inject_org_dynamic.sh
#
set -euo pipefail
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt-5-mini}"
DEVICE="${DEVICE:-cpu}"
QA_FILE="${QA_FILE:-data/atm-bench/atm-bench-hard.json}"
OUT="${OUT:-output/answers_inject_hard/${MODEL}}"
KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key | tr -d '[:space:]')}"

python3 scripts/eval_with_best_config_complete.py \
  --qa-file "${QA_FILE}" \
  --config-file auto \
  --device "${DEVICE}" \
  --use-reranker --reranker-model BAAI/bge-reranker-base \
  --rerank-top-k 50 --retrieval-top-k 10 \
  --provider openai --api-base https://api.openai.com/v1 --api-key "${KEY}" \
  --model "${MODEL}" --temperature 1 --max-tokens 256 --max-workers 4 \
  --output-dir "${OUT}"
# Note: gpt-5-family requires temperature=1; evidence is text-only here
# (insert_raw_images stays false — text answerers / agent-table comparability).
