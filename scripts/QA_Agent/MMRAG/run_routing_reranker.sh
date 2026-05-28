#!/usr/bin/env bash
# End-to-end routing retriever + reranker experiment.
#
# Runs the adaptive routing pipeline with TextReranker (Qwen3-Reranker-2B)
# on the full ATM-Bench set (1013 questions).
#
# Usage:
#   bash scripts/QA_Agent/MMRAG/run_routing_reranker.sh
#
# Environment variables:
#   DEVICE         - mps / cpu / cuda   (default: mps)
#   RERANK_TOP_K   - candidates to rerank (default: 50)
#   PRIMARY_TOP_K  - fusion pool size    (default: 100)
#   QA_FILE        - path to QA JSON     (default: full set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/../../.."

QA_FILE="${QA_FILE:-./data/atm-bench/atm-bench.json}"
DEVICE="${DEVICE:-mps}"
RERANK_TOP_K="${RERANK_TOP_K:-50}"
PRIMARY_TOP_K="${PRIMARY_TOP_K:-100}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
RERANKER_BATCH_SIZE="${RERANKER_BATCH_SIZE:-32}"
TEXT_EMBED_MODEL="${TEXT_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

echo "=========================================="
echo " Routing + Reranker Experiment"
echo "=========================================="
echo "  QA:          ${QA_FILE}"
echo "  Device:      ${DEVICE}"
echo "  Reranker:    ${RERANKER_MODEL}"
echo "  Rerank K:    ${RERANK_TOP_K}"
echo "  Primary K:   ${PRIMARY_TOP_K}"
echo "  Batch size:  ${RERANKER_BATCH_SIZE}"
echo "  Dense model: ${TEXT_EMBED_MODEL}"
echo ""

python scripts/QA_Agent/MMRAG/run_routing_reranker.py \
  --qa-file "${QA_FILE}" \
  --device "${DEVICE}" \
  --reranker-model "${RERANKER_MODEL}" \
  --rerank-top-k "${RERANK_TOP_K}" \
  --primary-top-k "${PRIMARY_TOP_K}" \
  --reranker-batch-size "${RERANKER_BATCH_SIZE}" \
  --text-embedding-model "${TEXT_EMBED_MODEL}"
