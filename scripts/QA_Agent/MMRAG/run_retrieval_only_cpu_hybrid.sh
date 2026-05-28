#!/usr/bin/env bash
# Retrieval-only hybrid run (metadata + BM25 + dense) on CPU.
#
# Uses sentence_transformer + all-MiniLM-L6-v2 as the dense sub-retriever
# plus BM25 and metadata filtering. No GPU or LLM answerer needed.
#
# Compare against the dense-only baseline from run_retrieval_only_cpu.sh
# to measure the hybrid retrieval improvement on R@k.
#
# Run from repo root via split-specific wrappers:
#   bash scripts/QA_Agent/MMRAG/run_retrieval_only_cpu_hybrid_hard.sh
#   bash scripts/QA_Agent/MMRAG/run_retrieval_only_cpu_hybrid_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/../../.."

QA_FILE="${QA_FILE:-./data/atm-bench/atm-bench-hard.json}"
SPLIT_NAME="${SPLIT_NAME:-hard}"
TOP_K="${TOP_K:-10}"
MAX_K="${MAX_K:-200}"
TEXT_EMBED_MODEL="${TEXT_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
RETRIEVER_BATCH_SIZE="${RETRIEVER_BATCH_SIZE:-64}"

# Hybrid-specific config
HYBRID_DENSE="${HYBRID_DENSE:-sentence_transformer}"
HYBRID_FUSION="${HYBRID_FUSION:-rrf}"
HYBRID_RRF_K="${HYBRID_RRF_K:-60}"
HYBRID_WEIGHT_META="${HYBRID_WEIGHT_META:-0.3}"
HYBRID_WEIGHT_SPARSE="${HYBRID_WEIGHT_SPARSE:-0.4}"
HYBRID_WEIGHT_DENSE="${HYBRID_WEIGHT_DENSE:-0.3}"
HYBRID_FILTER_MODE="${HYBRID_FILTER_MODE:-soft}"

METHOD_NAME="${METHOD_NAME:-mmrag_hybrid_cpu_${SPLIT_NAME}_${HYBRID_FUSION}_m${HYBRID_WEIGHT_META}_s${HYBRID_WEIGHT_SPARSE}_d${HYBRID_WEIGHT_DENSE}_${HYBRID_FILTER_MODE}}"
OUTPUT_BASE="${OUTPUT_BASE:-output/QA_Agent/MMRAG/hybrid_cpu}"

EMAIL_FILE="./data/raw_memory/email/emails.json"
IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"

METHOD_DIR="${OUTPUT_BASE}/${METHOD_NAME}"
mkdir -p "${OUTPUT_BASE}"

echo "[hybrid-retrieval] split=${SPLIT_NAME}"
echo "[hybrid-retrieval] qa_file=${QA_FILE}"
echo "[hybrid-retrieval] dense_sub=${HYBRID_DENSE}  text_embed=${TEXT_EMBED_MODEL}"
echo "[hybrid-retrieval] fusion=${HYBRID_FUSION}  rrf_k=${HYBRID_RRF_K}"
echo "[hybrid-retrieval] weights: meta=${HYBRID_WEIGHT_META} sparse=${HYBRID_WEIGHT_SPARSE} dense=${HYBRID_WEIGHT_DENSE}"
echo "[hybrid-retrieval] filter_mode=${HYBRID_FILTER_MODE}"
echo "[hybrid-retrieval] batch_size=${RETRIEVER_BATCH_SIZE}  top_k=${TOP_K}  max_k=${MAX_K}"
echo "[hybrid-retrieval] writing under ${METHOD_DIR}/"

python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file "${QA_FILE}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --retriever hybrid \
  --hybrid-dense-retriever "${HYBRID_DENSE}" \
  --hybrid-fusion "${HYBRID_FUSION}" \
  --hybrid-rrf-k "${HYBRID_RRF_K}" \
  --hybrid-weight-metadata "${HYBRID_WEIGHT_META}" \
  --hybrid-weight-sparse "${HYBRID_WEIGHT_SPARSE}" \
  --hybrid-weight-dense "${HYBRID_WEIGHT_DENSE}" \
  --hybrid-filter-mode "${HYBRID_FILTER_MODE}" \
  --text-embedding-model "${TEXT_EMBED_MODEL}" \
  --retriever-batch-size "${RETRIEVER_BATCH_SIZE}" \
  --retrieval-top-k "${TOP_K}" \
  --retrieval-max-k "${MAX_K}" \
  --no-reuse-retrieval-results \
  --no-evidence \
  --provider openai \
  --output-dir-base "${OUTPUT_BASE}" \
  --method-name "${METHOD_NAME}"

# Print summary
SUM="${METHOD_DIR}/retrieval_recall_summary.json"
DETAILS="${METHOD_DIR}/retrieval_recall_details.json"
if [[ -f "${SUM}" ]]; then
  echo
  echo "[hybrid-retrieval] experiment summary:"
  python - "${SUM}" "${DETAILS}" "${SPLIT_NAME}" "${QA_FILE}" "${METHOD_NAME}" "${OUTPUT_BASE}" "${TEXT_EMBED_MODEL}" "hybrid" "${TOP_K}" "${MAX_K}" "${RETRIEVER_BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    details_path,
    split_name,
    qa_file,
    method_name,
    output_base,
    text_embed_model,
    retriever,
    top_k,
    max_k,
    batch_size,
) = sys.argv[1:]

summary = json.loads(Path(summary_path).read_text())
recall = summary.get("recall", summary)
details_file = Path(details_path)
details = json.loads(details_file.read_text()) if details_file.exists() else []

ks = [1, 5, 10, 25, 50, 100, 200]
gold_counts = [len(d.get("gt_evidence_ids", [])) for d in details]
retrieved_counts = [len(d.get("retrieval_ids", [])) for d in details]

def avg(values):
    return sum(values) / len(values) if values else 0.0

def full_coverage_at(k):
    if not details:
        return 0.0
    ok = 0
    for d in details:
        gt = set(d.get("gt_evidence_ids", []))
        retrieved = set(d.get("retrieval_ids", [])[:k])
        if gt and gt.issubset(retrieved):
            ok += 1
    return ok / len(details)

def any_hit_at(k):
    if not details:
        return 0.0
    ok = 0
    for d in details:
        gt = set(d.get("gt_evidence_ids", []))
        retrieved = set(d.get("retrieval_ids", [])[:k])
        if gt and gt.intersection(retrieved):
            ok += 1
    return ok / len(details)

report = {
    "config": {
        "split": split_name,
        "qa_file": qa_file,
        "method_name": method_name,
        "retriever": retriever,
        "text_embedding_model": text_embed_model,
        "retriever_batch_size": int(batch_size),
        "media_source": "batch_results",
        "answerer": "disabled (--no-evidence)",
        "retrieval_top_k_for_answerer": int(top_k),
        "retrieval_max_k_for_metrics": int(max_k),
    },
    "dataset": {
        "question_count": len(details) or recall.get("count"),
        "avg_gold_evidence_per_question": round(avg(gold_counts), 3),
        "max_gold_evidence_per_question": max(gold_counts) if gold_counts else 0,
        "avg_retrieved_candidates": round(avg(retrieved_counts), 3),
    },
    "recall": {k: round(float(v), 6) for k, v in recall.items()},
    "coverage": {
        f"any_hit@{k}": round(any_hit_at(k), 6)
        for k in ks
        if k <= int(max_k)
    },
    "full_coverage": {
        f"all_gold_found@{k}": round(full_coverage_at(k), 6)
        for k in ks
        if k <= int(max_k)
    },
    "outputs": {
        "method_dir": str(Path(output_base) / method_name),
        "summary": summary_path,
        "details": details_path,
        "answers": str(Path(output_base) / method_name / "mmrag_answers.jsonl"),
    },
}

print(json.dumps(report, indent=2, ensure_ascii=False))
PY
fi
