#!/usr/bin/env bash
#
# Memory-organization experiment: DYNAMIC per-question OFFLINE-organise + INJECT.
#
# Distinct from org_dynamic (agent self-organises at run time). Here an LLM
# organises each question's top-N retrieved items into events OFFLINE, and the
# harness injects that question's events as memory/query_events.json at run time.
# On the original Qwen3.6-27B / Pi setting this offline-inject variant was the
# token/QS winner (organise cost paid once, tiny per item; answerer context stays
# small). See remmi/organize/dynamic_inject.py and docs.
#
# Prereqs (from repo root):
#   # 1. Build the SGM sandbox for org_inject (one-time)
#   AGSYS_MEMORY_MODE=org_inject python3 agent_systems/prepare_sandbox.py
#
#   # 2. Build per-question events offline (needs a retrieval file with top-N ids
#   #    per question, e.g. from REMMI's hybrid retriever / an mmrag run, and an
#   #    OpenAI-compatible endpoint for the organiser LLM).
#   python3 -m remmi.organize.dynamic_inject \
#       --retrieval <path/to/retrieval_ids.jsonl> \
#       --questions data/atm-bench/atm-bench-hard.json \
#       --out-dir agent_systems/eval_root_orgi/memory/dynamic_events \
#       --base-url "${ORGANISER_BASE_URL:-http://localhost:8000/v1}" \
#       --model    "${ORGANISER_MODEL:-gpt-5-mini}" --topn 50
#
# Usage (from repo root; pass a question id for a single-question smoke):
#   bash experiments/memory_organization/run_pi_org_inject.sh [<question_id>]
#
set -o pipefail

export AGSYS_MEMORY_MODE="org_inject"
# Per-question offline-organised events injected as memory/query_events.json.
export AGSYS_DYNAMIC_EVENTS_DIR="${AGSYS_DYNAMIC_EVENTS_DIR:-$(pwd)/agent_systems/eval_root_orgi/memory/dynamic_events}"

# Pi + OpenAI-compatible endpoint (point at your served answerer, e.g. Qwen3.6-27B).
export PI_OPENAI_BASE_URL="${PI_OPENAI_BASE_URL:-http://localhost:8000/v1}"
export PI_OPENAI_MODEL="${PI_OPENAI_MODEL:-Qwen/Qwen3.6-27B-FP8}"

if [[ ! -d "${AGSYS_DYNAMIC_EVENTS_DIR}" ]]; then
  echo "ERROR: AGSYS_DYNAMIC_EVENTS_DIR not found: ${AGSYS_DYNAMIC_EVENTS_DIR}" >&2
  echo "       Build it first — see the header of this script (step 2)." >&2
  exit 1
fi

bash agent_systems/scripts/pi/run_pi_openai_compatible.sh "$@"
