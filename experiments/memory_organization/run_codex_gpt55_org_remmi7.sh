#!/usr/bin/env bash
#
# Memory-organization experiment: REMMI retrieval as an in-sandbox agent tool.
# The answering agent discovers evidence via `python3 memory/search.py` (BM25 +
# date/city/type filters over a compact corpus projection — the sparse+metadata
# channels of the REMMI hybrid retriever), verifies details with targeted reads,
# organises a timeline, then answers. Tests whether replacing raw-file grep
# with real retrieval cuts the dynamic mode's discovery cost.
#
# One-time sandbox build (builds the corpus projection, no LLM needed):
#   AGSYS_MEMORY_MODE=org_remmi python3 agent_systems/prepare_sandbox.py
#
# Usage (from repo root):
#   bash experiments/memory_organization/run_codex_gpt55_org_remmi.sh [<question_id>]
#

set -o pipefail

export AGSYS_MEMORY_MODE="org_remmi7"
export AGSYS_CODEX_MODEL="${AGSYS_CODEX_MODEL:-gpt-5.5}"
export AGSYS_CODEX_MODEL_TAG="${AGSYS_CODEX_MODEL_TAG:-gpt-5.5-medium}"
export AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}"

bash agent_systems/scripts/codex/run_codex.sh "$@"
