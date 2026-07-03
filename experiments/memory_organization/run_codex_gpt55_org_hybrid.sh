#!/usr/bin/env bash
#
# Memory-organization experiment: HYBRID (index-guided dynamic organisation).
# The answering agent first LOCATES candidate events in the prebuilt index,
# drills down only into those events' items, builds a compact timeline, then
# answers. Tests whether index-guided location cuts the dynamic mode's
# per-question organisation cost (and whether a smaller working context
# also improves answer quality).
#
# One-time sandbox build (reuses the pure static index):
#   AGSYS_MEMORY_MODE=org_hybrid \
#   AGSYS_ORGANIZED_MEMORY=output/organized/agent_static_pure/organized_memory.json \
#   python3 agent_systems/prepare_sandbox.py
#
# Usage (from repo root):
#   bash experiments/memory_organization/run_codex_gpt55_org_hybrid.sh [<question_id>]
#

set -o pipefail

export AGSYS_MEMORY_MODE="org_hybrid"
export AGSYS_CODEX_MODEL="${AGSYS_CODEX_MODEL:-gpt-5.5}"
export AGSYS_CODEX_MODEL_TAG="${AGSYS_CODEX_MODEL_TAG:-gpt-5.5-medium}"
export AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}"

bash agent_systems/scripts/codex/run_codex.sh "$@"
