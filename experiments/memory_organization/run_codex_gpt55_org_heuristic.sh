#!/usr/bin/env bash
#
# Memory-organization experiment: HEURISTIC organisation (day-gap event/trip index).
# Deterministic clustering builds memory/organized_memory.json; the agent reads
# the compact event index first, then drills into per-item files.
#
# One-time sandbox build (runs the heuristic organiser, no LLM needed):
#   AGSYS_MEMORY_MODE=org_heuristic python3 agent_systems/prepare_sandbox.py
#
# Usage (from repo root):
#   bash experiments/memory_organization/run_codex_gpt55_org_heuristic.sh [<question_id>]
#

set -o pipefail

export AGSYS_MEMORY_MODE="org_heuristic"
export AGSYS_CODEX_MODEL="${AGSYS_CODEX_MODEL:-gpt-5.5}"
export AGSYS_CODEX_MODEL_TAG="${AGSYS_CODEX_MODEL_TAG:-gpt-5.5-medium}"
export AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}"

bash agent_systems/scripts/codex/run_codex.sh "$@"
