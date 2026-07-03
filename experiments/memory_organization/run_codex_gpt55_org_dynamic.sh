#!/usr/bin/env bash
#
# Memory-organization experiment: DYNAMIC self-organisation (prompt-only variant).
# The answering agent explores the memory relevant to each question and
# organises it into a scratch timeline before answering. Memory files = SGM.
#
# One-time sandbox build:
#   AGSYS_MEMORY_MODE=org_dynamic python3 agent_systems/prepare_sandbox.py
#
# Usage (from repo root):
#   bash experiments/memory_organization/run_codex_gpt55_org_dynamic.sh [<question_id>]
#

set -o pipefail

export AGSYS_MEMORY_MODE="org_dynamic"
export AGSYS_CODEX_MODEL="${AGSYS_CODEX_MODEL:-gpt-5.5}"
export AGSYS_CODEX_MODEL_TAG="${AGSYS_CODEX_MODEL_TAG:-gpt-5.5-medium}"
export AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}"

bash agent_systems/scripts/codex/run_codex.sh "$@"
