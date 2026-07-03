#!/usr/bin/env bash
#
# Memory-organization experiment: STATIC agent organisation (full upfront pass).
# An LLM organiser reworks the whole corpus into events offline; the answering
# agent then reads that LLM-organized index.
#
# Step 1 — build the organized index (needs an OpenAI-compatible endpoint):
#   ORGANIZER_BASE_URL=https://api.openai.com/v1 ORGANIZER_MODEL=gpt-5.5 \
#   python3 -m remmi.organize.agent_static \
#     --image-source output/image/qwen3vl2b/batch_results.json \
#     --video-source output/video/qwen3vl2b/batch_results.json \
#     --emails-source data/raw_memory/email/emails.json \
#     --out output/organized/agent_static/organized_memory.json
#
# Step 2 — build the sandbox from it:
#   AGSYS_MEMORY_MODE=org_static \
#   AGSYS_ORGANIZED_MEMORY=output/organized/agent_static/organized_memory.json \
#   python3 agent_systems/prepare_sandbox.py
#
# Usage (from repo root):
#   bash experiments/memory_organization/run_codex_gpt55_org_static.sh [<question_id>]
#

set -o pipefail

export AGSYS_MEMORY_MODE="org_static"
export AGSYS_CODEX_MODEL="${AGSYS_CODEX_MODEL:-gpt-5.5}"
export AGSYS_CODEX_MODEL_TAG="${AGSYS_CODEX_MODEL_TAG:-gpt-5.5-medium}"
export AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}"

bash agent_systems/scripts/codex/run_codex.sh "$@"
