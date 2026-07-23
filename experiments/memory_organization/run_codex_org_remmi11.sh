#!/usr/bin/env bash
#
# org_remmi11 = the org_remmi10 prompt (UNION-then-PRUNE for list questions,
# on the tool-OPTIONAL v9 base) + the OCR-indexed search corpus: ocr_text[:600]
# and email detail[:600] folded into the BM25 index, --show ocr window 600.
# Amount/duration questions were unanswerable before — the values live on the
# receipt (ocr_text) or in the email body, never in the one-line caption.
#
# One-time sandbox build (rebuilds the corpus WITH ocr — required, do not reuse
# an older org_remmi* sandbox):
#   AGSYS_MEMORY_MODE=org_remmi11 python3 agent_systems/prepare_sandbox.py
#
# Usage (from repo root):
#   bash experiments/memory_organization/run_codex_org_remmi11.sh [<question_id>]        # gpt-5.5
#   MODEL=gpt-5-mini bash experiments/memory_organization/run_codex_org_remmi11.sh [...] # gpt-5-mini
#

set -o pipefail

MODEL="${MODEL:-gpt-5.5}"

export AGSYS_MEMORY_MODE="org_remmi11"
export AGSYS_CODEX_MODEL="${AGSYS_CODEX_MODEL:-${MODEL}}"
export AGSYS_CODEX_MODEL_TAG="${AGSYS_CODEX_MODEL_TAG:-${MODEL}-medium}"
export AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}"

bash agent_systems/scripts/codex/run_codex.sh "$@"
