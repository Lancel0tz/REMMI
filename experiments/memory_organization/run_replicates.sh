#!/usr/bin/env bash
#
# Replicate driver for the final mean±sd results table.
#
# Runs a list of memory modes as INDEPENDENT replicates by switching
# AGSYS_RUN_TAG (atm-bench-hard-r2, -r3, ...). Sandboxes are reused; run dirs,
# collected answers and eval live under the new tag, so nothing overwrites the
# original atm-bench-hard run (= replicate 1). Legs resume: already-answered
# questions under the same tag are skipped, so rerunning this script after a
# quota/network failure continues where it stopped.
#
# Usage (from repo root):
#   MODEL=gpt-5-mini REPS="2 3" \
#     MODES="sgm org_heuristic org_static org_hybrid org_dynamic org_remmi org_remmi8 org_remmi9" \
#     bash experiments/memory_organization/run_replicates.sh
#   MODEL=gpt-5.5 REPS="2 3" MODES="sgm org_dynamic org_remmi8 org_remmi9" \
#     bash experiments/memory_organization/run_replicates.sh
#
# Use CANONICAL mode ids (org_dynamic, org_remmi8, ...) — NOT the short
# aliases (orgd, orgr8): config.py maps the raw env value to the default
# eval_root, and an alias silently falls back to eval_root_sgm and errors out.
# Here org_dynamic = the agentic timeline variant (org_timeline in
# LEADERBOARD.md), NOT the inject pipeline.
#
# Aggregate afterwards with:
#   python3 experiments/memory_organization/compare_modes.py --run-tag atm-bench-hard-r2 --model-base <tag>
#
set -o pipefail

MODEL="${MODEL:-gpt-5-mini}"
REPS="${REPS:-2 3}"
MODES="${MODES:-sgm org_heuristic org_static org_hybrid org_dynamic org_remmi org_remmi8 org_remmi9}"

for REP in ${REPS}; do
  for MODE in ${MODES}; do
    echo "=== replicate r${REP} | ${MODEL} | ${MODE} ==="
    AGSYS_RUN_TAG="atm-bench-hard-r${REP}" \
    AGSYS_MEMORY_MODE="${MODE}" \
    AGSYS_CODEX_MODEL="${MODEL}" \
    AGSYS_CODEX_MODEL_TAG="${MODEL}-medium" \
    AGSYS_CODEX_REASONING_EFFORT="${AGSYS_CODEX_REASONING_EFFORT:-medium}" \
      bash agent_systems/scripts/codex/run_codex.sh
    echo "=== done r${REP}/${MODE} (exit $?) ==="
  done
done
