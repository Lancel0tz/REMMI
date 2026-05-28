#!/usr/bin/env bash
# Hybrid retrieval-only on the full ATM-Bench split.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SPLIT_NAME="${SPLIT_NAME:-full}"
export QA_FILE="${QA_FILE:-./data/atm-bench/atm-bench.json}"

bash "${SCRIPT_DIR}/run_retrieval_only_cpu_hybrid.sh"
