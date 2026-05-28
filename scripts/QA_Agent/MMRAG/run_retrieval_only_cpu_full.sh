#!/usr/bin/env bash
# Retrieval-only MiniLM baseline on the full ATM-Bench split.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SPLIT_NAME="${SPLIT_NAME:-full}"
export QA_FILE="${QA_FILE:-./data/atm-bench/atm-bench.json}"
export METHOD_NAME="${METHOD_NAME:-mmrag_retrieval_only_cpu_full}"

bash "${SCRIPT_DIR}/run_retrieval_only_cpu.sh"
