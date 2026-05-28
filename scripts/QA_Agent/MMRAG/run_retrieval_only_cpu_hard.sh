#!/usr/bin/env bash
# Retrieval-only MiniLM baseline on ATM-Bench-Hard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SPLIT_NAME="${SPLIT_NAME:-hard}"
export QA_FILE="${QA_FILE:-./data/atm-bench/atm-bench-hard.json}"
export METHOD_NAME="${METHOD_NAME:-mmrag_retrieval_only_cpu_hard}"

bash "${SCRIPT_DIR}/run_retrieval_only_cpu.sh"
