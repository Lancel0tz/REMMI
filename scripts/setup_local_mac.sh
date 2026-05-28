#!/usr/bin/env bash
# Local setup helper for reproducing ATM-Bench on a Mac (no GPU, OpenAI API).
#
# Run from the repo root:
#   bash scripts/setup_local_mac.sh
#
# This is the project-proposal Day 1–14 deliverable: install dependencies,
# stage the dataset, and verify the QA / batch_results schemas. Heavier
# baselines (vLLM-served Qwen3-VL-8B) need a GPU and are NOT attempted here;
# Oracle/MMRAG with the OpenAI provider DO work locally.
#
# What this script does:
#   1. Verify conda + python availability and create the `atmbench` env if absent.
#   2. Install the core requirements (torch + transformers stack; CPU build).
#   3. Install the project package in editable mode.
#   4. Install rank_bm25 (needed by the hybrid extension and A-Mem baseline).
#   5. Stage api_keys/.openai_key from $OPENAI_API_KEY if it is set.
#   6. Run scripts/download_data.sh  (≈3.3 GB; downloads the HF dataset).
#   7. Run a small schema check on data/atm-bench/atm-bench-hard.json.
#
# Anything that fails is reported with the exact next-step you can take.

set -uo pipefail

# Use POSIX printf escapes (bash 3.2 on macOS does not support $'...' arrays).
RED=$(printf '\033[0;31m')
GREEN=$(printf '\033[0;32m')
YELLOW=$(printf '\033[1;33m')
BLUE=$(printf '\033[0;34m')
RESET=$(printf '\033[0m')

step() {
  printf "\n${BLUE}==>${RESET} %s\n" "$1"
}
ok() {
  printf "${GREEN}OK${RESET}    %s\n" "$1"
}
warn() {
  printf "${YELLOW}WARN${RESET}  %s\n" "$1"
}
fail() {
  printf "${RED}FAIL${RESET}  %s\n" "$1"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Conda + env
# ---------------------------------------------------------------------------
step "1. Check conda and atmbench env"
if ! command -v conda >/dev/null 2>&1; then
  fail "conda not found in PATH."
  cat <<EOF
        Install Miniforge (recommended on Apple Silicon):
          https://github.com/conda-forge/miniforge
        Or Miniconda:
          https://docs.conda.io/en/latest/miniconda.html
        Then re-run this script.
EOF
  exit 1
fi
ok "conda found: $(command -v conda)"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "atmbench"; then
  ok "conda env 'atmbench' already exists"
else
  step "Creating conda env 'atmbench' (Python 3.11)"
  conda create -n atmbench python=3.11 -y
fi

conda activate atmbench
ok "Activated env: $(python -c 'import sys; print(sys.executable)')"

# ---------------------------------------------------------------------------
# 2. Apple Silicon notes
# ---------------------------------------------------------------------------
ARCH="$(uname -m)"
step "2. Detected architecture: ${ARCH}"
if [[ "$ARCH" == "arm64" ]]; then
  cat <<EOF
        Note for Apple Silicon: torch on macOS uses the MPS / CPU backend,
        which is fine for retrieval-only experiments. vLLM is NOT supported
        on macOS — heavy answerer runs (Qwen3-VL-8B) must go on HPC.
EOF
fi

# ---------------------------------------------------------------------------
# 3. Core deps
# ---------------------------------------------------------------------------
step "3. Install core requirements (this may take a few minutes)"
# vllm and decord are in requirements.txt but cannot be installed reliably on
# macOS/Apple Silicon via pip, so we filter them for the local OpenAI path.
TMP_REQ="$(mktemp)"
# Using POSIX-portable grep + sed; works across Linux/macOS.
grep -v -i -e '^vllm' -e '^decord' requirements.txt > "$TMP_REQ"
pip install --upgrade pip >/dev/null
pip install -r "$TMP_REQ"
rm -f "$TMP_REQ"
ok "Core requirements installed (vllm/decord filtered out for macOS)"

step "4. Editable install of memqa package"
pip install -e . --no-deps >/dev/null
ok "memqa installed"

step "5. rank_bm25 for hybrid extension and A-Mem baseline"
pip install rank_bm25 >/dev/null
ok "rank_bm25 installed"

# ---------------------------------------------------------------------------
# 4. API keys
# ---------------------------------------------------------------------------
step "6. Stage OpenAI API key"
mkdir -p api_keys
if [[ -f api_keys/.openai_key ]]; then
  ok "api_keys/.openai_key already exists"
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  printf "%s" "$OPENAI_API_KEY" > api_keys/.openai_key
  chmod 600 api_keys/.openai_key
  ok "Wrote api_keys/.openai_key from \$OPENAI_API_KEY"
else
  warn "OPENAI_API_KEY not set in environment. To enable Oracle/MMRAG GPT-5 runs,
        either export OPENAI_API_KEY, or place your key in api_keys/.openai_key
        manually before running the QA scripts."
fi

# ---------------------------------------------------------------------------
# 5. Dataset download
# ---------------------------------------------------------------------------
step "7. Download dataset (≈3.3 GB) from Hugging Face"
if [[ -f data/atm-bench/atm-bench-hard.json && -f data/raw_memory/email/emails.json ]]; then
  ok "Dataset already staged under data/. Skipping download."
else
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    pip install huggingface_hub >/dev/null
  fi
  if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
    warn "Not logged into Hugging Face. If 'Jingbiao/ATM-Bench' is gated you need to:
          1) accept the license on https://huggingface.co/datasets/Jingbiao/ATM-Bench
          2) run 'huggingface-cli login' (paste a read token from hf.co/settings/tokens)
          Then re-run this script."
  fi
  bash scripts/download_data.sh
  ok "Dataset download attempted"
fi

# ---------------------------------------------------------------------------
# 6. Schema check
# ---------------------------------------------------------------------------
step "8. Schema sanity check"
python scripts/check_data_schema.py
echo
ok "All steps attempted. See output above for any WARN/FAIL lines."
