#!/usr/bin/env bash
#SBATCH -J atmbench-best-cfg
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=06:00:00
#SBATCH --output=logs/best_config_%j.out
#SBATCH --error=logs/best_config_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate /home/kz345/miniconda3/envs/atmbench

export TMPDIR=/local/$USER/tmp
export PIP_CACHE_DIR=/local/$USER/pip_cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" logs

# Avoid Triton/CUDA illegal-instruction with system compilers.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export AS=/usr/bin/as
export LD=/usr/bin/ld
export AR=/usr/bin/ar
export PATH=/usr/bin:/bin:$PATH

# OpenAI key for LLM-judge evaluation.
OPENAI_KEY_FILE="api_keys/.openai_key"
if [[ -f "${OPENAI_KEY_FILE}" ]]; then
  export OPENAI_API_KEY="$(cat "${OPENAI_KEY_FILE}")"
else
  echo "[WARN] ${OPENAI_KEY_FILE} not found — eval judge will use \$OPENAI_API_KEY if set"
fi

# ── vLLM answerer server ──────────────────────────────────────────────────────
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
export VLLM_API_KEY="${VLLM_API_KEY:-dummy}"

echo "[vLLM] Starting server for ${VLLM_MODEL} ..."
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model "${VLLM_MODEL}" \
  > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} >/dev/null 2>&1 || true' EXIT

# Wait up to 10 min for the server to be ready.
python - <<'PY'
import socket, sys, time
host, port, deadline = "127.0.0.1", 8000, time.time() + 600
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            sys.exit(0)
    except OSError:
        time.sleep(5)
print("vLLM server did not start in time", file=sys.stderr)
sys.exit(1)
PY
echo "[vLLM] Server ready."

cd /home/kz345/rds/hpc-work/ATM-Bench

# ── Standard set: hybrid soft filter, VL dense weight = 0.20 ──────────────
echo ""
echo "======================================================="
echo "  Stage 1/2 — Standard set (hybrid soft VL w=0.20)"
echo "======================================================="
bash scripts/QA_Agent/MMRAG/run_hybrid_soft_vl020_qwen3vl8b.sh

# ── Hard set: hybrid hard filter, VL dense weight = 0.15 ──────────────────
echo ""
echo "======================================================="
echo "  Stage 2/2 — Hard set (hybrid hard VL w=0.15)"
echo "======================================================="
bash scripts/QA_Agent/MMRAG/run_hybrid_hard_cond_vl015_qwen3vl8b.sh

echo ""
echo "======================================================="
echo "  All done. Job ${SLURM_JOB_ID} complete."
echo "======================================================="
