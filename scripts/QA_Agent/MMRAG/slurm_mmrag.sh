#!/usr/bin/env bash
#SBATCH -J atmbench-mmrag
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mmrag_%j.out
#SBATCH --error=logs/mmrag_%j.err

set -euo pipefail

# Use home miniconda and the atmbench environment.
source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate /home/kz345/miniconda3/envs/atmbench

# Prefer local disk for temporary files and pip cache.
export TMPDIR=/local/$USER/tmp
export PIP_CACHE_DIR=/local/$USER/pip_cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" logs

# Use GCC for Triton/CUDA helper builds (avoid icx illegal instruction).
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export AS=/usr/bin/as
export LD=/usr/bin/ld
export AR=/usr/bin/ar
export PATH=/usr/bin:/bin:$PATH

# Start vLLM server locally for the run.
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
export VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
python -m vllm.entrypoints.openai.api_server \
	--host 127.0.0.1 \
	--port 8000 \
	--model "${VLLM_MODEL}" \
	> "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} >/dev/null 2>&1 || true' EXIT

# Wait for the server to accept connections (model load can take several minutes).
python - <<'PY'
import socket
import sys
import time

host = "127.0.0.1"
port = 8000
deadline = time.time() + 600
while time.time() < deadline:
	try:
		with socket.create_connection((host, port), timeout=1):
			sys.exit(0)
	except OSError:
		time.sleep(5)
print("vLLM server did not start in time", file=sys.stderr)
sys.exit(1)
PY

cd /home/kz345/rds/hpc-work/ATM-Bench
bash scripts/QA_Agent/MMRAG/run.sh
