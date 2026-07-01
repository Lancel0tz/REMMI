#!/usr/bin/env bash
#SBATCH -J decompose
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/decompose_%j.out
#SBATCH --error=logs/decompose_%j.err

set -uo pipefail
REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"; mkdir -p logs
source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate atmbench
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="/usr/bin:/bin:${CONDA_PREFIX}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/rds/user/kz345/hpc-work/hf_cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR="/local/${USER}/tmp"; export TRITON_CACHE_DIR="/local/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/local/${USER}/inductor_cache"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

QA_FILE="${QA_FILE:-data/atm-bench/atm-bench-hard.json}"
CONFIG_FILE="${CONFIG_FILE:-config/best_routing_config_hard.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/decompose_hard}"
ROUTER_MODEL="${ROUTER_MODEL:-Qwen/Qwen3-14B}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.7}"
VLLM_PORT="${VLLM_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

echo "=== Decompose recall — ${QA_FILE} | model=${ROUTER_MODEL} | port=${VLLM_PORT} ==="
fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true; sleep 1
python -m vllm.entrypoints.openai.api_server \
  --model "${ROUTER_MODEL}" --tensor-parallel-size 1 \
  --gpu-memory-utilization "${VLLM_GPU_MEM}" --port "${VLLM_PORT}" \
  --trust-remote-code --max-model-len 8192 --enforce-eager \
  > logs/vllm_decompose_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "waiting for vLLM..."; sleep 15
for i in {1..300}; do
  curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1 && { echo "✅ vLLM ready ($((15+i*2))s)"; break; }
  kill -0 ${VLLM_PID} 2>/dev/null || { echo "❌ vLLM died"; tail -40 logs/vllm_decompose_${SLURM_JOB_ID}.log; exit 1; }
  [[ $i -eq 300 ]] && { echo "❌ timeout"; kill ${VLLM_PID}; exit 1; }
  sleep 2
done

VL_ADAPTIVE_FLAG=""
if [[ "${VL_ADAPTIVE:-false}" == "true" || "${VL_ADAPTIVE:-}" == "1" ]]; then
  VL_ADAPTIVE_FLAG="--vl-adaptive --weight-vl-visual ${WEIGHT_VL_VISUAL:-0.25} --weight-vl-base ${WEIGHT_VL_BASE:-0}"
fi

python scripts/eval_decompose_recall.py \
  --qa-file "${QA_FILE}" --config-file "${CONFIG_FILE}" --device cuda \
  --output-dir "${OUTPUT_DIR}" --model "${ROUTER_MODEL}" \
  --api-base "http://127.0.0.1:${VLLM_PORT}/v1" --workers 8 \
  ${VL_ADAPTIVE_FLAG}
RESULT=$?
kill ${VLLM_PID} 2>/dev/null || true
echo "exit=${RESULT} output=${OUTPUT_DIR}"
