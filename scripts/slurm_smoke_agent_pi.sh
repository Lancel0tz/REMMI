#!/usr/bin/env bash
#SBATCH -J smoke-agent-pi
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/smoke_agent_pi_%j.out
#SBATCH --error=logs/smoke_agent_pi_%j.err

set -uo pipefail

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"
mkdir -p logs

source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate atmbench

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# node (conda) + pi (npm-global) on PATH; system tools for bwrap
export PATH="${HOME}/.npm-global/bin:${CONDA_PREFIX}/bin:/usr/bin:/bin:${PATH}"

# Force node-local disk for temp/compile caches (avoid /tmp tmpfs filling)
export TMPDIR="/local/${USER}/tmp"
export TRITON_CACHE_DIR="/local/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/local/${USER}/inductor_cache"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ── Parameters ───────────────────────────────────────────────────────────────
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.8}"
# agent 读 memory 时上下文可达 ~46K，32K 会触发 400；给到 64K 留余量
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
# Question to run (default: first hard question)
QID="${QID:-$(head -1 agent_systems/eval_root_sgm/question_ids.txt)}"

echo "================================================================================"
echo "  SMOKE TEST: pi agent on ATM-Bench-hard (self-hosted vLLM, free)"
echo "  Job ${SLURM_JOB_ID} on $(hostname)  |  QID=${QID}"
echo "================================================================================"
node --version; ~/.npm-global/bin/pi --version 2>&1 | head -1

# ── Start vLLM (OpenAI-compatible endpoint pi will hit) ──────────────────────
echo "[vLLM] starting ${VLLM_MODEL} on :${VLLM_PORT}"
fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true
sleep 1
python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${VLLM_GPU_MEM}" \
  --port "${VLLM_PORT}" \
  --trust-remote-code \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > logs/vllm_agent_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "  waiting for vLLM..."
sleep 15
for i in {1..300}; do
  if curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
    echo "  ✅ vLLM ready ($((15 + i*2))s)"; break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "  ❌ vLLM died"; tail -40 logs/vllm_agent_${SLURM_JOB_ID}.log; exit 1
  fi
  [[ $i -eq 300 ]] && { echo "  ❌ vLLM timeout"; tail -40 logs/vllm_agent_${SLURM_JOB_ID}.log; kill ${VLLM_PID}; exit 1; }
  sleep 2
done

# ── Run pi on a single question via the OpenAI-compatible preset ─────────────
echo ""
echo "[pi] running question ${QID} ..."
export PI_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export PI_OPENAI_MODEL="${VLLM_MODEL}"
export PI_OPENAI_CONTEXT_WINDOW="${VLLM_MAX_MODEL_LEN}"
export PI_OPENAI_MAX_TOKENS="2048"
# Qwen3-VL-8B via vLLM 的 OpenAI 端点不支持 reasoning/thinking API → 必须关闭，
# 否则 pi 走 reasoning 代码路径、根本不发请求 (vLLM 收到 0 请求, 空回复)
export AGSYS_PI_THINKING="${AGSYS_PI_THINKING:-off}"
# enforce-eager 下 8B 较慢 + 多轮工具调用，900s 不够，给 40 分钟
export AGSYS_PI_TIMEOUT_S="${AGSYS_PI_TIMEOUT_S:-2400}"
export AGSYS_SKIP_EVAL=1   # smoke: skip the ATM judge step

bash agent_systems/scripts/pi/run_pi_openai_compatible.sh "${QID}"
PI_RESULT=$?

# ── Cleanup + show answer ────────────────────────────────────────────────────
kill ${VLLM_PID} 2>/dev/null || true
sleep 2

echo ""
echo "================================================================================"
echo "  RESULT"
echo "================================================================================"
ANS=$(find agent_systems/eval_root_sgm/runs -path "*${QID}*/output/answer.json" 2>/dev/null | head -1)
if [[ -n "${ANS}" && -f "${ANS}" ]]; then
  echo "✅ answer.json: ${ANS}"
  cat "${ANS}"
  echo ""
  echo "真实答案 (gold):"
  python3 -c "import json;d={q['id']:q for q in json.load(open('data/atm-bench/atm-bench-hard.json'))};q=d.get('${QID}',{});print('  Q:',q.get('question'));print('  GOLD:',q.get('answer'))"
else
  echo "❌ no answer.json produced (pi exit=${PI_RESULT})"
  echo "trace/stderr:"
  find agent_systems/eval_root_sgm/runs -path "*${QID}*/output/*" 2>/dev/null | head
  STDERR=$(find agent_systems/eval_root_sgm/runs -path "*${QID}*/output/stderr.log" 2>/dev/null | head -1)
  [[ -n "${STDERR}" ]] && tail -30 "${STDERR}"
fi
echo "================================================================================"
