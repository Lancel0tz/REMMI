#!/usr/bin/env bash
#SBATCH -J agent-pi-all
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/agent_pi_all_%j.out
#SBATCH --error=logs/agent_pi_all_%j.err

set -uo pipefail

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"
mkdir -p logs

source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate atmbench

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="${HOME}/.npm-global/bin:${CONDA_PREFIX}/bin:/usr/bin:/bin:${PATH}"

export TMPDIR="/local/${USER}/tmp"
export TRITON_CACHE_DIR="/local/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/local/${USER}/inductor_cache"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ── Parameters ───────────────────────────────────────────────────────────────
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"

echo "================================================================================"
echo "  pi agent on ALL ATM-Bench-hard (31 q) — self-hosted vLLM (free)"
echo "  Job ${SLURM_JOB_ID} on $(hostname)"
echo "================================================================================"
node --version; ~/.npm-global/bin/pi --version 2>&1 | head -1

# ── Start vLLM (with tool-calling for the agent) ─────────────────────────────
echo "[vLLM] starting ${VLLM_MODEL} on :${VLLM_PORT} (tool-calling enabled)"
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

# ── Run pi on ALL questions (no qid) → collects answers.jsonl ────────────────
echo ""
echo "[pi] running ALL 31 questions ..."
export PI_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export PI_OPENAI_MODEL="${VLLM_MODEL}"
export PI_OPENAI_CONTEXT_WINDOW="${VLLM_MAX_MODEL_LEN}"
export PI_OPENAI_MAX_TOKENS="2048"
export AGSYS_PI_THINKING="${AGSYS_PI_THINKING:-off}"
export AGSYS_PI_TIMEOUT_S="${AGSYS_PI_TIMEOUT_S:-1200}"   # 20min/题上限，防卡死
export AGSYS_SKIP_EVAL=1   # 评测放登录节点单独跑 (计算节点可能无外网)

bash agent_systems/scripts/pi/run_pi_openai_compatible.sh
PI_RESULT=$?

kill ${VLLM_PID} 2>/dev/null || true
sleep 2

# ── Locate the collected answers.jsonl ───────────────────────────────────────
echo ""
echo "================================================================================"
echo "  DONE (pi exit=${PI_RESULT})"
echo "================================================================================"
ANSWERS=$(find output/QA_Agent/AgentSystems -name answers.jsonl -newermt '-13 hours' 2>/dev/null | head -1)
if [[ -n "${ANSWERS}" && -f "${ANSWERS}" ]]; then
  echo "✅ answers.jsonl: ${ANSWERS}"
  echo "   行数: $(wc -l < "${ANSWERS}")"
  python3 -c "
import json,collections
c=collections.Counter()
for l in open('${ANSWERS}'):
    a=json.loads(l); ans=str(a.get('answer','')).strip().lower()
    c['unknown' if ans in ('unknown','') else 'answered']+=1
print('  已作答:',c['answered'],' Unknown/空:',c['unknown'])
"
  echo ""
  echo "🎯 下一步 (在登录节点跑 ATM 评测, 无需 GPU):"
  echo "   bash scripts/eval_agent_pi_atm.sh '${ANSWERS}'"
else
  echo "❌ 未找到 answers.jsonl (检查 run 目录)"
  find agent_systems/eval_root_sgm/runs -name answer.json 2>/dev/null | wc -l | xargs echo "  单题 answer.json 数:"
fi
echo "================================================================================"
