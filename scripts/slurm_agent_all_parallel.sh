#!/usr/bin/env bash
#SBATCH -J agent-pi-par
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/agent_pi_par_%j.out
#SBATCH --error=logs/agent_pi_par_%j.err

set -uo pipefail

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"
mkdir -p logs

source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate atmbench

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="${HOME}/.npm-global/bin:${CONDA_PREFIX}/bin:/usr/bin:/bin:${PATH}"

# ── HF cache adaptation: models live on RDS now (moved off the small home quota) ──
export HF_HOME="${HF_HOME:-/rds/user/kz345/hpc-work/hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # 模型已预下载, 离线加载更快/更稳

export TMPDIR="/local/${USER}/tmp"
export TRITON_CACHE_DIR="/local/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/local/${USER}/inductor_cache"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ── Parameters ───────────────────────────────────────────────────────────────
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
# 唯一端口, 避免和残留 vLLM 撞车 (上次 8000 撞车 → 新 vLLM 起不来, pi 连到残留服务报 401)
VLLM_PORT="${VLLM_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-hermes}"
CONCURRENCY="${CONCURRENCY:-4}"
EVAL_ROOT="${EVAL_ROOT:-agent_systems/eval_root_sgm}"

echo "================================================================================"
echo "  pi agent (CONCURRENCY=${CONCURRENCY}) on ALL ATM-Bench-hard — model=${VLLM_MODEL}"
echo "  Job ${SLURM_JOB_ID} on $(hostname)  |  HF_HOME=${HF_HOME}"
echo "================================================================================"
node --version; ~/.npm-global/bin/pi --version 2>&1 | head -1

# ── Start vLLM (tool-calling enabled) ────────────────────────────────────────
echo "[vLLM] starting ${VLLM_MODEL} (tool-parser=${VLLM_TOOL_PARSER})"
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
  --tool-call-parser "${VLLM_TOOL_PARSER}" \
  > logs/vllm_agent_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "  waiting for vLLM..."
sleep 15
for i in {1..300}; do
  if curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
    echo "  ✅ vLLM ready ($((15 + i*2))s)"; break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "  ❌ vLLM died (模型可能不被此 vLLM 版本支持)"; tail -50 logs/vllm_agent_${SLURM_JOB_ID}.log; exit 1
  fi
  [[ $i -eq 300 ]] && { echo "  ❌ vLLM timeout"; tail -50 logs/vllm_agent_${SLURM_JOB_ID}.log; kill ${VLLM_PID}; exit 1; }
  sleep 2
done

# ── Run pi on all questions, CONCURRENCY at a time ───────────────────────────
echo ""
echo "[pi] running ALL questions, ${CONCURRENCY} in parallel ..."
export PI_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export PI_OPENAI_MODEL="${VLLM_MODEL}"
export PI_OPENAI_CONTEXT_WINDOW="${VLLM_MAX_MODEL_LEN}"
export PI_OPENAI_MAX_TOKENS="2048"
export AGSYS_PI_THINKING="${AGSYS_PI_THINKING:-off}"
export AGSYS_PI_TIMEOUT_S="${AGSYS_PI_TIMEOUT_S:-1200}"
export AGSYS_SKIP_EVAL=1
MODEL_TAG="openai-compatible_${VLLM_MODEL//\//_}"
export AGSYS_PI_MODEL_TAG="${MODEL_TAG}"

# 每个 qid 是独立沙箱 → 可安全并行。xargs -P 控制并发数。
xargs -P "${CONCURRENCY}" -I {} bash agent_systems/scripts/pi/run_pi_openai_compatible.sh {} \
  < "${EVAL_ROOT}/question_ids.txt"
PI_RESULT=$?

# ── Collect answers.jsonl + usage ────────────────────────────────────────────
echo ""
echo "[collect] gathering answers.jsonl ..."
python3 agent_systems/collect_results.py --all --model-tag "${MODEL_TAG}" 2>&1 | tail -5 || true
python3 agent_systems/collect_usage.py   --all --model-tag "${MODEL_TAG}" 2>&1 | tail -3 || true

kill ${VLLM_PID} 2>/dev/null || true
sleep 2

# ── Report ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "  DONE (pi exit=${PI_RESULT})"
echo "================================================================================"
ANSWERS=$(find output/QA_Agent/AgentSystems -name answers.jsonl -path "*${MODEL_TAG}*" 2>/dev/null | head -1)
if [[ -n "${ANSWERS}" && -f "${ANSWERS}" ]]; then
  echo "✅ answers.jsonl: ${ANSWERS}  ($(wc -l < "${ANSWERS}") 行)"
  python3 -c "
import json,collections
c=collections.Counter()
for l in open('${ANSWERS}'):
    a=json.loads(l); ans=str(a.get('answer','')).strip().lower()
    c['unknown/空' if ans in ('unknown','') else '实际作答']+=1
print('  ', dict(c))
"
  echo ""
  echo "🎯 ATM 评测 (登录节点跑, 无需 GPU):"
  echo "   bash scripts/eval_agent_pi_atm.sh '${ANSWERS}'"
else
  echo "❌ 无 answers.jsonl; 单题 answer.json 数: $(find ${EVAL_ROOT}/runs -name answer.json 2>/dev/null | wc -l)"
fi
echo "================================================================================"
