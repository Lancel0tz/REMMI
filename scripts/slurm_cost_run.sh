#!/usr/bin/env bash
#SBATCH -J cost-run
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cost_%x_%j.out
#SBATCH --error=logs/cost_%x_%j.err
#
# One method, one isolated eval_root, cache reporting ON. Parametrised by env:
#   NAME       short tag for output dir  (base | orgs | remmi9)
#   EVAL_ROOT  agent_systems/eval_root_<name>
# The whole point of this run: vLLM is started with --enable-prompt-tokens-details
# so usage carries prompt_tokens_details.cached_tokens -> pi records cacheRead ->
# usage.json splits uncached vs cache_read -> scripts/cost.py prices it exactly.
# Also: the run dir is NOT wiped (that is how the earlier fair-table splits were
# lost), and each method has its OWN eval_root so parallel jobs never collide.

set -uo pipefail
REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
REMMI_ROOT="/rds/user/kz345/hpc-work/REMMI"
cd "${REPO_ROOT}"; mkdir -p logs
source /home/kz345/miniconda3/etc/profile.d/conda.sh
conda activate atmbench
export PYTHONPATH="${REPO_ROOT}:${REMMI_ROOT}:${PYTHONPATH:-}"
export PATH="${HOME}/.npm-global/bin:${CONDA_PREFIX}/bin:/usr/bin:/bin:${PATH}"
export HF_HOME="${HF_HOME:-/rds/user/kz345/hpc-work/hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR="/local/${USER}/tmp"
export TRITON_CACHE_DIR="/local/${USER}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/local/${USER}/inductor_cache"
export FLASHINFER_WORKSPACE_BASE="/local/${USER}/flashinfer"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

# flashinfer JIT needs the venv's CUDA 13 (external qwen36vllm env), not cluster 11.4
VLLM_PYTHON_PRE="${VLLM_PYTHON:-python}"
if [[ "${VLLM_PYTHON_PRE}" != "python" && -x "${VLLM_PYTHON_PRE}" ]]; then
  _VENV_ROOT="$(dirname "$(dirname "${VLLM_PYTHON_PRE}")")"
  _VENV_CUDA="${_VENV_ROOT}/lib/python3.11/site-packages/nvidia/cu13"
  if [[ -x "${_VENV_CUDA}/bin/nvcc" ]]; then
    unset CPATH CUDA_PATH FPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
    export LIBRARY_PATH="$(echo "${LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE 'cuda/11|cuda-11' | paste -sd: -)"
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE 'cuda/11|cuda-11' | paste -sd: -)"
    export CUDA_HOME="${_VENV_CUDA}"; export PATH="${_VENV_CUDA}/bin:${PATH}"
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
  fi
fi

NAME="${NAME:?set NAME=base|orgs|remmi9}"
EVAL_ROOT="${EVAL_ROOT:?set EVAL_ROOT=agent_systems/eval_root_<name>}"
EVAL_ROOT_ABS="${REPO_ROOT}/${EVAL_ROOT}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B-FP8}"
# Pick a genuinely free port. A job-id-derived fixed port collides with a stale
# vLLM left on the node by a prior job (fuser -k can't kill another user's proc).
pick_free_port() {
  local p
  for _ in $(seq 1 50); do
    p=$(( 20000 + (RANDOM % 20000) ))
    if ! (exec 3<>/dev/tcp/127.0.0.1/${p}) 2>/dev/null; then echo "${p}"; return 0; fi
    exec 3>&- 2>/dev/null || true
  done
  echo $(( 20000 + SLURM_JOB_ID % 20000 ))   # fallback
}
VLLM_PORT="${VLLM_PORT:-$(pick_free_port)}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-qwen3_xml}"
VLLM_PYTHON="${VLLM_PYTHON:-python}"
CONCURRENCY="${CONCURRENCY:-4}"
TAG="openai-compatible_${VLLM_MODEL//\//_}"
RUN_DIR="${EVAL_ROOT_ABS}/runs/atm-bench-hard/pi/${TAG}"
OUT="output/events/cost_${NAME}"; mkdir -p "${OUT}"

echo "======== cost-run [${NAME}] eval_root=${EVAL_ROOT} | job ${SLURM_JOB_ID} on $(hostname) ========"
echo "  prompt: $(head -1 ${EVAL_ROOT_ABS}/prompts/system_prompt.txt)"
echo "  memory: $(ls ${EVAL_ROOT_ABS}/memory | tr '\n' ' ')"

# ── vLLM with cache reporting ON ─────────────────────────────────────────────
fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true; sleep 1
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" --tensor-parallel-size 1 --gpu-memory-utilization "${VLLM_GPU_MEM}" \
  --port "${VLLM_PORT}" --trust-remote-code --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --enforce-eager --enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_PARSER}" \
  --enable-prompt-tokens-details \
  > logs/vllm_cost_${NAME}_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
echo "[vLLM] waiting (cache reporting ON)..."; sleep 15
for i in {1..900}; do
  curl -s http://127.0.0.1:${VLLM_PORT}/v1/models >/dev/null 2>&1 && { echo "  ✅ vLLM ready"; break; }
  kill -0 ${VLLM_PID} 2>/dev/null || { echo "  ❌ vLLM died"; tail -40 logs/vllm_cost_${NAME}_${SLURM_JOB_ID}.log; exit 1; }
  [[ $i -eq 900 ]] && { echo "  ❌ timeout"; kill ${VLLM_PID}; exit 1; }
  sleep 2
done

# quick probe: confirm the endpoint actually emits cached_tokens on a repeat prompt
python3 - "$VLLM_PORT" "$VLLM_MODEL" <<'PY' || true
import json,urllib.request,sys
port,model=sys.argv[1],sys.argv[2]
def ask(p):
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model":model,"messages":[{"role":"user","content":p}],"max_tokens":8}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    return json.loads(urllib.request.urlopen(req,timeout=120).read())["usage"]
long="Repeat context. "*400
ask(long+" first")                       # populate cache
u=ask(long+" second")                    # should hit cache
d=u.get("prompt_tokens_details") or {}
print(f"[probe] prompt_tokens={u.get('prompt_tokens')} cached_tokens={d.get('cached_tokens')} "
      f"-> cache reporting {'WORKS' if d.get('cached_tokens') else 'returned 0/absent'}")
PY

# ── run pi on all 31 (run dir NOT wiped) ─────────────────────────────────────
export AGSYS_EVAL_ROOT="${EVAL_ROOT_ABS}"
export AGSYS_SYSTEM_PROMPT="${EVAL_ROOT_ABS}/prompts/system_prompt.txt"
export AGSYS_QA_SCHEMA="${EVAL_ROOT_ABS}/prompts/qa_schema.json"
export PI_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export PI_OPENAI_MODEL="${VLLM_MODEL}"
export PI_OPENAI_CONTEXT_WINDOW="${VLLM_MAX_MODEL_LEN}"
export PI_OPENAI_MAX_TOKENS="2048"
export AGSYS_PI_THINKING="off"
export AGSYS_PI_TIMEOUT_S="${AGSYS_PI_TIMEOUT_S:-2400}"
export AGSYS_SKIP_EVAL=1
export AGSYS_PI_MODEL_TAG="${TAG}"

xargs -P "${CONCURRENCY}" -I {} bash agent_systems/scripts/pi/run_pi_openai_compatible.sh {} \
  < "${EVAL_ROOT_ABS}/question_ids.txt"

# Assemble answers.jsonl DIRECTLY from this job's isolated run dir — collect_results.py
# writes a shared output/QA_Agent/.../${TAG}/ path that concurrent replicate jobs (same
# model tag) would clobber. Reading RUN_DIR/*/output/answer.json is race-free.
python3 - "${RUN_DIR}" "${OUT}/answers.jsonl" <<'PY'
import json,glob,sys
RD,OUT=sys.argv[1],sys.argv[2]
rows=[]
for f in sorted(glob.glob(RD+"/*/output/answer.json")):
    try: rows.append(json.load(open(f)))
    except Exception: pass
with open(OUT,"w") as fh:
    for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
print("  assembled %d answers -> %s"%(len(rows),OUT))
PY

kill ${VLLM_PID} 2>/dev/null || true

# ── per-method usage split (uncached / cache_read / output) from THIS run dir ─
python3 - "${RUN_DIR}" "${OUT}" "${NAME}" <<'PY'
import json,glob,sys,os
RD,OUT,NAME=sys.argv[1],sys.argv[2],sys.argv[3]
ui=cr=cw=out=tot=n=turns=0
for f in glob.glob(RD+"/*/output/usage.json"):
    d=json.load(open(f)); n+=1
    ui += d.get("input_tokens_uncached") or d.get("input_tokens") or 0
    cr += d.get("cache_read_input_tokens") or 0
    cw += d.get("cache_creation_input_tokens") or 0
    out+= d.get("output_tokens") or 0
    tot+= d.get("total_tokens") or 0
for t in glob.glob(RD+"/*/output/trace.jsonl"):
    turns += open(t).read().count("turn_start")
rec=dict(name=NAME,n=n,turns=turns,uncached_in=ui,cache_read=cr,cache_write=cw,
         output=out,total=tot,
         cache_hit_rate=(cr/(ui+cr) if (ui+cr) else 0.0))
json.dump(rec,open(os.path.join(OUT,"usage_split.json"),"w"),indent=1)
print("\n======== cost-run [%s] usage split ========"%NAME)
print("  n=%d turns=%d"%(n,turns))
print("  uncached_in=%d  cache_read=%d  cache_write=%d  output=%d  total=%d"%(ui,cr,cw,out,tot))
print("  cache_hit_rate=%.4f  (cache_read / (uncached+cache_read))"%rec["cache_hit_rate"])
PY

# ── price it ─────────────────────────────────────────────────────────────────
python3 - "${OUT}/usage_split.json" "${OUT}" <<'PY'
import json,sys
u=json.load(open(sys.argv[1]))
IN,OUT_,CR,CW = 0.29/1e6, 3.20/1e6, 0.029/1e6, 0.29/1e6
cost=dict(uncached=u["uncached_in"]*IN, cache_read=u["cache_read"]*CR,
          cache_write=u["cache_write"]*CW, output=u["output"]*OUT_)
cost["total"]=sum(cost.values())
# also an "as if no cache" upper bound for comparison with the old h=0 numbers
full=(u["uncached_in"]+u["cache_read"]+u["cache_write"])*IN + u["output"]*OUT_
json.dump(dict(usage=u,cost=cost,cost_full_uncached=full),
          open(sys.argv[2]+"/cost.json","w"),indent=1)
print("  cost breakdown $: uncached=%.4f cache_read=%.4f output=%.4f -> TOTAL=%.4f"%(
      cost["uncached"],cost["cache_read"],cost["output"],cost["total"]))
print("  (if no cache existed: $%.4f — the old h=0 upper bound)"%full)
PY
echo "======== cost-run [${NAME}] DONE — judge: bash scripts/eval_agent_pi_atm.sh ${OUT}/answers.jsonl ========"
