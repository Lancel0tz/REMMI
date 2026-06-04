#!/usr/bin/env bash
# Run routing retrieval on ATM-Bench standard set with conditional VL (w=0.20) and no reranker.
#
# Submit:
#   sbatch scripts/QA_Agent/MMRAG/slurm_routing_cond_vl020_no_reranker.sh
#
# Useful overrides:
#   sbatch --time=06:00:00 scripts/QA_Agent/MMRAG/slurm_routing_cond_vl020_no_reranker.sh
#   FORCE_REBUILD=1 sbatch scripts/QA_Agent/MMRAG/slurm_routing_cond_vl020_no_reranker.sh

#SBATCH -J atm-routing-std-vl020
#SBATCH -A MLMI-kz345-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/routing_cond_vl020_%j.out
#SBATCH --error=logs/routing_cond_vl020_%j.err

set -euo pipefail

REPO_ROOT="/rds/user/kz345/hpc-work/ATM-Bench"
cd "${REPO_ROOT}"

mkdir -p logs

# Activate conda environment
if [[ -f /home/kz345/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/kz345/miniconda3/etc/profile.d/conda.sh
  conda activate atmbench
else
  echo "Could not find conda.sh" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Keep model/cache/temp writes on fast local or project storage.
export TMPDIR="${TMPDIR:-/local/${USER}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/local/${USER}/pip_cache}"
export HF_HOME="${HF_HOME:-/rds/user/kz345/hpc-work/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# Avoid occasional CUDA/Triton helper build issues on the cluster.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export AS=/usr/bin/as
export LD=/usr/bin/ld
export AR=/usr/bin/ar
export PATH=/usr/bin:/bin:${PATH}

FORCE_REBUILD_ARG=()
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  FORCE_REBUILD_ARG=(--force-rebuild)
fi

python scripts/QA_Agent/MMRAG/run_routing_reranker.py \
  --qa-file data/atm-bench/atm-bench.json \
  --device cuda \
  --conditional \
  --no-reranker \
  --vl-embedding-model openai/clip-vit-large-patch14 \
  --vl-weight 0.20 \
  --conditional-vl \
  --text-embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retriever-batch-size 64 \
  --primary-top-k 100 \
  --retrieval-max-k 200 \
  "${FORCE_REBUILD_ARG[@]}"
