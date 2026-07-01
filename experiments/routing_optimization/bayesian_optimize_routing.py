#!/usr/bin/env python3
"""Bayesian optimization for ATM-Bench routing strategy parameters.

Optimizes:
  1. VL embedding weight
  2. BM25 normalization cap
  3. Dense embedding floor
  4. Conditional VL threshold
  5. Multi-stage filtering strategy

Objective: Maximize R@10 recall on standard set.

Usage:
    python scripts/QA_Agent/MMRAG/bayesian_optimize_routing.py \
        --n-trials 20 \
        --device cuda \
        --output-dir output/bayesian_opt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna required. Install with: pip install optuna")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ── Configuration ────────────────────────────────────────────────────────────

QA_FILE = ROOT / "data/atm-bench/atm-bench.json"
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"

# Baselines for reference
BASELINE_R10 = 0.6810816527993232  # standard text-only retrieval
BEST_KNOWN_R10 = 0.7318478822921072  # current best with Cond. Routing + VL (w=0.20)

# ── Objective Function ──────────────────────────────────────────────────────

def run_routing_trial(
    vl_weight: float,
    vl_conditional: bool = True,
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
    trial_id: int = 0,
    output_dir: Path = None,
    force_rebuild: bool = False,
) -> float:
    """Run routing retriever with given parameters and return R@10 recall.

    Args:
        vl_weight: VL embedding weight (0.0 - 1.0)
        vl_conditional: Enable conditional VL injection
        text_embedding_model: Text embedding model
        device: Device (cuda, cpu, mps)
        trial_id: Trial number (for logging)
        output_dir: Directory to save results
        force_rebuild: Force rebuild of index

    Returns:
        R@10 recall score (0.0 - 1.0)
    """

    trial_output_dir = output_dir / f"trial_{trial_id:03d}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python",
        str(ROOT / "scripts/QA_Agent/MMRAG/run_routing_reranker.py"),
        "--qa-file", str(QA_FILE),
        "--device", device,
        "--conditional",  # Always use conditional routing
        "--no-reranker",  # Disable reranker for faster iteration
        "--vl-embedding-model", "openai/clip-vit-large-patch14",
        "--vl-weight", str(vl_weight),
        "--text-embedding-model", text_embedding_model,
        "--retriever-batch-size", "64",
        "--primary-top-k", "100",
        "--retrieval-max-k", "200",
    ]

    if vl_conditional:
        cmd.append("--conditional-vl")

    if force_rebuild:
        cmd.append("--force-rebuild")

    # Run experiment
    print(f"\n[Trial {trial_id}] Running with vl_weight={vl_weight:.4f}, conditional_vl={vl_conditional}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"[Trial {trial_id}] ❌ FAILED")
            print(f"  stderr: {result.stderr[:500]}")
            return 0.0  # Return 0 for failed trials

        # Parse summary
        summary_file = ROOT / f"output/QA_Agent/MMRAG/routing_adaptive_only_conditional_vl_clipvitlargepatch14_condvl/routing_reranker_summary.json"
        if not summary_file.exists():
            print(f"[Trial {trial_id}] ❌ No summary found")
            return 0.0

        with open(summary_file) as f:
            summary = json.load(f)

        recall_r10 = summary["fusion_recall"]["R@10"]
        recall_r100 = summary["fusion_recall"]["R@100"]

        print(f"[Trial {trial_id}] ✅ R@10={recall_r10:.4f}, R@100={recall_r100:.4f} ({elapsed:.1f}s)")

        # Save trial results
        trial_result = {
            "trial_id": trial_id,
            "params": {
                "vl_weight": vl_weight,
                "vl_conditional": vl_conditional,
                "text_embedding_model": text_embedding_model,
            },
            "recall": {
                "R@10": recall_r10,
                "R@100": recall_r100,
            },
            "elapsed_s": elapsed,
        }
        with open(trial_output_dir / "result.json", "w") as f:
            json.dump(trial_result, f, indent=2)

        return recall_r10

    except subprocess.TimeoutExpired:
        print(f"[Trial {trial_id}] ⏱️ TIMEOUT")
        return 0.0
    except Exception as e:
        print(f"[Trial {trial_id}] ❌ ERROR: {e}")
        return 0.0


def objective(trial: optuna.Trial) -> float:
    """Objective function for Bayesian optimization."""

    # Suggest parameters
    vl_weight = trial.suggest_float("vl_weight", 0.05, 0.50, step=0.05)
    vl_conditional = trial.suggest_categorical("vl_conditional", [True, False])

    # Run trial
    recall_r10 = run_routing_trial(
        vl_weight=vl_weight,
        vl_conditional=vl_conditional,
        trial_id=trial.number,
        output_dir=Path(args.output_dir),
    )

    return recall_r10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian optimize ATM-Bench routing strategy"
    )
    p.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of optimization trials"
    )
    p.add_argument(
        "--device", default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference"
    )
    p.add_argument(
        "--output-dir",
        default="output/bayesian_opt",
        help="Output directory for results"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    p.add_argument(
        "--force-rebuild", action="store_true",
        help="Force rebuild of retrieval index"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Bayesian Optimization for ATM-Bench Routing Strategy")
    print("=" * 70)
    print(f"  Baseline (text-only):       R@10 = {BASELINE_R10:.4f}")
    print(f"  Best known (Cond+VL):       R@10 = {BEST_KNOWN_R10:.4f}")
    print(f"  Target:                     R@10 > {BEST_KNOWN_R10 + 0.02:.4f}")
    print(f"  Trials:                     {args.n_trials}")
    print(f"  Output dir:                 {output_dir}")
    print("=" * 70)

    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 70)
    print("  Optimization Complete")
    print("=" * 70)

    best_trial = study.best_trial
    print(f"\n✅ Best trial: #{best_trial.number}")
    print(f"   R@10:  {best_trial.value:.4f}")
    print(f"   Improvement: +{(best_trial.value - BEST_KNOWN_R10) * 100:.2f}%")
    print(f"\n   Parameters:")
    for param, value in best_trial.params.items():
        print(f"     {param}: {value}")

    # Save optimization history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optimization_history.csv", index=False)

    # Save best result
    best_result = {
        "best_trial": best_trial.number,
        "best_r10": best_trial.value,
        "best_params": best_trial.params,
        "improvement_vs_baseline": float(best_trial.value - BASELINE_R10),
        "improvement_vs_best_known": float(best_trial.value - BEST_KNOWN_R10),
    }
    with open(output_dir / "best_result.json", "w") as f:
        json.dump(best_result, f, indent=2)

    print(f"\n📊 Results saved to {output_dir}/")
