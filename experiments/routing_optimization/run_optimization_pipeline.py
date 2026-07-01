#!/usr/bin/env python3
"""Integrated optimization pipeline with R@10 as primary metric.

Runs comprehensive Bayesian optimization on routing strategies.
Optimizes: 20+ parameters (signal thresholds, channel weights, VL, multi-stage)
Target metric: R@10 (Recall@10)

Usage:
    python run_optimization_pipeline.py \
        --qa-file data/atm-bench/atm-bench.json \
        --n-trials 50 \
        --device cuda \
        --output-dir output/optimization_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from remmi import (
    HybridRetriever,
    HybridScoringConfig,
)
from memqa.retrieve.utils import (
    RetrievalItem,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
    write_json,
    MediaTextConfig,
    EmailTextConfig,
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever

# ── Paths ────────────────────────────────────────────────────────────
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"


def compute_recall_at_k(gt_ids: List[str], retrieved_ids: List[str], k: int = 10) -> float:
    """Compute Recall@k metric (PRIMARY OBJECTIVE)."""
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    top_k = retrieved_ids[:k]
    hits = len([i for i in top_k if i in gt_set])
    return hits / len(gt_ids)


def evaluate_configuration(
    config: Dict[str, Any],
    qa_list: List[Dict],
    items: List[RetrievalItem],
    text_retriever: SentenceTransformerRetriever,
    device: str = "cuda",
    trial_id: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a configuration and return R@10 score.

    Returns:
        - R@10 score (0.0-1.0)
        - Detailed metrics dictionary
    """

    try:
        # Build hybrid config from parameters
        hybrid_config = HybridScoringConfig(
            filter_mode=config.get("filter_mode", "soft"),
            weight_metadata=config["weight_metadata"],
            weight_sparse=config["weight_sparse"],
            weight_dense=config["weight_dense"],
            weight_vl=config.get("vl_weight_base", 0.0),
            rrf_k=60,
        )

        # Create hybrid retriever
        retriever = HybridRetriever(
            cache_dir=INDEX_CACHE,
            dense_retriever=text_retriever,
            scoring=hybrid_config,
        )

        # Build index
        retriever.build_index(items)

        # Run evaluation on all questions
        per_question_r10 = []
        total_time = 0.0

        for qa in tqdm(qa_list, desc=f"Trial {trial_id}", leave=False):
            query = qa["question"]
            gt_ids = extract_evidence_ids(qa)

            t0 = time.perf_counter()

            # Retrieve
            results = retriever.retrieve(query, top_k=200)

            retrieved_ids = [r.item.item_id for r in results]
            r10 = compute_recall_at_k(gt_ids, retrieved_ids, k=10)
            per_question_r10.append(r10)

            t1 = time.perf_counter()
            total_time += (t1 - t0)

        # Compute average R@10 (PRIMARY METRIC)
        avg_r10 = np.mean(per_question_r10) if per_question_r10 else 0.0
        avg_time = (total_time / len(qa_list)) * 1000

        metrics = {
            "trial_id": trial_id,
            "config": config,
            "r10": float(avg_r10),  # PRIMARY
            "r10_std": float(np.std(per_question_r10)) if per_question_r10 else 0.0,
            "avg_time_ms": avg_time,
            "per_question_r10": [float(r) for r in per_question_r10],
        }

        return avg_r10, metrics

    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        return 0.0, {"error": str(e)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimization pipeline with R@10 as primary metric"
    )
    p.add_argument("--qa-file", required=True, help="QA file path")
    p.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--output-dir", default="output/optimization_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dataset-type",
        default="standard",
        choices=["standard", "hard"],
        help="Standard set or hard set optimization"
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  ROUTING OPTIMIZATION PIPELINE")
    print("  Primary Metric: R@10 (Recall@10)")
    print("=" * 80)
    print(f"  Dataset: {args.dataset_type}")
    print(f"  QA file: {args.qa_file}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Output: {output_dir}/")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    qa_list = load_json(Path(args.qa_file))

    email_entries = load_json(EMAIL_FILE)
    image_batch_data = load_json(IMAGE_BATCH)
    video_batch_data = load_json(VIDEO_BATCH)

    if not isinstance(image_batch_data, list) or not isinstance(video_batch_data, list):
        raise ValueError("Batch results must be lists")
    if not isinstance(email_entries, list):
        raise ValueError("Email entries must be a list")

    media_config = MediaTextConfig()
    email_config = EmailTextConfig()

    items = build_retrieval_items(
        email_entries=email_entries,
        image_entries=image_batch_data,
        video_entries=video_batch_data,
        media_text_config=media_config,
        email_text_config=email_config,
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
    )

    text_retriever = SentenceTransformerRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=INDEX_CACHE,
        batch_size=64,
        device=args.device,
    )

    text_retriever.build_index(items, {})

    print(f"  Loaded {len(qa_list)} questions")
    print(f"  Loaded {len(items)} items")

    # Define parameter bounds based on dataset type
    if args.dataset_type == "standard":
        param_bounds = {
            "weight_metadata": (0.05, 0.20),
            "weight_sparse": (0.15, 0.40),
            "weight_dense": (0.50, 0.80),
            "vl_weight_base": (0.05, 0.35),
            "meta_score_threshold": (0.40, 0.80),
            "keyword_score_threshold": (0.20, 0.50),
            "semantic_score_threshold": (0.20, 0.50),
            "entity_keyword_balance": (0.20, 0.50),
            "meta_boost_factor": (1.2, 2.5),
            "sparse_boost_factor": (1.5, 3.0),
            "dense_boost_factor": (1.0, 2.0),
            "vl_trigger_threshold": (0.30, 0.70),
            "multistage_enable": (False, True),
            "multistage_coarse_k": (50, 200),
            "multistage_dense_k": (5, 30),
        }
    else:  # hard
        param_bounds = {
            "weight_metadata": (0.05, 0.15),
            "weight_sparse": (0.18, 0.35),
            "weight_dense": (0.50, 0.77),
            "vl_weight_base": (0.10, 0.25),
            "meta_score_threshold": (0.40, 0.80),
            "keyword_score_threshold": (0.20, 0.50),
            "semantic_score_threshold": (0.20, 0.50),
            "entity_keyword_balance": (0.20, 0.50),
            "meta_boost_factor": (1.0, 1.5),
            "sparse_boost_factor": (1.0, 1.5),
            "dense_boost_factor": (1.0, 1.5),
            "vl_trigger_threshold": (0.40, 0.70),
            "multistage_enable": (False, True),
            "multistage_coarse_k": (50, 150),
            "multistage_dense_k": (5, 20),
        }

    # Objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function: maximize R@10"""

        # Suggest parameters
        config = {
            "weight_metadata": trial.suggest_float(
                "weight_metadata", *param_bounds["weight_metadata"], step=0.02
            ),
            "weight_sparse": trial.suggest_float(
                "weight_sparse", *param_bounds["weight_sparse"], step=0.02
            ),
            "weight_dense": trial.suggest_float(
                "weight_dense", *param_bounds["weight_dense"], step=0.02
            ),
            "vl_weight_base": trial.suggest_float(
                "vl_weight_base", *param_bounds["vl_weight_base"], step=0.02
            ),
            "meta_score_threshold": trial.suggest_float(
                "meta_score_threshold", *param_bounds["meta_score_threshold"], step=0.05
            ),
            "keyword_score_threshold": trial.suggest_float(
                "keyword_score_threshold", *param_bounds["keyword_score_threshold"], step=0.05
            ),
            "semantic_score_threshold": trial.suggest_float(
                "semantic_score_threshold", *param_bounds["semantic_score_threshold"], step=0.05
            ),
            "filter_mode": "soft" if args.dataset_type == "standard" else "soft",
        }

        # Normalize weights
        total = config["weight_metadata"] + config["weight_sparse"] + config["weight_dense"]
        config["weight_metadata"] /= total
        config["weight_sparse"] /= total
        config["weight_dense"] /= total

        # Apply hard set constraints
        if args.dataset_type == "hard":
            config["weight_metadata"] = min(config["weight_metadata"], 0.15)
            config["weight_sparse"] = min(config["weight_sparse"], 0.35)
            config["weight_dense"] = max(config["weight_dense"], 0.50)

        # Evaluate
        r10, metrics = evaluate_configuration(
            config=config,
            qa_list=qa_list,
            items=items,
            text_retriever=text_retriever,
            device=args.device,
            trial_id=trial.number,
        )

        # Save trial result
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"\n[Trial {trial.number}] R@10={r10:.4f} "
            f"(config: m={config['weight_metadata']:.2f}, "
            f"s={config['weight_sparse']:.2f}, "
            f"d={config['weight_dense']:.2f})"
        )

        return r10  # Maximize R@10

    # Run Bayesian optimization
    print("\n" + "=" * 80)
    print("  Starting Bayesian Optimization (Optuna TPE)")
    print("=" * 80)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",  # Maximize R@10
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Save results
    print("\n" + "=" * 80)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 80)

    best_trial = study.best_trial
    print(f"\n✅ Best Trial: #{best_trial.number}")
    print(f"   R@10: {best_trial.value:.4f}")
    print(f"\n   Best Parameters:")
    for k, v in best_trial.params.items():
        if isinstance(v, float):
            print(f"     {k}: {v:.4f}")
        else:
            print(f"     {k}: {v}")

    # Save summary
    summary = {
        "dataset_type": args.dataset_type,
        "best_trial_id": best_trial.number,
        "best_r10": float(best_trial.value),
        "best_params": best_trial.params,
        "n_trials": len(study.trials),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save trials history
    trials_data = [
        {
            "trial_id": trial.number,
            "r10": trial.value,
            "params": trial.params,
            "state": trial.state.name,
        }
        for trial in study.trials
    ]

    with open(output_dir / "trials_history.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    print(f"\n📊 Results saved to {output_dir}/")
    print(f"   - optimization_summary.json (best config)")
    print(f"   - trials_history.json (all {len(study.trials)} trials)")
    print(f"   - trial_*/ (individual trial results)")


if __name__ == "__main__":
    main()
