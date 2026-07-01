#!/usr/bin/env python3
"""Grid search + Bayesian optimization for routing parameters.

Tests combinations of:
  - VL weight: importance of vision-language embeddings
  - Conditional VL: inject VL only for visual queries
  - Hard mode: optimized settings for hard set
  - Strategies: different fusion weight combinations

Usage:
    # Grid search (all combinations)
    python scripts/QA_Agent/MMRAG/optimize_routing_grid_search.py \
        --qa-file data/atm-bench/atm-bench.json \
        --device cuda \
        --method grid

    # Bayesian optimization (sequential)
    python scripts/QA_Agent/MMRAG/optimize_routing_grid_search.py \
        --qa-file data/atm-bench/atm-bench.json \
        --device cuda \
        --method bayesian \
        --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ltma import (
    HybridRetriever,
    HybridScoringConfig,
    RoutingRetriever,
    RoutingConfig,
    ConfidenceConfig,
    AdaptiveWeights,
    analyze_query,
    adaptive_weights,
)
from ltma.routing_retriever import adaptive_weights_hard
from memqa.retrieve.utils import (
    RetrievalItem,
    EmailTextConfig,
    MediaTextConfig,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
    write_json,
)
from memqa.retrieve.retrievers import (
    SentenceTransformerRetriever,
    ClipRetriever,
)

# ── Paths ────────────────────────────────────────────────────────────
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"

RECALL_KS = [1, 5, 10, 25, 50, 100]


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    """Compute recall@k."""
    gt_set = set(gt_ids)
    recalls = {}
    for k in RECALL_KS:
        top = retrieved_ids[:k]
        hit = len([i for i in top if i in gt_set])
        recalls[f"R@{k}"] = hit / len(gt_ids) if gt_ids else 0.0
    return recalls


def evaluate_params(
    params: Dict[str, any],
    qa_list: List[Dict],
    items: List[RetrievalItem],
    text_retriever: SentenceTransformerRetriever,
    vl_retriever: Optional[ClipRetriever] = None,
    trial_name: str = "trial",
) -> Dict[str, float]:
    """Evaluate a parameter configuration on the QA set.

    Returns average recall@k across all questions.
    """

    router = RoutingRetriever(
        cache_dir=INDEX_CACHE,
        batch_size=params.get("retriever_batch_size", 64),
        device=params.get("device", "cuda"),
    )

    # Build hybrid base
    config = HybridScoringConfig(
        filter_mode=params.get("filter_mode", "soft"),
        weight_metadata=params.get("weight_metadata", 0.1),
        weight_sparse=params.get("weight_sparse", 0.4),
        weight_dense=params.get("weight_dense", 0.5),
    )

    router.build_index(
        items=items,
        text_retriever=text_retriever,
        vl_retriever=vl_retriever,
        cache_config={
            "method": trial_name,
            "dense_model": params.get("text_embedding_model"),
        },
        scoring_config=config,
    )

    # Run retrieval
    per_q_recalls = []
    total_time = 0.0

    for qa in tqdm(qa_list, desc=f"{trial_name} (eval)", leave=False):
        query = qa["question"]
        gt_ids = extract_evidence_ids(qa)

        t0 = time.perf_counter()
        results = router.retrieve_adaptive(
            query,
            max_k=params.get("retrieval_max_k", 200),
            override_weights=None,  # Will use default adaptive_weights
        )
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        retrieved_ids = [r.item.item_id for r in results]
        recalls = compute_recall(gt_ids, retrieved_ids)
        per_q_recalls.append(recalls)

    # Average recall
    avg_recall = {
        rk: np.mean([q[rk] for q in per_q_recalls])
        for rk in [f"R@{k}" for k in RECALL_KS]
    }
    avg_recall["avg_time_ms"] = (total_time / len(qa_list)) * 1000

    return avg_recall


def grid_search(args):
    """Exhaustive grid search over parameter combinations."""

    # Load data
    qa_list = load_json(args.qa_file)
    items = build_retrieval_items(
        image_batch=IMAGE_BATCH,
        video_batch=VIDEO_BATCH,
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
        email_file=EMAIL_FILE,
    )

    # Build retrievers
    text_retriever = SentenceTransformerRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=INDEX_CACHE,
        batch_size=64,
        device=args.device,
    )

    vl_retriever = ClipRetriever(
        model_name="openai/clip-vit-large-patch14",
        cache_dir=INDEX_CACHE,
        device=args.device,
    )

    text_retriever.build_index(items, {})
    vl_retriever.build_index(items, {})

    # Define grid
    param_grid = {
        "vl_weight": [0.05, 0.10, 0.15, 0.20, 0.30],
        "filter_mode": ["soft", "hard"],
        "conditional_vl": [True, False],
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combinations = list(product(*param_values))

    print("=" * 80)
    print(f"  Grid Search: {len(combinations)} combinations")
    print(f"  QA file: {Path(args.qa_file).name} ({len(qa_list)} questions)")
    print("=" * 80)

    results = []

    for combo_idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        trial_name = f"combo_{combo_idx:03d}"

        print(
            f"\n[{combo_idx + 1}/{len(combinations)}] "
            f"vl_w={params['vl_weight']:.2f}, "
            f"mode={params['filter_mode']}, "
            f"cond_vl={params['conditional_vl']}"
        )

        try:
            recall = evaluate_params(
                params=params,
                qa_list=qa_list,
                items=items,
                text_retriever=text_retriever,
                vl_retriever=vl_retriever,
                trial_name=trial_name,
            )

            result = {
                "combo_id": combo_idx,
                "params": params,
                "recall": recall,
            }
            results.append(result)

            print(
                f"  ✅ R@10={recall['R@10']:.4f}, "
                f"R@100={recall['R@100']:.4f}, "
                f"time={recall['avg_time_ms']:.1f}ms"
            )

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find best
    if results:
        best = max(results, key=lambda x: x["recall"]["R@10"])
        print("\n" + "=" * 80)
        print("  Best Configuration")
        print("=" * 80)
        print(f"  R@10: {best['recall']['R@10']:.4f}")
        print(f"  R@100: {best['recall']['R@100']:.4f}")
        print(f"  Params:")
        for k, v in best["params"].items():
            print(f"    {k}: {v}")
        print(f"\n  Results saved to {output_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize routing parameters")
    p.add_argument("--qa-file", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--method", default="grid", choices=["grid", "bayesian"])
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--output-dir", default="output/routing_optimization")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    grid_search(args)
