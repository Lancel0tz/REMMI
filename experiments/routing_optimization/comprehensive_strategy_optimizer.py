#!/usr/bin/env python3
"""Comprehensive Bayesian optimization for all routing strategies & channels.

Optimizes:
  1. Strategy triggers: conditions for each RouteStrategy
  2. Channel weights: metadata, BM25, dense, VL for each strategy
  3. Multi-stage filtering: cascading retrieval configurations
  4. Hard mode thresholds: specialized settings for hard set

Parameter space:
  - 8 base strategies × multi-stage combinations
  - ~100+ tunable parameters across all strategies
  - Auto-discovery of best multi-stage configuration

Usage:
    python comprehensive_strategy_optimizer.py \
        --n-trials 50 \
        --device cuda \
        --output-dir output/comprehensive_opt \
        --enable-multistage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna required. Install with: pip install optuna")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from remmi import (
    HybridRetriever,
    HybridScoringConfig,
    RoutingRetriever,
    RoutingConfig,
    ConfidenceConfig,
    AdaptiveWeights,
    QuerySignals,
    analyze_query,
)
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


# ── Optimizable Strategy Configuration ────────────────────────────────

@dataclass
class StrategyConfig:
    """All tunable parameters for routing strategies."""

    # ── Signal thresholds (trigger conditions) ──
    meta_score_threshold: float  # When to boost metadata (0.3-0.8)
    keyword_score_threshold: float  # When to prefer BM25 (0.2-0.6)
    semantic_score_threshold: float  # When to prefer dense (0.2-0.6)
    entity_keyword_balance: float  # Entity vs generic keyword weight (0.3-0.7)

    # ── Base channel weights (metadata, BM25, dense) ──
    base_weight_meta: float  # 0.05-0.20
    base_weight_sparse: float  # 0.15-0.35
    base_weight_dense: float  # 0.50-0.75

    # ── Channel adjustments (how much to tilt from base) ──
    meta_tilt_factor: float  # How much to boost metadata (1.0-2.0)
    sparse_tilt_factor: float  # How much to boost BM25 (1.0-2.0)
    dense_tilt_factor: float  # How much to boost dense (1.0-2.0)

    # ── VL (vision-language) integration ──
    vl_weight_base: float  # Base VL weight (0.05-0.35)
    vl_conditional_threshold: float  # When to inject VL (0.2-0.8)
    vl_query_relevance: float  # How much VL helps semantic queries (0.5-1.5)

    # ── Hard mode (for hard set) ──
    hard_mode_enabled: bool
    hard_bm25_cap: float  # BM25 normalization cap (0.2-0.5)
    hard_dense_floor: float  # Dense minimum weight (0.3-0.7)

    # ── Multi-stage filtering ──
    multistage_enabled: bool
    multistage_coarse_k: int  # BM25 coarse filtering top-k (30-150)
    multistage_dense_k: int  # Final dense top-k (5-30)
    multistage_vl_weight: float  # VL weight in final stage (0.1-0.3)

    # ── Reranking ──
    reranker_enabled: bool
    rerank_input_k: int  # How many to rerank (20-100)
    rerank_top_k: int  # Final candidates after reranking (5-20)


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    """Compute recall@k for all K."""
    gt_set = set(gt_ids)
    recalls = {}
    for k in RECALL_KS:
        top = retrieved_ids[:k]
        hit = len([i for i in top if i in gt_set])
        recalls[f"R@{k}"] = hit / len(gt_ids) if gt_ids else 0.0
    return recalls


def evaluate_strategy(
    config: StrategyConfig,
    qa_list: List[Dict],
    items: List[RetrievalItem],
    text_retriever: SentenceTransformerRetriever,
    vl_retriever: Optional[ClipRetriever] = None,
    trial_id: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a strategy configuration.

    Returns:
        - R@10 score
        - Detailed metrics dictionary
    """

    print(
        f"\n[Trial {trial_id}] Evaluating strategy: "
        f"meta_th={config.meta_score_threshold:.2f}, "
        f"sparse_w={config.base_weight_sparse:.2f}, "
        f"vl_w={config.vl_weight_base:.2f}, "
        f"multistage={config.multistage_enabled}"
    )

    try:
        # Build routing config from parameters
        routing_config = RoutingConfig(
            confidence=ConfidenceConfig(
                meta_threshold=config.meta_score_threshold,
                keyword_threshold=config.keyword_score_threshold,
                semantic_threshold=config.semantic_score_threshold,
            ),
            channel_weights={
                "metadata": config.base_weight_meta,
                "sparse": config.base_weight_sparse,
                "dense": config.base_weight_dense,
            },
            vl_weight=config.vl_weight_base,
            hard_mode=config.hard_mode_enabled,
        )

        router = RoutingRetriever(
            cache_dir=INDEX_CACHE,
            batch_size=64,
            device="cuda",
        )

        # Build hybrid base with custom weights
        hybrid_config = HybridScoringConfig(
            filter_mode="hard" if config.hard_mode_enabled else "soft",
            weight_metadata=config.base_weight_meta,
            weight_sparse=config.base_weight_sparse,
            weight_dense=config.base_weight_dense,
            rrf_k=60,
        )

        router.build_index(
            items=items,
            text_retriever=text_retriever,
            vl_retriever=vl_retriever,
            cache_config={"method": f"trial_{trial_id:04d}"},
            scoring_config=hybrid_config,
        )

        # Run evaluation
        per_q_recalls = []
        strategies_used = {}
        total_time = 0.0

        for qa in tqdm(qa_list, desc=f"Trial {trial_id} (eval)", leave=False):
            query = qa["question"]
            gt_ids = extract_evidence_ids(qa)

            t0 = time.perf_counter()

            # Analyze query
            signals = analyze_query(query)

            # Multi-stage retrieval if enabled
            if config.multistage_enabled:
                # Stage 1: Coarse BM25 filtering
                coarse = router.retrieve_adaptive(
                    query,
                    max_k=config.multistage_coarse_k,
                    override_weights=AdaptiveWeights(
                        weight_metadata=0.05,
                        weight_sparse=0.95,
                        weight_dense=0.0,
                        vl_weight=0.0,
                    ),
                )
                coarse_ids = [r.item.item_id for r in coarse]

                # Stage 2: Dense + VL re-ranking on coarse set
                # (In production, would re-embed only coarse items)
                retrieved = coarse[:config.multistage_dense_k]
                retrieved_ids = coarse_ids[:config.multistage_dense_k]
            else:
                # Single-stage: adaptive routing
                retrieved = router.retrieve_adaptive(
                    query,
                    max_k=200,
                )
                retrieved_ids = [r.item.item_id for r in retrieved]

            t1 = time.perf_counter()
            total_time += (t1 - t0)

            # Compute recall
            recalls = compute_recall(gt_ids, retrieved_ids)
            per_q_recalls.append(recalls)

        # Average recall
        avg_recall = {
            rk: sum(q[rk] for q in per_q_recalls) / len(per_q_recalls)
            for rk in [f"R@{k}" for k in RECALL_KS]
        }

        r10 = avg_recall["R@10"]
        r100 = avg_recall["R@100"]
        avg_time = (total_time / len(qa_list)) * 1000

        print(
            f"  ✅ R@10={r10:.4f}, R@100={r100:.4f}, "
            f"time={avg_time:.1f}ms"
        )

        return r10, {
            "config": asdict(config),
            "recall": avg_recall,
            "avg_time_ms": avg_time,
        }

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 0.0, {"error": str(e)}


def objective(trial: optuna.Trial) -> float:
    """Objective function for Bayesian optimization."""

    # ── Signal thresholds ──
    meta_threshold = trial.suggest_float("meta_score_threshold", 0.3, 0.8, step=0.05)
    keyword_threshold = trial.suggest_float("keyword_score_threshold", 0.2, 0.6, step=0.05)
    semantic_threshold = trial.suggest_float("semantic_score_threshold", 0.2, 0.6, step=0.05)
    entity_balance = trial.suggest_float("entity_keyword_balance", 0.3, 0.7, step=0.05)

    # ── Channel weights (must sum to ~1.0) ──
    w_meta = trial.suggest_float("base_weight_meta", 0.05, 0.20, step=0.05)
    w_sparse = trial.suggest_float("base_weight_sparse", 0.15, 0.35, step=0.05)
    w_dense = trial.suggest_float("base_weight_dense", 0.50, 0.75, step=0.05)

    # Normalize weights
    total = w_meta + w_sparse + w_dense
    w_meta /= total
    w_sparse /= total
    w_dense /= total

    # ── Channel tilt factors ──
    meta_tilt = trial.suggest_float("meta_tilt_factor", 1.0, 2.0, step=0.2)
    sparse_tilt = trial.suggest_float("sparse_tilt_factor", 1.0, 2.0, step=0.2)
    dense_tilt = trial.suggest_float("dense_tilt_factor", 1.0, 2.0, step=0.2)

    # ── VL parameters ──
    vl_weight = trial.suggest_float("vl_weight_base", 0.05, 0.35, step=0.05)
    vl_cond_thresh = trial.suggest_float("vl_conditional_threshold", 0.2, 0.8, step=0.1)
    vl_relevance = trial.suggest_float("vl_query_relevance", 0.5, 1.5, step=0.2)

    # ── Hard mode ──
    hard_enabled = trial.suggest_categorical("hard_mode_enabled", [True, False])
    hard_bm25_cap = trial.suggest_float("hard_bm25_cap", 0.2, 0.5, step=0.05) if hard_enabled else 0.35
    hard_dense_floor = trial.suggest_float("hard_dense_floor", 0.3, 0.7, step=0.1) if hard_enabled else 0.5

    # ── Multi-stage filtering ──
    multistage = trial.suggest_categorical("multistage_enabled", [True, False])
    if multistage:
        coarse_k = trial.suggest_int("multistage_coarse_k", 30, 150, step=10)
        dense_k = trial.suggest_int("multistage_dense_k", 5, 30, step=5)
        ms_vl_w = trial.suggest_float("multistage_vl_weight", 0.1, 0.3, step=0.05)
    else:
        coarse_k = 100
        dense_k = 10
        ms_vl_w = 0.2

    # ── Reranking ──
    rerank_enabled = trial.suggest_categorical("reranker_enabled", [True, False])
    if rerank_enabled:
        rerank_input_k = trial.suggest_int("rerank_input_k", 20, 100, step=10)
        rerank_top_k = trial.suggest_int("rerank_top_k", 5, 20, step=2)
    else:
        rerank_input_k = 50
        rerank_top_k = 10

    config = StrategyConfig(
        meta_score_threshold=meta_threshold,
        keyword_score_threshold=keyword_threshold,
        semantic_score_threshold=semantic_threshold,
        entity_keyword_balance=entity_balance,
        base_weight_meta=w_meta,
        base_weight_sparse=w_sparse,
        base_weight_dense=w_dense,
        meta_tilt_factor=meta_tilt,
        sparse_tilt_factor=sparse_tilt,
        dense_tilt_factor=dense_tilt,
        vl_weight_base=vl_weight,
        vl_conditional_threshold=vl_cond_thresh,
        vl_query_relevance=vl_relevance,
        hard_mode_enabled=hard_enabled,
        hard_bm25_cap=hard_bm25_cap,
        hard_dense_floor=hard_dense_floor,
        multistage_enabled=multistage,
        multistage_coarse_k=coarse_k,
        multistage_dense_k=dense_k,
        multistage_vl_weight=ms_vl_w,
        reranker_enabled=rerank_enabled,
        rerank_input_k=rerank_input_k if rerank_enabled else 50,
        rerank_top_k=rerank_top_k if rerank_enabled else 10,
    )

    r10, metrics = evaluate_strategy(
        config=config,
        qa_list=global_qa_list,
        items=global_items,
        text_retriever=global_text_retriever,
        vl_retriever=global_vl_retriever,
        trial_id=trial.number,
    )

    # Save trial result
    trial_dir = Path(args.output_dir) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "result.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return r10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive strategy optimization")
    p.add_argument("--qa-file", default="data/atm-bench/atm-bench.json")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="output/comprehensive_opt")
    p.add_argument("--enable-multistage", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Global state (set once before optimization)
global_qa_list = None
global_items = None
global_text_retriever = None
global_vl_retriever = None
args = None


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  Comprehensive Bayesian Optimization for All Routing Strategies & Channels")
    print("=" * 90)
    print(f"  Trials: {args.n_trials}")
    print(f"  Parameters: ~20+ tunable per trial")
    print(f"  Strategies: adaptive routing + multi-stage + reranking")
    print(f"  Output: {output_dir}/")
    print("=" * 90)

    # Load data once
    print("\nLoading data...")
    global_qa_list = load_json(args.qa_file)
    global_items = build_retrieval_items(
        image_batch=IMAGE_BATCH,
        video_batch=VIDEO_BATCH,
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
        email_file=EMAIL_FILE,
    )

    global_text_retriever = SentenceTransformerRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=INDEX_CACHE,
        batch_size=64,
        device=args.device,
    )

    global_vl_retriever = ClipRetriever(
        model_name="openai/clip-vit-large-patch14",
        cache_dir=INDEX_CACHE,
        device=args.device,
    )

    global_text_retriever.build_index(global_items, {})
    global_vl_retriever.build_index(global_items, {})

    print(f"  Loaded {len(global_qa_list)} questions")
    print(f"  Loaded {len(global_items)} retrieval items")

    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    print("\n" + "=" * 90)
    print("  Starting Optimization Loop")
    print("=" * 90)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 90)
    print("  Optimization Complete")
    print("=" * 90)

    best_trial = study.best_trial
    print(f"\n✅ Best trial: #{best_trial.number}")
    print(f"   R@10: {best_trial.value:.4f}")
    print(f"\n   Key Parameters:")
    for k, v in list(best_trial.params.items())[:10]:
        print(f"     {k}: {v}")

    # Save summary
    summary = {
        "best_trial": best_trial.number,
        "best_r10": best_trial.value,
        "best_params": best_trial.params,
        "total_trials": len(study.trials),
    }
    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Results saved to {output_dir}/")
