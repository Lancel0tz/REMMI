#!/usr/bin/env python3
"""Meta-learning strategy discovery: automatically find best strategy combinations.

Key idea: Instead of optimizing a single fixed configuration, learn:
  1. Which queries benefit from which strategy
  2. Per-query strategy routing (meta-classifier)
  3. Automatic multi-stage scheduling
  4. Channel composition optimization

This creates an ensemble of strategies, each specialized for different query types.

Usage:
    python meta_strategy_discovery.py \
        --n-strategies 5 \
        --n-generations 20 \
        --device cuda \
        --output-dir output/meta_discovery
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chronicle import (
    HybridRetriever,
    HybridScoringConfig,
    RoutingRetriever,
    AdaptiveWeights,
    QuerySignals,
    analyze_query,
)
from memqa.retrieve.utils import (
    RetrievalItem,
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


@dataclass
class Strategy:
    """A specialized routing strategy for a subset of queries."""

    strategy_id: int
    name: str  # e.g., "keyword_optimized", "visual_optimized"

    # Channel weights
    weight_metadata: float
    weight_sparse: float
    weight_dense: float
    weight_vl: float

    # Trigger conditions (when to use this strategy)
    trigger_keyword_threshold: float  # Prefer if keyword_score > this
    trigger_semantic_threshold: float  # Prefer if semantic_score > this
    trigger_meta_threshold: float  # Prefer if meta_score > this

    # Multi-stage config
    use_multistage: bool
    multistage_coarse_k: int
    multistage_dense_k: int

    # Performance metrics
    accuracy_on_specialized: Optional[float] = None  # R@10 on queries it's specialized for
    coverage: Optional[float] = None  # % of queries it handles well


def compute_recall(gt_ids: List[str], retrieved_ids: List[str], k: int = 10) -> float:
    """Compute recall@k."""
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    top = retrieved_ids[:k]
    hit = len([i for i in top if i in gt_set])
    return hit / len(gt_ids)


def evaluate_strategy(
    strategy: Strategy,
    qa_list: List[Dict],
    items: List[RetrievalItem],
    text_retriever: SentenceTransformerRetriever,
    vl_retriever: ClipRetriever,
    device: str = "cuda",
) -> Tuple[float, int, List[float]]:
    """Evaluate a strategy on QA set.

    Returns:
        - Average R@10
        - Number of questions handled by this strategy
        - Per-question recall scores
    """

    router = RoutingRetriever(
        cache_dir=INDEX_CACHE,
        batch_size=64,
        device=device,
    )

    config = HybridScoringConfig(
        filter_mode="soft",
        weight_metadata=strategy.weight_metadata,
        weight_sparse=strategy.weight_sparse,
        weight_dense=strategy.weight_dense,
    )

    try:
        router.build_index(
            items=items,
            text_retriever=text_retriever,
            vl_retriever=vl_retriever,
            cache_config={"strategy": strategy.name},
            scoring_config=config,
        )
    except:
        return 0.0, 0, []

    per_question_recalls = []
    handled_count = 0

    for qa in qa_list:
        query = qa["question"]
        gt_ids = extract_evidence_ids(qa)

        # Analyze query
        signals = analyze_query(query)

        # Check if this strategy should handle this query
        is_keyword_query = signals.keyword_score > strategy.trigger_keyword_threshold
        is_semantic_query = signals.semantic_score > strategy.trigger_semantic_threshold
        is_meta_query = signals.meta_score > strategy.trigger_meta_threshold

        # Strategy activation: if any trigger matches, use this strategy
        uses_strategy = is_keyword_query or is_semantic_query or is_meta_query

        if not uses_strategy:
            continue  # This strategy doesn't handle this query

        handled_count += 1

        # Retrieve
        try:
            results = router.retrieve_adaptive(
                query,
                max_k=200,
                override_weights=AdaptiveWeights(
                    weight_metadata=strategy.weight_metadata,
                    weight_sparse=strategy.weight_sparse,
                    weight_dense=strategy.weight_dense,
                    vl_weight=strategy.weight_vl,
                ),
            )

            retrieved_ids = [r.item.item_id for r in results]
            recall_at_10 = compute_recall(gt_ids, retrieved_ids, k=10)
            per_question_recalls.append(recall_at_10)

        except:
            per_question_recalls.append(0.0)

    if handled_count == 0:
        return 0.0, 0, []

    avg_recall = np.mean(per_question_recalls) if per_question_recalls else 0.0
    return avg_recall, handled_count, per_question_recalls


def generate_strategy(strategy_id: int, seed: Optional[int] = None) -> Strategy:
    """Generate a random strategy."""

    if seed is not None:
        np.random.seed(seed)

    # Specialized strategies
    strategy_types = [
        "keyword_optimized",  # Heavy on BM25
        "semantic_optimized",  # Heavy on dense
        "metadata_optimized",  # Heavy on metadata filter
        "visual_optimized",  # Heavy on VL
        "balanced",  # All equal
    ]

    strategy_type = strategy_types[strategy_id % len(strategy_types)]

    if strategy_type == "keyword_optimized":
        w_m, w_s, w_d, w_vl = 0.1, 0.6, 0.25, 0.05
        t_k, t_s, t_m = 0.4, 0.15, 0.2
    elif strategy_type == "semantic_optimized":
        w_m, w_s, w_d, w_vl = 0.1, 0.2, 0.6, 0.1
        t_k, t_s, t_m = 0.15, 0.5, 0.2
    elif strategy_type == "metadata_optimized":
        w_m, w_s, w_d, w_vl = 0.35, 0.3, 0.25, 0.1
        t_k, t_s, t_m = 0.3, 0.3, 0.6
    elif strategy_type == "visual_optimized":
        w_m, w_s, w_d, w_vl = 0.1, 0.2, 0.4, 0.3
        t_k, t_s, t_m = 0.2, 0.5, 0.2
    else:  # balanced
        w_m, w_s, w_d, w_vl = 0.2, 0.3, 0.35, 0.15
        t_k, t_s, t_m = 0.3, 0.3, 0.3

    # Add some randomization
    noise = np.random.normal(0, 0.05, 4)
    w_m = np.clip(w_m + noise[0], 0.05, 0.4)
    w_s = np.clip(w_s + noise[1], 0.1, 0.7)
    w_d = np.clip(w_d + noise[2], 0.15, 0.7)
    w_vl = np.clip(w_vl + noise[3], 0.0, 0.3)

    # Normalize
    total = w_m + w_s + w_d + w_vl
    w_m, w_s, w_d, w_vl = w_m / total, w_s / total, w_d / total, w_vl / total

    return Strategy(
        strategy_id=strategy_id,
        name=f"{strategy_type}_{strategy_id:02d}",
        weight_metadata=w_m,
        weight_sparse=w_s,
        weight_dense=w_d,
        weight_vl=w_vl,
        trigger_keyword_threshold=np.clip(t_k + np.random.normal(0, 0.05), 0.1, 0.6),
        trigger_semantic_threshold=np.clip(t_s + np.random.normal(0, 0.05), 0.1, 0.6),
        trigger_meta_threshold=np.clip(t_m + np.random.normal(0, 0.05), 0.1, 0.8),
        use_multistage=np.random.rand() > 0.5,
        multistage_coarse_k=np.random.randint(50, 150),
        multistage_dense_k=np.random.randint(8, 25),
    )


def mutate_strategy(strategy: Strategy, mutation_rate: float = 0.3) -> Strategy:
    """Create a mutated copy of a strategy."""

    mutant = deepcopy(strategy)

    # Mutate weights
    if np.random.rand() < mutation_rate:
        noise = np.random.normal(0, 0.1, 4)
        mutant.weight_metadata = np.clip(mutant.weight_metadata + noise[0], 0.05, 0.4)
        mutant.weight_sparse = np.clip(mutant.weight_sparse + noise[1], 0.1, 0.7)
        mutant.weight_dense = np.clip(mutant.weight_dense + noise[2], 0.15, 0.7)
        mutant.weight_vl = np.clip(mutant.weight_vl + noise[3], 0.0, 0.3)

        # Normalize
        total = mutant.weight_metadata + mutant.weight_sparse + mutant.weight_dense + mutant.weight_vl
        mutant.weight_metadata /= total
        mutant.weight_sparse /= total
        mutant.weight_dense /= total
        mutant.weight_vl /= total

    # Mutate thresholds
    if np.random.rand() < mutation_rate:
        mutant.trigger_keyword_threshold = np.clip(
            mutant.trigger_keyword_threshold + np.random.normal(0, 0.05), 0.1, 0.6
        )
    if np.random.rand() < mutation_rate:
        mutant.trigger_semantic_threshold = np.clip(
            mutant.trigger_semantic_threshold + np.random.normal(0, 0.05), 0.1, 0.6
        )
    if np.random.rand() < mutation_rate:
        mutant.trigger_meta_threshold = np.clip(
            mutant.trigger_meta_threshold + np.random.normal(0, 0.05), 0.1, 0.8
        )

    # Mutate multi-stage config
    if np.random.rand() < mutation_rate / 2:
        mutant.use_multistage = not mutant.use_multistage
        mutant.multistage_coarse_k = np.random.randint(50, 150)
        mutant.multistage_dense_k = np.random.randint(8, 25)

    return mutant


def main():
    parser = argparse.ArgumentParser(description="Meta-strategy discovery")
    parser.add_argument("--qa-file", default="data/atm-bench/atm-bench.json")
    parser.add_argument("--n-strategies", type=int, default=5, help="Population size")
    parser.add_argument("--n-generations", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="output/meta_discovery")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print("=" * 90)
    print("  Meta-Learning Strategy Discovery")
    print("=" * 90)
    print(f"  Population: {args.n_strategies} strategies")
    print(f"  Generations: {args.n_generations}")
    print(f"  Output: {output_dir}/")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    qa_list = load_json(args.qa_file)
    items = build_retrieval_items(
        image_batch=IMAGE_BATCH,
        video_batch=VIDEO_BATCH,
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
        email_file=EMAIL_FILE,
    )

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

    print(f"  Loaded {len(qa_list)} questions")
    print(f"  Loaded {len(items)} items")

    # Initialize population
    print("\nInitializing strategies...")
    population = [
        generate_strategy(i, seed=args.seed + i)
        for i in range(args.n_strategies)
    ]

    history = []

    # Evolution loop
    for gen in range(args.n_generations):
        print(f"\n[Generation {gen + 1}/{args.n_generations}]")

        # Evaluate each strategy
        fitness_scores = []
        for strategy in tqdm(population, desc="Evaluating"):
            r10, handled, _ = evaluate_strategy(
                strategy, qa_list, items, text_retriever, vl_retriever, args.device
            )
            strategy.accuracy_on_specialized = r10
            strategy.coverage = handled / len(qa_list) if qa_list else 0
            fitness_scores.append(r10 * (1 + strategy.coverage))  # Fitness = accuracy × coverage

        fitness_scores = np.array(fitness_scores)

        # Select top performers
        top_indices = np.argsort(fitness_scores)[-args.n_strategies // 2 :]
        top_strategies = [population[i] for i in top_indices]

        # Print generation results
        best_idx = top_indices[-1]
        best_strategy = population[best_idx]
        print(
            f"  Best: {best_strategy.name} (R@10={best_strategy.accuracy_on_specialized:.4f}, "
            f"coverage={best_strategy.coverage:.1%})"
        )

        # Save generation
        gen_result = {
            "generation": gen,
            "best_strategy": {
                "name": best_strategy.name,
                "r10": best_strategy.accuracy_on_specialized,
                "coverage": best_strategy.coverage,
                "weights": {
                    "metadata": best_strategy.weight_metadata,
                    "sparse": best_strategy.weight_sparse,
                    "dense": best_strategy.weight_dense,
                    "vl": best_strategy.weight_vl,
                },
                "triggers": {
                    "keyword_threshold": best_strategy.trigger_keyword_threshold,
                    "semantic_threshold": best_strategy.trigger_semantic_threshold,
                    "meta_threshold": best_strategy.trigger_meta_threshold,
                },
            },
            "population_fitness": fitness_scores.tolist(),
        }
        history.append(gen_result)

        # Create next generation (mutation + crossover)
        new_population = []
        for _ in range(args.n_strategies):
            parent = np.random.choice(top_strategies)
            child = mutate_strategy(parent)
            child.strategy_id = len(new_population)
            new_population.append(child)

        population = new_population

    # Save final results
    with open(output_dir / "evolution_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save best strategies
    best_strategies = sorted(
        population, key=lambda s: s.accuracy_on_specialized * (1 + s.coverage), reverse=True
    )[:3]

    final_results = {
        "best_strategies": [
            {
                "name": s.name,
                "r10": s.accuracy_on_specialized,
                "coverage": s.coverage,
                "weights": {
                    "metadata": s.weight_metadata,
                    "sparse": s.weight_sparse,
                    "dense": s.weight_dense,
                    "vl": s.weight_vl,
                },
            }
            for s in best_strategies
        ]
    }

    with open(output_dir / "final_strategies.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Discovery complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
