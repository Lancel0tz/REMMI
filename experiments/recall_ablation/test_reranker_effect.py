#!/usr/bin/env python3
"""Test the effect of adding a reranker to the best routing configuration.

Compares:
  1. HybridRetriever only (77.50% R@10)
  2. HybridRetriever + TextReranker (?)

Usage:
    python scripts/test_reranker_effect.py \
        --qa-file data/atm-bench/atm-bench.json \
        --device cuda \
        --rerank-top-k 50
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ROOT should be ATM-Bench directory
script_path = Path(__file__).resolve()
ROOT = script_path.parents[1]  # parents[1] = ATM-Bench (scripts is at ATM-Bench/scripts)
sys.path.insert(0, str(ROOT))

from chronicle import HybridRetriever, HybridScoringConfig
from memqa.retrieve.utils import (
    RetrievalItem,
    EmailTextConfig,
    MediaTextConfig,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
    write_json,
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever
from memqa.retrieve.rerankers import TextReranker

# ── Paths ────────────────────────────────────────────────────────────
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"
# Try both paths to handle different working directories
CONFIG_FILE = Path("config/best_routing_config.json")
if not CONFIG_FILE.exists():
    CONFIG_FILE = ROOT / "config/best_routing_config.json"


def compute_recall_at_k(gt_ids: List[str], retrieved_ids: List[str], k: int = 10) -> float:
    """Compute Recall@k metric."""
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    top_k = retrieved_ids[:k]
    hits = len([i for i in top_k if i in gt_set])
    return hits / len(gt_ids)


def main():
    p = argparse.ArgumentParser(description="Test reranker effect")
    p.add_argument("--qa-file", required=True, help="QA file path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--rerank-top-k", type=int, default=50)
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    p.add_argument("--output-dir", default="output/reranker_test")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🧪 RERANKER EFFECT TEST")
    print("=" * 80)

    # Load config
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Load data
    print("\n📂 Loading data...")
    qa_list = load_json(Path(args.qa_file))
    email_entries = load_json(EMAIL_FILE)
    image_batch_data = load_json(IMAGE_BATCH)
    video_batch_data = load_json(VIDEO_BATCH)

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

    print(f"   ✓ Loaded {len(qa_list)} questions")
    print(f"   ✓ Loaded {len(items)} items")

    # Build retrievers
    print("\n🔧 Building retrievers...")
    text_retriever = SentenceTransformerRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=INDEX_CACHE,
        batch_size=64,
        device=args.device,
    )

    # Best config from optimization
    best_config = config["hybrid_scoring_config"]
    scoring_config = HybridScoringConfig(
        filter_mode=best_config["filter_mode"],
        weight_metadata=best_config["weight_metadata"],
        weight_sparse=best_config["weight_sparse"],
        weight_dense=best_config["weight_dense"],
        weight_vl=best_config["weight_vl"],
        rrf_k=best_config["rrf_k"],
    )

    retriever = HybridRetriever(
        cache_dir=INDEX_CACHE,
        dense_retriever=text_retriever,
        scoring=scoring_config,
    )

    print("   ✓ Building index...")
    retriever.build_index(items)

    # Build reranker
    print(f"   ✓ Loading reranker ({args.reranker_model})...")
    reranker = TextReranker(
        model_name=args.reranker_model,
        device=args.device,
        batch_size=8,
    )

    # Evaluation
    print("\n📊 Running evaluation...")
    print(f"   Method 1: HybridRetriever only")
    print(f"   Method 2: HybridRetriever + Reranker (top-{args.rerank_top_k})")

    results_without_reranker = []
    results_with_reranker = []

    for qa in tqdm(qa_list, desc="Questions"):
        query = qa["question"]
        gt_ids = extract_evidence_ids(qa)

        # Method 1: Retrieval only
        ret_results = retriever.retrieve(query, top_k=200)
        retrieved_ids = [r.item.item_id for r in ret_results]

        r1_ret = compute_recall_at_k(gt_ids, retrieved_ids, k=1)
        r5_ret = compute_recall_at_k(gt_ids, retrieved_ids, k=5)
        r10_ret = compute_recall_at_k(gt_ids, retrieved_ids, k=10)
        r25_ret = compute_recall_at_k(gt_ids, retrieved_ids, k=25)

        results_without_reranker.append({
            "query": query,
            "r1": r1_ret,
            "r5": r5_ret,
            "r10": r10_ret,
            "r25": r25_ret,
        })

        # Method 2: Retrieval + Reranker
        top_k_for_rerank = min(args.rerank_top_k, len(ret_results))
        top_candidates = ret_results[:top_k_for_rerank]

        if top_candidates:
            # Rerank top-K - reranker expects items with .text attribute
            # Truncate text to avoid model length limit issues (BGE max ~512 tokens)
            items_for_rerank = [r.item for r in top_candidates]
            # Limit text length to prevent model overflow
            for item in items_for_rerank:
                if hasattr(item, 'text') and item.text:
                    item.text = item.text[:1024]  # Truncate to ~1024 chars (~256 tokens)

            try:
                rerank_results = reranker.rerank(query, items_for_rerank)
                # Extract scores from RerankResult objects
                if rerank_results and hasattr(rerank_results[0], 'score'):
                    reranked_scores = np.array([r.score for r in rerank_results])
                else:
                    # Fallback if format is different
                    reranked_scores = np.array(rerank_results) if isinstance(rerank_results, (list, np.ndarray)) else np.array([rerank_results])
            except Exception as e:
                # If reranking fails, fall back to original scores
                print(f"  ⚠️ Reranker error: {e}, using retrieval scores")
                reranked_scores = np.array([0.5] * len(items_for_rerank))

            # Reconstruct retrieved IDs in reranked order
            reranked_indices = np.argsort(-reranked_scores)
            reranked_ids = [top_candidates[i].item.item_id for i in reranked_indices]

            # Append the rest in original order
            rest_ids = [r.item.item_id for r in ret_results[top_k_for_rerank:]]
            reranked_ids.extend(rest_ids)
        else:
            reranked_ids = retrieved_ids

        r1_rer = compute_recall_at_k(gt_ids, reranked_ids, k=1)
        r5_rer = compute_recall_at_k(gt_ids, reranked_ids, k=5)
        r10_rer = compute_recall_at_k(gt_ids, reranked_ids, k=10)
        r25_rer = compute_recall_at_k(gt_ids, reranked_ids, k=25)

        results_with_reranker.append({
            "query": query,
            "r1": r1_rer,
            "r5": r5_rer,
            "r10": r10_rer,
            "r25": r25_rer,
        })

    # Summary
    r1_ret = np.mean([r["r1"] for r in results_without_reranker])
    r5_ret = np.mean([r["r5"] for r in results_without_reranker])
    r10_ret = np.mean([r["r10"] for r in results_without_reranker])
    r25_ret = np.mean([r["r25"] for r in results_without_reranker])

    r1_rer = np.mean([r["r1"] for r in results_with_reranker])
    r5_rer = np.mean([r["r5"] for r in results_with_reranker])
    r10_rer = np.mean([r["r10"] for r in results_with_reranker])
    r25_rer = np.mean([r["r25"] for r in results_with_reranker])

    print("\n" + "=" * 80)
    print("📈 RESULTS")
    print("=" * 80)

    print(f"\n┌────────┬──────────────┬──────────────┬───────────────┐")
    print(f"│  K    │  Retrieval   │  + Reranker  │    Gain       │")
    print(f"├────────┼──────────────┼──────────────┼───────────────┤")
    print(f"│  R@1  │   {r1_ret:.4f}    │   {r1_rer:.4f}    │  {(r1_rer-r1_ret):+.4f} ({(r1_rer-r1_ret)*100:+.2f}%) │")
    print(f"│  R@5  │   {r5_ret:.4f}    │   {r5_rer:.4f}    │  {(r5_rer-r5_ret):+.4f} ({(r5_rer-r5_ret)*100:+.2f}%) │")
    print(f"│  R@10 │   {r10_ret:.4f}    │   {r10_rer:.4f}    │  {(r10_rer-r10_ret):+.4f} ({(r10_rer-r10_ret)*100:+.2f}%) │")
    print(f"│ R@25  │   {r25_ret:.4f}    │   {r25_rer:.4f}    │  {(r25_rer-r25_ret):+.4f} ({(r25_rer-r25_ret)*100:+.2f}%) │")
    print(f"└────────┴──────────────┴──────────────┴───────────────┘")

    # Save results
    summary = {
        "reranker_model": args.reranker_model,
        "rerank_top_k": args.rerank_top_k,
        "metrics": {
            "retrieval_only": {
                "r1": float(r1_ret),
                "r5": float(r5_ret),
                "r10": float(r10_ret),
                "r25": float(r25_ret),
            },
            "with_reranker": {
                "r1": float(r1_rer),
                "r5": float(r5_rer),
                "r10": float(r10_rer),
                "r25": float(r25_rer),
            },
            "improvement": {
                "r1": float(r1_rer - r1_ret),
                "r5": float(r5_rer - r5_ret),
                "r10": float(r10_rer - r10_ret),
                "r25": float(r25_rer - r25_ret),
            },
        },
    }

    output_file = output_dir / "reranker_comparison.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
