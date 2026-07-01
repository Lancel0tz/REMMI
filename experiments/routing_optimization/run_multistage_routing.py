#!/usr/bin/env python3
"""Multi-stage cascading routing: BM25 coarse filter → dense re-ranking.

Strategy:
  1. Coarse stage:  BM25 retrieval (sparse, fast) → top K_coarse candidates
  2. Dense stage:   Dense embedding on coarse results → top K_dense
  3. Optional rerank: TextReranker on top-K_final

This reduces computation (dense encoding only on BM25-filtered set) while
often improving precision.

Usage:
    python scripts/QA_Agent/MMRAG/run_multistage_routing.py \
        --qa-file data/atm-bench/atm-bench.json \
        --device cuda \
        --coarse-top-k 50 \
        --dense-top-k 10 \
        --vl-weight 0.20

Config to optimize:
  - coarse_top_k: BM25 candidates to retrieve (50-200)
  - dense_top_k: Final candidates to return (5-50)
  - vl_weight: Vision-language embedding weight (0.05-0.40)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ltma import (
    HybridRetriever,
    HybridScoringConfig,
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


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    """Compute recall@k for all K in RECALL_KS."""
    gt_set = set(gt_ids)
    recalls = {}
    for k in RECALL_KS:
        top = retrieved_ids[:k]
        hit = len([i for i in top if i in gt_set])
        recalls[f"R@{k}"] = hit / len(gt_ids) if gt_ids else 0.0
    return recalls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage cascading routing")
    p.add_argument("--qa-file", required=True, help="QA file")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--coarse-top-k", type=int, default=100,
                   help="BM25 candidates to retrieve")
    p.add_argument("--dense-top-k", type=int, default=10,
                   help="Dense re-ranking top-k")
    p.add_argument("--text-embedding-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vl-embedding-model", default="openai/clip-vit-large-patch14",
                   help="VL embedding model")
    p.add_argument("--vl-weight", type=float, default=0.20)
    p.add_argument("--retriever-batch-size", type=int, default=64)
    p.add_argument("--output-dir", default="output/QA_Agent/MMRAG/multistage")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Multi-Stage Cascading Routing")
    print(f"  Coarse (BM25):  top-{args.coarse_top_k}")
    print(f"  Dense rerank:   top-{args.dense_top_k}")
    print(f"  VL weight:      {args.vl_weight:.3f}")
    print("=" * 70)

    # Load QA and media
    qa_list = load_json(args.qa_file)
    print(f"\nLoaded {len(qa_list)} questions from {Path(args.qa_file).name}")

    # Build retrieval items
    items = build_retrieval_items(
        image_batch=IMAGE_BATCH,
        video_batch=VIDEO_BATCH,
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
        email_file=EMAIL_FILE,
        text_config=MediaTextConfig(
            include_id=True,
            include_timestamp=True,
            include_location=True,
            include_short_caption=True,
            include_caption=False,
            include_ocr_text=False,
        ),
        email_config=EmailTextConfig(
            include_id=True,
            include_summary=True,
            include_detail=False,
        ),
    )
    print(f"Loaded {len(items)} retrieval items")

    # ── Stage 1: Coarse BM25 retrieval ──
    print(f"\n[Stage 1/2] Building hybrid retriever (BM25 + text dense)...")
    t0 = time.perf_counter()
    hybrid = HybridRetriever(
        cache_dir=INDEX_CACHE,
        batch_size=args.retriever_batch_size,
        device=args.device,
    )

    # Dense only from text (no VL for initial coarse stage)
    text_retriever = SentenceTransformerRetriever(
        model_name=args.text_embedding_model,
        cache_dir=INDEX_CACHE,
        batch_size=args.retriever_batch_size,
        device=args.device,
    )

    config = HybridScoringConfig(
        filter_mode="soft",
        weight_metadata=0.1,
        weight_sparse=0.6,
        weight_dense=0.3,
        rrf_k=60,
    )

    hybrid.build_index(
        items=items,
        dense_retriever=text_retriever,
        cache_config={
            "retriever": "text_hybrid",
            "dense_model": args.text_embedding_model,
            "media_source": "batch_results",
        },
        scoring_config=config,
        force_rebuild=args.force_rebuild,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Index built in {elapsed:.1f}s")

    # ── Stage 2: Dense re-ranking with VL ──
    print(f"\n[Stage 2/2] Building VL re-ranker...")
    t0 = time.perf_counter()
    vl_retriever = ClipRetriever(
        model_name=args.vl_embedding_model,
        cache_dir=INDEX_CACHE,
        batch_size=16,
        device=args.device,
    )
    vl_retriever.build_index(
        items=items,
        cache_config={
            "retriever": "vl",
            "vl_model": args.vl_embedding_model,
        },
        force_rebuild=args.force_rebuild,
    )
    elapsed = time.perf_counter() - t0
    print(f"  VL index built in {elapsed:.1f}s")

    # ── Run inference on all questions ──
    print(f"\n[Inference] Running on {len(qa_list)} questions...")
    per_question = []
    total_time = 0.0

    for q_idx, qa in enumerate(tqdm(qa_list, desc="Questions")):
        query = qa["question"]
        gt_ids = extract_evidence_ids(qa)

        t_query_start = time.perf_counter()

        # Stage 1: Coarse BM25
        coarse_results = hybrid.retrieve(query, args.coarse_top_k)
        coarse_ids = [r.item.item_id for r in coarse_results]

        # Stage 2: Re-rank coarse results with VL
        # Recompute embeddings for coarse set only
        coarse_items = [r.item for r in coarse_results]
        vl_embs = vl_retriever.encode_items(coarse_items)
        text_embs = text_retriever.encode_items(coarse_items)

        # Fuse: text (0.5) + VL (0.5 * vl_weight)
        combined = (
            0.5 * text_embs +
            (0.5 * args.vl_weight) * vl_embs
        )

        # Sort by combined scores
        import torch
        import torch.nn.functional as F

        combined = F.normalize(combined, p=2, dim=1)
        query_emb = (
            0.5 * text_retriever.encode_query(query) +
            0.5 * args.vl_weight * vl_retriever.encode_query(query)
        )
        query_emb = F.normalize(query_emb, p=2, dim=1)

        scores = torch.matmul(query_emb, combined.t()).squeeze(0)
        _, indices = torch.topk(scores, min(args.dense_top_k, len(coarse_ids)))

        final_ids = [coarse_ids[i] for i in indices.cpu().numpy()]

        t_query_end = time.perf_counter()
        query_time = t_query_end - t_query_start
        total_time += query_time

        # Compute recall
        recalls = compute_recall(gt_ids, final_ids)

        per_question.append({
            "question_id": qa["question_id"],
            "coarse_stage_ids": coarse_ids,
            "final_ids": final_ids,
            "gt_ids": gt_ids,
            "recalls": recalls,
            "query_time_ms": query_time * 1000,
        })

    avg_time_ms = (total_time / len(qa_list)) * 1000
    print(f"\n  Avg query time: {avg_time_ms:.1f}ms")

    # ── Save results ──
    avg_recall = {
        rk: sum(q["recalls"].get(rk, 0) for q in per_question) / len(per_question)
        for rk in [f"R@{k}" for k in RECALL_KS]
    }

    summary = {
        "config": {
            "coarse_top_k": args.coarse_top_k,
            "dense_top_k": args.dense_top_k,
            "vl_weight": args.vl_weight,
            "text_embedding_model": args.text_embedding_model,
            "vl_embedding_model": args.vl_embedding_model,
            "num_questions": len(qa_list),
        },
        "recall": avg_recall,
        "avg_query_time_ms": avg_time_ms,
        "total_time_s": total_time,
    }

    with open(output_dir / "multistage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "multistage_details.json", "w") as f:
        json.dump(per_question, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    for rk, v in avg_recall.items():
        print(f"  {rk}: {v:.4f}")
    print(f"\n  Output: {output_dir}/")


if __name__ == "__main__":
    main()
