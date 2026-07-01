#!/usr/bin/env python3
"""Generate answers using the best routing configuration.

Usage:
    # 使用优化配置生成答案 (不使用 reranker)
    python scripts/eval_with_best_config.py \
        --qa-file data/atm-bench/atm-bench.json \
        --config-file config/best_routing_config.json \
        --device cuda \
        --output-dir output/answers_with_best_config

    # 使用优化配置 + reranker 生成答案
    python scripts/eval_with_best_config.py \
        --qa-file data/atm-bench/atm-bench.json \
        --config-file config/best_routing_config.json \
        --use-reranker \
        --reranker-model BAAI/bge-reranker-base \
        --device cuda \
        --output-dir output/answers_with_reranker
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltma import HybridRetriever, HybridScoringConfig
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


def load_config(config_file: Path) -> Dict[str, Any]:
    """Load routing configuration."""
    with open(config_file) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description="Generate answers using best routing configuration"
    )
    p.add_argument("--qa-file", required=True, help="QA file path")
    p.add_argument(
        "--config-file",
        default="config/best_routing_config.json",
        help="Best routing config file",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use reranker to rerank top-K results",
    )
    p.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
        help="Reranker model name",
    )
    p.add_argument("--rerank-top-k", type=int, default=50, help="Top-K for reranking")
    p.add_argument("--output-dir", required=True, help="Output directory")
    args = p.parse_args()

    config_file = Path(args.config_file)
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 ANSWER GENERATION WITH BEST ROUTING CONFIG")
    print("=" * 80)

    # Load config
    print(f"\n📂 Loading config from: {config_file}")
    config = load_config(config_file)

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

    # Use best config from optimization
    best_config = config["hybrid_scoring_config"]
    scoring_config = HybridScoringConfig(
        filter_mode=best_config.get("filter_mode", "soft"),
        weight_metadata=best_config["weight_metadata"],
        weight_sparse=best_config["weight_sparse"],
        weight_dense=best_config["weight_dense"],
        weight_vl=best_config.get("weight_vl", 0.0),
        rrf_k=best_config.get("rrf_k", 60),
    )

    retriever = HybridRetriever(
        cache_dir=INDEX_CACHE,
        dense_retriever=text_retriever,
        scoring=scoring_config,
    )

    print("   ✓ Building index...")
    retriever.build_index(items)

    # Build reranker if needed
    reranker = None
    if args.use_reranker:
        print(f"   ✓ Loading reranker ({args.reranker_model})...")
        reranker = TextReranker(
            model_name=args.reranker_model,
            device=args.device,
            batch_size=8,
        )

    # Generate answers
    print("\n📊 Generating answers...")
    answers = []

    for qa in tqdm(qa_list, desc="Processing questions"):
        qa_id = qa.get("id") or qa.get("question_id")
        question = qa["question"]
        gt_ids = extract_evidence_ids(qa)

        # Retrieve
        ret_results = retriever.retrieve(question, top_k=200)
        retrieved_ids = [r.item.item_id for r in ret_results]

        # Rerank if needed
        if reranker is not None:
            top_k_for_rerank = min(args.rerank_top_k, len(ret_results))
            top_candidates = ret_results[:top_k_for_rerank]

            if top_candidates:
                items_for_rerank = [r.item for r in top_candidates]
                for item in items_for_rerank:
                    if hasattr(item, "text") and item.text:
                        item.text = item.text[:1024]

                try:
                    rerank_results = reranker.rerank(question, items_for_rerank)
                    if rerank_results and hasattr(rerank_results[0], "score"):
                        reranked_scores = np.array([r.score for r in rerank_results])
                    else:
                        reranked_scores = np.array(
                            rerank_results
                            if isinstance(rerank_results, (list, np.ndarray))
                            else [rerank_results]
                        )
                except Exception as e:
                    print(f"  ⚠️ Reranker error for {qa_id}: {e}, using retrieval scores")
                    reranked_scores = np.array([0.5] * len(items_for_rerank))

                reranked_indices = np.argsort(-reranked_scores)
                reranked_ids = [
                    top_candidates[i].item.item_id for i in reranked_indices
                ]
                rest_ids = [r.item.item_id for r in ret_results[top_k_for_rerank:]]
                reranked_ids.extend(rest_ids)
                retrieved_ids = reranked_ids

        # Build answer entry - format: {id, answer} for evaluation
        # Extract item texts for answer
        answer_texts = []
        for item_id in retrieved_ids[:10]:  # Top-10 items
            for item in items:
                if item.item_id == item_id:
                    if hasattr(item, 'text') and item.text:
                        answer_texts.append(item.text[:200])  # Truncate for readability
                    break

        # Combine retrieved items into answer
        answer_text = " | ".join(answer_texts) if answer_texts else "No relevant information found"

        answer_entry = {
            "id": qa_id,
            "answer": answer_text,
        }
        answers.append(answer_entry)

    # Save answers
    output_file = output_dir / "answers.jsonl"
    print(f"\n💾 Saving answers to: {output_file}")
    with open(output_file, "w") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")

    # Save config info
    config_info = {
        "config_file": str(config_file),
        "scoring_config": {
            "filter_mode": scoring_config.filter_mode,
            "weight_metadata": scoring_config.weight_metadata,
            "weight_sparse": scoring_config.weight_sparse,
            "weight_dense": scoring_config.weight_dense,
            "weight_vl": scoring_config.weight_vl,
            "rrf_k": scoring_config.rrf_k,
        },
        "optimization_info": config.get("optimization", {}),
        "use_reranker": args.use_reranker,
        "reranker_model": args.reranker_model if args.use_reranker else None,
        "total_questions": len(answers),
    }

    config_file_out = output_dir / "config_info.json"
    with open(config_file_out, "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"\n✅ Complete!")
    print("=" * 80)
    print(f"📄 Answers: {output_file}")
    print(f"📄 Config: {config_file_out}")
    print(f"📊 Total: {len(answers)} answers")

    if args.use_reranker:
        print(f"\n🎯 Generated with:")
        print(f"   - HybridRetriever (optimized weights)")
        print(f"   - {args.reranker_model} reranker")
    else:
        print(f"\n🎯 Generated with:")
        print(f"   - HybridRetriever (optimized weights)")

    print("=" * 80)


if __name__ == "__main__":
    main()
