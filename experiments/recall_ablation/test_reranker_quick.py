#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ltma import HybridRetriever, HybridScoringConfig
from memqa.retrieve.utils import (
    EmailTextConfig, MediaTextConfig,
    build_retrieval_items, extract_evidence_ids, load_json
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever
from memqa.retrieve.rerankers import TextReranker

# Paths
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"
CONFIG_FILE = Path("config/best_routing_config.json").resolve()

def compute_recall_at_k(gt_ids: List[str], retrieved_ids: List[str], k: int = 10) -> float:
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    top_k = retrieved_ids[:k]
    hits = len([i for i in top_k if i in gt_set])
    return hits / len(gt_ids)

print("=" * 80)
print("🧪 RERANKER QUICK TEST (100 questions only)")
print("=" * 80)
print("")

# Load config
with open(CONFIG_FILE) as f:
    config = json.load(f)

# Load data
print("📂 Loading data...")
qa_list = load_json(Path("data/atm-bench/atm-bench.json"))
qa_list = qa_list[:100]  # Quick test: only 100 questions
print(f"   ✓ Loaded {len(qa_list)} questions (quick sample)")

email_entries = load_json(EMAIL_FILE)
image_batch_data = load_json(IMAGE_BATCH)
video_batch_data = load_json(VIDEO_BATCH)

items = build_retrieval_items(
    email_entries=email_entries,
    image_entries=image_batch_data,
    video_entries=video_batch_data,
    media_text_config=MediaTextConfig(),
    email_text_config=EmailTextConfig(),
    image_root=IMAGE_ROOT,
    video_root=VIDEO_ROOT,
)

print("🔧 Building retrievers...")
text_retriever = SentenceTransformerRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=INDEX_CACHE,
    batch_size=64,
    device="cpu",
)

best_config = config["hybrid_scoring_config"]
scoring_config = HybridScoringConfig(**best_config)
retriever = HybridRetriever(
    cache_dir=INDEX_CACHE,
    dense_retriever=text_retriever,
    scoring=scoring_config,
)

retriever.build_index(items)
print("   ✓ Loading reranker (BAAI/bge-reranker-base)...")
reranker = TextReranker(model_name="BAAI/bge-reranker-base", device="cpu", batch_size=8)

# Evaluation
print("\n📊 Running evaluation...")
results_retrieval = []
results_reranker = []

for qa in tqdm(qa_list, desc="Questions"):
    query = qa["question"]
    gt_ids = extract_evidence_ids(qa)
    
    # Retrieval only
    ret_results = retriever.retrieve(query, top_k=200)
    retrieved_ids = [r.item.item_id for r in ret_results]
    
    r10_ret = compute_recall_at_k(gt_ids, retrieved_ids, k=10)
    results_retrieval.append(r10_ret)
    
    # With reranker
    top_candidates = ret_results[:50]
    if top_candidates:
        items_for_rerank = [r.item for r in top_candidates]
        for item in items_for_rerank:
            if hasattr(item, 'text') and item.text:
                item.text = item.text[:1024]
        
        try:
            rerank_results = reranker.rerank(query, items_for_rerank)
            if rerank_results and hasattr(rerank_results[0], 'score'):
                reranked_scores = np.array([r.score for r in rerank_results])
            else:
                reranked_scores = np.array(rerank_results) if isinstance(rerank_results, (list, np.ndarray)) else np.array([rerank_results])
        except Exception as e:
            reranked_scores = np.array([0.5] * len(items_for_rerank))
        
        reranked_indices = np.argsort(-reranked_scores)
        reranked_ids = [top_candidates[i].item.item_id for i in reranked_indices]
        rest_ids = [r.item.item_id for r in ret_results[50:]]
        reranked_ids.extend(rest_ids)
    else:
        reranked_ids = retrieved_ids
    
    r10_rer = compute_recall_at_k(gt_ids, reranked_ids, k=10)
    results_reranker.append(r10_rer)

# Summary
r10_ret = np.mean(results_retrieval)
r10_rer = np.mean(results_reranker)

print("\n" + "=" * 80)
print("📈 RESULTS (Quick test - 100 questions)")
print("=" * 80)
print(f"\nRetrieval Only:    {r10_ret:.4f} ({r10_ret*100:.2f}%)")
print(f"With Reranker:     {r10_rer:.4f} ({r10_rer*100:.2f}%)")
print(f"Improvement:       {r10_rer-r10_ret:+.4f} ({(r10_rer-r10_ret)*100:+.2f}%)")

# Save results
output_dir = Path("output/reranker_test")
output_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "test_mode": "quick_sample",
    "sample_size": len(qa_list),
    "reranker_model": "BAAI/bge-reranker-base",
    "rerank_top_k": 50,
    "metrics": {
        "retrieval_only": {"r10": float(r10_ret)},
        "with_reranker": {"r10": float(r10_rer)},
        "improvement": {"r10": float(r10_rer - r10_ret)},
    },
}

with open(output_dir / "reranker_comparison.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Results saved to: {output_dir}/reranker_comparison.json")
print("=" * 80)
