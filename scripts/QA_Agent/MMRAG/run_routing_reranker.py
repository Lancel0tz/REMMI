#!/usr/bin/env python3
"""End-to-end routing retriever + reranker experiment.

Runs the full pipeline on ATM-Bench:
  1. Build retrieval index (HybridRetriever with BM25 + dense)
  2. For each question, run adaptive fusion via RoutingRetriever
  3. Apply TextReranker (Qwen/Qwen3-Reranker-2B) to reorder the top candidates
  4. Evaluate R@K and compare against cached baselines

The reranker only reorders the top `--rerank-top-k` candidates from the
adaptive fusion pool. Items beyond that rank are appended in their original
fusion order, so R@50/R@100 still make sense.

Usage:
    python scripts/QA_Agent/MMRAG/run_routing_reranker.py \
        --qa-file data/atm-bench/atm-bench.json \
        --device mps \
        --rerank-top-k 50

    # CPU-only (slow but no GPU needed):
    python scripts/QA_Agent/MMRAG/run_routing_reranker.py \
        --qa-file data/atm-bench/atm-bench.json \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chronicle import (
    HybridRetriever,
    HybridScoringConfig,
    RoutingRetriever,
    RoutingConfig,
    ConfidenceConfig,
    AdaptiveWeights,
    analyze_query,
    adaptive_weights,
)
from chronicle.routing_retriever import adaptive_weights_hard
from memqa.retrieve.utils import (
    RetrievalItem,
    EmailTextConfig,
    MediaTextConfig,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
    write_json,
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever, VisionRetriever
from memqa.retrieve.rerankers import TextReranker

# ── Paths ────────────────────────────────────────────────────────────────
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"

# Baselines (cached from hybrid sweep)
BASELINE_DIR = ROOT / "output/QA_Agent/MMRAG/hybrid_sweep"
BM25_DETAILS = BASELINE_DIR / "sweep_full_rrf_m0.0_s1.0_d0.0_soft/retrieval_recall_details.json"
DENSE_DETAILS = BASELINE_DIR / "sweep_full_rrf_m0.0_s0.0_d1.0_soft/retrieval_recall_details.json"
HYBRID_DETAILS = BASELINE_DIR / "sweep_full_rrf_m0.1_s0.2_d0.7_soft/retrieval_recall_details.json"

RECALL_KS = [1, 5, 10, 25, 50, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routing retriever + reranker experiment")
    p.add_argument("--qa-file", required=True, help="Path to QA JSON")
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"],
                   help="Device for the reranker model")
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base",
                   help="Reranker model name (BAAI/bge-reranker-base, BAAI/bge-reranker-v2-m3, etc.)")
    p.add_argument("--rerank-top-k", type=int, default=50,
                   help="How many candidates to send to the reranker")
    p.add_argument("--reranker-batch-size", type=int, default=8,
                   help="Batch size for the reranker")
    p.add_argument("--reranker-max-length", type=int, default=4096,
                   help="Max token length for the reranker")
    p.add_argument("--primary-top-k", type=int, default=100,
                   help="Adaptive fusion retrieval pool size")
    p.add_argument("--retrieval-max-k", type=int, default=200,
                   help="Max K for recall evaluation")
    p.add_argument("--text-embedding-model",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Dense embedding model")
    p.add_argument("--retriever-batch-size", type=int, default=64)
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: auto-generated)")
    p.add_argument("--no-reranker", action="store_true",
                   help="Skip reranker (adaptive fusion only, for comparison)")
    p.add_argument("--conditional", action="store_true",
                   help="Enable conditional routing: fall back to hybrid for "
                        "email-dominated queries; skip reranker for meta+dense")
    p.add_argument("--email-threshold", type=float, default=0.95,
                   help="Fraction of top-10 results that are email items to "
                        "trigger hybrid fallback (default: 0.95)")
    p.add_argument("--skip-rerank-strategies", nargs="*",
                   default=["meta+dense", "meta+dense_rel"],
                   help="Strategy names for which reranking is skipped")
    p.add_argument("--vl-embedding-model", default=None,
                   help="Vision-language model for 4th channel "
                        "(e.g. openai/clip-vit-base-patch32, google/siglip-base-patch16-224). "
                        "If unset, VL channel is disabled.")
    p.add_argument("--vl-weight", type=float, default=0.15,
                   help="Base VL channel weight when --vl-embedding-model is set (default: 0.15)")
    p.add_argument("--vl-batch-size", type=int, default=16,
                   help="Batch size for VL image encoding")
    p.add_argument("--conditional-vl", action="store_true",
                   help="Only inject VL weight for queries that are likely to "
                        "benefit from visual retrieval (semantic queries, list_recall, "
                        "or queries mentioning photos/images/videos). "
                        "Avoids hurting email/number queries where VL gives zero scores.")
    p.add_argument("--hard-mode", action="store_true",
                   help="Hard-set optimised routing: dense-heavy weights, "
                        "BM25 cap=0.35, dense floor=0.50, NO reranking. "
                        "Designed for media-heavy multi-evidence queries.")
    p.add_argument("--fixed-hybrid", action="store_true",
                   help="Use fixed hybrid weights (0.1/0.2/0.7) instead of "
                        "adaptive routing. For ablation: 'Hybrid + Reranker'.")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Force rebuild the retrieval index")
    return p.parse_args()


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    gt_set = set(gt_ids)
    recalls = {}
    for k in RECALL_KS:
        top = retrieved_ids[:k]
        hit = len([i for i in top if i in gt_set])
        recalls[f"R@{k}"] = hit / len(gt_ids) if gt_ids else 0.0
    return recalls


def evidence_modality(eids: List[str]) -> str:
    has_email = any(e.startswith("email") for e in eids)
    has_media = any(not e.startswith("email") for e in eids)
    if has_email and has_media:
        return "mixed"
    return "email" if has_email else "media"


def _detect_baseline_paths(qa_file: str) -> Dict[str, Path]:
    """Auto-detect baseline paths based on QA file (full vs hard set)."""
    qa_stem = Path(qa_file).stem.lower()
    if "hard" in qa_stem:
        prefix = "sweep_hard_rrf"
    else:
        prefix = "sweep_full_rrf"
    return {
        "bm25":   BASELINE_DIR / f"{prefix}_m0.0_s1.0_d0.0_soft/retrieval_recall_details.json",
        "dense":  BASELINE_DIR / f"{prefix}_m0.0_s0.0_d1.0_soft/retrieval_recall_details.json",
        "hybrid": BASELINE_DIR / f"{prefix}_m0.1_s0.2_d0.7_soft/retrieval_recall_details.json",
    }


def load_baselines(qa_file: str = "") -> Dict[str, Dict[str, Any]]:
    """Load cached baseline retrieval results for comparison."""
    paths = _detect_baseline_paths(qa_file) if qa_file else {
        "bm25": BM25_DETAILS, "dense": DENSE_DETAILS, "hybrid": HYBRID_DETAILS,
    }
    baselines = {}
    for name, path in paths.items():
        if path.exists():
            data = load_json(path)
            baselines[name] = {str(d["id"]): d for d in data}
            print(f"  [baseline] {name}: {len(baselines[name])} questions loaded")
        else:
            print(f"  [baseline] {name}: NOT FOUND at {path}")
    return baselines


def build_data(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[RetrievalItem]]:
    """Load QA list and build retrieval items."""
    qa_data = load_json(Path(args.qa_file))
    if isinstance(qa_data, list):
        qas = qa_data
    elif isinstance(qa_data, dict) and "qas" in qa_data:
        qas = qa_data["qas"]
    else:
        raise ValueError("Unsupported QA schema")

    media_config = MediaTextConfig()  # full text config (all fields True)
    email_config = EmailTextConfig()

    email_entries = load_json(EMAIL_FILE)
    image_entries = load_json(IMAGE_BATCH)
    video_entries = load_json(VIDEO_BATCH)

    items = build_retrieval_items(
        email_entries, image_entries, video_entries,
        media_config, email_config,
        IMAGE_ROOT, VIDEO_ROOT,
    )
    print(f"  [data] {len(qas)} questions, {len(items)} retrieval items")
    return qas, items


def build_retriever_stack(
    items: List[RetrievalItem],
    args: argparse.Namespace,
) -> Tuple[RoutingRetriever, Optional[TextReranker]]:
    """Build HybridRetriever → RoutingRetriever (+ optional TextReranker)."""

    # ── Dense sub-retriever ──
    print(f"  [model] Loading dense embedder: {args.text_embedding_model}")
    t0 = time.perf_counter()
    dense = SentenceTransformerRetriever(
        model_name=args.text_embedding_model,
        cache_dir=INDEX_CACHE,
        batch_size=args.retriever_batch_size,
    )
    # Build dense index (uses cache if available)
    cache_config = {
        "retriever": "hybrid",
        "text_embedding_model": args.text_embedding_model,
        "media_source": "batch_results",
    }
    dense.build_index(items, cache_config, force_rebuild=args.force_rebuild)
    print(f"  [model] Dense index ready ({time.perf_counter() - t0:.1f}s)")

    # ── VL sub-retriever (optional 4th channel) ──
    vl = None
    vl_weight = 0.0
    if args.vl_embedding_model:
        print(f"  [model] Loading VL embedder: {args.vl_embedding_model}")
        t0 = time.perf_counter()
        vl = VisionRetriever(
            model_name=args.vl_embedding_model,
            cache_dir=INDEX_CACHE,
            batch_size=args.vl_batch_size,
        )
        vl_cache_config = {
            "retriever": "vl",
            "vl_embedding_model": args.vl_embedding_model,
            "media_source": "batch_results",
        }
        vl.build_index(items, vl_cache_config, force_rebuild=args.force_rebuild)
        print(f"  [model] VL index ready ({time.perf_counter() - t0:.1f}s)")
        vl_weight = args.vl_weight

    # ── Hybrid retriever (BM25 + dense + metadata [+ VL]) ──
    # Use the best fixed hybrid config as the baseline scoring
    scoring = HybridScoringConfig(
        fusion="rrf",
        rrf_k=60,
        weight_metadata=0.1,
        weight_sparse=0.2,
        weight_dense=0.7,
        weight_vl=vl_weight,
        filter_mode="soft",
    )
    hybrid = HybridRetriever(
        cache_dir=INDEX_CACHE,
        dense_retriever=dense,
        vl_retriever=vl,
        scoring=scoring,
    )
    hybrid.build_index(items, cache_config, force_rebuild=args.force_rebuild)
    channels = "BM25 + dense + metadata"
    if vl:
        channels += f" + VL({args.vl_embedding_model})"
    print(f"  [model] Hybrid index ready ({channels})")

    # ── Reranker ──
    reranker = None
    if not args.no_reranker:
        print(f"  [model] Loading reranker: {args.reranker_model} (device={args.device})")
        t0 = time.perf_counter()
        reranker = TextReranker(
            model_name=args.reranker_model,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            device=args.device,
        )
        print(f"  [model] Reranker ready ({time.perf_counter() - t0:.1f}s)")

    # ── Routing retriever ──
    routing_config = RoutingConfig(
        primary_top_k=args.primary_top_k,
        expand_top_k=min(args.retrieval_max_k, 200),
        confidence=ConfidenceConfig(
            min_top1_score=0.001,
            min_candidates=3,
            max_expand_rounds=1,
        ),
        rerank_top_k=args.rerank_top_k,
        final_top_k=args.retrieval_max_k,
        verbose=False,
    )
    router = RoutingRetriever(
        hybrid=hybrid,
        reranker=None,  # We'll apply reranker manually for more control
        config=routing_config,
    )
    return router, reranker


def run_experiment(
    qas: List[Dict[str, Any]],
    router: RoutingRetriever,
    reranker: Optional[TextReranker],
    baselines: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Run retrieval + reranking on all questions and evaluate."""

    per_question: List[Dict[str, Any]] = []
    strategy_counter: Counter = Counter()
    rerank_times: List[float] = []
    fusion_times: List[float] = []

    # Conditional routing counters
    email_fallback_count = 0
    skip_rerank_count = 0

    for qa in tqdm(qas, desc="Routing+Rerank"):
        qa_id = str(qa.get("id") or qa.get("qa_id"))
        question = qa.get("question", "")
        gt_ids = extract_evidence_ids(qa)
        qtype = qa.get("qtype", "?")
        mod = evidence_modality(gt_ids)

        if not qa_id or not question:
            continue

        # ── Adaptive fusion (no reranker) ──
        t0 = time.perf_counter()
        signals = analyze_query(question)
        if args.fixed_hybrid:
            # Ablation: fixed hybrid weights, no adaptive routing
            aw = AdaptiveWeights(
                weight_metadata=0.1,
                weight_sparse=0.2,
                weight_dense=0.7,
                weight_vl=0.0,
                filter_mode="soft",
                strategy_name="fixed_hybrid",
            )
        elif args.hard_mode:
            aw = adaptive_weights_hard(signals)
        else:
            aw = adaptive_weights(signals)

        # ── Inject VL weight if VL channel is active ──
        if args.vl_embedding_model and args.vl_weight > 0:
            # Conditional VL: only enable for queries that benefit from
            # visual retrieval. Analysis showed:
            # - VL helps list_recall (+5.7% R@10) and explicit visual queries
            # - VL hurts email (0 score), number, and non-visual open_end
            # - "I remember" semantic signal alone is NOT enough (e.g.
            #   "Sofitel hotels" has sem=0.40 but evidence is all email)
            # Strategy: require explicit visual keywords or list_recall hint.
            use_vl = True
            if args.conditional_vl:
                use_vl = (
                    signals.has_visual_hint       # explicit: photo/image/video/visual objects
                    or signals.qtype_hint == "list_recall"  # list_recall strongly benefits from VL
                )
            if use_vl:
                vl_w = args.vl_weight
                # Scale down existing 3-channel weights to make room for VL
                scale = 1.0 - vl_w
                aw = AdaptiveWeights(
                    weight_metadata=aw.weight_metadata * scale,
                    weight_sparse=aw.weight_sparse * scale,
                    weight_dense=aw.weight_dense * scale,
                    weight_vl=vl_w,
                    filter_mode=aw.filter_mode,
                    strategy_name=aw.strategy_name,
                )

        strategy_counter[aw.strategy_name] += 1

        # Get fusion results (without reranker)
        fusion_results = router.retrieve_adaptive(question, args.retrieval_max_k, override_weights=aw)
        fusion_ms = (time.perf_counter() - t0) * 1000
        fusion_times.append(fusion_ms)

        fusion_ids = [r.item.item_id for r in fusion_results]
        fusion_recall = compute_recall(gt_ids, fusion_ids)

        # ── Conditional: email-dominated → hybrid fallback ──
        used_fallback = False
        if args.conditional and fusion_results:
            top_10_ids = fusion_ids[:10]
            email_frac = sum(1 for iid in top_10_ids if iid.startswith("email")) / max(len(top_10_ids), 1)
            if email_frac >= args.email_threshold:
                # Fall back to cached hybrid baseline results (no extra retrieval needed)
                if "hybrid" in baselines and qa_id in baselines["hybrid"]:
                    hybrid_cached_ids = baselines["hybrid"][qa_id]["retrieval_ids"]
                    fusion_ids = hybrid_cached_ids
                    fusion_recall = compute_recall(gt_ids, fusion_ids)
                    # Clear fusion_results so reranker uses fusion_ids directly
                    fusion_results = []
                    used_fallback = True
                    email_fallback_count += 1

        # ── Reranking (on top-k of fusion results) ──
        reranked_ids = fusion_ids  # default: no change
        reranked_recall = fusion_recall
        rerank_ms = 0.0

        # Conditional: skip reranking for certain strategies or hybrid fallback
        # Hard-mode: always skip reranking (reranker hurts hard set by -8.1%)
        skip_rerank = args.hard_mode or (
            args.conditional and (
                used_fallback
                or (args.skip_rerank_strategies
                    and aw.strategy_name in args.skip_rerank_strategies)
            )
        )

        if reranker and fusion_results and not skip_rerank:
            t0 = time.perf_counter()
            # Take top-k for reranking
            rerank_pool = fusion_results[:args.rerank_top_k]
            rerank_items = [r.item for r in rerank_pool]

            try:
                reranked = reranker.rerank(question, rerank_items)
                reranked_sorted = sorted(reranked, key=lambda r: r.score, reverse=True)

                # Build final list: reranked top-k + remaining tail in original order
                reranked_top_ids = [r.item.item_id for r in reranked_sorted]
                tail_ids = [rid for rid in fusion_ids[args.rerank_top_k:]
                           if rid not in set(reranked_top_ids)]
                reranked_ids = reranked_top_ids + tail_ids
                reranked_recall = compute_recall(gt_ids, reranked_ids)
            except Exception as exc:
                print(f"\n  [WARN] Reranker failed for {qa_id}: {exc}")
                reranked_ids = fusion_ids
                reranked_recall = fusion_recall

            rerank_ms = (time.perf_counter() - t0) * 1000
            rerank_times.append(rerank_ms)
        elif skip_rerank:
            skip_rerank_count += 1

        # ── Collect baseline recalls ──
        entry: Dict[str, Any] = {
            "id": qa_id,
            "question": question,
            "qtype": qtype,
            "modality": mod,
            "adaptive_strategy": aw.strategy_name,
            "adaptive_weights": list(aw.as_tuple_4ch() if aw.weight_vl > 0 else aw.as_tuple()),
            "signals": {
                "kw": signals.keyword_score,
                "sem": signals.semantic_score,
                "meta": signals.meta_score,
                "entity": signals.has_entity,
                "rel_date": signals.has_relative_date,
                "non_latin": signals.has_non_latin,
            },
            "fusion_recall": fusion_recall,
            "reranked_recall": reranked_recall,
            "fusion_ids": fusion_ids[:50],  # save top-50 for debugging
            "reranked_ids": reranked_ids[:50],
            "fusion_ms": round(fusion_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "used_hybrid_fallback": used_fallback,
            "skipped_rerank": skip_rerank if args.conditional else False,
        }

        # Add baseline recalls if available
        for bname, bdata in baselines.items():
            if qa_id in bdata:
                entry[f"{bname}_recall"] = compute_recall(
                    gt_ids, bdata[qa_id]["retrieval_ids"]
                )

        per_question.append(entry)

    # Print timing stats
    if fusion_times:
        print(f"\n  [timing] Fusion: avg={sum(fusion_times)/len(fusion_times):.1f}ms, "
              f"total={sum(fusion_times)/1000:.1f}s")
    if rerank_times:
        print(f"  [timing] Rerank: avg={sum(rerank_times)/len(rerank_times):.1f}ms, "
              f"total={sum(rerank_times)/1000:.1f}s, "
              f"per-candidate={sum(rerank_times)/len(rerank_times)/args.rerank_top_k:.1f}ms")

    print(f"\n  [strategies] {dict(strategy_counter.most_common())}")

    if args.conditional:
        print(f"\n  [conditional] email→hybrid fallback: {email_fallback_count} questions, "
              f"skip-rerank: {skip_rerank_count} questions")

    return per_question


def print_results(per_question: List[Dict[str, Any]], baselines: Dict[str, Dict[str, Any]]):
    """Print formatted comparison tables."""
    n = len(per_question)
    if not n:
        print("No results!")
        return

    def avg_key(key: str, subset=None) -> Dict[str, float]:
        items = subset or per_question
        items = [q for q in items if key in q]
        if not items:
            return {}
        return {
            rk: sum(q[key].get(rk, 0) for q in items) / len(items)
            for rk in [f"R@{k}" for k in RECALL_KS]
        }

    # ── Overall R@K comparison ──
    print(f"\n{'=' * 110}")
    print("OVERALL R@K COMPARISON")
    print(f"{'=' * 110}")

    headers = ["R@K"]
    for bname in ["bm25", "dense", "hybrid"]:
        if any(f"{bname}_recall" in q for q in per_question):
            headers.append(bname.upper())
    headers.extend(["Adaptive", "Reranked", "Δ(Rr-Hy)", "Δ(Rr-Ad)"])

    print("  ".join(f"{h:>9}" for h in headers))
    print("-" * (11 * len(headers)))

    for rk in [f"R@{k}" for k in RECALL_KS]:
        row = [rk]
        hyb_val = 0
        for bname in ["bm25", "dense", "hybrid"]:
            key = f"{bname}_recall"
            if any(key in q for q in per_question):
                val = avg_key(key).get(rk, 0)
                row.append(f"{val:.4f}")
                if bname == "hybrid":
                    hyb_val = val

        ad_val = avg_key("fusion_recall").get(rk, 0)
        rr_val = avg_key("reranked_recall").get(rk, 0)
        delta_hy = rr_val - hyb_val if hyb_val else 0
        delta_ad = rr_val - ad_val

        row.append(f"{ad_val:.4f}")
        row.append(f"{rr_val:.4f}")
        row.append(f"{delta_hy:+.4f}")
        row.append(f"{delta_ad:+.4f}")

        print("  ".join(f"{v:>9}" for v in row))

    # ── Per-category R@10 breakdown ──
    print(f"\n{'=' * 110}")
    print("R@10 BY CATEGORY")
    print(f"{'=' * 110}")

    cat_header = f"{'Category':<22} {'N':>5}"
    for bname in ["bm25", "dense", "hybrid"]:
        if any(f"{bname}_recall" in q for q in per_question):
            cat_header += f"  {bname.upper():>7}"
    cat_header += f"  {'Adapt':>7}  {'Rerank':>7}  {'Δ(Rr-Hy)':>9}  {'Δ(Rr-Ad)':>9}"
    print(cat_header)
    print("-" * 110)

    categories = [
        ("BY QTYPE", lambda q: q["qtype"], ["number", "open_end", "list_recall"]),
        ("BY MODALITY", lambda q: q["modality"], ["email", "media", "mixed"]),
    ]

    for label, cat_fn, cats in categories:
        for cat in cats:
            subset = [q for q in per_question if cat_fn(q) == cat]
            if not subset:
                continue
            ns = len(subset)

            line = f"{cat:<22} {ns:>5}"
            hyb_val = 0
            for bname in ["bm25", "dense", "hybrid"]:
                key = f"{bname}_recall"
                if any(key in q for q in per_question):
                    vals = [q[key]["R@10"] for q in subset if key in q]
                    val = sum(vals) / len(vals) if vals else 0
                    line += f"  {val:>7.1%}"
                    if bname == "hybrid":
                        hyb_val = val

            ad = sum(q["fusion_recall"]["R@10"] for q in subset) / ns
            rr = sum(q["reranked_recall"]["R@10"] for q in subset) / ns
            d_hy = rr - hyb_val if hyb_val else 0
            d_ad = rr - ad
            line += f"  {ad:>7.1%}  {rr:>7.1%}  {d_hy:>+8.1%}  {d_ad:>+8.1%}"
            print(line)
        print()

    # Total line
    line = f"{'ALL':<22} {n:>5}"
    hyb_val = 0
    for bname in ["bm25", "dense", "hybrid"]:
        key = f"{bname}_recall"
        if any(key in q for q in per_question):
            vals = [q[key]["R@10"] for q in per_question if key in q]
            val = sum(vals) / len(vals) if vals else 0
            line += f"  {val:>7.1%}"
            if bname == "hybrid":
                hyb_val = val
    ad = sum(q["fusion_recall"]["R@10"] for q in per_question) / n
    rr = sum(q["reranked_recall"]["R@10"] for q in per_question) / n
    d_hy = rr - hyb_val if hyb_val else 0
    d_ad = rr - ad
    line += f"  {ad:>7.1%}  {rr:>7.1%}  {d_hy:>+8.1%}  {d_ad:>+8.1%}"
    print(line)

    # ── Reranker impact analysis ──
    if any(q["reranked_recall"] != q["fusion_recall"] for q in per_question):
        print(f"\n{'=' * 110}")
        print("RERANKER IMPACT (per-question R@10 changes)")
        print(f"{'=' * 110}")

        improved = [q for q in per_question
                    if q["reranked_recall"]["R@10"] > q["fusion_recall"]["R@10"] + 1e-9]
        degraded = [q for q in per_question
                    if q["reranked_recall"]["R@10"] < q["fusion_recall"]["R@10"] - 1e-9]
        unchanged = [q for q in per_question
                     if abs(q["reranked_recall"]["R@10"] - q["fusion_recall"]["R@10"]) < 1e-9]

        print(f"  Improved:  {len(improved):>5} ({len(improved)/n*100:.1f}%)")
        print(f"  Degraded:  {len(degraded):>5} ({len(degraded)/n*100:.1f}%)")
        print(f"  Unchanged: {len(unchanged):>5} ({len(unchanged)/n*100:.1f}%)")

        if improved:
            avg_gain = sum(
                q["reranked_recall"]["R@10"] - q["fusion_recall"]["R@10"]
                for q in improved
            ) / len(improved)
            print(f"  Avg gain (improved):  {avg_gain:+.4f}")
        if degraded:
            avg_loss = sum(
                q["reranked_recall"]["R@10"] - q["fusion_recall"]["R@10"]
                for q in degraded
            ) / len(degraded)
            print(f"  Avg loss (degraded):  {avg_loss:+.4f}")

        # Breakdown of improvements by category
        if improved:
            print(f"\n  Improved by qtype:")
            for qtype in ["number", "open_end", "list_recall"]:
                sub = [q for q in improved if q["qtype"] == qtype]
                if sub:
                    print(f"    {qtype:<15} {len(sub)}")
            print(f"  Improved by modality:")
            for mod in ["email", "media", "mixed"]:
                sub = [q for q in improved if q["modality"] == mod]
                if sub:
                    print(f"    {mod:<15} {len(sub)}")

        if degraded:
            print(f"\n  Degraded by qtype:")
            for qtype in ["number", "open_end", "list_recall"]:
                sub = [q for q in degraded if q["qtype"] == qtype]
                if sub:
                    print(f"    {qtype:<15} {len(sub)}")

        # Sample improvements
        if improved:
            print(f"\n  Sample improvements (top 5 by gain):")
            by_gain = sorted(improved,
                           key=lambda q: q["reranked_recall"]["R@10"] - q["fusion_recall"]["R@10"],
                           reverse=True)
            for q in by_gain[:5]:
                f_r = q["fusion_recall"]["R@10"]
                r_r = q["reranked_recall"]["R@10"]
                print(f"    [{q['qtype']:<12} {q['modality']:<6}] "
                      f"Fusion={f_r:.2f} → Rerank={r_r:.2f} ({r_r - f_r:+.2f})  "
                      f"strategy={q['adaptive_strategy']}")
                print(f"      Q: {q['question'][:100]}")

        # Sample degradations
        if degraded:
            print(f"\n  Sample degradations (top 5 by loss):")
            by_loss = sorted(degraded,
                           key=lambda q: q["fusion_recall"]["R@10"] - q["reranked_recall"]["R@10"],
                           reverse=True)
            for q in by_loss[:5]:
                f_r = q["fusion_recall"]["R@10"]
                r_r = q["reranked_recall"]["R@10"]
                print(f"    [{q['qtype']:<12} {q['modality']:<6}] "
                      f"Fusion={f_r:.2f} → Rerank={r_r:.2f} ({r_r - f_r:+.2f})  "
                      f"strategy={q['adaptive_strategy']}")
                print(f"      Q: {q['question'][:100]}")


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("ROUTING RETRIEVER + RERANKER EXPERIMENT")
    print("=" * 70)
    print(f"  QA file:        {args.qa_file}")
    print(f"  Reranker:       {args.reranker_model} (device={args.device})")
    print(f"  Rerank top-k:   {args.rerank_top_k}")
    print(f"  Primary top-k:  {args.primary_top_k}")
    print(f"  Dense model:    {args.text_embedding_model}")
    print(f"  VL model:       {args.vl_embedding_model or 'None (disabled)'}")
    if args.vl_embedding_model:
        print(f"    VL weight:    {args.vl_weight}")
        print(f"    Cond. VL:     {args.conditional_vl}")
    print(f"  No reranker:    {args.no_reranker}")
    print(f"  Conditional:    {args.conditional}")
    if args.conditional:
        print(f"    email thresh: {args.email_threshold}")
        print(f"    skip-rerank:  {args.skip_rerank_strategies}")
    print(f"  Hard mode:      {args.hard_mode}")
    if args.hard_mode:
        print(f"    ↳ dense-heavy weights, BM25 cap, NO reranking")
    print(f"  Fixed hybrid:   {args.fixed_hybrid}")
    if args.fixed_hybrid:
        print(f"    ↳ weights=(0.1, 0.2, 0.7), no adaptive routing")
    print()

    # ── Load data ──
    print("[1/4] Loading data...")
    qas, items = build_data(args)

    # ── Load baselines ──
    print("[2/4] Loading baselines...")
    baselines = load_baselines(args.qa_file)

    # ── Build retriever stack ──
    print("[3/4] Building retriever stack...")
    router, reranker = build_retriever_stack(items, args)

    # ── Run experiment ──
    print(f"[4/4] Running experiment on {len(qas)} questions...")
    t_start = time.perf_counter()
    per_question = run_experiment(qas, router, reranker, baselines, args)
    total_s = time.perf_counter() - t_start
    print(f"\n  Total experiment time: {total_s:.1f}s ({total_s/60:.1f}min)")

    # ── Print results ──
    print_results(per_question, baselines)

    # ── Save detailed results ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        if args.fixed_hybrid:
            tag = "hybrid_reranker" if not args.no_reranker else "hybrid_only"
        else:
            tag = "reranker" if not args.no_reranker else "adaptive_only"
        if args.conditional:
            tag += "_conditional"
        if args.hard_mode:
            tag += "_hardmode"
        if args.vl_embedding_model:
            vl_short = args.vl_embedding_model.split("/")[-1].replace("-", "")
            tag += f"_vl_{vl_short}"
            if args.conditional_vl:
                tag += "_condvl"
        out_dir = ROOT / f"output/QA_Agent/MMRAG/routing_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-question details
    details_path = out_dir / "routing_reranker_details.json"
    write_json(details_path, per_question)
    print(f"\nDetailed results → {details_path}")

    # Save summary
    n = len(per_question)
    summary = {
        "config": {
            "reranker_model": args.reranker_model if not args.no_reranker else None,
            "rerank_top_k": args.rerank_top_k,
            "primary_top_k": args.primary_top_k,
            "device": args.device,
            "text_embedding_model": args.text_embedding_model,
            "num_questions": n,
            "conditional": args.conditional,
            "email_threshold": args.email_threshold if args.conditional else None,
            "skip_rerank_strategies": args.skip_rerank_strategies if args.conditional else None,
            "hard_mode": args.hard_mode,
            "vl_embedding_model": args.vl_embedding_model,
            "vl_weight": args.vl_weight if args.vl_embedding_model else None,
            "conditional_vl": args.conditional_vl if args.vl_embedding_model else None,
        },
        "fusion_recall": {
            rk: sum(q["fusion_recall"].get(rk, 0) for q in per_question) / n
            for rk in [f"R@{k}" for k in RECALL_KS]
        },
        "reranked_recall": {
            rk: sum(q["reranked_recall"].get(rk, 0) for q in per_question) / n
            for rk in [f"R@{k}" for k in RECALL_KS]
        },
        "total_time_s": round(total_s, 1),
    }
    for bname in baselines:
        key = f"{bname}_recall"
        vals = [q[key] for q in per_question if key in q]
        if vals:
            summary[key] = {
                rk: sum(v.get(rk, 0) for v in vals) / len(vals)
                for rk in [f"R@{k}" for k in RECALL_KS]
            }

    write_json(out_dir / "routing_reranker_summary.json", summary)

    # Also save retrieval details in the standard format for downstream compatibility
    recall_details = []
    for q in per_question:
        recall_details.append({
            "id": q["id"],
            "question": q["question"],
            "gt_evidence_ids": extract_evidence_ids(
                next(qa for qa in qas if str(qa.get("id", qa.get("qa_id"))) == q["id"])
            ),
            "retrieval_ids": q["reranked_ids"],
            "retrieval_recall": q["reranked_recall"],
        })
    write_json(out_dir / "retrieval_recall_details.json", recall_details)

    print(f"Summary → {out_dir / 'routing_reranker_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
