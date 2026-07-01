#!/usr/bin/env python3
"""Offline evaluation of the routing retriever using cached BM25/Dense results.

This script does NOT re-run any retrieval models. It reuses the per-question
retrieval result caches from the hybrid sweep (pure BM25: s1.0_d0.0, pure
Dense: s0.0_d1.0) and simulates the routing retriever's decisions to produce
an "oracle-aware" upper-bound and a "rule-based router" estimate.

Usage:
    python scripts/QA_Agent/MMRAG/eval_routing_retriever.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chronicle.routing_retriever import (
    QuerySignals,
    RouteStrategy,
    analyze_query,
    route,
    adaptive_weights,
    AdaptiveWeights,
)

BM25_DETAILS = ROOT / "output/QA_Agent/MMRAG/hybrid_sweep/sweep_full_rrf_m0.0_s1.0_d0.0_soft/retrieval_recall_details.json"
DENSE_DETAILS = ROOT / "output/QA_Agent/MMRAG/hybrid_sweep/sweep_full_rrf_m0.0_s0.0_d1.0_soft/retrieval_recall_details.json"
# Also load the best hybrid config for comparison
HYBRID_DETAILS = ROOT / "output/QA_Agent/MMRAG/hybrid_sweep/sweep_full_rrf_m0.1_s0.2_d0.7_soft/retrieval_recall_details.json"
QA_FILE = ROOT / "data/atm-bench/atm-bench.json"

RECALL_KS = [1, 5, 10, 25, 50, 100]


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def evidence_modality(eids: List[str]) -> str:
    has_email = any(e.startswith("email") for e in eids)
    has_media = any(not e.startswith("email") for e in eids)
    if has_email and has_media:
        return "mixed"
    return "email" if has_email else "media"


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    gt_set = set(gt_ids)
    recalls = {}
    for k in RECALL_KS:
        top = retrieved_ids[:k]
        hit = len([i for i in top if i in gt_set])
        recalls[f"R@{k}"] = hit / len(gt_ids) if gt_ids else 0.0
    return recalls


def pick_routed_results(
    strategy: RouteStrategy,
    bm25_entry: Dict[str, Any],
    dense_entry: Dict[str, Any],
) -> List[str]:
    """Given a routing strategy, return the retrieval_ids to use."""
    if strategy == RouteStrategy.BM25_ONLY:
        return bm25_entry["retrieval_ids"]
    elif strategy == RouteStrategy.DENSE_ONLY:
        return dense_entry["retrieval_ids"]
    elif strategy in (RouteStrategy.META_THEN_BM25, RouteStrategy.FULL_HYBRID):
        # For META_THEN_BM25 and FULL_HYBRID we simulate RRF merge
        # Use interleaved merge (round-robin) as a proxy for RRF
        return _interleave(bm25_entry["retrieval_ids"], dense_entry["retrieval_ids"])
    elif strategy == RouteStrategy.META_THEN_DENSE:
        # Dense primary, BM25 fallback
        return _interleave(dense_entry["retrieval_ids"], bm25_entry["retrieval_ids"])
    return dense_entry["retrieval_ids"]


def _interleave(primary: List[str], secondary: List[str], max_k: int = 200) -> List[str]:
    """Round-robin interleave, deduped, primary first."""
    seen: Set[str] = set()
    result: List[str] = []
    i = j = 0
    while len(result) < max_k and (i < len(primary) or j < len(secondary)):
        if i < len(primary):
            if primary[i] not in seen:
                seen.add(primary[i])
                result.append(primary[i])
            i += 1
        if j < len(secondary):
            if secondary[j] not in seen:
                seen.add(secondary[j])
                result.append(secondary[j])
            j += 1
    return result


def pick_adaptive_results(
    aw: AdaptiveWeights,
    bm25_entry: Dict[str, Any],
    dense_entry: Dict[str, Any],
    rrf_k: int = 60,
    max_k: int = 200,
) -> List[str]:
    """Simulate adaptive-weight RRF fusion using cached rank lists.

    We don't have access to the actual metadata channel scores in the
    cache, so we use bm25 rank + dense rank fused with the adaptive
    sparse/dense weights (metadata weight is folded into the dominant
    channel for simulation purposes).
    """
    bm25_ids = bm25_entry["retrieval_ids"][:max_k]
    dense_ids = dense_entry["retrieval_ids"][:max_k]

    # Build rank maps (1-indexed; missing items get rank = max_k + 1)
    bm25_rank = {item_id: rank + 1 for rank, item_id in enumerate(bm25_ids)}
    dense_rank = {item_id: rank + 1 for rank, item_id in enumerate(dense_ids)}
    default_rank = max_k + 1

    # Collect all candidate IDs
    all_ids = list(dict.fromkeys(bm25_ids + dense_ids))  # dedupe, preserve order

    # Effective weights: fold metadata into sparse or dense
    ws = aw.weight_sparse + aw.weight_metadata * (
        aw.weight_sparse / max(aw.weight_sparse + aw.weight_dense, 1e-9)
    )
    wd = aw.weight_dense + aw.weight_metadata * (
        aw.weight_dense / max(aw.weight_sparse + aw.weight_dense, 1e-9)
    )

    # RRF scores
    scored = []
    for item_id in all_ids:
        br = bm25_rank.get(item_id, default_rank)
        dr = dense_rank.get(item_id, default_rank)
        score = ws / (rrf_k + br) + wd / (rrf_k + dr)
        scored.append((item_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scored[:max_k]]


def _oracle_best(
    bm25_entry: Dict[str, Any],
    dense_entry: Dict[str, Any],
    gt_ids: List[str],
    k: int,
) -> List[str]:
    """Oracle: pick whichever retriever gives higher R@k for this question."""
    bm25_r = compute_recall(gt_ids, bm25_entry["retrieval_ids"])[f"R@{k}"]
    dense_r = compute_recall(gt_ids, dense_entry["retrieval_ids"])[f"R@{k}"]
    if bm25_r >= dense_r:
        return bm25_entry["retrieval_ids"]
    return dense_entry["retrieval_ids"]


# ──────────────────────────────────────────────────────────────────────


def main():
    bm25_data = {str(d["id"]): d for d in load(BM25_DETAILS)}
    dense_data = {str(d["id"]): d for d in load(DENSE_DETAILS)}

    hybrid_data = {}
    if HYBRID_DETAILS.exists():
        hybrid_data = {str(d["id"]): d for d in load(HYBRID_DETAILS)}

    qa_list = load(QA_FILE)
    qa_map = {str(q["id"]): q for q in qa_list}
    all_ids = sorted(set(bm25_data) & set(dense_data))
    print(f"Total questions: {len(all_ids)}\n")

    # ── Run routing on every question ────────────────────────────────
    strategy_counter = Counter()
    adaptive_strategy_counter = Counter()
    per_question: List[Dict[str, Any]] = []

    for qid in all_ids:
        qa = qa_map[qid]
        question = qa["question"]
        gt_ids = qa["evidence_ids"]
        qtype = qa.get("qtype", "?")
        mod = evidence_modality(gt_ids)

        signals = analyze_query(question)
        strategy = route(signals)
        strategy_counter[strategy.name] += 1

        aw = adaptive_weights(signals)
        adaptive_strategy_counter[aw.strategy_name] += 1

        routed_ids = pick_routed_results(strategy, bm25_data[qid], dense_data[qid])
        adaptive_ids = pick_adaptive_results(aw, bm25_data[qid], dense_data[qid])
        oracle_ids = _oracle_best(bm25_data[qid], dense_data[qid], gt_ids, 10)

        per_question.append({
            "id": qid,
            "question": question,
            "qtype": qtype,
            "mod": mod,
            "strategy": strategy.name,
            "adaptive_strategy": aw.strategy_name,
            "adaptive_weights": aw.as_tuple(),
            "signals": {
                "kw": signals.keyword_score,
                "sem": signals.semantic_score,
                "meta": signals.meta_score,
                "entity": signals.has_entity,
                "rel_date": signals.has_relative_date,
                "non_latin": signals.has_non_latin,
            },
            "bm25_recall": compute_recall(gt_ids, bm25_data[qid]["retrieval_ids"]),
            "dense_recall": compute_recall(gt_ids, dense_data[qid]["retrieval_ids"]),
            "routed_recall": compute_recall(gt_ids, routed_ids),
            "adaptive_recall": compute_recall(gt_ids, adaptive_ids),
            "oracle_recall": compute_recall(gt_ids, oracle_ids),
            "hybrid_recall": compute_recall(gt_ids, hybrid_data[qid]["retrieval_ids"]) if qid in hybrid_data else {},
        })

    # ── Aggregate results ────────────────────────────────────────────
    def avg_recall(key: str, subset=None) -> Dict[str, float]:
        items = subset or per_question
        items = [q for q in items if q.get(key)]
        if not items:
            return {}
        avgs = {}
        for rk in [f"R@{k}" for k in RECALL_KS]:
            avgs[rk] = sum(q[key].get(rk, 0) for q in items) / len(items)
        return avgs

    print("=" * 100)
    print("STRATEGY DISTRIBUTION")
    print("=" * 100)
    print("\n  Hard routing:")
    for strat, count in strategy_counter.most_common():
        print(f"    {strat:<20} {count:>5} ({count/len(all_ids)*100:.1f}%)")
    print("\n  Adaptive (soft) routing:")
    for strat, count in adaptive_strategy_counter.most_common():
        print(f"    {strat:<20} {count:>5} ({count/len(all_ids)*100:.1f}%)")

    # Full R@K table — the main comparison
    print(f"\n{'=' * 100}")
    print("OVERALL R@K COMPARISON")
    print(f"{'=' * 100}")
    print(f"{'R@K':<8} {'BM25':>8} {'Dense':>8} {'Hybrid':>8} {'HardRt':>8} {'Adaptive':>8} {'Oracle':>8}  {'Δ(Ad-Hy)':>9}")
    print("-" * 80)
    for rk in [f"R@{k}" for k in RECALL_KS]:
        bm = avg_recall("bm25_recall").get(rk, 0)
        dn = avg_recall("dense_recall").get(rk, 0)
        rt = avg_recall("routed_recall").get(rk, 0)
        ad = avg_recall("adaptive_recall").get(rk, 0)
        orc = avg_recall("oracle_recall").get(rk, 0)
        hyb = avg_recall("hybrid_recall").get(rk, 0) if hybrid_data else 0
        delta = ad - hyb if hyb else 0
        print(f"{rk:<8} {bm:>8.4f} {dn:>8.4f} {hyb:>8.4f} {rt:>8.4f} {ad:>8.4f} {orc:>8.4f}  {delta:>+9.4f}")

    # Per-adaptive-strategy breakdown
    print(f"\n{'=' * 100}")
    print("PER-ADAPTIVE-STRATEGY R@10")
    print(f"{'=' * 100}")
    print(f"{'Strategy':<22} {'N':>5}  {'BM25':>7} {'Dense':>7} {'Hybrid':>7} {'Adaptive':>8} {'Oracle':>7}  {'Δ(Ad-Hy)':>9}")
    print("-" * 100)
    for strat in sorted(adaptive_strategy_counter):
        subset = [q for q in per_question if q["adaptive_strategy"] == strat]
        n = len(subset)
        bm = sum(q["bm25_recall"]["R@10"] for q in subset) / n
        dn = sum(q["dense_recall"]["R@10"] for q in subset) / n
        ad = sum(q["adaptive_recall"]["R@10"] for q in subset) / n
        orc = sum(q["oracle_recall"]["R@10"] for q in subset) / n
        hyb = sum(q["hybrid_recall"].get("R@10", 0) for q in subset) / n if hybrid_data else 0
        delta = ad - hyb if hyb else 0
        print(f"{strat:<22} {n:>5}  {bm:>7.1%} {dn:>7.1%} {hyb:>7.1%} {ad:>8.1%} {orc:>7.1%}  {delta:>+8.1%}")
    # Total
    n = len(per_question)
    print("-" * 100)
    bm = sum(q["bm25_recall"]["R@10"] for q in per_question) / n
    dn = sum(q["dense_recall"]["R@10"] for q in per_question) / n
    ad = sum(q["adaptive_recall"]["R@10"] for q in per_question) / n
    orc = sum(q["oracle_recall"]["R@10"] for q in per_question) / n
    hyb = sum(q["hybrid_recall"].get("R@10", 0) for q in per_question) / n if hybrid_data else 0
    delta = ad - hyb if hyb else 0
    print(f"{'ALL':<22} {n:>5}  {bm:>7.1%} {dn:>7.1%} {hyb:>7.1%} {ad:>8.1%} {orc:>7.1%}  {delta:>+8.1%}")

    # Per qtype × modality
    print(f"\n{'=' * 100}")
    print("R@10 BY QTYPE × MODALITY")
    print(f"{'=' * 100}")
    print(f"{'Category':<25} {'N':>5}  {'BM25':>7} {'Dense':>7} {'Hybrid':>7} {'Adaptive':>8} {'Oracle':>7}  {'Δ(Ad-Hy)':>9}")
    print("-" * 95)
    for cat_fn, cats in [
        (lambda q: q["qtype"], ["number", "open_end", "list_recall"]),
        (lambda q: q["mod"], ["email", "media"]),
        (lambda q: f"{q['qtype']}×{q['mod']}", None),
    ]:
        if cats is None:
            cats = sorted(set(f"{q['qtype']}×{q['mod']}" for q in per_question))
        for cat in cats:
            subset = [q for q in per_question if cat_fn(q) == cat]
            if not subset:
                continue
            n = len(subset)
            bm = sum(q["bm25_recall"]["R@10"] for q in subset) / n
            dn = sum(q["dense_recall"]["R@10"] for q in subset) / n
            ad = sum(q["adaptive_recall"]["R@10"] for q in subset) / n
            orc = sum(q["oracle_recall"]["R@10"] for q in subset) / n
            hyb = sum(q["hybrid_recall"].get("R@10", 0) for q in subset) / n if hybrid_data else 0
            delta = ad - hyb if hyb else 0
            print(f"{cat:<25} {n:>5}  {bm:>7.1%} {dn:>7.1%} {hyb:>7.1%} {ad:>8.1%} {orc:>7.1%}  {delta:>+8.1%}")

    # Misrouted questions analysis
    print(f"\n{'=' * 90}")
    print("MISROUTED QUESTIONS (Router picked worse channel)")
    print(f"{'=' * 90}")
    misrouted = []
    for q in per_question:
        best_single = max(q["bm25_recall"]["R@10"], q["dense_recall"]["R@10"])
        if q["routed_recall"]["R@10"] < best_single - 1e-9:
            misrouted.append(q)

    print(f"Total misrouted: {len(misrouted)} / {len(per_question)} ({len(misrouted)/len(per_question)*100:.1f}%)\n")

    # Break down misrouted by strategy
    mis_by_strat = Counter(q["strategy"] for q in misrouted)
    print("  By strategy:")
    for strat, cnt in mis_by_strat.most_common():
        print(f"    {strat:<20} {cnt}")

    # Break down by what should have been used
    print("\n  Samples (router chose wrong):")
    for q in misrouted[:10]:
        bm = q["bm25_recall"]["R@10"]
        dn = q["dense_recall"]["R@10"]
        rt = q["routed_recall"]["R@10"]
        better = "BM25" if bm > dn else "Dense"
        print(
            f"  [{q['strategy']:<18}] {q['qtype']:<12} {q['mod']:<6} "
            f"BM25={bm:.2f} Dense={dn:.2f} Routed={rt:.2f} "
            f"(should use {better})"
        )
        print(f"    Q: {q['question'][:100]}")

    # Dump detailed results
    out_path = ROOT / "output/QA_Agent/MMRAG/routing_eval_results.json"
    out_path.write_text(json.dumps(per_question, indent=2, ensure_ascii=False))
    print(f"\nDetailed results → {out_path}")


if __name__ == "__main__":
    main()
