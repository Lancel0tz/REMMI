#!/usr/bin/env python3
"""Error analysis: BM25-only vs Dense-only retrieval on ATM-Bench (full)."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

BM25_DETAILS = ROOT / "output/QA_Agent/MMRAG/hybrid_sweep/sweep_full_rrf_m0.0_s1.0_d0.0_soft/retrieval_recall_details.json"
DENSE_DETAILS = ROOT / "output/QA_Agent/MMRAG/hybrid_sweep/sweep_full_rrf_m0.0_s0.0_d1.0_soft/retrieval_recall_details.json"
QA_FILE = ROOT / "data/atm-bench/atm-bench.json"

HARD_FILE = ROOT / "data/atm-bench/atm-bench-hard.json"

K = 10  # primary R@K to compare


def load(path):
    return json.loads(path.read_text())


def evidence_modality(evidence_ids):
    has_email = any(eid.startswith("email") for eid in evidence_ids)
    has_media = any(not eid.startswith("email") for eid in evidence_ids)
    if has_email and has_media:
        return "mixed"
    return "email" if has_email else "media"


def main():
    bm25_data = {str(d["id"]): d for d in load(BM25_DETAILS)}
    dense_data = {str(d["id"]): d for d in load(DENSE_DETAILS)}
    qa_list = load(QA_FILE)
    qa_map = {str(q["id"]): q for q in qa_list}

    hard_ids = set()
    if HARD_FILE.exists():
        hard_ids = {str(q["id"]) for q in load(HARD_FILE)}

    all_ids = sorted(set(bm25_data) & set(dense_data))
    print(f"Total questions with both BM25 and Dense results: {len(all_ids)}")
    print(f"Using R@{K} as primary metric\n")

    bm25_wins = []
    dense_wins = []
    both_good = []
    both_bad = []

    for qid in all_ids:
        bm25_r = bm25_data[qid]["retrieval_recall"].get(f"R@{K}", 0.0)
        dense_r = dense_data[qid]["retrieval_recall"].get(f"R@{K}", 0.0)
        qa = qa_map.get(qid, {})

        entry = {
            "id": qid,
            "question": bm25_data[qid]["question"],
            "qtype": qa.get("qtype", "?"),
            "evidence_modality": evidence_modality(qa.get("evidence_ids", [])),
            "num_evidence": len(qa.get("evidence_ids", [])),
            "is_hard": qid in hard_ids,
            "bm25_recall": bm25_r,
            "dense_recall": dense_r,
            "gt_ids": qa.get("evidence_ids", []),
        }

        if bm25_r == 1.0 and dense_r == 1.0:
            both_good.append(entry)
        elif bm25_r == 1.0 and dense_r < 1.0:
            bm25_wins.append(entry)
        elif dense_r == 1.0 and bm25_r < 1.0:
            dense_wins.append(entry)
        elif bm25_r > dense_r + 1e-9:
            bm25_wins.append(entry)
        elif dense_r > bm25_r + 1e-9:
            dense_wins.append(entry)
        else:
            if bm25_r >= 1.0 - 1e-9:
                both_good.append(entry)
            else:
                both_bad.append(entry)

    print("=" * 70)
    print("OVERVIEW")
    print("=" * 70)
    print(f"  BM25 wins (BM25 > Dense):  {len(bm25_wins)}")
    print(f"  Dense wins (Dense > BM25):  {len(dense_wins)}")
    print(f"  Both perfect (R@{K}=1.0):    {len(both_good)}")
    print(f"  Both equal & imperfect:     {len(both_bad)}")
    print()

    def analyze_group(name, group):
        print(f"\n{'=' * 70}")
        print(f"{name} ({len(group)} questions)")
        print("=" * 70)

        qtypes = Counter(e["qtype"] for e in group)
        modalities = Counter(e["evidence_modality"] for e in group)
        hard_count = sum(1 for e in group if e["is_hard"])
        num_ev = Counter(e["num_evidence"] for e in group)

        print(f"\n  By qtype:")
        for k, v in qtypes.most_common():
            print(f"    {k}: {v} ({v/len(group)*100:.1f}%)")

        print(f"\n  By evidence modality:")
        for k, v in modalities.most_common():
            print(f"    {k}: {v} ({v/len(group)*100:.1f}%)")

        print(f"\n  Hard subset: {hard_count}/{len(group)} ({hard_count/len(group)*100:.1f}%)")

        print(f"\n  By num_evidence:")
        for k, v in sorted(num_ev.items()):
            print(f"    {k}: {v}")

        if name in ("BM25 WINS", "DENSE WINS"):
            avg_winner = sum(
                e["bm25_recall"] if "BM25" in name else e["dense_recall"] for e in group
            ) / len(group)
            avg_loser = sum(
                e["dense_recall"] if "BM25" in name else e["bm25_recall"] for e in group
            ) / len(group)
            print(f"\n  Avg winner R@{K}: {avg_winner:.4f}")
            print(f"  Avg loser  R@{K}: {avg_loser:.4f}")
            print(f"  Avg gap:          {avg_winner - avg_loser:.4f}")

    # Overall baseline stats
    total_qa = len(qa_list)
    print(f"\n--- Baseline distribution (all {total_qa} questions) ---")
    all_qtypes = Counter(q["qtype"] for q in qa_list)
    for k, v in all_qtypes.most_common():
        print(f"  {k}: {v} ({v/total_qa*100:.1f}%)")
    all_mod = Counter(evidence_modality(q["evidence_ids"]) for q in qa_list)
    for k, v in all_mod.most_common():
        print(f"  {k}: {v} ({v/total_qa*100:.1f}%)")

    analyze_group("BM25 WINS", bm25_wins)
    analyze_group("DENSE WINS", dense_wins)
    analyze_group("BOTH PERFECT", both_good)
    analyze_group("BOTH BAD (equal & imperfect)", both_bad)

    # Exclusive wins: questions where one method gets perfect recall and the other gets 0
    print(f"\n\n{'=' * 70}")
    print("EXCLUSIVE WINS (R@{K}=1.0 vs R@{K}=0.0)")
    print("=" * 70)
    bm25_exclusive = [e for e in bm25_wins if e["bm25_recall"] >= 1.0 - 1e-9 and e["dense_recall"] < 1e-9]
    dense_exclusive = [e for e in dense_wins if e["dense_recall"] >= 1.0 - 1e-9 and e["bm25_recall"] < 1e-9]
    print(f"  BM25 perfect & Dense zero: {len(bm25_exclusive)}")
    print(f"  Dense perfect & BM25 zero: {len(dense_exclusive)}")

    # Sample questions from each exclusive group
    def print_samples(name, group, n=5):
        print(f"\n  --- Sample {name} (up to {n}) ---")
        for e in group[:n]:
            print(f"  [{e['qtype']}|{e['evidence_modality']}|ev={e['num_evidence']}] {e['question']}")
            print(f"    BM25 R@{K}={e['bm25_recall']:.2f}  Dense R@{K}={e['dense_recall']:.2f}")
            print(f"    GT: {e['gt_ids']}")

    print_samples("BM25 exclusive wins", bm25_exclusive, 8)
    print_samples("Dense exclusive wins", dense_exclusive, 8)

    # Complementarity analysis: if we could use an oracle to pick the better retriever per question
    print(f"\n\n{'=' * 70}")
    print("COMPLEMENTARITY (Oracle best-of-two)")
    print("=" * 70)
    bm25_total = sum(bm25_data[qid]["retrieval_recall"].get(f"R@{K}", 0) for qid in all_ids) / len(all_ids)
    dense_total = sum(dense_data[qid]["retrieval_recall"].get(f"R@{K}", 0) for qid in all_ids) / len(all_ids)
    oracle_total = sum(
        max(
            bm25_data[qid]["retrieval_recall"].get(f"R@{K}", 0),
            dense_data[qid]["retrieval_recall"].get(f"R@{K}", 0),
        )
        for qid in all_ids
    ) / len(all_ids)
    print(f"  BM25 avg R@{K}:   {bm25_total:.4f}")
    print(f"  Dense avg R@{K}:  {dense_total:.4f}")
    print(f"  Oracle avg R@{K}: {oracle_total:.4f}")
    print(f"  Oracle headroom:  +{oracle_total - max(bm25_total, dense_total):.4f} over best single")

    # Per-modality & qtype breakdown
    print(f"\n\n{'=' * 70}")
    print("PER-CATEGORY R@{K} COMPARISON")
    print("=" * 70)

    for dim_name, dim_fn in [
        ("qtype", lambda e: qa_map.get(e, {}).get("qtype", "?")),
        ("evidence_modality", lambda e: evidence_modality(qa_map.get(e, {}).get("evidence_ids", []))),
        ("is_hard", lambda e: "hard" if e in hard_ids else "non-hard"),
    ]:
        print(f"\n  --- By {dim_name} ---")
        groups = defaultdict(list)
        for qid in all_ids:
            groups[dim_fn(qid)].append(qid)

        print(f"  {'Category':<15} {'N':>5}  {'BM25':>8}  {'Dense':>8}  {'Delta':>8}  {'Oracle':>8}")
        for cat in sorted(groups):
            ids = groups[cat]
            n = len(ids)
            bm = sum(bm25_data[qid]["retrieval_recall"].get(f"R@{K}", 0) for qid in ids) / n
            dn = sum(dense_data[qid]["retrieval_recall"].get(f"R@{K}", 0) for qid in ids) / n
            orc = sum(max(
                bm25_data[qid]["retrieval_recall"].get(f"R@{K}", 0),
                dense_data[qid]["retrieval_recall"].get(f"R@{K}", 0),
            ) for qid in ids) / n
            print(f"  {cat:<15} {n:>5}  {bm:>8.4f}  {dn:>8.4f}  {dn-bm:>+8.4f}  {orc:>8.4f}")

    # Extended R@K comparison
    print(f"\n\n{'=' * 70}")
    print("R@K COMPARISON (all K values)")
    print("=" * 70)
    for rk in ["R@1", "R@5", "R@10", "R@25", "R@50", "R@100", "R@200"]:
        bm = sum(bm25_data[qid]["retrieval_recall"].get(rk, 0) for qid in all_ids) / len(all_ids)
        dn = sum(dense_data[qid]["retrieval_recall"].get(rk, 0) for qid in all_ids) / len(all_ids)
        orc = sum(max(
            bm25_data[qid]["retrieval_recall"].get(rk, 0),
            dense_data[qid]["retrieval_recall"].get(rk, 0),
        ) for qid in all_ids) / len(all_ids)
        print(f"  {rk:<6}  BM25={bm:.4f}  Dense={dn:.4f}  Delta={dn-bm:+.4f}  Oracle={orc:.4f}")

    # Dump detailed results to JSON for further analysis
    output = {
        "bm25_wins": bm25_wins,
        "dense_wins": dense_wins,
        "both_good": both_good,
        "both_bad": both_bad,
    }
    out_path = ROOT / "output/QA_Agent/MMRAG/error_analysis_bm25_vs_dense.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()
