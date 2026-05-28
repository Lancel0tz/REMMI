#!/usr/bin/env python3
"""Comprehensive hybrid retrieval sweep analysis.

Reads all sweep_*/ directories under the sweep output base,
computes multi-k recall comparison across hard and full splits,
and recommends optimal weight configurations.

Usage:
    python scripts/QA_Agent/MMRAG/analyze_hybrid_sweep.py
    python scripts/QA_Agent/MMRAG/analyze_hybrid_sweep.py --output-base output/QA_Agent/MMRAG/hybrid_sweep
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KS = [1, 5, 10, 25, 50, 100, 200]


@dataclass
class SweepResult:
    split: str
    fusion: str
    filter_mode: str
    w_meta: float
    w_sparse: float
    w_dense: float
    recall: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    @property
    def config_key(self) -> str:
        return f"{self.fusion}_m{self.w_meta}_s{self.w_sparse}_d{self.w_dense}_{self.filter_mode}"

    @property
    def short_label(self) -> str:
        parts = []
        if self.w_meta == 0 and self.w_sparse == 0:
            parts.append("dense-only")
        elif self.w_meta == 0 and self.w_dense == 0:
            parts.append("BM25-only")
        else:
            parts.append(f"m={self.w_meta} s={self.w_sparse} d={self.w_dense}")
        parts.append(self.fusion)
        if self.filter_mode == "hard":
            parts.append("hard-filter")
        return " | ".join(parts)


def parse_dir_name(dirname: str) -> Optional[Tuple[str, str, float, float, float, str]]:
    m = re.match(
        r"sweep_(hard|full)_(rrf|weighted_sum)_m([\d.]+)_s([\d.]+)_d([\d.]+)_(soft|hard)$",
        dirname,
    )
    if not m:
        return None
    split, fusion, wm, ws, wd, fmode = m.groups()
    return split, fusion, float(wm), float(ws), float(wd), fmode


def load_results(base: Path) -> List[SweepResult]:
    results = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("sweep_"):
            continue
        parsed = parse_dir_name(d.name)
        if parsed is None:
            continue
        split, fusion, wm, ws, wd, fmode = parsed

        summary_file = d / "retrieval_recall_summary.json"
        if not summary_file.exists():
            continue

        data = json.loads(summary_file.read_text())
        recall_raw = data.get("recall", {})
        recall = {}
        for k, v in recall_raw.items():
            if k.startswith("R@"):
                recall[k] = float(v)
        count = recall_raw.get("count", data.get("count", 0))

        results.append(SweepResult(
            split=split,
            fusion=fusion,
            filter_mode=fmode,
            w_meta=wm,
            w_sparse=ws,
            w_dense=wd,
            recall=recall,
            count=count,
        ))
    return results


def composite_score(r: SweepResult, weights: Optional[Dict[int, float]] = None) -> float:
    if weights is None:
        weights = {1: 0.10, 5: 0.15, 10: 0.25, 25: 0.20, 50: 0.15, 100: 0.10, 200: 0.05}
    score = 0.0
    for k, w in weights.items():
        score += w * r.recall.get(f"R@{k}", 0.0)
    return score


def delta_str(val: float, baseline: float) -> str:
    diff = (val - baseline) * 100
    if diff > 0:
        return f"+{diff:.2f}"
    elif diff < 0:
        return f"{diff:.2f}"
    return "  0.00"


def print_split_table(split: str, rows: List[SweepResult]):
    if not rows:
        print(f"\n  (no results for {split} split)\n")
        return

    baseline = None
    for r in rows:
        if r.w_meta == 0 and r.w_sparse == 0 and r.w_dense == 1.0:
            baseline = r
            break

    rows_scored = [(r, composite_score(r)) for r in rows]
    rows_scored.sort(key=lambda x: -x[1])

    baseline_cs = composite_score(baseline) if baseline else 0.0

    print(f"\n{'='*120}")
    print(f"  {split.upper()} split  (n={rows[0].count})")
    print(f"{'='*120}")

    header = (
        f"{'#':>2} {'Config':<42} "
        f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@25':>6} {'R@50':>6} {'R@100':>6} {'R@200':>6} "
        f"{'Comp':>6} {'Δcomp':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, (r, cs) in enumerate(rows_scored, 1):
        vals = []
        for k in KS:
            v = r.recall.get(f"R@{k}", 0.0) * 100
            vals.append(f"{v:6.2f}")
        cs_pct = cs * 100
        delta = delta_str(cs, baseline_cs) if baseline else "  N/A"
        marker = " *" if r == baseline else ""
        print(
            f"{rank:>2} {r.short_label:<42} "
            f"{'  '.join(vals)} "
            f"{cs_pct:6.2f} {delta:>7}{marker}"
        )

    if baseline:
        print(f"\n  * = dense-only baseline | Comp = weighted composite "
              f"(R@1×10% + R@5×15% + R@10×25% + R@25×20% + R@50×15% + R@100×10% + R@200×5%)")
    print()


def print_cross_split_ranking(hard_results: List[SweepResult], full_results: List[SweepResult]):
    hard_map = {r.config_key: r for r in hard_results}
    full_map = {r.config_key: r for r in full_results}

    all_keys = set(hard_map.keys()) | set(full_map.keys())

    hard_baseline_cs = 0.0
    full_baseline_cs = 0.0
    for key in all_keys:
        if key.startswith("rrf_m0.0_s0.0_d1.0"):
            if key in hard_map:
                hard_baseline_cs = composite_score(hard_map[key])
            if key in full_map:
                full_baseline_cs = composite_score(full_map[key])

    rows = []
    for key in all_keys:
        h = hard_map.get(key)
        f = full_map.get(key)
        h_cs = composite_score(h) if h else None
        f_cs = composite_score(f) if f else None

        if h_cs is not None and f_cs is not None:
            avg_cs = (h_cs + f_cs) / 2
        elif h_cs is not None:
            avg_cs = h_cs
        else:
            avg_cs = f_cs

        label = h.short_label if h else f.short_label
        rows.append((key, label, h_cs, f_cs, avg_cs))

    rows.sort(key=lambda x: -(x[4] or 0))

    print(f"\n{'='*100}")
    print("  CROSS-SPLIT RANKING (by average composite score)")
    print(f"{'='*100}")

    header = (
        f"{'#':>2} {'Config':<42} "
        f"{'Hard':>7} {'Δhard':>7} {'Full':>7} {'Δfull':>7} {'Avg':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, (key, label, h_cs, f_cs, avg_cs) in enumerate(rows, 1):
        h_str = f"{h_cs*100:7.2f}" if h_cs is not None else "    N/A"
        f_str = f"{f_cs*100:7.2f}" if f_cs is not None else "    N/A"
        avg_str = f"{avg_cs*100:7.2f}" if avg_cs is not None else "    N/A"
        dh = delta_str(h_cs, hard_baseline_cs) if h_cs is not None else "    N/A"
        df = delta_str(f_cs, full_baseline_cs) if f_cs is not None else "    N/A"
        print(f"{rank:>2} {label:<42} {h_str} {dh:>7} {f_str} {df:>7} {avg_str}")

    print(f"\n  Composite = R@1×10% + R@5×15% + R@10×25% + R@25×20% + R@50×15% + R@100×10% + R@200×5%")
    print()


def print_per_k_best(hard_results: List[SweepResult], full_results: List[SweepResult]):
    print(f"\n{'='*80}")
    print("  BEST CONFIG PER R@k (both splits)")
    print(f"{'='*80}\n")

    for k in KS:
        key = f"R@{k}"
        print(f"  --- {key} ---")
        for split_name, results in [("Hard", hard_results), ("Full", full_results)]:
            if not results:
                continue
            best = max(results, key=lambda r: r.recall.get(key, 0.0))
            baseline = next((r for r in results if r.w_meta == 0 and r.w_sparse == 0), None)
            bl_val = baseline.recall.get(key, 0.0) if baseline else 0.0
            best_val = best.recall.get(key, 0.0)
            delta = (best_val - bl_val) * 100
            print(f"    {split_name:>4}: {best.short_label:<40}  {best_val*100:6.2f}%  (Δ {delta:+.2f}%)")
        print()


def print_recommendation(hard_results: List[SweepResult], full_results: List[SweepResult]):
    hard_map = {r.config_key: r for r in hard_results}
    full_map = {r.config_key: r for r in full_results}
    common_keys = set(hard_map.keys()) & set(full_map.keys())

    if not common_keys:
        print("\n  (Need both splits to make a recommendation)\n")
        return

    best_key = None
    best_avg = -1.0
    for key in common_keys:
        avg = (composite_score(hard_map[key]) + composite_score(full_map[key])) / 2
        if avg > best_avg:
            best_avg = avg
            best_key = key

    best_h = hard_map[best_key]
    best_f = full_map[best_key]

    print(f"\n{'='*80}")
    print("  RECOMMENDATION")
    print(f"{'='*80}\n")
    print(f"  Best overall config: {best_h.short_label}")
    print(f"    fusion={best_h.fusion}  filter={best_h.filter_mode}")
    print(f"    weights: meta={best_h.w_meta}  sparse={best_h.w_sparse}  dense={best_h.w_dense}")
    print()
    print(f"    Hard split (n={best_h.count}):")
    for k in KS:
        v = best_h.recall.get(f"R@{k}", 0.0) * 100
        print(f"      R@{k:<3} = {v:.2f}%")
    print()
    print(f"    Full split (n={best_f.count}):")
    for k in KS:
        v = best_f.recall.get(f"R@{k}", 0.0) * 100
        print(f"      R@{k:<3} = {v:.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze hybrid retrieval sweep results")
    parser.add_argument(
        "--output-base",
        default="output/QA_Agent/MMRAG/hybrid_sweep",
        help="Base directory containing sweep_*/ result dirs",
    )
    args = parser.parse_args()
    base = Path(args.output_base)

    if not base.exists():
        print(f"Error: {base} does not exist")
        return

    results = load_results(base)
    if not results:
        print("No results found.")
        return

    hard_results = [r for r in results if r.split == "hard"]
    full_results = [r for r in results if r.split == "full"]

    print(f"\nLoaded {len(hard_results)} hard + {len(full_results)} full configs\n")

    print_split_table("hard", hard_results)
    print_split_table("full", full_results)
    print_cross_split_ranking(hard_results, full_results)
    print_per_k_best(hard_results, full_results)
    print_recommendation(hard_results, full_results)


if __name__ == "__main__":
    main()
