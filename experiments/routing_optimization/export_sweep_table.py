#!/usr/bin/env python3
"""Export hybrid sweep results to a comprehensive CSV + markdown summary table.

Outputs:
  1. Full CSV with all R@k values, deltas, composite scores for both splits
  2. Markdown table for paper/report inclusion
  3. Pretty-printed terminal table

Usage:
    python scripts/QA_Agent/MMRAG/export_sweep_table.py
    python scripts/QA_Agent/MMRAG/export_sweep_table.py --output-base output/QA_Agent/MMRAG/hybrid_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


KS = [1, 5, 10, 25, 50, 100]
COMPOSITE_WEIGHTS = {1: 0.10, 5: 0.15, 10: 0.25, 25: 0.20, 50: 0.15, 100: 0.10, 200: 0.05}


@dataclass
class Result:
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
    def label(self) -> str:
        if self.w_meta == 0 and self.w_sparse == 0:
            tag = "dense-only"
        elif self.w_meta == 0 and self.w_dense == 0:
            tag = "BM25-only"
        else:
            tag = f"m={self.w_meta} s={self.w_sparse} d={self.w_dense}"
        parts = [tag, self.fusion]
        if self.filter_mode == "hard":
            parts.append("hard-filter")
        return " | ".join(parts)

    def composite(self) -> float:
        return sum(w * self.recall.get(f"R@{k}", 0.0) for k, w in COMPOSITE_WEIGHTS.items())


def parse_dir_name(name: str) -> Optional[Tuple[str, str, float, float, float, str]]:
    m = re.match(
        r"sweep_(hard|full)_(rrf|weighted_sum)_m([\d.]+)_s([\d.]+)_d([\d.]+)_(soft|hard)$",
        name,
    )
    if not m:
        return None
    split, fusion, wm, ws, wd, fmode = m.groups()
    return split, fusion, float(wm), float(ws), float(wd), fmode


def load_results(base: Path) -> List[Result]:
    results = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("sweep_"):
            continue
        parsed = parse_dir_name(d.name)
        if not parsed:
            continue
        split, fusion, wm, ws, wd, fmode = parsed
        sf = d / "retrieval_recall_summary.json"
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        recall_raw = data.get("recall", {})
        recall = {k: float(v) for k, v in recall_raw.items() if k.startswith("R@")}
        results.append(Result(
            split=split, fusion=fusion, filter_mode=fmode,
            w_meta=wm, w_sparse=ws, w_dense=wd,
            recall=recall, count=recall_raw.get("count", 0),
        ))
    return results


def fmt_pct(v: Optional[float]) -> str:
    return f"{v * 100:.2f}" if v is not None else ""


def fmt_delta(v: Optional[float], bl: Optional[float]) -> str:
    if v is None or bl is None:
        return ""
    d = (v - bl) * 100
    return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"


def build_rows(results: List[Result]) -> List[Dict[str, Any]]:
    hard_map: Dict[str, Result] = {}
    full_map: Dict[str, Result] = {}
    for r in results:
        (hard_map if r.split == "hard" else full_map)[r.config_key] = r

    hard_bl = next((r for r in hard_map.values() if r.w_meta == 0 and r.w_sparse == 0 and r.w_dense == 1.0), None)
    full_bl = next((r for r in full_map.values() if r.w_meta == 0 and r.w_sparse == 0 and r.w_dense == 1.0), None)

    all_keys = sorted(set(hard_map) | set(full_map), key=lambda k: (
        not (hard_map.get(k) or full_map.get(k)).w_meta == 0 and
        (hard_map.get(k) or full_map.get(k)).w_sparse == 0,
        k,
    ))

    rows = []
    for key in all_keys:
        h = hard_map.get(key)
        f = full_map.get(key)
        ref = h or f

        row: Dict[str, Any] = {
            "config": ref.label,
            "fusion": ref.fusion,
            "filter": ref.filter_mode,
            "w_meta": ref.w_meta,
            "w_sparse": ref.w_sparse,
            "w_dense": ref.w_dense,
        }

        for k in KS:
            rk = f"R@{k}"
            h_val = h.recall.get(rk) if h else None
            f_val = f.recall.get(rk) if f else None
            h_bl_val = hard_bl.recall.get(rk) if hard_bl else None
            f_bl_val = full_bl.recall.get(rk) if full_bl else None

            row[f"hard_R@{k}"] = h_val
            row[f"hard_Δ@{k}"] = (h_val - h_bl_val) if (h_val is not None and h_bl_val is not None) else None
            row[f"full_R@{k}"] = f_val
            row[f"full_Δ@{k}"] = (f_val - f_bl_val) if (f_val is not None and f_bl_val is not None) else None

        h_cs = h.composite() if h else None
        f_cs = f.composite() if f else None
        h_bl_cs = hard_bl.composite() if hard_bl else None
        f_bl_cs = full_bl.composite() if full_bl else None

        row["hard_comp"] = h_cs
        row["hard_Δcomp"] = (h_cs - h_bl_cs) if (h_cs is not None and h_bl_cs is not None) else None
        row["full_comp"] = f_cs
        row["full_Δcomp"] = (f_cs - f_bl_cs) if (f_cs is not None and f_bl_cs is not None) else None

        if h_cs is not None and f_cs is not None:
            row["avg_comp"] = (h_cs + f_cs) / 2
            row["avg_Δcomp"] = ((h_cs - (h_bl_cs or 0)) + (f_cs - (f_bl_cs or 0))) / 2
        elif h_cs is not None:
            row["avg_comp"] = h_cs
            row["avg_Δcomp"] = (h_cs - (h_bl_cs or 0))
        else:
            row["avg_comp"] = f_cs
            row["avg_Δcomp"] = (f_cs - (f_bl_cs or 0)) if f_cs is not None else None

        rows.append(row)

    rows.sort(key=lambda r: -(r["avg_comp"] or 0))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path):
    fieldnames = [
        "rank", "config", "fusion", "filter", "w_meta", "w_sparse", "w_dense",
    ]
    for k in KS:
        fieldnames += [f"hard_R@{k}(%)", f"hard_Δ@{k}(pp)"]
    fieldnames += ["hard_comp(%)", "hard_Δcomp(pp)"]
    for k in KS:
        fieldnames += [f"full_R@{k}(%)", f"full_Δ@{k}(pp)"]
    fieldnames += ["full_comp(%)", "full_Δcomp(pp)", "avg_comp(%)", "avg_Δcomp(pp)"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            out = {
                "rank": i,
                "config": row["config"],
                "fusion": row["fusion"],
                "filter": row["filter"],
                "w_meta": row["w_meta"],
                "w_sparse": row["w_sparse"],
                "w_dense": row["w_dense"],
            }
            for k in KS:
                out[f"hard_R@{k}(%)"] = fmt_pct(row.get(f"hard_R@{k}"))
                out[f"hard_Δ@{k}(pp)"] = fmt_delta(row.get(f"hard_R@{k}"), row.get(f"hard_R@{k}") - row[f"hard_Δ@{k}"] if row.get(f"hard_Δ@{k}") is not None else None) if row.get(f"hard_Δ@{k}") is not None else ""
            out["hard_comp(%)"] = fmt_pct(row.get("hard_comp"))
            out["hard_Δcomp(pp)"] = fmt_delta(row.get("hard_comp"), row["hard_comp"] - row["hard_Δcomp"] if row.get("hard_Δcomp") is not None else None) if row.get("hard_Δcomp") is not None else ""
            for k in KS:
                out[f"full_R@{k}(%)"] = fmt_pct(row.get(f"full_R@{k}"))
                out[f"full_Δ@{k}(pp)"] = fmt_delta(row.get(f"full_R@{k}"), row["full_R@{k}"] - row[f"full_Δ@{k}"] if row.get(f"full_Δ@{k}") is not None else None) if row.get(f"full_Δ@{k}") is not None else ""
            out["full_comp(%)"] = fmt_pct(row.get("full_comp"))
            out["full_Δcomp(pp)"] = fmt_delta(row.get("full_comp"), row["full_comp"] - row["full_Δcomp"] if row.get("full_Δcomp") is not None else None) if row.get("full_Δcomp") is not None else ""
            out["avg_comp(%)"] = fmt_pct(row.get("avg_comp"))
            out["avg_Δcomp(pp)"] = fmt_delta(row.get("avg_comp"), row["avg_comp"] - row["avg_Δcomp"] if row.get("avg_Δcomp") is not None else None) if row.get("avg_Δcomp") is not None else ""
            writer.writerow(out)
    print(f"CSV written to {path}")


def write_csv_simple(rows: List[Dict[str, Any]], path: Path):
    """Simpler CSV with cleaner delta computation."""
    fieldnames = [
        "rank", "config", "fusion", "filter", "w_meta", "w_sparse", "w_dense",
    ]
    for k in KS:
        fieldnames += [f"hard_R@{k}(%)", f"hard_Δ@{k}(pp)"]
    fieldnames += ["hard_comp(%)", "hard_Δcomp(pp)"]
    for k in KS:
        fieldnames += [f"full_R@{k}(%)", f"full_Δ@{k}(pp)"]
    fieldnames += ["full_comp(%)", "full_Δcomp(pp)", "avg_comp(%)", "avg_Δcomp(pp)"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            out: Dict[str, Any] = {
                "rank": i,
                "config": row["config"],
                "fusion": row["fusion"],
                "filter": row["filter"],
                "w_meta": row["w_meta"],
                "w_sparse": row["w_sparse"],
                "w_dense": row["w_dense"],
            }
            for k in KS:
                v = row.get(f"hard_R@{k}")
                d = row.get(f"hard_Δ@{k}")
                out[f"hard_R@{k}(%)"] = f"{v*100:.2f}" if v is not None else ""
                out[f"hard_Δ@{k}(pp)"] = (f"+{d*100:.2f}" if d >= 0 else f"{d*100:.2f}") if d is not None else ""
            v = row.get("hard_comp")
            d = row.get("hard_Δcomp")
            out["hard_comp(%)"] = f"{v*100:.2f}" if v is not None else ""
            out["hard_Δcomp(pp)"] = (f"+{d*100:.2f}" if d >= 0 else f"{d*100:.2f}") if d is not None else ""

            for k in KS:
                v = row.get(f"full_R@{k}")
                d = row.get(f"full_Δ@{k}")
                out[f"full_R@{k}(%)"] = f"{v*100:.2f}" if v is not None else ""
                out[f"full_Δ@{k}(pp)"] = (f"+{d*100:.2f}" if d >= 0 else f"{d*100:.2f}") if d is not None else ""
            v = row.get("full_comp")
            d = row.get("full_Δcomp")
            out["full_comp(%)"] = f"{v*100:.2f}" if v is not None else ""
            out["full_Δcomp(pp)"] = (f"+{d*100:.2f}" if d >= 0 else f"{d*100:.2f}") if d is not None else ""

            v = row.get("avg_comp")
            d = row.get("avg_Δcomp")
            out["avg_comp(%)"] = f"{v*100:.2f}" if v is not None else ""
            out["avg_Δcomp(pp)"] = (f"+{d*100:.2f}" if d >= 0 else f"{d*100:.2f}") if d is not None else ""
            writer.writerow(out)
    print(f"CSV written to {path}")


def write_markdown(rows: List[Dict[str, Any]], path: Path):
    lines = []
    lines.append("# Hybrid Retrieval Weight Sensitivity Sweep Results\n")
    lines.append(f"Configs: {len(rows)} | Splits: Hard (n=31), Full (n=1013)\n")
    lines.append("Composite = R@1×10% + R@5×15% + R@10×25% + R@25×20% + R@50×15% + R@100×10% + R@200×5%\n")
    lines.append("Δ = difference vs dense-only baseline (percentage points)\n")
    lines.append("")

    # --- Compact table: key metrics only ---
    lines.append("## Summary Table (R@1, R@5, R@10, R@50, Composite)\n")

    hdr = "| # | Config | Fusion | Filter | w_m | w_s | w_d |"
    hdr += " Hard R@1 | Hard R@5 | Hard R@10 | Hard R@50 | Hard Comp |"
    hdr += " Full R@1 | Full R@5 | Full R@10 | Full R@50 | Full Comp |"
    hdr += " Avg Comp | Avg Δ |"
    lines.append(hdr)

    sep = "|---|--------|--------|--------|-----|-----|-----|"
    sep += "----------|----------|-----------|-----------|-----------|"
    sep += "----------|----------|-----------|-----------|-----------|"
    sep += "----------|-------|"
    lines.append(sep)

    for i, row in enumerate(rows, 1):
        def pct(key):
            v = row.get(key)
            return f"{v*100:.2f}" if v is not None else "—"

        def delta(key):
            v = row.get(key)
            if v is None:
                return "—"
            d = v * 100
            return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"

        line = f"| {i} | {row['config']} | {row['fusion']} | {row['filter']} | {row['w_meta']} | {row['w_sparse']} | {row['w_dense']} |"
        for k in [1, 5, 10, 50]:
            line += f" {pct(f'hard_R@{k}')} |"
        line += f" {pct('hard_comp')} |"
        for k in [1, 5, 10, 50]:
            line += f" {pct(f'full_R@{k}')} |"
        line += f" {pct('full_comp')} |"
        line += f" {pct('avg_comp')} | {delta('avg_Δcomp')} |"
        lines.append(line)

    lines.append("")
    lines.append("")

    # --- Full detail table ---
    lines.append("## Full Detail Table (all R@k with Δ)\n")

    hdr2 = "| # | Config |"
    for k in KS:
        hdr2 += f" Hard R@{k} | Δ |"
    hdr2 += " Hard Comp | Δ |"
    for k in KS:
        hdr2 += f" Full R@{k} | Δ |"
    hdr2 += " Full Comp | Δ | Avg Comp | Avg Δ |"
    lines.append(hdr2)

    sep2 = "|---|--------|"
    for _ in KS:
        sep2 += "----------|-----|"
    sep2 += "-----------|-----|"
    for _ in KS:
        sep2 += "----------|-----|"
    sep2 += "-----------|-----|----------|-------|"
    lines.append(sep2)

    for i, row in enumerate(rows, 1):
        def pct(key):
            v = row.get(key)
            return f"{v*100:.2f}" if v is not None else "—"

        def delta(key):
            v = row.get(key)
            if v is None:
                return "—"
            d = v * 100
            return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"

        line = f"| {i} | {row['config']} |"
        for k in KS:
            line += f" {pct(f'hard_R@{k}')} | {delta(f'hard_Δ@{k}')} |"
        line += f" {pct('hard_comp')} | {delta('hard_Δcomp')} |"
        for k in KS:
            line += f" {pct(f'full_R@{k}')} | {delta(f'full_Δ@{k}')} |"
        line += f" {pct('full_comp')} | {delta('full_Δcomp')} |"
        line += f" {pct('avg_comp')} | {delta('avg_Δcomp')} |"
        lines.append(line)

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown written to {path}")


def print_terminal(rows: List[Dict[str, Any]]):
    def pct(v):
        return f"{v*100:6.2f}" if v is not None else "     —"

    def delta(v):
        if v is None:
            return "     —"
        d = v * 100
        return f"+{d:5.2f}" if d >= 0 else f"{d:6.2f}"

    print()
    print("=" * 160)
    print("  COMPREHENSIVE SWEEP RESULTS — sorted by avg composite score")
    print("=" * 160)
    print()

    # Print in two blocks for readability: Hard then Full
    print(f"{'#':>2} {'Config':<42} │ {'Hard':^60} │ {'Full':^60} │ {'Avg':^14}")
    sub = (
        f"{'':>2} {'':42} │ "
        + " ".join(f"{'R@'+str(k):>6}" for k in KS) + f" {'Comp':>6} {'Δ':>6}"
        + " │ "
        + " ".join(f"{'R@'+str(k):>6}" for k in KS) + f" {'Comp':>6} {'Δ':>6}"
        + " │ " + f"{'Comp':>6} {'Δ':>6}"
    )
    print(sub)
    print("─" * 160)

    for i, row in enumerate(rows, 1):
        h_vals = " ".join(pct(row.get(f"hard_R@{k}")) for k in KS)
        h_comp = pct(row.get("hard_comp"))
        h_delta = delta(row.get("hard_Δcomp"))

        f_vals = " ".join(pct(row.get(f"full_R@{k}")) for k in KS)
        f_comp = pct(row.get("full_comp"))
        f_delta = delta(row.get("full_Δcomp"))

        a_comp = pct(row.get("avg_comp"))
        a_delta = delta(row.get("avg_Δcomp"))

        bl = " ←BL" if row["w_meta"] == 0 and row["w_sparse"] == 0 and row["w_dense"] == 1.0 else ""
        print(
            f"{i:>2} {row['config']:<42} │ {h_vals} {h_comp} {h_delta} │ {f_vals} {f_comp} {f_delta} │ {a_comp} {a_delta}{bl}"
        )

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", default="output/QA_Agent/MMRAG/hybrid_sweep")
    args = parser.parse_args()

    base = Path(args.output_base)
    results = load_results(base)
    if not results:
        print("No results found.")
        return

    rows = build_rows(results)

    out_dir = base
    csv_path = out_dir / "sweep_comprehensive.csv"
    md_path = out_dir / "sweep_comprehensive.md"

    write_csv_simple(rows, csv_path)
    write_markdown(rows, md_path)
    print_terminal(rows)


if __name__ == "__main__":
    main()
