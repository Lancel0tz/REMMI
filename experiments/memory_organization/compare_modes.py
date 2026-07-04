#!/usr/bin/env python3
"""Compare memory-organization modes: tokens, QS, and auxiliary metrics.

Scans the per-question run artifacts of each mode (answer.json + usage.json)
and, when ATM evaluation has been run, joins the judge scores. Emits a
markdown table plus per-question extremes.

Usage (from repo root):
    python3 experiments/memory_organization/compare_modes.py
    python3 experiments/memory_organization/compare_modes.py --run-tag atm-bench-hard --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

def modes_for(base_tag: str) -> list[tuple[str, str, str]]:
    return [
        # label, eval_root, model_tag
        ("sgm (baseline)", "agent_systems/eval_root_sgm", base_tag),
        ("org_heuristic", "agent_systems/eval_root_orgh", f"{base_tag}-orgh"),
        ("org_static", "agent_systems/eval_root_orgs", f"{base_tag}-orgs"),
        ("org_dynamic", "agent_systems/eval_root_orgd", f"{base_tag}-orgd"),
        ("org_hybrid", "agent_systems/eval_root_orgx", f"{base_tag}-orgx"),
        ("org_remmi", "agent_systems/eval_root_orgr", f"{base_tag}-orgr"),
        ("org_remmi2", "agent_systems/eval_root_orgr2", f"{base_tag}-orgr2"),
    ]


RESULTS_ROOT = "output/QA_Agent/AgentSystems"
AGENT = "codex"


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_mode(eval_root: str, model_tag: str, run_tag: str) -> dict[str, Any]:
    base = Path(eval_root) / "runs" / run_tag / AGENT / model_tag
    rows: list[dict[str, Any]] = []
    for qdir in sorted(base.glob("*/")):
        out = qdir / "output"
        row: dict[str, Any] = {"qid": qdir.name}
        answer_file = out / "answer.json"
        usage_file = out / "usage.json"
        if answer_file.exists():
            try:
                answer = read_json(answer_file).get("answer", "")
            except Exception:
                answer = ""
            row["answered"] = bool(str(answer).strip())
            row["unknown"] = str(answer).strip().lower() in ("unknown", "unknown.")
            row["answer_len"] = len(str(answer))
        if usage_file.exists():
            usage = read_json(usage_file)
            row["input_tokens"] = usage.get("input_tokens") or 0
            row["output_tokens"] = usage.get("output_tokens") or 0
            row["total_tokens"] = usage.get("total_tokens") or 0
        trace_file = out / "trace.jsonl"
        if trace_file.exists():
            # billed input includes cached context re-sends; uncached = new content
            uncached = 0
            try:
                with open(trace_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if '"turn.completed"' not in line:
                            continue
                        turn_usage = json.loads(line).get("usage", {})
                        uncached += turn_usage.get("input_tokens", 0) - turn_usage.get("cached_input_tokens", 0)
                row["uncached_input_tokens"] = uncached
            except Exception:
                pass
        rows.append(row)
    return {"rows": rows}


def find_qs(model_tag: str, run_tag: str) -> tuple[float | None, dict[str, float]]:
    """Return (overall QS, per-question scores) from ATM eval outputs, if present."""
    eval_dir = Path(RESULTS_ROOT) / run_tag / AGENT / model_tag / "eval"
    per_question: dict[str, float] = {}
    overall: float | None = None
    for summary_path in sorted(eval_dir.glob("atm_*_summary.json")):
        summary = read_json(summary_path)
        flat = json.dumps(summary)
        for key in ("accuracy", "overall_score", "atm_score", "average_score", "score", "qs"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                overall = float(value)
                break
        if overall is None and isinstance(summary.get("overall"), dict):
            for key in ("score", "atm", "qs", "mean"):
                value = summary["overall"].get(key)
                if isinstance(value, (int, float)):
                    overall = float(value)
                    break
        _ = flat
    for detail_path in sorted(eval_dir.glob("atm_*.json")):
        if detail_path.name.endswith("_summary.json"):
            continue
        try:
            for entry in read_json(detail_path):
                qid = entry.get("id") or entry.get("qa_id") or entry.get("question_id")
                score = entry.get("accuracy", entry.get("score", entry.get("atm_score")))
                if qid is not None and isinstance(score, (int, float)):
                    per_question[str(qid)] = float(score)
        except Exception:
            continue
    if overall is None and per_question:
        overall = statistics.mean(per_question.values())
    return overall, per_question


def fmt_tokens(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="atm-bench-hard")
    parser.add_argument("--model-base", default="gpt-5-mini-medium", help="base model tag (mode suffixes appended)")
    parser.add_argument("--json", type=Path, help="also dump the raw comparison as JSON")
    args = parser.parse_args()

    table: list[dict[str, Any]] = []
    for label, eval_root, model_tag in modes_for(args.model_base):
        data = collect_mode(eval_root, model_tag, args.run_tag)
        rows = [r for r in data["rows"] if r.get("total_tokens")]
        if not rows:
            table.append({"label": label, "n": 0})
            continue
        qs, per_q = find_qs(model_tag, args.run_tag)
        totals = [r["total_tokens"] for r in rows]
        outs = [r["output_tokens"] for r in rows]
        uncached = [r["uncached_input_tokens"] + r["output_tokens"] for r in rows if "uncached_input_tokens" in r]
        entry: dict[str, Any] = {
            "label": label,
            "n": len(rows),
            "answered": sum(1 for r in rows if r.get("answered")),
            "unknown_rate": sum(1 for r in rows if r.get("unknown")) / len(rows),
            "total_tokens": sum(totals),
            "mean_tokens": statistics.mean(totals),
            "median_tokens": statistics.median(totals),
            "max_tokens": max(totals),
            "output_tokens": sum(outs),
            "mean_uncached": statistics.mean(uncached) if uncached else None,
            "qs": qs,
            "per_question_qs": per_q,
        }
        if qs:
            entry["tokens_per_qs_point"] = sum(totals) / (qs * 100 if qs <= 1 else qs)
        table.append(entry)

    print(f"\n## Memory-organization comparison — {args.run_tag} (codex / {args.model_base})\n")
    print("| Mode | Qs | QS | Billed tok | Mean/Q | Uncached/Q | Max/Q | Unknown% | Tok/QS-pt |")
    print("|------|---:|---:|-----------:|-------:|-----------:|------:|---------:|----------:|")
    for e in table:
        if not e.get("n"):
            print(f"| {e['label']} | 0 | — | — | — | — | — | — | — |")
            continue
        qs_str = f"{e['qs'] * 100:.1f}" if e.get("qs") and e["qs"] <= 1 else (f"{e['qs']:.1f}" if e.get("qs") else "pending")
        tps = fmt_tokens(e["tokens_per_qs_point"]) if e.get("tokens_per_qs_point") else "—"
        unc = fmt_tokens(e["mean_uncached"]) if e.get("mean_uncached") else "—"
        print(
            f"| {e['label']} | {e['n']} | {qs_str} | {fmt_tokens(e['total_tokens'])} "
            f"| {fmt_tokens(e['mean_tokens'])} | {unc} "
            f"| {fmt_tokens(e['max_tokens'])} | {e['unknown_rate'] * 100:.0f}% | {tps} |"
        )

    print("\n(QS 'pending' = ATM judge not yet run for that mode.)")

    if args.json:
        args.json.write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
        print(f"raw comparison -> {args.json}")


if __name__ == "__main__":
    main()
