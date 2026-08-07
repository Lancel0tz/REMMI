#!/usr/bin/env python3
"""Dissertation analyses over the 3-replicate memory-organization runs.

Subcommands (all read-only over existing run artifacts):
  frontier  (e)  cost-vs-QS scatter, n=3 error bars on both axes -> figures/
  tools     (b)  tool-choice statistics from codex traces (search.py vs direct scan)
  variance  (#1) run-to-run variance structure by question type
  evidence  (#3) gold-evidence accessibility on number questions (v9, v11)
  all            run everything

Usage (from repo root):
    python3 experiments/memory_organization/analysis.py all
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "memory_organization"))
from compare_modes import load_pricing, rates_for  # noqa: E402

TAGS = ["atm-bench-hard", "atm-bench-hard-r2", "atm-bench-hard-r3"]
MODELS = ["gpt-5-mini-medium", "gpt-5.5-medium"]
# short id -> (display label, model-tag suffix, family)
MODES = {
    "sgm":   ("sgm (baseline)", "", "baseline"),
    "orgh":  ("org_heuristic", "-orgh", "index"),
    "orgs":  ("org_static", "-orgs", "index"),
    "orgx":  ("org_hybrid", "-orgx", "index"),
    "orgd":  ("org_timeline", "-orgd", "self-org"),
    "orgr":  ("org_remmi (v1)", "-orgr", "tool"),
    "orgr8": ("remmi_weak (v8)", "-orgr8", "tool"),
    "orgr9": ("remmi (v9)", "-orgr9", "tool"),
}
QTYPES = ("number", "list_recall", "open_end")
FIGDIR = REPO / "experiments" / "memory_organization" / "figures"


def load_questions():
    qs = json.load(open(REPO / "data/atm-bench/atm-bench-hard.json"))
    qtype = {q["id"]: q["qtype"] for q in qs}
    evidence = {}
    for q in qs:
        raw = q.get("evidence_ids") or "[]"
        try:
            evidence[q["id"]] = [str(e) for e in ast.literal_eval(raw)] if isinstance(raw, str) else list(raw)
        except Exception:
            evidence[q["id"]] = []
    return qtype, evidence


def eval_scores(tag, model_tag):
    p = REPO / f"output/QA_Agent/AgentSystems/{tag}/codex/{model_tag}/eval/atm_gpt-5-mini.json"
    if not p.exists():
        return {}
    return {r["id"]: float(r["accuracy"]) for r in json.load(open(p))}


def run_dir(short, tag, model_tag, qid):
    return REPO / f"agent_systems/eval_root_{short}/runs/{tag}/codex/{model_tag}/{qid}"


def per_run_cost(short, tag, model_tag, pricing):
    """Mean USD/question for one leg, replicating compare_modes' formula."""
    rates = rates_for(model_tag, pricing)
    base = REPO / f"agent_systems/eval_root_{short}/runs/{tag}/codex/{model_tag}"
    costs = []
    for qdir in sorted(base.glob("*/")):
        tr = qdir / "output/trace.jsonl"
        if not tr.exists():
            continue
        unc = cached = out = 0
        for line in open(tr):
            if '"turn.completed"' not in line:
                continue
            u = json.loads(line).get("usage", {})
            cached += u.get("cached_input_tokens", 0)
            unc += u.get("input_tokens", 0) - u.get("cached_input_tokens", 0)
            out += u.get("output_tokens", 0)
        if unc + cached + out:
            costs.append(unc / 1e6 * rates["input"] + cached / 1e6 * rates["cache_read"] + out / 1e6 * rates["output"])
    return statistics.mean(costs) if costs else None


def mode_runs(model_tag_base):
    """{short: [(qs_overall, {qtype: score}, tag), ...]} using per-question evals."""
    qtype, _ = load_questions()
    out = defaultdict(list)
    for short, (_label, suffix, _fam) in MODES.items():
        mt = model_tag_base + suffix
        for tag in TAGS:
            scores = eval_scores(tag, mt)
            if len(scores) < 31:
                continue
            by = defaultdict(list)
            for qid, acc in scores.items():
                by[qtype.get(qid, "?")].append(acc)
            out[short].append((
                100 * statistics.mean(scores.values()),
                {qt: 100 * statistics.mean(v) for qt, v in by.items()},
                tag,
            ))
    return out


# ---------------------------------------------------------------- frontier (e)

def cmd_frontier():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pricing = load_pricing()
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    FAMC = {"baseline": "#777777", "index": "#1f77b4", "self-org": "#2ca02c", "tool": "#d62728"}

    rows = []
    for ax, base, title in ((axes[0], "gpt-5-mini-medium", "gpt-5-mini"),
                            (axes[1], "gpt-5.5-medium", "gpt-5.5")):
        runs = mode_runs(base)
        for short, (label, suffix, fam) in MODES.items():
            rr = runs.get(short, [])
            if len(rr) < 2:
                continue
            qss = [r[0] for r in rr]
            costs = [per_run_cost(short, r[2], base + suffix, pricing) for r in rr]
            costs = [c for c in costs if c]
            mq, sq = statistics.mean(qss), statistics.stdev(qss)
            mc = statistics.mean(costs)
            sc = statistics.stdev(costs) if len(costs) > 1 else 0
            ax.errorbar(mc, mq, xerr=sc, yerr=sq, fmt="o", color=FAMC[fam], capsize=3, ms=6)
            ax.annotate(label.split(" (")[0], (mc, mq), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
            rows.append((title, label, mq, sq, mc, sc))
        # inject reference (server, n=3); 5.5 subscription run has no billed tokens
        if base.startswith("gpt-5-mini"):
            ax.errorbar(0.0067, 14.1, yerr=1.6, fmt="s", color="#9467bd", capsize=3, ms=6)
            ax.annotate("inject (org_dynamic)", (0.0067, 14.1), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
            rows.append((title, "org_dynamic (inject)", 14.1, 1.6, 0.0067, 0.0))
        ax.set_xscale("log")
        ax.set_xlabel("Cost per question (USD, API-equivalent)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("QS (mean of 3 runs)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMC.items()]
               + [Line2D([], [], marker="s", ls="", color="#9467bd", label="inject")],
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("ATM-Bench-Hard: cost vs quality (n=3, ±sd both axes)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"frontier.{ext}", dpi=200)
    print(f"\n## (e) frontier -> {FIGDIR}/frontier.png|pdf\n")
    print("| model | system | QS | ±sd | Cost/Q | ±sd |")
    print("|---|---|---:|---:|---:|---:|")
    for m, label, mq, sq, mc, sc in rows:
        print(f"| {m} | {label} | {mq:.1f} | {sq:.1f} | ${mc:.4f} | {sc:.4f} |")
    print("\n(5.5 inject omitted from the cost axis: subscription run, no billed tokens.)")


# ------------------------------------------------------------------- tools (b)

SEARCH_RE = re.compile(r"search\.py\s+(?!--show)")
SHOW_RE = re.compile(r"search\.py\s+--show")
DIRECT_RE = re.compile(r"\b(rg|grep|jq|sed|awk)\b[^|]*memory/(image_metadata|video_metadata|emails)"
                       r"|open\(['\"]memory/(image_metadata|video_metadata|emails)"
                       r"|json\.load\([^)]*memory/")


def commands_for(short, tag, model_tag, qid):
    tr = run_dir(short, tag, model_tag, qid) / "output/trace.jsonl"
    cmds = []
    if not tr.exists():
        return cmds
    for line in open(tr):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "command_execution":
            cmds.append((e["item"].get("command") or "", e["item"].get("aggregated_output") or ""))
    return cmds


def cmd_tools():
    qtype, _ = load_questions()
    print("\n## (b) tool-choice statistics (codex traces, 3 runs each)\n")
    print("| model | mode | adoption %Q | search/Q | --show/Q | direct-scan/Q |")
    print("|---|---|---:|---:|---:|---:|")
    detail = {}
    for base in MODELS:
        for short in ("orgr8", "orgr9"):
            label, suffix, _ = MODES[short]
            mt = base + suffix
            per_q = defaultdict(lambda: [0, 0, 0, 0])  # search, show, direct, runs-seen
            for tag in TAGS:
                for qid in qtype:
                    cmds = commands_for(short, tag, mt, qid)
                    if not cmds:
                        continue
                    s = sum(1 for c, _o in cmds if SEARCH_RE.search(c))
                    sh = sum(1 for c, _o in cmds if SHOW_RE.search(c))
                    d = sum(1 for c, _o in cmds if DIRECT_RE.search(c))
                    per_q[qid][0] += s; per_q[qid][1] += sh; per_q[qid][2] += d; per_q[qid][3] += 1
            n_runs = sum(v[3] for v in per_q.values())
            if not n_runs:
                continue
            adopt = 100 * sum(1 for v in per_q.values() if v[0] + v[1] > 0) / len(per_q)
            tot = lambda i: sum(v[i] for v in per_q.values()) / n_runs
            print(f"| {base.split('-medium')[0]} | {label} | {adopt:.0f}% | {tot(0):.1f} | {tot(1):.1f} | {tot(2):.1f} |")
            detail[(base, short)] = per_q
    # qtype split for v9
    print("\nremmi (v9) by question type (per-question-run means):\n")
    print("| model | qtype | search/Q | direct/Q | search-share |")
    print("|---|---|---:|---:|---:|")
    for base in MODELS:
        per_q = detail.get((base, "orgr9"), {})
        agg = defaultdict(lambda: [0, 0, 0])
        for qid, (s, sh, d, n) in per_q.items():
            qt = qtype[qid]
            agg[qt][0] += s + sh; agg[qt][1] += d; agg[qt][2] += n
        for qt in QTYPES:
            tool, direct, n = agg[qt]
            if not n:
                continue
            share = 100 * tool / (tool + direct) if tool + direct else 0
            print(f"| {base.split('-medium')[0]} | {qt} | {tool/n:.1f} | {direct/n:.1f} | {share:.0f}% |")


# ---------------------------------------------------------------- variance (#1)

def cmd_variance():
    print("\n## (#1) run-to-run variance structure (sd over 3 runs)\n")
    print("| model | mode | sd(QS) | sd(number) | sd(recall) | sd(open) |")
    print("|---|---|---:|---:|---:|---:|")
    open_sds, num_sds, rec_sds = [], [], []
    fam_sd = defaultdict(list)
    for base in MODELS:
        runs = mode_runs(base)
        for short, (label, _suf, fam) in MODES.items():
            rr = runs.get(short, [])
            if len(rr) < 3:
                continue
            sq = statistics.stdev([r[0] for r in rr])
            sds = {qt: statistics.stdev([r[1].get(qt, 0) for r in rr]) for qt in QTYPES}
            print(f"| {base.split('-medium')[0]} | {label} | {sq:.1f} | {sds['number']:.1f} "
                  f"| {sds['list_recall']:.1f} | {sds['open_end']:.1f} |")
            num_sds.append(sds["number"]); rec_sds.append(sds["list_recall"]); open_sds.append(sds["open_end"])
            if base.startswith("gpt-5-mini"):
                fam_sd["mandate" if short in ("orgr", "orgr8") else
                       "free" if short in ("orgr9", "orgd") else "other"].append(sq)
    med = statistics.median
    print(f"\nmedian sd across all 16 mode×model cells — number: {med(num_sds):.1f}, "
          f"recall: {med(rec_sds):.1f}, open_end: {med(open_sds):.1f}")
    if fam_sd["mandate"] and fam_sd["free"]:
        print(f"mini QS-sd, mandate family (v1,v8): {[f'{x:.1f}' for x in fam_sd['mandate']]} "
              f"vs free family (v9,timeline): {[f'{x:.1f}' for x in fam_sd['free']]}")
    print("inject (server, n=3 per family): number = 16.7 ± 0.0 in all 9 runs; "
          "open_end sd 4.4–8.9 (see LEADERBOARD).")


# ---------------------------------------------------------------- evidence (#3)

def cmd_evidence():
    qtype, evidence = load_questions()
    numbers = [q for q, t in qtype.items() if t == "number"]
    print("\n## (#3) gold-evidence accessibility on number questions "
          f"({len(numbers)} Qs; surfaced = any gold id appears in a command output)\n")
    print("| model | mode | runs×Q | correct | wrong, evidence surfaced | wrong, never surfaced |")
    print("|---|---|---:|---:|---:|---:|")
    detail_rows = []
    for base in MODELS:
        for short in ("orgr9", "orgr11"):
            label = "remmi (v9)" if short == "orgr9" else "org_remmi11 (ocr corpus)"
            mt = base + ("-orgr9" if short == "orgr9" else "-orgr11")
            tags = TAGS if short == "orgr9" else ["atm-bench-hard"]
            c = w_s = w_n = 0
            for tag in tags:
                scores = eval_scores(tag, mt)
                if not scores:
                    continue
                for qid in numbers:
                    cmds = commands_for(short, tag, mt, qid)
                    if not cmds:
                        continue
                    blob = "\n".join(cc + "\n" + oo for cc, oo in cmds)
                    surfaced = any(g in blob for g in evidence.get(qid, []))
                    ok = scores.get(qid, 0) >= 0.99
                    if ok:
                        c += 1
                    elif surfaced:
                        w_s += 1
                    else:
                        w_n += 1
                    detail_rows.append((base.split("-medium")[0], short, tag[-2:] if tag[-2] == 'r' else 'r1',
                                        qid[:8], "✓" if ok else ("surfaced" if surfaced else "MISSED")))
            tot = c + w_s + w_n
            if tot:
                print(f"| {base.split('-medium')[0]} | {label} | {tot} | {c} | {w_s} | {w_n} |")
    print("\nReading: 'wrong, evidence surfaced' = the agent SAW a gold item and still "
          "answered wrong (aggregation/reasoning failure); 'never surfaced' = the gold "
          "evidence was never in any command output (access failure).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["frontier", "tools", "variance", "evidence", "all"])
    args = ap.parse_args()
    fns = {"frontier": cmd_frontier, "tools": cmd_tools, "variance": cmd_variance, "evidence": cmd_evidence}
    if args.cmd == "all":
        for fn in fns.values():
            fn()
    else:
        fns[args.cmd]()


if __name__ == "__main__":
    main()
