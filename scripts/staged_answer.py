#!/usr/bin/env python3
"""Staged INJECT+ answerer - org_dynamic without an agent loop.

Measured motivation (ATM-Bench-Hard, Qwen3.6-27B):
  * org_dynamic AGENT costs $2.00/run, 97% of it in the answer phase
    (5.2M tokens over 254 turns) because context grows every turn.
  * Injecting only retrieval top-10 covers just 37% of gold evidence, which is
    why the no-agent INJECT pipeline scores low and says "Unknown" a lot.
  * The retrieval file already holds 200 candidates/question and top-200 covers
    70% - i.e. what the agent buys with 8.2 turns of grepping is mostly just
    "37% -> 70% evidence coverage", which is FREE to hand over.

Answers in 1-2 fixed-size calls instead of an agent loop:
  Layer 1  type-conditioned evidence budget (wide ranked pool + full records
           where raw detail, not the caption, decides the answer)
  Layer 2  deterministic metadata filter, zero LLM cost - the pre-extracted
           query map's locations/date-window, applied to the whole corpus.
           This is the agent's grep, for free, and reaches past the retriever.
  Layer 3  sufficiency gate: the call also returns {sufficient, missing}; if
           insufficient we widen the evidence once and re-ask. Bounded at 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/rds/user/kz345/hpc-work/REMMI")

from remmi.organize.agent_static import USAGE_TOTALS, _parse_json_object, chat

# Budgets from measured gold-coverage: number @200=72%, list @200=74%,
# open @200=67% (vs @10 = 37% overall). Gold sets are small (median 3-7), so a
# wide ranked pool is cheap and is what lifts coverage.
import os as _os

BUDGET = {
    "number":      {"compact": 200, "full": 30},
    "list_recall": {"compact": 200, "full": 0},
    "open_end":    {"compact": 150, "full": 12},
}
DEFAULT_BUDGET = {"compact": 150, "full": 12}

# Env override so the same script can run ablations (e.g. SA_BUDGET="10,0"
# reproduces the plain INJECT top-10 single-call baseline).
_B = _os.environ.get("SA_BUDGET")
if _B:
    _c, _f = [int(x) for x in _B.split(",")]
    BUDGET = {k: {"compact": _c, "full": _f} for k in BUDGET}
    DEFAULT_BUDGET = {"compact": _c, "full": _f}
MAX_METADATA_ADD = int(_os.environ.get("SA_MAXADD", 120))


def _stem(p):
    return Path(p).stem if p else ""


def load_items(image_src, video_src, email_src):
    out = {}
    for src, key, kind in ((image_src, "image_path", "image"), (video_src, "video_path", "video")):
        for r in json.loads(Path(src).read_text()):
            iid = _stem(r.get(key, ""))
            if not iid:
                continue
            out[iid] = {"id": iid, "kind": kind,
                        "ts": str(r.get("timestamp") or "")[:16],
                        "city": str(r.get("city") or ""),
                        "loc": str(r.get("location_name") or ""),
                        "cap": str(r.get("short_caption") or r.get("caption") or ""),
                        "tags": ", ".join(r.get("tags") or [])[:120],
                        "detail": str(r.get("ocr_text") or "")[:300]}
    for r in json.loads(Path(email_src).read_text()):
        eid = str(r.get("id") or "")
        if not eid:
            continue
        out[eid] = {"id": eid, "kind": "email",
                    "ts": str(r.get("timestamp") or "")[:16],
                    "city": "", "loc": "",
                    "cap": str(r.get("short_summary") or ""),
                    "tags": "", "detail": str(r.get("detail") or "")[:600]}
    return out


def line_compact(rank, it):
    return "[%d] %s | %-5s | %-16s | %-22s | %s" % (
        rank, it["id"], it["kind"], it["ts"], (it["city"] or it["loc"])[:22], it["cap"][:70])


def line_full(it):
    bits = ["== %s (%s) %s %s %s" % (it["id"], it["kind"], it["ts"], it["city"], it["loc"]),
            "   caption: %s" % it["cap"][:200]]
    if it["tags"]:
        bits.append("   tags: %s" % it["tags"])
    if it["detail"]:
        bits.append("   detail: %s" % it["detail"])
    return "\n".join(bits)


def metadata_matches(items, constraints):
    """LAYER 2 - deterministic filter, no LLM."""
    locs = [str(x).lower() for x in (constraints.get("locations") or []) if x]
    ds, de = constraints.get("date_start"), constraints.get("date_end")
    if not locs and not ds and not de:
        return []
    hits = []
    for iid, it in items.items():
        if ds and (not it["ts"] or it["ts"][:10] < ds):
            continue
        if de and (not it["ts"] or it["ts"][:10] > de):
            continue
        if locs:
            hay = ("%s %s %s %s" % (it["city"], it["loc"], it["cap"], it["tags"])).lower()
            if not any(l in hay for l in locs):
                continue
        hits.append(iid)
    return hits


PROMPT = """You answer questions about a personal memory archive.

EVIDENCE - memory items retrieved for THIS question, ranked by relevance
(rank 1 = most relevant). Lower-ranked items are still often correct; judge by
the content, not only the rank.
{compact}
{full_block}{events_block}
QUESTION: {question}

ANSWER RULES
- Use ONLY the evidence above. If it genuinely does not contain the answer, say so
  via "sufficient": false rather than guessing.
- Item ID = the id shown in the evidence (e.g. 20240701_120945, email202201010001).
{type_rule}

Reply with ONLY a JSON object:
{{"answer": "<the answer>", "sufficient": true|false, "missing": "<what evidence is still needed>"}}"""

TYPE_RULE = {
    "list_recall": ("- LIST question, scored by set overlap: a missing id and a wrong id cost the\n"
                    "  same. Include every id that truly satisfies the question and NO others -\n"
                    "  do not pad with 'probably related' items.\n"
                    "- `answer` must be ONLY the ids, comma-separated."),
    "number": ("- NUMBER question. Gather every relevant record first, list the values you are\n"
               "  combining, then compute explicitly (count DISTINCT days, sum amounts).\n"
               "- `answer` must be the number (with units if applicable)."),
    "open_end": "- Answer concisely and concretely, grounded in the evidence.",
}


def build_prompt(q, ordered, items, events, widen):
    qtype = q.get("qtype", "open_end")
    b = dict(BUDGET.get(qtype, DEFAULT_BUDGET))
    if widen:
        b["compact"] = min(len(ordered), int(b["compact"] * 1.8) + 50)
        b["full"] = b["full"] + 20
    sel = [i for i in ordered if i in items][: b["compact"]]
    compact = "\n".join(line_compact(n, items[i]) for n, i in enumerate(sel, 1))
    full_block = ""
    if b["full"]:
        fu = [line_full(items[i]) for i in sel[: b["full"]]]
        full_block = "\nFULL RECORDS for the top candidates (detail matters here):\n" + "\n".join(fu) + "\n"
    events_block = ""
    if events and events.get("events"):
        ev = "\n".join("  - %s: %s" % (e.get("label", "event"),
                                       ", ".join(x["id"] for x in e.get("items", [])[:12]))
                       for e in events["events"][:8])
        events_block = "\nPRE-ORGANISED EVENTS for this question:\n" + ev + "\n"
    return PROMPT.format(compact=compact, full_block=full_block, events_block=events_block,
                         question=q["question"], type_rule=TYPE_RULE.get(qtype, TYPE_RULE["open_end"]))


def answer_one(q, ordered, items, events, base_url, model, extra_body, retry):
    calls = 0
    for widen in (False, True):
        prompt = build_prompt(q, ordered, items, events, widen)
        raw = chat([{"role": "user", "content": prompt}], base_url=base_url, model=model,
                   extra_body=extra_body)
        calls += 1
        parsed = _parse_json_object(raw) or {}
        ans = str(parsed.get("answer") or "").strip()
        ok = bool(parsed.get("sufficient", True))
        if ok and ans:
            return {"id": q["id"], "question": q["question"], "answer": ans,
                    "calls": calls, "widened": widen, "sufficient": True}
        if not retry or widen:
            return {"id": q["id"], "question": q["question"], "answer": ans or "Unknown",
                    "calls": calls, "widened": widen, "sufficient": ok,
                    "missing": str(parsed.get("missing") or "")[:200]}
    return {"id": q["id"], "question": q["question"], "answer": "Unknown", "calls": calls}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--query-map", default="")
    ap.add_argument("--events-dir", default="")
    ap.add_argument("--image-source", default="output/image/qwen3vl2b/batch_results.json")
    ap.add_argument("--video-source", default="output/video/qwen3vl2b/batch_results.json")
    ap.add_argument("--emails-source", default="data/raw_memory/email/emails.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-layer2", action="store_true")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--price-in", type=float, default=0.29)
    ap.add_argument("--price-out", type=float, default=3.20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else None

    items = load_items(Path(args.image_source), Path(args.video_source), Path(args.emails_source))
    questions = json.loads(Path(args.questions).read_text())
    ret = {}
    for line in Path(args.retrieval).read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            ret[x["id"]] = x.get("retrieval_ids", [])
    qmap = {}
    if args.query_map and Path(args.query_map).exists():
        qm = json.loads(Path(args.query_map).read_text())
        qmap = qm.get("by_qid", qm)
    events_dir = Path(args.events_dir) if args.events_dir else None
    print("corpus=%d questions=%d retrieval=%d qmap=%d" % (len(items), len(questions), len(ret), len(qmap)))

    def work(q):
        ordered = list(ret.get(q["id"], []))
        added = 0
        if not args.no_layer2:
            seen = set(ordered)
            for iid in metadata_matches(items, qmap.get(q["id"], {}) or {}):
                if iid not in seen and added < MAX_METADATA_ADD:
                    ordered.append(iid)
                    seen.add(iid)
                    added += 1
        ev = None
        if events_dir and (events_dir / ("%s.json" % q["id"])).exists():
            try:
                ev = json.loads((events_dir / ("%s.json" % q["id"])).read_text())
            except Exception:
                ev = None
        r = answer_one(q, ordered, items, ev, args.base_url, args.model, extra_body, not args.no_retry)
        r["layer2_added"] = added
        return r

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, q) for q in questions]):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["id"])

    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "question": r["question"],
                                 "answer": r["answer"]}, ensure_ascii=False) + "\n")
    ti, to = USAGE_TOTALS["input_tokens"], USAGE_TOTALS["output_tokens"]
    cost = ti * args.price_in / 1e6 + to * args.price_out / 1e6
    stats = {"n": len(rows), "calls": sum(r.get("calls", 1) for r in rows),
             "widened": sum(1 for r in rows if r.get("widened")),
             "insufficient": sum(1 for r in rows if r.get("sufficient") is False),
             "layer2_added_mean": round(sum(r.get("layer2_added", 0) for r in rows) / max(len(rows), 1), 1),
             "input_tokens": ti, "output_tokens": to, "cost_usd": round(cost, 4),
             "model": args.model, "layer2": not args.no_layer2, "retry": not args.no_retry}
    json.dump(stats, open(out_dir / "usage.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
