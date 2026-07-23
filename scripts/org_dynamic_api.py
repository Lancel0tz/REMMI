#!/usr/bin/env python3
"""org_dynamic (canonical INJECT) with an OpenAI-family answerer.

Pipeline, matching the canonical definition used elsewhere in this repo:
    retrieve (frozen, offline)  ->  ORGANISE top-N into question-relevant events
    (one stateless LLM call)    ->  INJECT those events as the evidence
                                ->  ANSWER in ONE call. No agent loop.

Both stages use the SAME model, per the project's fairness rule (an organiser
stronger than the answerer inflates the method).

Two answerer backends:
  --backend api     OpenAI-compatible HTTP (gpt-5-mini; billed per token)
  --backend codex   Codex CLI with the ChatGPT-account login (gpt-5.5; draws on
                    the subscription, no per-token bill). Used for text-only
                    completion, so codex's broken-on-this-cluster command sandbox
                    is never exercised.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/rds/user/kz345/hpc-work/REMMI")
from remmi.organize.agent_static import USAGE_TOTALS, _parse_json_object, chat
from remmi.organize.dynamic import organize_question, _meta_map

ANSWER_PROMPT = """You answer questions about a personal memory archive.

The relevant memory has ALREADY been retrieved and organised into events for THIS
question. Answer from it.

PRE-ORGANISED EVENTS:
{events}

QUESTION: {question}

RULES
- Use ONLY the evidence above; if it does not contain the answer, say "Unknown".
- Item ID = the id shown (e.g. 20240701_120945, email202201010001).
{type_rule}

Reply with ONLY: {{"answer": "<the answer>"}}"""

TYPE_RULE = {
    "list_recall": "- LIST question: `answer` = ONLY the ids, comma-separated. A missing id and a\n  wrong id cost the same, so include exactly those that match.",
    "number": ("- NUMBER question. The value you need is usually inside a receipt's raw OCR\n"
               "  text or an email body shown under `raw:`, and often must be SUMMED across\n"
               "  several records (e.g. three dental receipts of 50 + 70 + 70).\n"
               "- Do the arithmetic silently. `answer` must contain ONLY the final value --\n"
               "  never the working. Write \"\u00a3190\", NOT \"50 + 70 + 70 = \u00a3190\";\n"
               "  write \"9 days\", NOT \"5 days + 4 days = 9 days\".\n"
               "- Keep the archive's own format: ORIGINAL currency symbol (\u00a3190, \u20ac62.8),\n"
               "  no padded decimals (\u00a3190 not 190.00), and a unit for durations.\n"
               "- If several currencies appear, convert to the one the question implies\n"
               "  rather than returning a mixed expression."),
    "open_end": "- Answer concisely and concretely.",
}


def codex_complete(prompt: str, model: str, codex_bin: str, timeout: int = 900) -> str:
    """One text-only completion through the Codex CLI (ChatGPT-account quota)."""
    with tempfile.TemporaryDirectory(prefix="cx-") as ws:
        p = subprocess.run(
            [codex_bin, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
             "-m", model, "-c", "model_reasoning_effort=low", prompt],
            cwd=ws, capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def render_events(ev: dict, detail_map: dict | None = None, qtype: str = "open_end") -> str:
    """Render organised events as evidence.

    The captions alone are NOT enough for number questions: prices, totals and
    durations live in a receipt photo's `ocr_text` or an email's `detail`, never
    in the one-line caption. Dropping those fields made every number question
    answer "Unknown" even when the gold receipts were ranked 1-2-3. So attach the
    raw detail, with a budget: number questions get it for many items, list/open
    questions stay compact (they are scored on ids/prose, not on amounts).
    """
    n_detail = {"number": 25, "open_end": 0, "list_recall": 0}.get(qtype, 0)
    out, shown = [], 0
    for e in ev.get("events", []):
        items = ", ".join("%s(%s %s)" % (i["id"], i.get("time", ""), i.get("place", ""))
                          for i in e.get("items", [])[:25])
        out.append("- %s: %s" % (e.get("label", "event"), items))
        for i in e.get("items", [])[:25]:
            if i.get("caption"):
                out.append("    %s: %s" % (i["id"], i["caption"][:90]))
            d = (detail_map or {}).get(i["id"])
            if d and shown < n_detail:
                out.append("      raw: %s" % d[:700].replace("\n", " | "))
                shown += 1
    return "\n".join(out) if out else "(the organiser produced no events for this question)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=("api", "codex"), default="api")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--codex-bin", default=os.path.expanduser("~/.npm-global/bin/codex"))
    ap.add_argument("--image-source", default="output/image/qwen3vl2b/batch_results.json")
    ap.add_argument("--video-source", default="output/video/qwen3vl2b/batch_results.json")
    ap.add_argument("--emails-source", default="data/raw_memory/email/emails.json")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--price-in", type=float, default=0.25)
    ap.add_argument("--price-out", type=float, default=2.00)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    evd = out / "events"; evd.mkdir(exist_ok=True)
    meta = _meta_map(Path(args.image_source), Path(args.video_source), Path(args.emails_source))
    # id -> raw detail (ocr_text for photos/videos, body for emails) so amounts survive
    detail_map = {}
    for src, key in ((args.image_source, "image_path"), (args.video_source, "video_path")):
        for r in json.loads(Path(src).read_text()):
            iid = Path(str(r.get(key, ""))).stem
            if iid and r.get("ocr_text"):
                detail_map[iid] = str(r["ocr_text"])
    for r in json.loads(Path(args.emails_source).read_text()):
        if r.get("id") and r.get("detail"):
            detail_map[str(r["id"])] = str(r["detail"])
    print("  detail_map: %d items carry raw ocr/body text" % len(detail_map))
    qs = json.loads(Path(args.questions).read_text())
    ret = {}
    for l in Path(args.retrieval).read_text().splitlines():
        if l.strip():
            x = json.loads(l); ret[x["id"]] = x.get("retrieval_ids", [])
    print("meta=%d questions=%d backend=%s model=%s" % (len(meta), len(qs), args.backend, args.model))

    # ---- stage 1: organise (same model as the answerer) --------------------
    def org(q):
        ev = organize_question(q["id"], q["question"], ret.get(q["id"], []), meta,
                               base_url=args.base_url, model=args.model, topn=args.topn)
        (evd / ("%s.json" % q["id"])).write_text(json.dumps(ev, ensure_ascii=False, indent=1))
        return q["id"], ev
    events = {}
    if args.backend == "api":
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(org, q) for q in qs]):
                i, e = f.result(); events[i] = e
    else:
        # Organise through the SAME subscription model as the answerer. Reusing
        # another model's events would reintroduce the organiser/answerer
        # mismatch the project explicitly rules out.
        ORG_PROMPT = ("Given a QUESTION and retrieved memory items (id, time, place, caption), "
                      "group ONLY these items into the events relevant to answering it. "
                      "An item may join several events.\n"
                      'Output ONLY JSON {"events":[{"label":str,"item_ids":[id]}]}.\n\n'
                      "QUESTION: %s\n\nRetrieved items:\n%s")
        def org_cx(q):
            ids = [i for i in ret.get(q["id"], [])[:args.topn] if i in meta]
            lines = "\n".join("[%s] %s | %s | %s" % (i, meta[i][0], meta[i][1], meta[i][2])
                               for i in ids)
            raw = codex_complete(ORG_PROMPT % (q["question"], lines), args.model, args.codex_bin)
            g = _parse_json_object(raw) or {}
            rich = []
            for e in g.get("events", []):
                its = [{"id": i, "time": meta[i][0], "place": meta[i][1], "caption": meta[i][2]}
                       for i in e.get("item_ids", []) if i in meta]
                if its:
                    rich.append({"label": e.get("label", "event"), "items": its})
            ev = {"question": q["question"], "n_events": len(rich), "events": rich}
            (evd / ("%s.json" % q["id"])).write_text(json.dumps(ev, ensure_ascii=False, indent=1))
            return q["id"], ev
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(org_cx, q) for q in qs]):
                i, e = f.result(); events[i] = e
        print("  [codex] organised with %s (same model as answerer)" % args.model)
    org_in, org_out = USAGE_TOTALS["input_tokens"], USAGE_TOTALS["output_tokens"]
    empty = sum(1 for e in events.values() if not e.get("events"))
    print("  organise done: empty=%d/%d  in=%d out=%d" % (empty, len(qs), org_in, org_out))

    # ---- stage 2: answer, one call ----------------------------------------
    def ans(q):
        p = ANSWER_PROMPT.format(events=render_events(events.get(q["id"], {}), detail_map,
                                                     q.get("qtype", "open_end")),
                                 question=q["question"],
                                 type_rule=TYPE_RULE.get(q.get("qtype", "open_end"),
                                                         TYPE_RULE["open_end"]))
        if args.backend == "codex":
            raw = codex_complete(p, args.model, args.codex_bin)
        else:
            raw = chat([{"role": "user", "content": p}], base_url=args.base_url, model=args.model)
        d = _parse_json_object(raw) or {}
        return {"id": q["id"], "question": q["question"],
                "answer": str(d.get("answer") or "").strip() or "Unknown"}
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(ans, q) for q in qs]):
            rows.append(f.result())
    rows.sort(key=lambda r: r["id"])
    with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ti, to = USAGE_TOTALS["input_tokens"], USAGE_TOTALS["output_tokens"]
    st = {"n": len(rows), "backend": args.backend, "model": args.model, "topn": args.topn,
          "organise_empty": empty, "organise_in": org_in, "organise_out": org_out,
          "input_tokens": ti, "output_tokens": to,
          "cost_usd": (round(ti * args.price_in / 1e6 + to * args.price_out / 1e6, 4)
                       if args.backend == "api" else None),
          "note": ("subscription quota, no per-token bill; answer-stage tokens not "
                   "reported by codex CLI" if args.backend == "codex" else "")}
    json.dump(st, open(out / "usage.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
