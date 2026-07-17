"""Dynamic memory organisation: offline, per question (``org_dynamic``).

Sibling of :mod:`remmi.organize.agent_static`. Both organise offline, outside the
agent loop; they differ in scope — static organises the whole corpus once, dynamic
organises per question. Neither is the ``org_remmi*`` line, where the answering
agent organises in-session (grep/search + ``timeline.md``) and so re-sends its
growing context on every organise turn.

Here, for each question we:
  1. take the top-N retrieved-relevant memory items (from a precomputed
     retrieval file, or REMMI's hybrid retriever),
  2. call an LLM ONCE to organise ONLY those items into question-relevant events,
  3. write a per-question ``<qid>.json`` file.

At agent run time the harness injects ``<qid>.json`` into the sandbox as
``memory/query_events.json`` (see ``AGSYS_DYNAMIC_EVENTS_DIR`` in
``agent_systems/scripts/pi/run_pi.sh``). The answering agent reads a focused,
pre-organised shortlist instead of paying to organise in-session.

Qwen3.6-27B / Pi result on ATM-Bench-Hard: the token/QS winner. Organising outside
the loop costs one stateless call per question (~0.03M total) against ~4.10M for
the in-session org_remmi line — a 137x gap that is the whole token story, since the
answer phase costs both roughly the same. See ``experiments/memory_organization``.

CLI:
    python -m remmi.organize.dynamic \
        --retrieval output/.../mmrag_answers.jsonl \
        --questions data/atm-bench/atm-bench-hard.json \
        --out-dir agent_systems/eval_root_orgd/memory/dynamic_events \
        --base-url http://localhost:8000/v1 --model <served-model> --topn 50
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from remmi.organize.agent_static import USAGE_TOTALS, _parse_json_object, chat
from remmi.organize.heuristic import _clean_city, _stem

# Token accounting comes from the endpoint via agent_static.USAGE_TOTALS (real
# prompt/completion tokens), incremented inside chat().


def _meta_map(
    image_source: Path, video_source: Path, emails_source: Path | None = None
) -> dict[str, tuple[str, str, str]]:
    """id -> (timestamp[:16], city|'email', text[:60]) for photos, videos AND emails.

    Emails matter: on ATM-Bench-Hard ~43% of the hybrid retriever's top-50 ids are
    emails, and number questions (nights, counts, prices) are usually answerable
    only from booking/receipt mail. Dropping them silently caps the organiser.
    """
    out: dict[str, tuple[str, str, str]] = {}
    for source, key in ((image_source, "image_path"), (video_source, "video_path")):
        with open(source, "r", encoding="utf-8") as handle:
            for r in json.load(handle):
                iid = _stem(r.get(key, ""))
                if not iid:
                    continue
                out[iid] = (
                    str(r.get("timestamp") or "")[:16],
                    _clean_city(r.get("city")) or "?",
                    str(r.get("short_caption") or r.get("caption") or "")[:60],
                )
    if emails_source is not None and Path(emails_source).exists():
        with open(emails_source, "r", encoding="utf-8") as handle:
            for r in json.load(handle):
                eid = str(r.get("id") or "")
                if not eid:
                    continue
                out[eid] = (
                    str(r.get("timestamp") or "")[:16],
                    "email",
                    str(r.get("short_summary") or "")[:60],
                )
    return out


_PROMPT = """Given a QUESTION and the retrieved memory items (id, time, place, caption), organize ONLY these items into the events relevant to answering the question. An item may join multiple events.
Output ONLY JSON {{"events":[{{"label":str,"item_ids":[id]}}]}}.

QUESTION: {q}

Retrieved items:
{lines}"""


def organize_question(
    qid: str,
    question: str,
    retrieval_ids: list[str],
    meta: dict[str, tuple[str, str, str]],
    *,
    base_url: str,
    model: str,
    topn: int,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LLM-organise the top-N retrieved items for one question into events."""
    ids = [i for i in retrieval_ids[:topn] if i in meta]
    if not ids:
        return {"question": question, "n_events": 0, "events": []}
    lines = "\n".join(f"[{i}] {meta[i][0]} | {meta[i][1]} | {meta[i][2]}" for i in ids)
    prompt = _PROMPT.format(q=question, lines=lines)
    raw = chat([{"role": "user", "content": prompt}], base_url=base_url, model=model,
               extra_body=extra_body)
    # _parse_json_object strips <think> blocks: a reasoning organiser (Qwen3.x) puts
    # braces in its trace, which a naive first-{-to-last-} slice never parses.
    grouping = _parse_json_object(raw)
    if grouping is None:
        return {"question": question, "n_events": 0, "events": [], "parse_failed": True,
                "raw_reply": raw[:2000]}
    events = grouping.get("events", [])
    rich = []
    for event in events:
        items = [
            {"id": i, "time": meta[i][0], "place": meta[i][1], "caption": meta[i][2]}
            for i in event.get("item_ids", [])
            if i in meta
        ]
        if items:
            rich.append({"label": event.get("label", "event"), "items": items})
    return {"question": question, "n_events": len(rich), "events": rich}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval", required=True,
                    help="JSONL with {id, retrieval_ids:[...]} per question (top-N relevant item ids)")
    ap.add_argument("--questions", required=True, help="ATM-Bench questions JSON (list of {id, question})")
    ap.add_argument("--image-source", default="output/image/qwen3vl2b/batch_results.json")
    ap.add_argument("--video-source", default="output/video/qwen3vl2b/batch_results.json")
    ap.add_argument("--emails-source", default="data/raw_memory/email/emails.json",
                    help="emails.json — retrieved email ids are organised too (~43%% of top-50)")
    ap.add_argument("--out-dir", required=True, help="dir to write per-question <qid>.json (query_events)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-think", action="store_true",
                    help="vLLM only: disable the organiser's reasoning trace "
                         "(chat_template_kwargs.enable_thinking=false). Matches the "
                         "answerer's no-think regime and cuts organise output tokens.")
    args = ap.parse_args()

    extra_body = ({"chat_template_kwargs": {"enable_thinking": False}}
                  if args.no_think else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta_map(Path(args.image_source), Path(args.video_source), Path(args.emails_source))
    with open(args.questions, "r", encoding="utf-8") as handle:
        qtext = {g["id"]: g["question"] for g in json.load(handle)}
    ret: dict[str, list[str]] = {}
    with open(args.retrieval, "r", encoding="utf-8") as handle:
        for line in handle:
            x = json.loads(line)
            ret[x["id"]] = x.get("retrieval_ids", [])

    def work(qid: str) -> tuple[str, int, bool]:
        payload = organize_question(
            qid, qtext[qid], ret.get(qid, []), meta,
            base_url=args.base_url, model=args.model, topn=args.topn,
            extra_body=extra_body,
        )
        with open(out_dir / f"{qid}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        return qid, payload["n_events"], bool(payload.get("parse_failed"))

    written = empty = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, q) for q in qtext]):
            qid, n_events, parse_failed = fut.result()
            written += 1
            failed += parse_failed
            if not n_events:
                empty += 1
                print(f"  ⚠️  {qid}: 0 events{' (reply did not parse)' if parse_failed else ''}")

    # Real usage from the endpoint (prompt/completion tokens), not a char estimate.
    ti, to = USAGE_TOTALS["input_tokens"], USAGE_TOTALS["output_tokens"]
    total = ti + to
    print(f"dynamic-inject: {written} questions -> {out_dir}")
    print(f"  ORGANISE cost (measured): in={ti} out={to} total={total} "
          f"({total / 1000:.1f}K over {USAGE_TOTALS['calls']} calls, topn={args.topn}, model={args.model})")
    with open(out_dir / "_organise_usage.json", "w", encoding="utf-8") as fh:
        json.dump({"input_tokens": ti, "output_tokens": to, "total": total,
                   "calls": USAGE_TOTALS["calls"], "model": args.model,
                   "topn": args.topn, "questions": written,
                   "empty": empty, "parse_failed": failed}, fh, indent=1)

    # Fail loudly. An empty query_events.json silently degrades the agent to the
    # baseline (free exploration), so the run still "succeeds" and scores like the
    # baseline — a whole GPU-day spent measuring nothing. Never let that ship.
    if empty:
        print(f"\n❌ {empty}/{written} questions organised into 0 events "
              f"({failed} replies did not parse). See raw_reply in the <qid>.json files.")
        if failed:
            print("   A reasoning organiser (Qwen3.x) with no no-think chat template is "
                  "the usual cause — its <think> trace breaks naive JSON extraction.")
        raise SystemExit(1 if empty == written else 0)


if __name__ == "__main__":
    main()
