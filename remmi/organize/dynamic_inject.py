"""Dynamic per-question memory organisation with OFFLINE pre-compute + INJECTION.

This is the "inject" variant of dynamic organisation (distinct from ``org_dynamic``
where the answering agent self-organises at run time via grep + ``timeline.md``).

Here, for each question we:
  1. take the top-N retrieved-relevant memory items (from a precomputed
     retrieval file, or REMMI's hybrid retriever),
  2. call an LLM ONCE to organise ONLY those items into question-relevant events,
  3. write a per-question ``<qid>.json`` file.

At agent run time the harness injects ``<qid>.json`` into the sandbox as
``memory/query_events.json`` (see ``AGSYS_DYNAMIC_EVENTS_DIR`` in
``agent_systems/scripts/pi/run_pi.sh``). The answering agent reads a focused,
pre-organised shortlist instead of paying to organise in-session.

Original Qwen3.6-27B / Pi result on ATM-Bench-Hard: this offline-inject variant
was the token/QS winner (organise cost paid once offline, tiny per item; the
answerer's context stays small). See ``experiments/memory_organization``.

CLI:
    python -m remmi.organize.dynamic_inject \
        --retrieval output/.../mmrag_answers.jsonl \
        --questions data/atm-bench/atm-bench-hard.json \
        --out-dir agent_systems/eval_root_orgi/memory/dynamic_events \
        --base-url http://localhost:8000/v1 --model <served-model> --topn 50
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from remmi.organize.agent_static import chat
from remmi.organize.heuristic import _clean_city, _stem

# Rough token accounting (chars // 4), reported at the end.
_TOK = {"in": 0, "out": 0, "calls": 0}


def _est(text: str) -> int:
    return len(text) // 4


def _meta_map(image_source: Path, video_source: Path) -> dict[str, tuple[str, str, str]]:
    """id -> (timestamp[:16], city, short_caption[:60]) for photos and videos."""
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
) -> dict[str, Any]:
    """LLM-organise the top-N retrieved items for one question into events."""
    ids = [i for i in retrieval_ids[:topn] if i in meta]
    if not ids:
        return {"question": question, "n_events": 0, "events": []}
    lines = "\n".join(f"[{i}] {meta[i][0]} | {meta[i][1]} | {meta[i][2]}" for i in ids)
    prompt = _PROMPT.format(q=question, lines=lines)
    _TOK["in"] += _est(prompt)
    _TOK["calls"] += 1
    raw = chat([{"role": "user", "content": prompt}], base_url=base_url, model=model)
    _TOK["out"] += _est(raw or "")
    grouping = None
    try:
        grouping = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        pass
    events = (grouping or {}).get("events", []) if grouping else []
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
    ap.add_argument("--out-dir", required=True, help="dir to write per-question <qid>.json (query_events)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta_map(Path(args.image_source), Path(args.video_source))
    with open(args.questions, "r", encoding="utf-8") as handle:
        qtext = {g["id"]: g["question"] for g in json.load(handle)}
    ret: dict[str, list[str]] = {}
    with open(args.retrieval, "r", encoding="utf-8") as handle:
        for line in handle:
            x = json.loads(line)
            ret[x["id"]] = x.get("retrieval_ids", [])

    def work(qid: str) -> tuple[str, int]:
        payload = organize_question(
            qid, qtext[qid], ret.get(qid, []), meta,
            base_url=args.base_url, model=args.model, topn=args.topn,
        )
        with open(out_dir / f"{qid}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        return qid, payload["n_events"]

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, q) for q in qtext]):
            _qid, _n = fut.result()
            written += 1

    total = _TOK["in"] + _TOK["out"]
    print(f"dynamic-inject: {written} questions -> {out_dir}")
    print(f"  organise cost: in={_TOK['in']} out={_TOK['out']} total={total} "
          f"({total / 1000:.0f}K over {_TOK['calls']} calls, topn={args.topn})")


if __name__ == "__main__":
    main()
