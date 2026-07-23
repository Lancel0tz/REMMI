#!/usr/bin/env python3
"""Lean agent loop - bounded context instead of Pi's growing transcript.

Measured on remmi10 (Qwen3.6-27B, hard-31), per question:
    10 LLM calls, context grows 2.9k -> 30k (10.3x), total input 175k
    => 87% of all input tokens are RE-SENT history.
Total input 7.95M costs $2.31 of remmi10's $2.89.

Pi re-sends the whole transcript every turn, so cost is triangular:
    total ~= n_calls * avg_context.
This loop keeps context CONSTANT instead:

    [system] + [question] + [FINDINGS (agent-maintained, capped)] + [last tool result]

The agent carries state forward in an explicit `findings` block it rewrites each
turn, so old tool output is dropped rather than re-sent. Cost becomes linear:
    total ~= n_calls * fixed_context.
Same tools, same memory files, same answer format as the Pi runs - only the
context policy differs, so QS is comparable and any cost delta is attributable.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/rds/user/kz345/hpc-work/REMMI")
from remmi.organize.agent_static import USAGE_TOTALS, _parse_json_object, chat

MAX_TOOL_CHARS = 6000      # truncate a single tool result before it enters context
MAX_FINDINGS_CHARS = 4000  # cap the carried-forward state

SYSTEM = """You answer questions about a personal memory archive by running shell commands.

Files in the workspace (memory/):
  image_metadata.json, video_metadata.json  - fields: image_path/video_path,
      timestamp, location_name, city, short_caption, caption, tags, ocr_text
  emails.json - fields: id, timestamp, short_summary, detail
{tool_line}
Item ID = filename stem for photos/videos (20240701_120945), or the `id` field
for emails (email202201010001).

HOW THIS LOOP WORKS - read carefully, it is not a normal chat:
You do NOT see earlier tool output again. Each turn you get: the question, your
own FINDINGS notes, and the MOST RECENT tool result only. Therefore you must copy
anything you want to keep into FINDINGS - ids, values, dates. Anything not in
FINDINGS is lost.

Each turn reply with ONLY a JSON object:
  {{"findings": "<compact notes to carry forward - ids/values you have confirmed>",
   "action": "run" | "answer",
   "command": "<shell command, when action=run>",
   "answer": "<final answer, when action=answer>"}}

Keep commands targeted (grep/jq with filters, head to limit output). You have at
most {max_turns} turns; plan them.
{type_rule}"""

TYPE_RULE = {
    "list_recall": ("\nThis is a LIST question, scored by set overlap - a missing id and a wrong id\n"
                    "cost the same. Scan exhaustively, then prune to ids that truly match.\n"
                    "Final `answer` = ONLY the ids, comma-separated."),
    "number": ("\nThis is a NUMBER question. Collect every relevant record into FINDINGS, then\n"
               "compute explicitly (count DISTINCT days, sum amounts). `answer` = the number."),
    "open_end": "\nAnswer concisely and concretely from the evidence you gathered.",
}


def run_cmd(cmd: str, cwd: str, timeout: int = 60) -> str:
    try:
        p = subprocess.run(["bash", "-lc", cmd], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n[stderr] " + p.stderr) if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        out = "[timeout]"
    except Exception as exc:
        out = "[error] %r" % (exc,)
    if len(out) > MAX_TOOL_CHARS:
        out = out[:MAX_TOOL_CHARS] + "\n[...truncated, narrow your command...]"
    return out or "[no output]"


def solve(q: dict, workspace: str, base_url: str, model: str, extra_body,
          max_turns: int, has_tool: bool) -> dict:
    qtype = q.get("qtype", "open_end")
    sys_prompt = SYSTEM.format(
        tool_line=("  search.py - ranked BM25 search: python3 memory/search.py \"query\" --top-k 20\n"
                   if has_tool else ""),
        max_turns=max_turns, type_rule=TYPE_RULE.get(qtype, TYPE_RULE["open_end"]))
    findings = ""
    last_tool = ""
    turns = 0
    for turns in range(1, max_turns + 1):
        user = "QUESTION: %s\n\nFINDINGS SO FAR:\n%s\n\nMOST RECENT TOOL RESULT:\n%s" % (
            q["question"], findings or "(none yet)", last_tool or "(no command run yet)")
        raw = chat([{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user}],
                   base_url=base_url, model=model, extra_body=extra_body)
        d = _parse_json_object(raw) or {}
        nf = str(d.get("findings") or "").strip()
        if nf:
            findings = nf[:MAX_FINDINGS_CHARS]
        if d.get("action") == "answer" or d.get("answer"):
            return {"id": q["id"], "question": q["question"],
                    "answer": str(d.get("answer") or "").strip() or "Unknown", "turns": turns}
        cmd = str(d.get("command") or "").strip()
        if not cmd:
            return {"id": q["id"], "question": q["question"], "answer": "Unknown", "turns": turns}
        last_tool = run_cmd(cmd, workspace)
    # out of turns: force an answer from findings
    user = ("QUESTION: %s\n\nFINDINGS:\n%s\n\nTurn budget exhausted - answer now from FINDINGS.\n"
            'Reply ONLY {"action":"answer","answer":"..."}' % (q["question"], findings))
    raw = chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
               base_url=base_url, model=model, extra_body=extra_body)
    d = _parse_json_object(raw) or {}
    return {"id": q["id"], "question": q["question"],
            "answer": str(d.get("answer") or "").strip() or "Unknown", "turns": turns}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--memory-dir", required=True, help="dir holding image/video/emails json (+search.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--price-in", type=float, default=0.29)
    ap.add_argument("--price-out", type=float, default=3.20)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else None
    qs = json.loads(Path(args.questions).read_text())
    mem = Path(args.memory_dir).resolve()
    has_tool = (mem / "search.py").exists()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = []

    def work(q):
        with tempfile.TemporaryDirectory(prefix="lean-") as ws:
            os.symlink(str(mem), os.path.join(ws, "memory"))
            return solve(q, ws, args.base_url, args.model, extra_body, args.max_turns, has_tool)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, q) for q in qs]):
            rows.append(f.result())
    rows.sort(key=lambda r: r["id"])
    with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "question": r["question"],
                                 "answer": r["answer"]}, ensure_ascii=False) + "\n")
    ti, to = USAGE_TOTALS["input_tokens"], USAGE_TOTALS["output_tokens"]
    st = {"n": len(rows), "llm_calls": USAGE_TOTALS["calls"],
          "turns_total": sum(r.get("turns", 0) for r in rows),
          "input_tokens": ti, "output_tokens": to,
          "cost_usd": round(ti * args.price_in / 1e6 + to * args.price_out / 1e6, 4),
          "model": args.model, "max_turns": args.max_turns, "has_search_tool": has_tool}
    json.dump(st, open(out / "usage.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
