"""Static agent organiser: an LLM organises the FULL memory corpus upfront.

This is the "static" strategy: one offline organisation pass over all memory
items, producing the same ``organized_memory.json`` schema as the heuristic
organiser, but with LLM-written event boundaries, titles, and summaries.

Pipeline (map-reduce over an OpenAI-compatible endpoint):

1. *Seed* — items are pre-grouped by the deterministic heuristic (cheap,
   keeps every item id accounted for).
2. *Map* — each batch of seed events is sent to the LLM, which may merge,
   split, retitle, and summarise them into refined events.
3. *Reduce* — adjacent-batch boundary events are offered for merging in a
   final pass.

The LLM never sees or edits item ids in free text; it works on numbered seed
events and returns grouping decisions, so ids can never be corrupted.

Endpoint configuration (any OpenAI-compatible /v1/chat/completions server —
vLLM, OpenRouter, OpenAI, ...):

    ORGANIZER_BASE_URL   default http://localhost:8000/v1
    ORGANIZER_MODEL      default gpt-5.5 (set to your served model id)
    ORGANIZER_API_KEY    bearer token; falls back to OPENAI_API_KEY, then to
                         api_keys/.openai_key if present

Usage:
    python -m remmi.organize.agent_static \
        --image-source output/image/qwen3vl2b/batch_results.json \
        --video-source output/video/qwen3vl2b/batch_results.json \
        --emails-source data/raw_memory/email/emails.json \
        --out output/organized/agent_static/organized_memory.json
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from remmi.organize.heuristic import (
    HeuristicConfig,
    attach_email_ids,
    cluster_events,
    home_city,
    load_media_items,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "gpt-5.5"

SYSTEM_PROMPT = """You organise a personal photo/video memory corpus into coherent events.
You receive numbered SEED EVENTS (chronological), each with dates, city, top tags/entities and sample captions.
Decide the final event grouping: merge seeds that belong to one real-life event or trip; keep others as-is.
For each final event, write a short `title` (<= 8 words) and a 1-2 sentence `summary`.
Respond with ONLY a JSON array; each element: {"seed_ids": [<int>, ...], "title": "...", "summary": "..."}.
Every seed id you were given must appear in exactly one element. No other text."""


def _api_key() -> str:
    for env in ("ORGANIZER_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    key_file = Path("api_keys/.openai_key")
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


def chat(messages: list[dict[str, str]], *, base_url: str, model: str, retries: int = 3) -> str:
    # No temperature override: gpt-5.x chat completions reject non-default values.
    payload = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - retry then surface
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Organiser endpoint failed after {retries} attempts: {last_error}")


def _seed_line(index: int, event: dict[str, Any]) -> str:
    return json.dumps(
        {
            "seed_id": index,
            "kind": event["kind"],
            "start": event["start"],
            "end": event["end"],
            "city": event["city"],
            "item_count": event["item_count"],
            "top_tags": event["top_tags"][:5],
            "top_entities": event["top_entities"][:5],
            "sample_captions": event["sample_captions"][:3],
        },
        ensure_ascii=False,
    )


def _parse_grouping(raw: str, valid_ids: set[int]) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("[") :]
    groups = json.loads(text[text.find("[") : text.rfind("]") + 1])
    seen: set[int] = set()
    cleaned: list[dict[str, Any]] = []
    for group in groups:
        seed_ids = [int(s) for s in group.get("seed_ids", []) if int(s) in valid_ids and int(s) not in seen]
        if not seed_ids:
            continue
        seen.update(seed_ids)
        cleaned.append(
            {
                "seed_ids": sorted(seed_ids),
                "title": str(group.get("title", "")).strip(),
                "summary": str(group.get("summary", "")).strip(),
            }
        )
    # Any seed the model dropped survives as its own event (ids must never vanish).
    for missing in sorted(valid_ids - seen):
        cleaned.append({"seed_ids": [missing], "title": "", "summary": ""})
    cleaned.sort(key=lambda g: g["seed_ids"][0])
    return cleaned


def _merge_seed_events(seeds: list[dict[str, Any]], group: dict[str, Any], event_id: str) -> dict[str, Any]:
    members = [seeds[i] for i in group["seed_ids"]]
    merged: dict[str, Any] = {
        "event_id": event_id,
        "kind": "trip" if any(m["kind"] == "trip" for m in members) else "event",
        "title": group["title"],
        "summary": group["summary"],
        "start": min(m["start"] for m in members),
        "end": max(m["end"] for m in members),
        "days": sum(m["days"] for m in members),
        "city": members[0]["city"],
        "cities": sorted({c for m in members for c in m["cities"]}),
        "top_locations": [loc for m in members for loc in m["top_locations"]][:4],
        "item_count": sum(m["item_count"] for m in members),
        "item_ids": [i for m in members for i in m["item_ids"]],
        "top_tags": [t for m in members for t in m["top_tags"]][:8],
        "top_entities": [e for m in members for e in m["top_entities"]][:8],
        "sample_captions": [c for m in members for c in m["sample_captions"]][:5],
    }
    return merged


def organize_with_agent(
    image_source: Path,
    video_source: Path,
    emails_source: Path | None,
    *,
    base_url: str,
    model: str,
    batch_size: int = 40,
    gap_days: int = 2,
) -> dict[str, Any]:
    items = load_media_items(image_source, video_source)
    seeds = cluster_events(items, HeuristicConfig(gap_days=gap_days))

    refined: list[dict[str, Any]] = []
    for offset in range(0, len(seeds), batch_size):
        batch = seeds[offset : offset + batch_size]
        lines = "\n".join(_seed_line(offset + i, e) for i, e in enumerate(batch))
        raw = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"SEED EVENTS:\n{lines}"},
            ],
            base_url=base_url,
            model=model,
        )
        valid = set(range(offset, offset + len(batch)))
        for group in _parse_grouping(raw, valid):
            refined.append(_merge_seed_events(seeds, group, event_id=f"A{len(refined) + 1:04d}"))
        print(f"  map: seeds {offset}-{offset + len(batch) - 1} -> {len(refined)} events so far")

    if emails_source is not None:
        with open(emails_source, "r", encoding="utf-8") as handle:
            attach_email_ids(refined, json.load(handle))

    undated = [i.item_id for i in items if i.day is None]
    return {
        "generated_by": "remmi.organize.agent_static",
        "params": {"model": model, "batch_size": batch_size, "gap_days": gap_days},
        "home_city": home_city(items),
        "event_count": len(refined),
        "trip_count": sum(1 for e in refined if e["kind"] == "trip"),
        "undated_item_ids": undated,
        "events": refined,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Static agent organiser (LLM full-corpus organisation pass)")
    parser.add_argument("--image-source", required=True, type=Path)
    parser.add_argument("--video-source", required=True, type=Path)
    parser.add_argument("--emails-source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--gap-days", type=int, default=2)
    args = parser.parse_args()

    base_url = os.environ.get("ORGANIZER_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("ORGANIZER_MODEL", DEFAULT_MODEL)
    print(f"organiser endpoint: {base_url} model: {model}")

    payload = organize_with_agent(
        args.image_source,
        args.video_source,
        args.emails_source,
        base_url=base_url,
        model=model,
        batch_size=args.batch_size,
        gap_days=args.gap_days,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"organized {payload['event_count']} events ({payload['trip_count']} trips) -> {args.out}")


if __name__ == "__main__":
    main()
