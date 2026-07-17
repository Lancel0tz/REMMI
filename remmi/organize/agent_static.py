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
import re
import time
import urllib.request
from datetime import date
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

# Accumulated across all chat() calls of one organiser run (reported at the end).
USAGE_TOTALS = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

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


def chat(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    model: str,
    retries: int = 3,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """One chat completion. ``extra_body`` merges vendor fields into the request.

    Used to pass vLLM's ``chat_template_kwargs`` (e.g. no-think). Left empty by
    default: OpenAI endpoints reject unknown fields.
    """
    # No temperature override: gpt-5.x chat completions reject non-default values.
    body_fields: dict[str, Any] = {"model": model, "messages": messages}
    body_fields.update(extra_body or {})
    payload = json.dumps(body_fields).encode("utf-8")
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
            usage = body.get("usage") or {}
            USAGE_TOTALS["input_tokens"] += usage.get("prompt_tokens", 0)
            USAGE_TOTALS["output_tokens"] += usage.get("completion_tokens", 0)
            USAGE_TOTALS["calls"] += 1
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


PURE_SYSTEM_PROMPT = """You organise a personal photo/video memory corpus into real-life events.
You receive ITEMS in strict chronological order: `idx | timestamp | city | short_caption`.
Group them into contiguous events: an event is a run of consecutive items belonging to one
real-life episode (an outing, a trip day-or-multi-day, a dinner, a project session...).
YOU decide every boundary — split or join freely based on time, place, and content.
If PREVIOUS_OPEN_EVENT is given, the first items may continue it; mark that group with "continues_previous": true.
Respond with ONLY a JSON array, in order, covering EVERY index exactly once:
[{"start_idx": <int>, "end_idx": <int>, "kind": "event"|"trip", "title": "<= 8 words", "summary": "1-2 sentences", "continues_previous": <bool, optional>}]
No other text."""


def _item_line(index: int, item: Any) -> str:
    ts = item.timestamp.strftime("%Y-%m-%d %H:%M") if item.timestamp else "undated"
    return f"{index} | {ts} | {item.city} | {item.short_caption[:120]}"


def _parse_ranges(raw: str, lo: int, hi: int) -> list[dict[str, Any]]:
    """Parse chunk grouping; repair gaps/overlaps so [lo, hi] is covered exactly."""
    text = raw.strip().strip("`")
    groups = json.loads(text[text.find("[") : text.rfind("]") + 1])
    cleaned: list[dict[str, Any]] = []
    cursor = lo
    for group in sorted(groups, key=lambda g: int(g.get("start_idx", lo))):
        start = max(int(group.get("start_idx", cursor)), cursor)
        end = min(int(group.get("end_idx", start)), hi)
        if end < cursor or start > hi:
            continue
        if start > cursor:  # gap the model skipped -> untitled filler event
            cleaned.append({"start_idx": cursor, "end_idx": start - 1, "kind": "event", "title": "", "summary": ""})
        cleaned.append(
            {
                "start_idx": start,
                "end_idx": end,
                "kind": group.get("kind", "event"),
                "title": str(group.get("title", "")).strip(),
                "summary": str(group.get("summary", "")).strip(),
                "continues_previous": bool(group.get("continues_previous", False)),
            }
        )
        cursor = end + 1
    if cursor <= hi:
        cleaned.append({"start_idx": cursor, "end_idx": hi, "kind": "event", "title": "", "summary": ""})
    return cleaned


def organize_pure(
    image_source: Path,
    video_source: Path,
    emails_source: Path | None,
    *,
    base_url: str,
    model: str,
    chunk_items: int = 250,
) -> dict[str, Any]:
    """Pure LLM organisation: the model decides EVERY event boundary at item level.

    No heuristic seeding — items are streamed chronologically in chunks; the
    model groups them into contiguous events, with an open-event carry-over so
    events can span chunk boundaries.
    """
    items = load_media_items(image_source, video_source)
    dated = sorted((i for i in items if i.timestamp), key=lambda i: i.timestamp)
    undated = [i.item_id for i in items if not i.timestamp]

    events: list[dict[str, Any]] = []
    open_event: dict[str, Any] | None = None
    for lo in range(0, len(dated), chunk_items):
        hi = min(lo + chunk_items, len(dated)) - 1
        lines = "\n".join(_item_line(i, dated[i]) for i in range(lo, hi + 1))
        context = ""
        if open_event is not None:
            context = (
                "PREVIOUS_OPEN_EVENT: "
                + json.dumps({k: open_event[k] for k in ("kind", "title", "summary", "start", "end")})
                + "\n\n"
            )
        raw = chat(
            [
                {"role": "system", "content": PURE_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}ITEMS:\n{lines}"},
            ],
            base_url=base_url,
            model=model,
        )
        for order, group in enumerate(_parse_ranges(raw, lo, hi)):
            members = dated[group["start_idx"] : group["end_idx"] + 1]
            if order == 0 and group.get("continues_previous") and open_event is not None:
                open_event["item_ids"].extend(m.item_id for m in members)
                open_event["item_count"] += len(members)
                open_event["end"] = members[-1].timestamp.date().isoformat()
                open_event["cities"] = sorted(set(open_event["cities"]) | {m.city for m in members if m.city})
                continue
            if open_event is not None:
                events.append(open_event)
            cities = [m.city for m in members if m.city]
            open_event = {
                "event_id": f"P{len(events) + 1:04d}",
                "kind": group["kind"] if group["kind"] in ("event", "trip") else "event",
                "title": group["title"],
                "summary": group["summary"],
                "start": members[0].timestamp.date().isoformat(),
                "end": members[-1].timestamp.date().isoformat(),
                "days": 0,
                "city": max(set(cities), key=cities.count) if cities else "",
                "cities": sorted(set(cities)),
                "top_locations": [],
                "item_count": len(members),
                "item_ids": [m.item_id for m in members],
                "top_tags": [],
                "top_entities": [],
                "sample_captions": [m.short_caption for m in members[:3] if m.short_caption],
            }
        print(f"  chunk {lo}-{hi} -> {len(events) + 1} events so far")
    if open_event is not None:
        events.append(open_event)
    for index, event in enumerate(events, start=1):  # renumber after carry-merges
        event["event_id"] = f"P{index:04d}"

    if emails_source is not None:
        with open(emails_source, "r", encoding="utf-8") as handle:
            attach_email_ids(events, json.load(handle))

    return {
        "generated_by": "remmi.organize.agent_static:pure",
        "params": {"model": model, "chunk_items": chunk_items},
        "organiser_usage": dict(USAGE_TOTALS),
        "home_city": home_city(items),
        "event_count": len(events),
        "trip_count": sum(1 for e in events if e["kind"] == "trip"),
        "undated_item_ids": undated,
        "events": events,
    }


# --------------------------------------------------------------------------
# 2-pass event-graph strategy
# --------------------------------------------------------------------------
#
# Ported from ATM-Bench/scripts/build_event_graph.py. Two LLM passes build a
# multi-membership graph instead of a flat event list:
#
#   Pass 1 (per day) — the LLM segments one day's chronological items into
#       EVENTS; one item may join multiple events (a meal captured mid-walk).
#   Pass 2 (per trip) — trips are recovered by the repo's day-gap heuristic
#       (``cluster_events`` with ``kind == "trip"``); the LLM labels each trip
#       and extracts CROSS-CUTTING THEMES (food, talks, sightseeing...) that
#       span its events.
#
# Output schema mirrors the server: {nodes, edges, item2nodes (multi-membership
# map), tokens, n_trips, n_events}. node types are event | trip | theme.

GRAPH_PASS1_SYSTEM = """You segment a person's chronological photos on ONE day into EVENTS (a meal, a visit, a talk, a walk...).
One photo may belong to multiple events.
Respond with ONLY JSON: {"events": [{"label": "<short>", "members": [<idx>, ...]}, ...]} covering the given items. No other text."""

GRAPH_PASS2_SYSTEM = """A trip's sub-events are listed. Give the trip a short label, and find CROSS-CUTTING THEMES (food, talks, people, sightseeing...) that span its events.
Respond with ONLY JSON: {"trip_label": "<short>", "themes": [{"label": "<short>", "event_ids": ["<event id>", ...]}, ...]}. No other text."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Drop <think> blocks from a reasoning model's reply.

    Qwen3.x thinks by default. Its reasoning contains braces, so a naive
    find('{')..rfind('}') slice starts inside the think block and never parses.
    Also handles a reply whose opening <think> was swallowed by the template
    (vLLM pre-closes it), leaving a bare prefix terminated by </think>.
    """
    text = _THINK_RE.sub("", text)
    if "</think>" in text.lower():
        text = _THINK_OPEN_RE.sub("", text, count=1)
    return text


def _iter_json_objects(text: str):
    """Yield candidate JSON object substrings by brace-balanced scan (string-aware)."""
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a chat reply.

    Tolerant of reasoning traces, code fences and surrounding prose.
    """
    text = _strip_reasoning(raw or "")
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:  # fast path: the whole span is one object
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001 - fall through to the balanced scan
            pass
    for candidate in _iter_json_objects(text):
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _graph_day_line(index: int, item: Any) -> str:
    ts = item.timestamp.strftime("%H:%M") if item.timestamp else "??:??"
    city = (item.city or "?").split(",")[0]
    return f"[{index}] {ts} | {city} | {item.short_caption[:60]}"


def organize_graph_2pass(
    image_source: Path,
    video_source: Path,
    emails_source: Path | None,
    *,
    base_url: str,
    model: str,
    gap_days: int = 2,
) -> dict[str, Any]:
    """2-pass LLM event-graph organiser (per-day events -> per-trip themes).

    Builds a multi-membership graph rather than a flat event list. Item ids are
    never emitted in free text: the LLM segments by numbered index (Pass 1) and
    by event id (Pass 2), so ids can never be corrupted.
    """
    items = load_media_items(image_source, video_source)
    dated = sorted((i for i in items if i.day is not None), key=lambda i: i.timestamp)  # type: ignore[arg-type]

    # Group dated items by calendar day (Pass 1 input).
    by_day: dict[Any, list[Any]] = {}
    for item in dated:
        by_day.setdefault(item.day, []).append(item)
    days = sorted(by_day)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[list[str]] = []
    item2nodes: dict[str, list[str]] = {}

    def _add_membership(item_id: str, node_id: str) -> None:
        item2nodes.setdefault(item_id, []).append(node_id)

    # ── Pass 1: per-day segmentation into events (multi-membership) ──
    ev_by_day: dict[Any, list[str]] = {}
    ev_idx = 0
    for day in days:
        day_items = by_day[day]
        events: list[dict[str, Any]]
        if len(day_items) <= 1:
            label = (day_items[0].short_caption or "moment")[:40] if day_items else "empty"
            events = [{"label": label, "item_ids": [i.item_id for i in day_items]}]
        else:
            lines = "\n".join(_graph_day_line(i, m) for i, m in enumerate(day_items))
            raw = chat(
                [
                    {"role": "system", "content": GRAPH_PASS1_SYSTEM},
                    {"role": "user", "content": f"Photos:\n{lines}"},
                ],
                base_url=base_url,
                model=model,
            )
            parsed = _parse_json_object(raw)
            events = []
            for event in (parsed or {}).get("events", []) or []:
                ids = [day_items[i].item_id for i in event.get("members", []) if isinstance(i, int) and 0 <= i < len(day_items)]
                if ids:
                    events.append({"label": str(event.get("label", "event")), "item_ids": ids})
            if not events:  # fallback: whole day is one event, ids never lost
                events = [{"label": "day", "item_ids": [i.item_id for i in day_items]}]

        day_key = day.isoformat()
        ev_by_day[day] = []
        for event in events:
            eid = f"ev{ev_idx}"
            ev_idx += 1
            nodes[eid] = {"type": "event", "label": event["label"], "date": day_key, "members": event["item_ids"]}
            ev_by_day[day].append(eid)
            for item_id in event["item_ids"]:
                _add_membership(item_id, eid)
        print(f"  pass1: {day_key} -> {len(events)} events ({ev_idx} total)")

    # ── Pass 2: trip grouping (repo heuristic) + per-trip themes ──
    trips = [e for e in cluster_events(items, HeuristicConfig(gap_days=gap_days)) if e["kind"] == "trip"]
    trip_idx = 0
    for trip in trips:
        d0 = date.fromisoformat(trip["start"])
        d1 = date.fromisoformat(trip["end"])
        eids = [eid for day in days if d0 <= day <= d1 for eid in ev_by_day[day]]
        if not eids:
            continue
        lines = "\n".join(f"[{eid}] {nodes[eid]['date']} {nodes[eid]['label'][:50]}" for eid in eids)
        raw = chat(
            [
                {"role": "system", "content": GRAPH_PASS2_SYSTEM},
                {"role": "user", "content": f"Sub-events:\n{lines}"},
            ],
            base_url=base_url,
            model=model,
        )
        parsed = _parse_json_object(raw) or {}

        tid = f"trip{trip_idx}"
        trip_idx += 1
        members = sorted({m for eid in eids for m in nodes[eid]["members"]})
        nodes[tid] = {
            "type": "trip",
            "label": parsed.get("trip_label") or f"Trip {trip['start']}",
            "date_start": trip["start"],
            "date_end": trip["end"],
            "nights": (d1 - d0).days,
            "cities": trip["cities"],
            "event_ids": eids,
            "members": members,
        }
        for eid in eids:
            edges.append([eid, tid])
        for item_id in members:
            _add_membership(item_id, tid)

        for order, theme in enumerate(parsed.get("themes", []) or []):
            theme_eids = [e for e in theme.get("event_ids", []) if e in nodes]
            theme_members = sorted({m for e in theme_eids for m in nodes[e]["members"]})
            if not theme_members:
                continue
            thid = f"{tid}_th{order}"
            nodes[thid] = {
                "type": "theme",
                "label": str(theme.get("label", "theme")),
                "trip": tid,
                "event_ids": theme_eids,
                "members": theme_members,
            }
            for item_id in theme_members:
                _add_membership(item_id, thid)
    print(f"  pass2: {trip_idx} trips -> {len(nodes)} nodes")

    return {
        "generated_by": "remmi.organize.agent_static:graph_2pass",
        "params": {"model": model, "gap_days": gap_days},
        "home_city": home_city(items),
        "n_events": ev_idx,
        "n_trips": trip_idx,
        "nodes": nodes,
        "edges": edges,
        "item2nodes": item2nodes,
        "tokens": dict(USAGE_TOTALS),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Static agent organiser (LLM full-corpus organisation pass)")
    parser.add_argument("--image-source", required=True, type=Path)
    parser.add_argument("--video-source", required=True, type=Path)
    parser.add_argument("--emails-source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=("pure", "seeded", "graph_2pass"),
        default="pure",
        help="pure: the LLM decides every event boundary at item level (default). "
        "seeded: heuristic day-gap clustering first, LLM only merges/labels seeds. "
        "graph_2pass: 2-pass event graph (per-day events -> per-trip themes) with multi-membership.",
    )
    parser.add_argument("--batch-size", type=int, default=40, help="(seeded) seed events per LLM call")
    parser.add_argument("--gap-days", type=int, default=2, help="(seeded) heuristic gap")
    parser.add_argument("--chunk-items", type=int, default=250, help="(pure) items per LLM call")
    args = parser.parse_args()

    base_url = os.environ.get("ORGANIZER_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("ORGANIZER_MODEL", DEFAULT_MODEL)
    print(f"organiser endpoint: {base_url} model: {model} strategy: {args.strategy}")

    if args.strategy == "pure":
        payload = organize_pure(
            args.image_source,
            args.video_source,
            args.emails_source,
            base_url=base_url,
            model=model,
            chunk_items=args.chunk_items,
        )
    elif args.strategy == "graph_2pass":
        payload = organize_graph_2pass(
            args.image_source,
            args.video_source,
            args.emails_source,
            base_url=base_url,
            model=model,
            gap_days=args.gap_days,
        )
    else:
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
    if args.strategy == "graph_2pass":
        print(
            f"built graph: {len(payload['nodes'])} nodes, {payload['n_events']} events, "
            f"{payload['n_trips']} trips -> {args.out}"
        )
    else:
        print(f"organized {payload['event_count']} events ({payload['trip_count']} trips) -> {args.out}")
    print(
        f"organiser usage: {USAGE_TOTALS['calls']} calls, "
        f"{USAGE_TOTALS['input_tokens']:,} in + {USAGE_TOTALS['output_tokens']:,} out tokens"
    )


if __name__ == "__main__":
    main()
