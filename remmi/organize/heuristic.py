"""Heuristic event/trip organiser over SGM memory items.

Deterministic, LLM-free clustering:

1. Media items (images + videos) are sorted by ``timestamp`` and grouped by
   calendar day.
2. Consecutive days are merged into one *event* while the gap between them is
   ``<= gap_days`` **and** the day's dominant city matches the running event's
   dominant city (a city change always starts a new event, so a same-week
   Cambridge -> London hop still splits).
3. The corpus-level *home city* is the modal city over all dated items. An
   event whose dominant city differs from home is tagged ``kind="trip"``;
   otherwise ``kind="event"``.
4. Emails are not clustered (they are queried separately by the agent), but
   each event lists the email ids whose timestamps fall inside its date range.

The output index intentionally stays compact — per event: date range, cities,
member item ids, top tags/entities, and a few representative short captions.
Item ids are preserved verbatim (filename stem / email ``id``) because
recall-style questions must answer with exact memory item ids.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------
# Input records
# --------------------------------------------------------------------------


@dataclass
class MediaItem:
    """Minimal view of one SGM media record used for clustering."""

    item_id: str
    kind: str  # "image" | "video"
    timestamp: datetime | None
    city: str
    location_name: str
    short_caption: str
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    @property
    def day(self) -> date | None:
        return self.timestamp.date() if self.timestamp else None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt) + 2].strip(), fmt)
        except ValueError:
            continue
    return None


def _stem(path_value: Any) -> str:
    return Path(str(path_value)).stem


def _clean_city(value: Any) -> str:
    """Normalise the SGM ``city`` field.

    Most records hold ``"City, Country"``, but a few carry multiline prose
    (e.g. an address block followed by ``City: X`` / ``Country: Y`` lines).
    Rebuild ``"City, Country"`` from those markers when present; otherwise
    keep the first non-empty line.
    """
    text = str(value or "").strip()
    if "\n" not in text:
        return text
    city = country = ""
    for line in (l.strip() for l in text.splitlines()):
        if line.lower().startswith("city:"):
            city = line.split(":", 1)[1].strip()
        elif line.lower().startswith("country:"):
            country = line.split(":", 1)[1].strip()
    if city:
        return f"{city}, {country}" if country else city
    return next((l.strip() for l in text.splitlines() if l.strip()), "")


def media_item_from_record(record: dict[str, Any], path_key: str, kind: str) -> MediaItem:
    return MediaItem(
        item_id=_stem(record.get(path_key, "")),
        kind=kind,
        timestamp=_parse_ts(record.get("timestamp")),
        city=_clean_city(record.get("city")),
        location_name=str(record.get("location_name") or "").strip(),
        short_caption=str(record.get("short_caption") or "").strip(),
        tags=[str(t) for t in record.get("tags") or []],
        entities=[str(e) for e in record.get("entities") or []],
    )


def load_media_items(image_source: Path, video_source: Path) -> list[MediaItem]:
    items: list[MediaItem] = []
    with open(image_source, "r", encoding="utf-8") as handle:
        for record in json.load(handle):
            items.append(media_item_from_record(record, "image_path", "image"))
    with open(video_source, "r", encoding="utf-8") as handle:
        for record in json.load(handle):
            items.append(media_item_from_record(record, "video_path", "video"))
    return items


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


@dataclass
class HeuristicConfig:
    gap_days: int = 2          # max day gap merged into the same event
    max_caption_samples: int = 5
    max_top_tags: int = 8
    max_top_entities: int = 8
    max_location_names: int = 4


def _dominant(values: Iterable[str]) -> str:
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else ""


def home_city(items: list[MediaItem]) -> str:
    return _dominant(item.city for item in items if item.day is not None)


def cluster_events(items: list[MediaItem], config: HeuristicConfig | None = None) -> list[dict[str, Any]]:
    """Group dated media items into event dicts (undated items are skipped here)."""
    config = config or HeuristicConfig()
    dated = sorted((i for i in items if i.day is not None), key=lambda i: i.timestamp)  # type: ignore[arg-type]
    if not dated:
        return []

    home = home_city(dated)

    # Group by calendar day, each day gets a dominant city.
    days: list[tuple[date, str, list[MediaItem]]] = []
    for item in dated:
        if days and days[-1][0] == item.day:
            days[-1][2].append(item)
        else:
            days.append((item.day, "", [item]))  # type: ignore[arg-type]
    days = [(d, _dominant(i.city for i in members), members) for d, _, members in days]

    # Merge consecutive days into events: split on gap > gap_days or city change.
    events: list[list[tuple[date, str, list[MediaItem]]]] = []
    for entry in days:
        if events:
            prev_day, prev_city, _ = events[-1][-1]
            gap = (entry[0] - prev_day).days
            if gap <= config.gap_days and entry[1] == prev_city:
                events[-1].append(entry)
                continue
        events.append([entry])

    out: list[dict[str, Any]] = []
    for index, event_days in enumerate(events, start=1):
        members = [item for _, _, day_items in event_days for item in day_items]
        cities = [c for _, c, _ in event_days if c]
        dominant_city = _dominant(cities)
        kind = "trip" if (dominant_city and home and dominant_city != home) else "event"
        tags = Counter(t for m in members for t in m.tags)
        entities = Counter(e for m in members for e in m.entities)
        locations = Counter(m.location_name for m in members if m.location_name)
        captions = [m.short_caption for m in members if m.short_caption]
        step = max(1, len(captions) // config.max_caption_samples)
        out.append(
            {
                "event_id": f"E{index:04d}",
                "kind": kind,
                "start": event_days[0][0].isoformat(),
                "end": event_days[-1][0].isoformat(),
                "days": len(event_days),
                "city": dominant_city,
                "cities": sorted(set(cities)),
                "top_locations": [name for name, _ in locations.most_common(config.max_location_names)],
                "item_count": len(members),
                "item_ids": [m.item_id for m in members],
                "top_tags": [t for t, _ in tags.most_common(config.max_top_tags)],
                "top_entities": [e for e, _ in entities.most_common(config.max_top_entities)],
                "sample_captions": captions[::step][: config.max_caption_samples],
            }
        )
    return out


def attach_email_ids(events: list[dict[str, Any]], emails: list[dict[str, Any]]) -> None:
    """Add ``email_ids`` to each event for emails dated inside its range."""
    dated_emails = []
    for email in emails:
        ts = _parse_ts(email.get("timestamp"))
        if ts is not None and email.get("id"):
            dated_emails.append((ts.date(), str(email["id"])))
    dated_emails.sort()
    for event in events:
        start = date.fromisoformat(event["start"])
        end = date.fromisoformat(event["end"])
        event["email_ids"] = [eid for d, eid in dated_emails if start <= d <= end]


# --------------------------------------------------------------------------
# Top-level build
# --------------------------------------------------------------------------


def build_organized_memory(
    image_source: Path,
    video_source: Path,
    emails_source: Path | None = None,
    config: HeuristicConfig | None = None,
) -> dict[str, Any]:
    """Build the full ``organized_memory.json`` payload."""
    config = config or HeuristicConfig()
    items = load_media_items(image_source, video_source)
    events = cluster_events(items, config)
    if emails_source is not None:
        with open(emails_source, "r", encoding="utf-8") as handle:
            attach_email_ids(events, json.load(handle))
    undated = [i.item_id for i in items if i.day is None]
    return {
        "generated_by": "remmi.organize.heuristic",
        "params": {"gap_days": config.gap_days},
        "home_city": home_city(items),
        "event_count": len(events),
        "trip_count": sum(1 for e in events if e["kind"] == "trip"),
        "undated_item_ids": undated,
        "events": events,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Heuristic event/trip organiser (day-gap clustering)")
    parser.add_argument("--image-source", required=True, type=Path)
    parser.add_argument("--video-source", required=True, type=Path)
    parser.add_argument("--emails-source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gap-days", type=int, default=2)
    args = parser.parse_args()

    payload = build_organized_memory(
        args.image_source,
        args.video_source,
        args.emails_source,
        HeuristicConfig(gap_days=args.gap_days),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(
        f"organized {payload['event_count']} events ({payload['trip_count']} trips), "
        f"home_city={payload['home_city']!r} -> {args.out}"
    )


if __name__ == "__main__":
    main()
