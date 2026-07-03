"""CPU-only unit tests for the heuristic organiser.

Run:
    python -m unittest remmi.organize.test_organize -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from remmi.organize.heuristic import (
    HeuristicConfig,
    attach_email_ids,
    build_organized_memory,
    cluster_events,
    home_city,
    media_item_from_record,
)


def _record(path: str, ts: str, city: str, caption: str = "", tags: list[str] | None = None) -> dict:
    return {
        "image_path": f"data/raw_memory/image/{path}",
        "timestamp": ts,
        "city": city,
        "location_name": f"Somewhere, {city}",
        "short_caption": caption or f"photo in {city}",
        "tags": tags or [],
        "entities": [],
    }


def _items(records: list[dict]) -> list:
    return [media_item_from_record(r, "image_path", "image") for r in records]


HOME = "Cambridge, United Kingdom"
AWAY = "Bangkok, Thailand"


class TestClustering(unittest.TestCase):
    def test_small_gap_same_city_merges_into_one_event(self) -> None:
        records = [
            _record("a.jpg", "2023-01-01 10:00:00", HOME),
            _record("b.jpg", "2023-01-02 10:00:00", HOME),
            _record("c.jpg", "2023-01-04 10:00:00", HOME),  # gap 2 <= gap_days
        ]
        events = cluster_events(_items(records), HeuristicConfig(gap_days=2))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["item_count"], 3)
        self.assertEqual(events[0]["start"], "2023-01-01")
        self.assertEqual(events[0]["end"], "2023-01-04")

    def test_large_gap_splits_events(self) -> None:
        records = [
            _record("a.jpg", "2023-01-01 10:00:00", HOME),
            _record("b.jpg", "2023-01-10 10:00:00", HOME),
        ]
        events = cluster_events(_items(records), HeuristicConfig(gap_days=2))
        self.assertEqual(len(events), 2)

    def test_city_change_splits_even_within_gap(self) -> None:
        records = [
            _record("a.jpg", "2023-01-01 10:00:00", HOME),
            _record("b.jpg", "2023-01-02 10:00:00", AWAY),
        ]
        events = cluster_events(_items(records), HeuristicConfig(gap_days=2))
        self.assertEqual(len(events), 2)

    def test_trip_detection_against_home_city(self) -> None:
        records = (
            [_record(f"h{i}.jpg", f"2023-01-{i:02d} 09:00:00", HOME) for i in range(1, 6)]
            + [_record(f"t{i}.jpg", f"2023-02-0{i} 09:00:00", AWAY) for i in range(1, 4)]
        )
        items = _items(records)
        self.assertEqual(home_city(items), HOME)
        events = cluster_events(items)
        kinds = {e["kind"] for e in events}
        self.assertIn("trip", kinds)
        trip = next(e for e in events if e["kind"] == "trip")
        self.assertEqual(trip["city"], AWAY)
        self.assertEqual(trip["item_count"], 3)

    def test_item_ids_are_preserved_exactly_and_exhaustively(self) -> None:
        records = [
            _record("20230101_010101.jpg", "2023-01-01 10:00:00", HOME),
            _record("20230115_020202.jpg", "2023-01-15 10:00:00", AWAY),
        ]
        events = cluster_events(_items(records))
        all_ids = sorted(i for e in events for i in e["item_ids"])
        self.assertEqual(all_ids, ["20230101_010101", "20230115_020202"])

    def test_email_ids_attach_by_date_range(self) -> None:
        records = [
            _record("a.jpg", "2023-01-01 10:00:00", HOME),
            _record("b.jpg", "2023-01-02 10:00:00", HOME),
        ]
        events = cluster_events(_items(records))
        emails = [
            {"id": "email1", "timestamp": "2023-01-01 12:00:00"},
            {"id": "email2", "timestamp": "2023-06-01 12:00:00"},  # outside range
        ]
        attach_email_ids(events, emails)
        self.assertEqual(events[0]["email_ids"], ["email1"])


class TestBuildPayload(unittest.TestCase):
    def test_end_to_end_build_and_compactness(self) -> None:
        images = [
            _record(f"h{i}.jpg", f"2023-01-0{i} 09:00:00", HOME, caption="long caption " * 20, tags=["home"])
            for i in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "img.json").write_text(json.dumps(images), encoding="utf-8")
            (root / "vid.json").write_text("[]", encoding="utf-8")
            (root / "emails.json").write_text(json.dumps([{"id": "email1", "timestamp": "2023-01-02 08:00:00"}]), encoding="utf-8")
            payload = build_organized_memory(root / "img.json", root / "vid.json", root / "emails.json")

        self.assertEqual(payload["home_city"], HOME)
        self.assertEqual(payload["event_count"], 1)
        self.assertEqual(payload["trip_count"], 0)
        self.assertEqual(payload["events"][0]["email_ids"], ["email1"])
        # Compactness: the index must be far smaller than the raw records it summarises.
        index_size = len(json.dumps(payload["events"][0]["sample_captions"]))
        raw_size = len(json.dumps([r["short_caption"] for r in images]))
        self.assertLess(index_size, raw_size)


if __name__ == "__main__":
    unittest.main()
