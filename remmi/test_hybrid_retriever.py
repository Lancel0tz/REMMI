"""Smoke tests for `HybridRetriever`.

These tests run on CPU in seconds. They use the synthetic corpus from
`hybrid_retriever.build_synthetic_items` to exercise:

  - `parse_query_constraints` (date + location heuristics).
  - The metadata, sparse, and (no-op) dense channels.
  - Both RRF and weighted-sum fusion.

Run with:

    python -m unittest remmi.test_hybrid_retriever -v

These are intended as developer smoke tests, not the full benchmark.
"""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from remmi import (
    HybridRetriever,
    HybridScoringConfig,
    parse_query_constraints,
)
from remmi.hybrid_retriever import build_synthetic_items


class TestQueryConstraints(unittest.TestCase):
    def test_extracts_year(self):
        c = parse_query_constraints("Where did I stay in 2023?")
        self.assertIsNotNone(c.date_range)
        start, end = c.date_range
        self.assertEqual(start, date(2023, 1, 1))
        self.assertEqual(end, date(2023, 12, 31))

    def test_extracts_year_and_month(self):
        c = parse_query_constraints("Show photos from April 2023")
        start, end = c.date_range
        self.assertEqual(start, date(2023, 4, 1))
        self.assertEqual(end, date(2023, 4, 30))

    def test_extracts_iso_date(self):
        c = parse_query_constraints("What did I do on 2023-04-02?")
        start, end = c.date_range
        self.assertEqual(start, date(2023, 4, 2))
        self.assertEqual(end, date(2023, 4, 2))

    def test_today_anchor(self):
        c = parse_query_constraints("Today is 2024-01-15. What did I do yesterday?")
        self.assertEqual(c.today, date(2024, 1, 15))

    def test_extracts_location_via_gazetteer(self):
        c = parse_query_constraints(
            "Where did I have dinner in Porto?",
            gazetteer=["Porto", "Lisbon", "Tokyo"],
        )
        self.assertIn("porto", c.locations)


class TestHybridRetrieve(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)
        self.items = build_synthetic_items()

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, **kwargs) -> HybridRetriever:
        scoring = HybridScoringConfig(**kwargs.pop("scoring_kwargs", {}))
        retriever = HybridRetriever(
            cache_dir=self.cache_dir,
            dense_retriever=None,  # CPU-only smoke; skip dense channel
            scoring=scoring,
            location_gazetteer=["Porto", "Lisbon", "Tokyo", "Cambridge"],
        )
        retriever.build_index(self.items)
        return retriever

    def test_bm25_only_finds_keyword(self):
        retriever = self._build()
        results = retriever.retrieve("Where did I have dinner with Grace?", top_k=3)
        self.assertGreater(len(results), 0)
        # The item containing "Grace" should be ranked first.
        self.assertEqual(results[0].item.item_id, "20230402_180000")

    def test_metadata_date_filter_soft(self):
        retriever = self._build(scoring_kwargs={"filter_mode": "soft"})
        results = retriever.retrieve(
            "Show photos from April 2023 in Porto",
            top_k=3,
        )
        ids = [r.item.item_id for r in results]
        self.assertIn("20230402_180000", ids[:2])

    def test_metadata_date_filter_hard_excludes_other_years(self):
        retriever = self._build(scoring_kwargs={"filter_mode": "hard"})
        results = retriever.retrieve(
            "Show photos from December 2022 in Cambridge",
            top_k=5,
        )
        ids = [r.item.item_id for r in results]
        # The Christmas dinner image must be top, and the Tokyo/Lisbon items
        # from 2023 must be excluded entirely.
        self.assertEqual(ids[0], "20221225_140000")
        self.assertNotIn("20231001_100000", ids)
        self.assertNotIn("20230401_120000", ids)

    def test_weighted_sum_fusion_matches_rrf_on_easy_query(self):
        rrf = self._build(scoring_kwargs={"fusion": "rrf"})
        ws = self._build(scoring_kwargs={"fusion": "weighted_sum"})
        q = "Where did I have ramen in Tokyo?"
        rrf_top = rrf.retrieve(q, top_k=1)[0].item.item_id
        ws_top = ws.retrieve(q, top_k=1)[0].item.item_id
        self.assertEqual(rrf_top, "20231001_100000")
        self.assertEqual(ws_top, "20231001_100000")

    def test_empty_corpus(self):
        retriever = HybridRetriever(cache_dir=self.cache_dir)
        retriever.build_index([])
        self.assertEqual(retriever.retrieve("anything", top_k=5), [])


if __name__ == "__main__":
    unittest.main()
