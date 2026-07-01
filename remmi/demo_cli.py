"""Quick-look CLI for the hybrid retriever.

Runs the synthetic 5-item corpus (no GPU, no network) so a supervisor can
see channel behavior at a glance:

    python -m remmi.demo_cli

To try a custom query:

    python -m remmi.demo_cli "Where did I have ramen in Tokyo?"

A real ATM-Bench run still goes through `mmrag_retrieve_answer.py`; the
hybrid retriever will be wired into that script as a new `--retriever
hybrid` choice in the next milestone.
"""

from __future__ import annotations

import sys
from pathlib import Path

from remmi import HybridRetriever, HybridScoringConfig
from remmi.hybrid_retriever import (
    build_synthetic_items,
    parse_query_constraints,
)


DEFAULT_QUERIES = [
    "Where did I have dinner with Grace?",
    "Show photos from December 2022 in Cambridge",
    "Where did I stay in 2023 in Portugal?",
    "Where did I have ramen in Tokyo?",
]


def _print_query(retriever: HybridRetriever, query: str) -> None:
    constraints = parse_query_constraints(query, gazetteer=retriever.gazetteer)
    print(f"\n=== Query: {query!r}")
    print(f"  parsed date_range = {constraints.date_range}, "
          f"locations = {constraints.locations}")
    results = retriever.retrieve(query, top_k=3)
    for rank, r in enumerate(results, 1):
        ts = (r.item.metadata or {}).get("timestamp", "")
        loc = (r.item.metadata or {}).get("location", "")
        print(f"  #{rank}  score={r.score:.4f}  id={r.item.item_id}  "
              f"ts={ts}  loc={loc}")


def main() -> int:
    queries = sys.argv[1:] or DEFAULT_QUERIES
    items = build_synthetic_items()
    retriever = HybridRetriever(
        cache_dir=Path("/tmp/atmbench-hybrid-demo"),
        dense_retriever=None,
        scoring=HybridScoringConfig(filter_mode="soft"),
        location_gazetteer=["Porto", "Lisbon", "Tokyo", "Cambridge", "Portugal", "Japan"],
    )
    retriever.build_index(items)
    print(f"Indexed {len(items)} synthetic SGM items.")
    for q in queries:
        _print_query(retriever, q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
