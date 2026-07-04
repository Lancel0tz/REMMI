#!/usr/bin/env python3
"""Memory search CLI — REMMI retrieval (sparse + metadata channels) as an agent tool.

Standalone, stdlib-only. Searches the compact corpus projection
`memory/search_corpus.json` (images + videos + emails) with BM25 plus optional
date/city/type filters, and prints ranked shortlists. With a rich corpus it can
also print full projected records (`--show`), replacing raw-file reads.

Usage (from the question workspace):
    python3 memory/search.py "ramen dinner Tokyo"                    # one query
    python3 memory/search.py "ramen Tokyo" "noodle restaurant"       # several queries in ONE call
    python3 memory/search.py "hotel booking" --type email --top-k 10
    python3 memory/search.py "beach" --start 2023-07-01 --end 2023-08-31
    python3 memory/search.py --show 20220507_150929,email202201010001   # full records by id

Search output columns:  rank | score | id | type | timestamp | city | text snippet
Prefer `--show <ids>` over grepping the raw metadata files — it prints the full
projected record (location, caption, ocr, tags / email detail) far more cheaply.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def bm25_scores(query_tokens: list[str], docs: list[list[str]], doc_freq: Counter, avg_len: float) -> list[float]:
    doc_count = len(docs)
    scores = [0.0] * doc_count
    for term in set(query_tokens):
        df = doc_freq.get(term)
        if not df:
            continue
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        for index, doc in enumerate(docs):
            tf = doc.count(term)
            if not tf:
                continue
            denom = tf + K1 * (1 - B + B * len(doc) / avg_len)
            scores[index] += idf * tf * (K1 + 1) / denom
    return scores


def show_records(items: list[dict], ids: list[str]) -> None:
    by_id = {item["id"]: item for item in items}
    for record_id in ids:
        item = by_id.get(record_id.strip())
        if item is None:
            print(f"-- {record_id}: NOT FOUND")
            continue
        print(f"== {item['id']} | {item['type']} | {item.get('ts') or '?'} | {item.get('city') or ''}")
        print(f"   text: {item['text']}")
        for key, value in (item.get("detail") or {}).items():
            if not value:
                continue
            if isinstance(value, list):
                value = ", ".join(value)
            print(f"   {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search personal memory (BM25 + metadata filters)")
    parser.add_argument("queries", nargs="*", help="one or MORE free-text queries (each searched separately)")
    parser.add_argument("--show", help="comma-separated ids: print full projected records instead of searching")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--start", help="only items on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end", help="only items on/before this date (YYYY-MM-DD)")
    parser.add_argument("--city", help="substring filter on city (case-insensitive)")
    parser.add_argument("--type", choices=("image", "video", "email", "any"), default="any")
    args = parser.parse_args()

    corpus_path = Path(__file__).parent / "search_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    if args.show:
        show_records(items, args.show.split(","))
        return
    if not args.queries:
        print("provide at least one query, or --show <ids>")
        sys.exit(1)

    def keep(item: dict) -> bool:
        if args.type != "any" and item["type"] != args.type:
            return False
        ts = item.get("ts") or ""
        if args.start and (not ts or ts[:10] < args.start):
            return False
        if args.end and (not ts or ts[:10] > args.end):
            return False
        if args.city and args.city.lower() not in (item.get("city") or "").lower():
            return False
        return True

    kept = [item for item in items if keep(item)]
    if not kept:
        print("no items match the filters; relax --start/--end/--city/--type")
        sys.exit(0)

    docs = [tokenize(item["text"]) for item in kept]
    doc_freq: Counter = Counter()
    for doc in docs:
        for term in set(doc):
            doc_freq[term] += 1
    avg_len = sum(len(d) for d in docs) / len(docs) if docs else 1.0

    for query in args.queries:
        if len(args.queries) > 1:
            print(f"### {query}")
        scores = bm25_scores(tokenize(query), docs, doc_freq, avg_len)
        ranked = sorted(zip(scores, kept), key=lambda pair: -pair[0])[: args.top_k]
        for rank, (score, item) in enumerate(ranked, start=1):
            if score <= 0 and rank > 1:
                break
            snippet = item["text"][:80].replace("\n", " ")
            print(f"{rank:2d} | {score:6.2f} | {item['id']} | {item['type']:5s} | {item.get('ts') or '?':16s} | {(item.get('city') or '')[:22]:22s} | {snippet}")


if __name__ == "__main__":
    main()
