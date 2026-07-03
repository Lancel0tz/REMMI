#!/usr/bin/env python3
"""Memory search CLI — REMMI retrieval (sparse + metadata channels) as an agent tool.

Standalone, stdlib-only (works in any sandbox). Searches the compact corpus
projection `memory/search_corpus.json` (images + videos + emails) with BM25
over text plus optional date/city/type filters, and prints a ranked shortlist.

Usage (from the question workspace):
    python3 memory/search.py "ramen dinner Tokyo"                 # top 20
    python3 memory/search.py "hotel booking" --type email --top-k 10
    python3 memory/search.py "beach" --start 2023-07-01 --end 2023-08-31
    python3 memory/search.py "museum" --city london

Output columns:  rank | score | id | type | timestamp | city | text snippet
Use the returned ids to drill into the full records in image_metadata.json /
video_metadata.json / emails.json — do NOT broad-grep those files for discovery.
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


def bm25_scores(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    doc_count = len(docs)
    doc_lens = [len(d) for d in docs]
    avg_len = (sum(doc_lens) / doc_count) if doc_count else 1.0
    doc_freq: Counter = Counter()
    for doc in docs:
        for term in set(doc):
            doc_freq[term] += 1
    scores = [0.0] * doc_count
    for term in query_tokens:
        df = doc_freq.get(term)
        if not df:
            continue
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        for index, doc in enumerate(docs):
            tf = doc.count(term)
            if not tf:
                continue
            denom = tf + K1 * (1 - B + B * doc_lens[index] / avg_len)
            scores[index] += idf * tf * (K1 + 1) / denom
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Search personal memory (BM25 + metadata filters)")
    parser.add_argument("query", help="free-text query (keywords work best)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--start", help="only items on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end", help="only items on/before this date (YYYY-MM-DD)")
    parser.add_argument("--city", help="substring filter on city (case-insensitive)")
    parser.add_argument("--type", choices=("image", "video", "email", "any"), default="any")
    args = parser.parse_args()

    corpus_path = Path(__file__).parent / "search_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as handle:
        items = json.load(handle)

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

    query_tokens = tokenize(args.query)
    docs = [tokenize(item["text"]) for item in kept]
    scores = bm25_scores(query_tokens, docs)
    ranked = sorted(zip(scores, kept), key=lambda pair: -pair[0])[: args.top_k]

    for rank, (score, item) in enumerate(ranked, start=1):
        if score <= 0 and rank > 1:
            break
        snippet = item["text"][:100].replace("\n", " ")
        print(f"{rank:2d} | {score:6.2f} | {item['id']} | {item['type']:5s} | {item.get('ts') or '?':16s} | {(item.get('city') or '')[:24]:24s} | {snippet}")


if __name__ == "__main__":
    main()
