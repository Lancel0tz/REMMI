"""Build the compact search corpus consumed by the in-sandbox search tool.

This is the projection layer of "REMMI retrieval as an agent tool": every
memory item (image / video / email) becomes one compact searchable record

    {"id", "type", "ts", "city", "text"}

where ``text`` concatenates the sparse-channel fields REMMI's hybrid retriever
uses (short_caption, tags, entities, location) — small enough to ship into the
per-question sandbox (~2 MB vs 29 MB raw SGM), rich enough for BM25 shortlists.
The dense / vision channels of the full REMMI retriever need model weights and
are intentionally excluded from the sandbox tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from remmi.organize.heuristic import _clean_city, _stem


def _media_record(record: dict[str, Any], path_key: str, kind: str, rich: bool = False) -> dict[str, Any]:
    # ocr_text belongs in the INDEXED text, not just in `detail`. The offline
    # hybrid retriever can afford to rank on captions alone, but an agent asks
    # value-shaped questions ("how much was the dentist", "invoice total") and
    # those tokens exist only on the receipt itself. With ocr out of the index a
    # search for an amount returns nothing, the agent trusts the tool and stops
    # -- measured: every number question the agent got wrong had its gold receipt
    # ranked 1-3 by the offline retriever but invisible to this tool.
    parts = [
        str(record.get("short_caption") or ""),
        " ".join(str(t) for t in record.get("tags") or []),
        " ".join(str(e) for e in record.get("entities") or []),
        str(record.get("location_name") or ""),
        str(record.get("ocr_text") or "")[:600],
    ]
    out = {
        "id": _stem(record.get(path_key, "")),
        "type": kind,
        "ts": str(record.get("timestamp") or "")[:16],
        "city": _clean_city(record.get("city")),
        "text": " ".join(p for p in parts if p).strip(),
    }
    if rich:
        # Fuller projection so `search.py --show` can replace raw-file
        # verification reads (the dominant token cost of org_remmi v1).
        out["detail"] = {
            "location": str(record.get("location_name") or "")[:160],
            "caption": str(record.get("caption") or "")[:400],
            "ocr": str(record.get("ocr_text") or "")[:600],
            "tags": [str(t) for t in (record.get("tags") or [])][:10],
            "entities": [str(e) for e in (record.get("entities") or [])][:10],
        }
    return out


def _email_record(record: dict[str, Any], rich: bool = False) -> dict[str, Any]:
    out = {
        "id": str(record.get("id") or ""),
        "type": "email",
        "ts": str(record.get("timestamp") or "")[:16],
        "city": "",
        # Same reasoning as media: booking/receipt mail carries the price in
        # `detail`, never in the one-line summary.
        "text": (str(record.get("short_summary") or "").strip() + " "
                 + str(record.get("detail") or "")[:600]).strip(),
    }
    if rich:
        out["detail"] = {"detail": str(record.get("detail") or "")[:500]}
    return out


def build_search_corpus(
    image_source: Path,
    video_source: Path,
    emails_source: Path,
    rich: bool = False,
) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    with open(image_source, "r", encoding="utf-8") as handle:
        corpus.extend(_media_record(r, "image_path", "image", rich) for r in json.load(handle))
    with open(video_source, "r", encoding="utf-8") as handle:
        corpus.extend(_media_record(r, "video_path", "video", rich) for r in json.load(handle))
    with open(emails_source, "r", encoding="utf-8") as handle:
        corpus.extend(_email_record(r, rich) for r in json.load(handle))
    return [c for c in corpus if c["id"]]
