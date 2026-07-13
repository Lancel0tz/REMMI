#!/usr/bin/env python3
"""Hybrid retriever for ATM-Bench SGM memory items.

Three channels:
  - metadata:  date-window + location-keyword filter / boost using
               `RetrievalItem.metadata` (timestamp, location, subject).
  - sparse:    BM25 over the rendered SGM text.
  - dense:     pluggable; any retriever exposing
               `retrieve(query, top_k) -> List[RetrievalResult]`
               or `BaseRetriever`-like with `embeddings` + `encode_query`.

Score fusion options:
  - RRF (default; robust, no per-channel calibration needed).
  - weighted_sum on min-max normalized per-channel scores.

The prototype intentionally avoids LLM calls; all metadata extraction is
done with lightweight, deterministic heuristics so we can iterate on a CPU
laptop. LLM-based extraction is a planned follow-up for ambiguous queries
(see project_proposal.pdf, Direction 3).
"""

from __future__ import annotations

import math
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

try:  # rank_bm25 is in A-Mem requirements, so it is acceptable here.
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "rank_bm25 is required for the hybrid retriever. "
        "Install with `pip install rank_bm25`."
    ) from exc

# We import RetrievalItem from a local lightweight shim that avoids pulling in
# torch when this module is used on its own (e.g. on a CPU sandbox during
# development). At runtime against the real codebase this falls through to
# `memqa.retrieve.utils.RetrievalItem`.
try:
    from memqa.retrieve.utils import RetrievalItem  # noqa: F401  (re-exported)
except Exception:  # pragma: no cover - sandbox fallback
    from remmi._retrieval_item import RetrievalItem  # type: ignore

# We deliberately do NOT import `BaseRetriever` from
# `memqa.retrieve.retrievers` because that module imports torch unconditionally.
# The hybrid retriever doesn't need any tensor operations; numpy is enough.
# It exposes the same `build_index` / `retrieve` interface as `BaseRetriever`.


@dataclass
class RetrievalResult:
    """Mirror of `memqa.retrieve.retrievers.RetrievalResult` to avoid the
    torch dependency at import time. Down-stream code can swap in the real
    class because both have identical fields.
    """

    item: "RetrievalItem"  # type: ignore[name-defined]
    score: float


class _DenseRetrieverProto(Protocol):
    """Structural type for a dense retriever the hybrid module can plug in.

    Anything with a ``retrieve(query, top_k) -> List[RetrievalResult-like]``
    method satisfies this contract — including the existing
    ``SentenceTransformerRetriever`` / ``Qwen3VLRetriever``.
    """

    def retrieve(self, query: str, top_k: int) -> List[Any]: ...


# ---------------------------------------------------------------------------
# Lightweight tokenization (mirrors A-Mem's simple_tokenize, but local)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# Query constraint parsing (heuristic; date + location)
# ---------------------------------------------------------------------------

_MONTH_LOOKUP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Year captured as 4 digits 1990-2099. Tighter than \d{4}.
_YEAR_RE = re.compile(r"\b(?P<year>(19|20)\d{2})\b")
_MONTH_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
# Numeric date forms YYYY-MM-DD or YYYY/MM/DD.
_ISO_DATE_RE = re.compile(
    r"\b(?P<year>(19|20)\d{2})[-/](?P<month>\d{1,2})(?:[-/](?P<day>\d{1,2}))?\b"
)
# "Today is YYYY-MM-DD" anchor used in ATM-Bench questions.
_TODAY_ANCHOR_RE = re.compile(
    r"today\s+is\s+(?P<year>(19|20)\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})",
    re.IGNORECASE,
)


@dataclass
class QueryConstraints:
    """Heuristic constraints extracted from a query.

    Attributes
    ----------
    date_range:
        Optional ``(start_date, end_date)`` inclusive window. ``None`` if no
        date hint was found.
    locations:
        Lowercased proper-noun tokens that may correspond to place names.
    today:
        If a "Today is YYYY-MM-DD" anchor was found, the resolved date.
    """

    date_range: Optional[Tuple[date, date]] = None
    locations: List[str] = field(default_factory=list)
    today: Optional[date] = None

    def has_date(self) -> bool:
        return self.date_range is not None

    def has_location(self) -> bool:
        return bool(self.locations)


def _maybe_year_month_to_range(
    year: int, month: Optional[int]
) -> Tuple[date, date]:
    if month is None:
        return date(year, 1, 1), date(year, 12, 31)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return date(year, month, 1), end


def _extract_locations(query: str, gazetteer: Optional[Sequence[str]] = None) -> List[str]:
    """Pick capitalized tokens that are not at the very start of the query
    and are not month names. If a gazetteer is provided, only tokens that
    appear in it (case-insensitive) are returned. Otherwise, capitalized
    bigrams are kept as a fallback.

    The fallback is intentionally conservative: it favors precision over
    recall so that the metadata channel rarely *hurts* baseline retrieval.
    """
    cleaned = query.strip()
    if not cleaned:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", cleaned)
    if not tokens:
        return []

    if gazetteer:
        gaz_lower = {g.lower() for g in gazetteer}
        return [t.lower() for t in tokens if t.lower() in gaz_lower]

    locations: List[str] = []
    for idx, tok in enumerate(tokens):
        if idx == 0:
            # The first token of a sentence is always capitalized; skip.
            continue
        if not tok[0].isupper():
            continue
        if tok.lower() in _MONTH_LOOKUP:
            continue
        if len(tok) < 3:
            continue
        locations.append(tok.lower())
    # de-dup preserving order
    seen: set = set()
    deduped: List[str] = []
    for loc in locations:
        if loc in seen:
            continue
        seen.add(loc)
        deduped.append(loc)
    return deduped


def parse_query_constraints(
    query: str,
    gazetteer: Optional[Sequence[str]] = None,
) -> QueryConstraints:
    """Best-effort extraction of date/location constraints from a question.

    The implementation is intentionally simple and side-effect free so it
    can be unit tested. It is *not* meant to be exhaustive — its purpose is
    to give the metadata channel something cheap and high-precision to work
    with. The proposal earmarks an LLM fallback for ambiguous queries.
    """
    q = query or ""

    today: Optional[date] = None
    today_match = _TODAY_ANCHOR_RE.search(q)
    if today_match:
        try:
            today = date(
                int(today_match.group("year")),
                int(today_match.group("month")),
                int(today_match.group("day")),
            )
        except ValueError:
            today = None

    date_range: Optional[Tuple[date, date]] = None
    iso_match = _ISO_DATE_RE.search(q)
    if iso_match:
        try:
            year = int(iso_match.group("year"))
            month = int(iso_match.group("month"))
            day = iso_match.group("day")
            if day:
                d = date(year, month, int(day))
                date_range = (d, d)
            else:
                date_range = _maybe_year_month_to_range(year, month)
        except ValueError:
            date_range = None
    else:
        year_match = _YEAR_RE.search(q)
        month_match = _MONTH_RE.search(q)
        if year_match and month_match:
            try:
                year = int(year_match.group("year"))
                month = _MONTH_LOOKUP[month_match.group("month").lower()]
                date_range = _maybe_year_month_to_range(year, month)
            except (KeyError, ValueError):
                date_range = None
        elif year_match:
            try:
                year = int(year_match.group("year"))
                date_range = _maybe_year_month_to_range(year, None)
            except ValueError:
                date_range = None

    locations = _extract_locations(q, gazetteer=gazetteer)

    # LLM query 理解: replace(默认)=LLM 抽到就替代; augment=heuristic 先跑+LLM 补漏
    # (地点并集, 日期 heuristic 优先、LLM 补漏)。设 ATM_QUERY_LLM 指向 by_text map 才生效。
    m = _query_llm_map()
    if m is not None and query in m:
        mode = os.environ.get("ATM_QUERY_MODE", "replace").lower()
        info = m[query]
        llm_locs = [str(x).strip().lower() for x in (info.get("locations") or []) if str(x).strip()]
        if mode == "replace":
            if llm_locs:
                locations = llm_locs
        else:  # augment: 并集 (heuristic ∪ LLM, 去重保序)
            locations = list(dict.fromkeys(list(locations) + llm_locs))
        ds, de = info.get("date_start"), info.get("date_end")
        if ds and de:
            try:
                llm_range = (date.fromisoformat(ds[:10]), date.fromisoformat(de[:10]))
                if mode == "replace" or date_range is None:
                    date_range = llm_range
            except ValueError:
                pass

    return QueryConstraints(date_range=date_range, locations=locations, today=today)


# ---------------------------------------------------------------------------
# Visual-query detection (for adaptive VL routing)
# ---------------------------------------------------------------------------

# 与 routing_retriever._VISUAL_QUERY_RE 同款：检测 query 是否提到
# photos/images/videos 或常见视觉物体/场景，VL embedding 对这类 query 有帮助。
# 在此处自包含复制一份，避免 hybrid_retriever ←→ routing_retriever 循环导入。
_VISUAL_QUERY_RE = re.compile(
    r"""
      \bphoto(?:s|graph)?\b
    | \bimage(?:s)?\b
    | \bpicture(?:s)?\b
    | \bvideo(?:s)?\b
    | \bsnapshot(?:s)?\b
    | \bscreenshot(?:s)?\b
    | \bselfie(?:s)?\b
    | \bsticker(?:s)?\b
    | \bcollect(?:ing|ion)\b
    | \bfind\s+all\b
    | \bsheep\b | \bcat\b | \bdog\b | \bflower\b | \bbird\b
    | \bbridge\b | \btower\b | \bcastle\b | \bmuseum\b
    | \bsunset\b | \bsunrise\b | \bbeach\b | \bmountain\b
    | \bfood\b | \bdish\b | \bdessert\b | \bcake\b
    | \blandmark\b | \bmonument\b | \bstatue\b
    | \blook(?:s|ed)?\s+like\b
    | \bwhat\s+(?:animal|building|object|place|scene)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# 收紧版门控 (ATM_VISUAL_GATE=tight, 默认): 只在出现具体视觉物体/场景 (靠外观辨识)
# 时触发, 砍掉食物词簇 (dessert/food/dish/cake → 餐厅/地点题误触发) 和过泛的通用媒体词
# (photo/video/collection/find all → 事件/人物/地点题误触发)。保留 list-recall 的物体识别类。
# hard set 上收紧后视觉子集 R@10 46→51, 整体转正 (+1.5); 设 ATM_VISUAL_GATE=default 回退 loose。
_VISUAL_QUERY_RE_TIGHT = re.compile(
    r"""
      \bsheep\b | \bcat\b | \bdog\b | \bflower\b | \bbird\b | \banimal\b
    | \bbridge\b | \btower\b | \bcastle\b | \bmuseum\b | \bbuilding\b
    | \bsunset\b | \bsunrise\b | \bbeach\b | \bmountain\b | \bscenery\b
    | \blandmark\b | \bmonument\b | \bstatue\b
    | \bbadge(?:s)?\b | \bposter(?:s)?\b | \bsign(?:s|board)?\b
    | \bslide(?:s)?\b | \blogo(?:s)?\b | \boutfit\b | \bdress\b | \bclothes\b
    | \blook(?:s|ed)?\s+like\b | \blooked?\b
    | \bwhat\s+(?:animal|building|object|place|scene|color|colour)\b
    | \bcolou?r\s+of\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── LLM query 理解 (替代正则/gazetteer 的 query 识别) ──────────────────────────
# 设 ATM_QUERY_LLM=<path> 指向 LLM query-extraction 输出 (by_text map:
# {query: {visual, locations, date_start, date_end}}), 则视觉/地点/日期改用 LLM 抽取;
# heuristic 仍作 fallback (query 不在 map 里)。channels/权重/融合一律不变。
_QUERY_LLM_MAP = None
_QUERY_LLM_LOADED = False


def _query_llm_map():
    global _QUERY_LLM_MAP, _QUERY_LLM_LOADED
    if not _QUERY_LLM_LOADED:
        _QUERY_LLM_LOADED = True
        path = os.environ.get("ATM_QUERY_LLM", "")
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                _QUERY_LLM_MAP = json.load(fh).get("by_text", {})
    return _QUERY_LLM_MAP


def _has_visual_hint(query: str) -> bool:
    gate = os.environ.get("ATM_VISUAL_GATE", "tight").lower()
    rx = _VISUAL_QUERY_RE if gate == "default" else _VISUAL_QUERY_RE_TIGHT
    heur = bool(rx.search(query or ""))
    m = _query_llm_map()
    if m is not None and query in m:
        llm = bool(m[query].get("visual"))
        mode = os.environ.get("ATM_QUERY_MODE", "replace").lower()
        # replace(默认) = 纯 LLM; augment = heuristic 先跑 + LLM 补漏 (并集 OR)
        return llm if mode == "replace" else (heur or llm)
    return heur


# ---------------------------------------------------------------------------
# Hybrid scoring config
# ---------------------------------------------------------------------------


@dataclass
class HybridScoringConfig:
    """Knobs for the hybrid retriever's score fusion."""

    fusion: str = "rrf"  # "rrf" or "weighted_sum"
    rrf_k: int = 60  # standard RRF constant
    weight_metadata: float = 0.3
    weight_sparse: float = 0.4
    weight_dense: float = 0.3
    weight_vl: float = 0.0  # vision-language channel (0 = disabled / non-visual queries)
    # VL routing: when `vl_adaptive` is True, the VL weight becomes per-query.
    # Queries with a visual hint (photos/images/videos/visual objects) use
    # `weight_vl_visual`; all other queries use `weight_vl`. The extra VL budget
    # on visual queries is taken from the dense channel (a "tilt"), keeping
    # metadata/sparse untouched. m/s/d 仍是你优化好的固定值。
    vl_adaptive: bool = False
    weight_vl_visual: float = 0.0  # VL weight for visual-hint queries (when vl_adaptive)
    # When `filter_mode = "hard"`, items failing date/location constraints are
    # dropped before fusion. When "soft" (default), the metadata channel only
    # contributes a positive boost.
    filter_mode: str = "soft"


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


def _parse_timestamp(value: Any) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # Common ATM-Bench timestamp shapes: "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD".
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    iso = _ISO_DATE_RE.search(text)
    if iso:
        try:
            return date(
                int(iso.group("year")),
                int(iso.group("month")),
                int(iso.group("day") or "1"),
            )
        except ValueError:
            return None
    return None


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return rank for every position; higher score → smaller rank (1-indexed).

    Items with score == -inf get rank == len(scores) + 1 (i.e. they
    contribute zero to RRF).
    """
    n = scores.shape[0]
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    # Items that were excluded (score = -inf) should contribute 0 in RRF.
    excluded = ~np.isfinite(scores)
    ranks[excluded] = n + 1
    return ranks


class HybridRetriever:
    """Hybrid retriever combining metadata, BM25, and dense channels.

    Exposes the same ``build_index`` / ``retrieve`` surface as
    ``memqa.retrieve.retrievers.BaseRetriever`` so it can be swapped into
    ``mmrag_retrieve_answer.py`` with only a small dispatch change.

    Parameters
    ----------
    cache_dir:
        Where caches would go (currently unused; reserved for parity with
        other retrievers and future BM25/index serialization).
    dense_retriever:
        Any object with a ``retrieve(query, top_k) -> List[...]`` method
        (e.g. ``SentenceTransformerRetriever`` after ``build_index`` was
        called on it). If ``None``, the dense channel is skipped.
    scoring:
        ``HybridScoringConfig`` instance.
    location_gazetteer:
        Optional set of place-name strings used to extract location hints
        from the query with high precision.
    """

    def __init__(
        self,
        cache_dir: Path,
        dense_retriever: Optional[_DenseRetrieverProto] = None,
        vl_retriever: Optional[_DenseRetrieverProto] = None,
        scoring: Optional[HybridScoringConfig] = None,
        location_gazetteer: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        del device  # accepted for API parity; no torch usage here
        self.cache_dir = Path(cache_dir)
        self.dense = dense_retriever
        self.vl = vl_retriever  # vision-language retriever (4th channel)
        self.scoring = scoring or HybridScoringConfig()
        self.gazetteer = list(location_gazetteer) if location_gazetteer else None
        self.items: List[RetrievalItem] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: List[List[str]] = []
        self._item_dates: List[Optional[date]] = []
        self._item_locations: List[str] = []

    # ---------------------------------------------------------------------
    # Index build
    # ---------------------------------------------------------------------

    def build_index(
        self,
        items: List[RetrievalItem],
        cache_config: Optional[Dict[str, Any]] = None,
        force_rebuild: bool = False,
    ) -> None:
        del cache_config, force_rebuild  # unused: hybrid index is cheap to rebuild
        self.items = list(items)
        if not items:
            self._bm25 = None
            self._tokenized = []
            self._item_dates = []
            self._item_locations = []
            return

        # BM25 over the SGM text, slightly augmented with metadata fields so
        # exact matches on entities/locations are easier.
        self._tokenized = [_tokenize(self._render_for_bm25(item)) for item in items]
        self._bm25 = BM25Okapi(self._tokenized)

        # Pre-extract metadata for the metadata channel.
        self._item_dates = [_parse_timestamp((it.metadata or {}).get("timestamp")) for it in items]
        self._item_locations = [
            ((it.metadata or {}).get("location") or "").lower() for it in items
        ]

    @staticmethod
    def _render_for_bm25(item: RetrievalItem) -> str:
        """Concatenate text + a few metadata fields so BM25 sees the full
        SGM signal.

        This stays consistent with how `mmrag_retrieve_answer.py` already
        formats memory items: timestamp + location + caption + ocr + tags.
        """
        meta = item.metadata or {}
        parts = [item.text or ""]
        for key in ("timestamp", "location", "subject"):
            val = meta.get(key)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    # ---------------------------------------------------------------------
    # Per-channel scoring
    # ---------------------------------------------------------------------

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self.items), dtype=np.float32)
        tokens = _tokenize(query)
        if not tokens:
            return np.zeros(len(self.items), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)

    def _dense_scores(self, query: str) -> np.ndarray:
        n = len(self.items)
        if self.dense is None or n == 0:
            return np.zeros(n, dtype=np.float32)
        # We re-use the dense retriever's index; align by item_id. The
        # dense retriever may return objects with `.item.item_id` (the
        # existing `RetrievalResult` shape) — we duck-type it.
        results = self.dense.retrieve(query, top_k=n)
        score_by_id: Dict[str, float] = {}
        for r in results:
            try:
                score_by_id[r.item.item_id] = float(r.score)
            except AttributeError:
                continue
        scores = np.zeros(n, dtype=np.float32)
        for i, item in enumerate(self.items):
            scores[i] = score_by_id.get(item.item_id, 0.0)
        return scores

    def _vl_scores(self, query: str) -> np.ndarray:
        """Vision-language channel scores (text query ↔ image embedding)."""
        n = len(self.items)
        if self.vl is None or n == 0:
            return np.zeros(n, dtype=np.float32)
        results = self.vl.retrieve(query, top_k=n)
        score_by_id: Dict[str, float] = {}
        for r in results:
            try:
                score_by_id[r.item.item_id] = float(r.score)
            except AttributeError:
                continue
        scores = np.zeros(n, dtype=np.float32)
        for i, item in enumerate(self.items):
            scores[i] = score_by_id.get(item.item_id, 0.0)
        return scores

    def _metadata_scores(
        self, constraints: QueryConstraints
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (raw_score, mask).

        ``mask[i] == False`` means item ``i`` violated a hard constraint and
        should be excluded when ``filter_mode == "hard"``.
        ``raw_score[i]`` is in [0, 1] and counts how many constraints item
        ``i`` satisfied.
        """
        n = len(self.items)
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.ones(0, dtype=bool)

        scores = np.zeros(n, dtype=np.float32)
        mask = np.ones(n, dtype=bool)
        contributing = 0

        if constraints.has_date():
            contributing += 1
            start, end = constraints.date_range  # type: ignore[misc]
            for i, d in enumerate(self._item_dates):
                if d is None:
                    # Don't penalize emails/items with missing timestamp.
                    scores[i] += 0.5
                elif start <= d <= end:
                    scores[i] += 1.0
                else:
                    mask[i] = False

        if constraints.has_location():
            contributing += 1
            for i, loc in enumerate(self._item_locations):
                if not loc:
                    scores[i] += 0.5  # neutral if metadata missing
                    continue
                if any(needle in loc for needle in constraints.locations):
                    scores[i] += 1.0
                else:
                    # location hard-fail keeps mask True (only date is hard);
                    # location is treated as a soft signal because gazetteer
                    # coverage is incomplete.
                    pass

        if contributing > 0:
            scores /= contributing
        return scores, mask

    # ---------------------------------------------------------------------
    # Retrieve
    # ---------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:  # type: ignore[override]
        if not self.items:
            return []

        constraints = parse_query_constraints(query, gazetteer=self.gazetteer)
        meta_scores, meta_mask = self._metadata_scores(constraints)
        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)

        # Per-query VL/dense weights (adaptive VL routing). When vl_adaptive is
        # off, this collapses to the fixed weight_vl / weight_dense behaviour.
        w_vl = self.scoring.weight_vl
        w_dense = self.scoring.weight_dense
        if self.scoring.vl_adaptive and self.vl is not None:
            if _has_visual_hint(query):
                w_vl = self.scoring.weight_vl_visual
                # Tilt: fund the extra VL budget from dense (m/s untouched).
                w_dense = max(0.0, self.scoring.weight_dense - (w_vl - self.scoring.weight_vl))
            else:
                w_vl = self.scoring.weight_vl  # base (e.g. 0 = VL off for non-visual)

        vl_scores = self._vl_scores(query) if (w_vl > 0 and self.vl is not None) else None

        if self.scoring.filter_mode == "hard":
            arrays = [meta_scores, bm25_scores, dense_scores]
            if vl_scores is not None:
                arrays.append(vl_scores)
            for arr in arrays:
                arr[~meta_mask] = -np.inf

        n = len(self.items)
        dummy_ranks = np.full(n, n + 1, dtype=np.float64)

        if self.scoring.fusion == "rrf":
            ranks_meta = _ranks_from_scores(meta_scores)
            ranks_sparse = _ranks_from_scores(bm25_scores)
            ranks_dense = (
                _ranks_from_scores(dense_scores)
                if self.dense is not None
                else dummy_ranks
            )
            ranks_vl = (
                _ranks_from_scores(vl_scores)
                if vl_scores is not None
                else dummy_ranks
            )
            k = max(1, int(self.scoring.rrf_k))
            fused = (
                self.scoring.weight_metadata / (k + ranks_meta)
                + self.scoring.weight_sparse / (k + ranks_sparse)
                + w_dense / (k + ranks_dense)
            )
            if w_vl > 0:
                fused += w_vl / (k + ranks_vl)
        elif self.scoring.fusion == "weighted_sum":
            fused = (
                self.scoring.weight_metadata * _minmax(meta_scores)
                + self.scoring.weight_sparse * _minmax(bm25_scores)
                + w_dense * _minmax(dense_scores)
            )
            if vl_scores is not None and w_vl > 0:
                fused += w_vl * _minmax(vl_scores)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.scoring.fusion}")

        if self.scoring.filter_mode == "hard":
            fused[~meta_mask] = -np.inf

        order = np.argsort(-fused, kind="stable")[: max(0, top_k)]
        return [
            RetrievalResult(item=self.items[idx], score=float(fused[idx]))
            for idx in order
            if math.isfinite(float(fused[idx]))
        ]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_synthetic_items(now: Optional[date] = None) -> List[RetrievalItem]:
    """A tiny in-memory corpus to smoke-test the retriever end-to-end.

    The corpus is *not* meant to look like real ATM-Bench data; it just
    exercises every channel (date, location, BM25 keyword) so unit tests
    and CI can run on CPU in seconds.
    """
    today = now or date(2023, 6, 15)

    def _ts(year: int, month: int, day: int) -> str:
        return f"{year:04d}-{month:02d}-{day:02d} 12:00:00"

    items: List[RetrievalItem] = []
    items.append(
        RetrievalItem(
            item_id="20230401_120000",
            modality="image",
            text=(
                "ID: 20230401_120000\nTimestamp: 2023-04-01 12:00:00\n"
                "Location: Lisbon, Portugal\nShort Caption: hotel lobby\n"
                "Tags: travel, hotel"
            ),
            metadata={"source": "image", "timestamp": _ts(2023, 4, 1), "location": "Lisbon, Portugal"},
        )
    )
    items.append(
        RetrievalItem(
            item_id="20230402_180000",
            modality="image",
            text=(
                "ID: 20230402_180000\nTimestamp: 2023-04-02 18:00:00\n"
                "Location: Porto, Portugal\nShort Caption: dinner with Grace\n"
                "Tags: food, family"
            ),
            metadata={"source": "image", "timestamp": _ts(2023, 4, 2), "location": "Porto, Portugal"},
        )
    )
    items.append(
        RetrievalItem(
            item_id="email00021",
            modality="email",
            text=(
                "ID: email00021\nTimestamp: 2023-03-20 09:00:00\n"
                "Summary: hotel booking confirmation\n"
                "Detail: Booking confirmation for two nights in Lisbon."
            ),
            metadata={"source": "email", "timestamp": _ts(2023, 3, 20), "location": ""},
        )
    )
    items.append(
        RetrievalItem(
            item_id="20221225_140000",
            modality="image",
            text=(
                "ID: 20221225_140000\nTimestamp: 2022-12-25 14:00:00\n"
                "Location: Cambridge, UK\nShort Caption: Christmas dinner\n"
                "Tags: holiday, food"
            ),
            metadata={"source": "image", "timestamp": _ts(2022, 12, 25), "location": "Cambridge, United Kingdom"},
        )
    )
    items.append(
        RetrievalItem(
            item_id="20231001_100000",
            modality="image",
            text=(
                "ID: 20231001_100000\nTimestamp: 2023-10-01 10:00:00\n"
                "Location: Tokyo, Japan\nShort Caption: ramen shop\n"
                "Tags: travel, food"
            ),
            metadata={"source": "image", "timestamp": _ts(2023, 10, 1), "location": "Tokyo, Japan"},
        )
    )
    return items
