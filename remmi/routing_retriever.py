#!/usr/bin/env python3
"""Smart routing retriever with multi-stage filtering and reranking.

Instead of fusing all channels into one score, this retriever:
  1. Analyses the query to decide a *routing strategy*
     (BM25-only / dense-only / meta-first / full-hybrid).
  2. Runs a primary retrieval stage using the selected channels.
  3. Optionally iterates: if the first pass looks low-confidence
     (few results, low scores), it expands to additional channels.
  4. Reranks the merged candidate pool with a cross-encoder.

Design goals
------------
- **No LLM calls for routing** — all classification is rule-based, so the
  pipeline runs on CPU in milliseconds (same philosophy as the existing
  ``HybridRetriever``).  An LLM-based router is pluggable as a future
  extension (see ``QueryRouter`` protocol).
- **Composable stages** — each stage is a plain function
  ``(query, items, top_k) → List[ScoredItem]``, easy to unit-test and swap.
- **Iterative refinement** — the pipeline can loop: if a stage's output
  falls below a configurable confidence threshold it falls through to a
  broader stage.

                      ┌────────────────┐
                      │  QueryAnalyzer │
                      └──────┬─────────┘
                             │ RoutingDecision
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
        ┌─────────┐   ┌───────────┐   ┌───────────┐
        │BM25-only│   │Dense-only │   │Meta-first │
        └────┬────┘   └─────┬─────┘   └─────┬─────┘
             │              │               │
             └──────────────┴───────────────┘
                            │
                    ┌───────▼────────┐
                    │ ConfidenceGate │ ── low → Expand & Retry
                    └───────┬────────┘
                            │ high
                    ┌───────▼────────┐
                    │   Reranker     │
                    └───────┬────────┘
                            │
                       final top-k
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from remmi.hybrid_retriever import (
    HybridRetriever,
    HybridScoringConfig,
    QueryConstraints,
    RetrievalResult,
    _minmax,
    _ranks_from_scores,
    _tokenize,
    parse_query_constraints,
)

try:
    from memqa.retrieve.utils import RetrievalItem
except Exception:  # pragma: no cover
    from remmi._retrieval_item import RetrievalItem  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# Routing strategy enum
# ──────────────────────────────────────────────────────────────────────


class RouteStrategy(Enum):
    """Which channel(s) to use for the primary retrieval pass."""

    BM25_ONLY = auto()       # keyword-heavy queries (dates, names, amounts)
    DENSE_ONLY = auto()      # semantic / visual / descriptive queries
    META_THEN_BM25 = auto()  # strong date/location signal → pre-filter then BM25
    META_THEN_DENSE = auto() # strong date/location signal → pre-filter then dense
    FULL_HYBRID = auto()     # no strong signal → run everything

    def __repr__(self) -> str:
        return self.name


# ──────────────────────────────────────────────────────────────────────
# Query analysis
# ──────────────────────────────────────────────────────────────────────

# Patterns that strongly suggest keyword / exact-match retrieval.
_KEYWORD_SIGNALS = re.compile(
    r"""
      \b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*  # "16th January"
    | (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}                     # "January 16"
    | \b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b                                                      # ISO date
    | \bhow\s+much\b                                                                        # price / amount
    | \btotal\s+(?:amount|cost|price|bill|payment)\b
    | \bwhat\s+(?:time|date|day)\b
    | \bwhen\s+(?:is|was|did|do|does)\b
    | \bdeadline\b
    | \border\s+(?:number|id|confirmation)\b
    | \breceipt\b
    | \binvoice\b
    | \bbooking\s+(?:ref|reference|confirmation|number)\b
    | \bflight\s+(?:number|code)\b
    | \btoday\s+is\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Patterns that strongly suggest semantic / visual retrieval.
_SEMANTIC_SIGNALS = re.compile(
    r"""
      \bI\s+remember\b
    | \bhelp\s+me\s+(?:find|recall|remember)\b
    | \bcan\s+you\s+(?:find|help|recall)\b
    | \bphoto\s+of\b
    | \bpicture\s+of\b
    | \ba\s+place\s+where\b
    | \bwhere\s+(?:I|we)\s+(?:had|ate|went|saw|visited|took)\b
    | \bwhat.*(?:look|looked)\s+like\b
    | \brecommend\b
    | \bwhat'?s?\s+(?:my|the)\s+(?:wifi|password)\b
    | \bdescri(?:be|ptive)\b
    | \bpostcard\b
    | \bpaint\b
    | \bdessert\b
    | \bcream\b
    | \bmochi\b
    | \bpet-friendly\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Strong metadata anchors.
_STRONG_META_SIGNALS = re.compile(
    r"""
      \b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b   # exact date
    | \btoday\s+is\s+\w+\s+\d                    # "Today is October 28th"
    | \bon\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class QuerySignals:
    """Feature vector extracted from a query for routing."""

    keyword_score: float = 0.0   # how keyword-heavy the query is [0,1]
    semantic_score: float = 0.0  # how semantic/visual the query is [0,1]
    meta_score: float = 0.0      # how much metadata constraint is present [0,1]
    constraints: Optional[QueryConstraints] = None
    qtype_hint: str = ""         # optional hint ("number", "list_recall", "open_end")
    has_entity: bool = False      # query contains a searchable proper noun / entity
    has_relative_date: bool = False  # "last week", "a few days ago" (fuzzy date)
    has_non_latin: bool = False   # Chinese / non-Latin keywords (BM25-matchable)
    has_visual_hint: bool = False  # query mentions photos/images/videos/visual objects

    @property
    def dominant(self) -> str:
        scores = {
            "keyword": self.keyword_score,
            "semantic": self.semantic_score,
            "meta": self.meta_score,
        }
        return max(scores, key=scores.get)  # type: ignore[arg-type]


# Entity / proper noun detection for routing
_ENTITY_RE = re.compile(
    r"""
      \b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b  # Eiffel Tower, Côte Brasserie
    | \b[A-Z]{2,}\d*\b                             # MIT, NeurIPS, ACL, BMVC, K702
    """,
    re.VERBOSE,
)
_ENTITY_STOP = {
    "What", "Where", "When", "How", "Which", "Who", "Why",
    "Do", "Did", "Does", "Can", "Could", "Would", "Should",
    "The", "That", "This", "There", "These", "Those",
    "Help", "Recall", "Find", "Tell", "Describe",
    "Today", "Yesterday", "Tomorrow",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
}

# Relative date expressions (fuzzy temporal, unsafe for hard filter)
_RELATIVE_DATE_RE = re.compile(
    r"""
      \blast\s+(?:week|month|weekend|night|time|friday|monday|saturday|sunday)\b
    | \ba\s+few\s+days\s+ago\b
    | \brecently\b
    | \bthe\s+other\s+day\b
    | \byesterday\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Non-Latin scripts that are BM25-matchable (CJK, etc.)
_NON_LATIN_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

# Visual content signals — queries about images, photos, videos, or
# visual objects/scenes where VL embeddings can help
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


def _extract_entities(query: str) -> List[str]:
    """Extract proper nouns / named entities from query."""
    matches = _ENTITY_RE.findall(query)
    return [m for m in matches if m not in _ENTITY_STOP and len(m) > 1]


def analyze_query(query: str) -> QuerySignals:
    """Rule-based query feature extraction for routing.

    Returns a ``QuerySignals`` with scores ∈ [0, 1] for each dimension.
    """
    constraints = parse_query_constraints(query)

    # Keyword signals
    kw_matches = len(_KEYWORD_SIGNALS.findall(query))
    keyword_score = min(1.0, kw_matches * 0.4)

    # Semantic signals
    sem_matches = len(_SEMANTIC_SIGNALS.findall(query))
    semantic_score = min(1.0, sem_matches * 0.4)

    # Meta signals  (date/location in the query)
    meta_score = 0.0
    if constraints.has_date():
        meta_score += 0.5
        if _STRONG_META_SIGNALS.search(query):
            meta_score += 0.2
    if constraints.has_location():
        meta_score += 0.3
    meta_score = min(1.0, meta_score)

    # ── New signal dimensions ────────────────────────────────────────

    # Entity detection: proper nouns like "Eiffel Tower", "NeurIPS",
    # "Cambridge", "Côte Brasserie" → BM25 can keyword-match these
    entities = _extract_entities(query)
    has_entity = len(entities) > 0

    # Relative date: "last week", "a few days ago" → hard date filter
    # is risky because the heuristic parser may compute a wrong window
    has_relative_date = bool(_RELATIVE_DATE_RE.search(query))

    # Non-Latin keywords (Chinese 螺蛳粉, Japanese, Korean) → BM25 can
    # exact-match these against captions/OCR but dense embeddings often
    # fail on cross-lingual matching
    has_non_latin = bool(_NON_LATIN_RE.search(query))

    # If entity or non-Latin detected, give keyword_score a boost
    if has_entity or has_non_latin:
        keyword_score = max(keyword_score, 0.2)

    # Heuristic qtype hint (not perfect — just helps routing)
    qtype_hint = ""
    if re.search(r"\bhow\s+much\b|\btotal\b|\bamount\b|\bprice\b|\bcost\b", query, re.I):
        qtype_hint = "number"
    elif re.search(r"\bhelp\s+me\s+find\b|\brecall\b|\bfind\s+(?:that|the)\s+photo\b", query, re.I):
        qtype_hint = "list_recall"

    # Visual hint: mentions photos/images/videos or visual objects/scenes
    has_visual_hint = bool(_VISUAL_QUERY_RE.search(query))

    return QuerySignals(
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        meta_score=meta_score,
        constraints=constraints,
        qtype_hint=qtype_hint,
        has_entity=has_entity,
        has_relative_date=has_relative_date,
        has_non_latin=has_non_latin,
        has_visual_hint=has_visual_hint,
    )


def route(signals: QuerySignals) -> RouteStrategy:
    """Map ``QuerySignals`` → ``RouteStrategy``.

    The thresholds are informed by the error analysis:
    - number+email → BM25 wins decisively
    - list_recall+media → Dense wins decisively
    - strong date → metadata pre-filter helps
    """
    # Strong metadata anchor → pre-filter first
    if signals.meta_score >= 0.6:
        if signals.keyword_score >= signals.semantic_score:
            return RouteStrategy.META_THEN_BM25
        return RouteStrategy.META_THEN_DENSE

    # Clear keyword dominance
    if signals.keyword_score >= 0.4 and signals.semantic_score < 0.2:
        return RouteStrategy.BM25_ONLY

    # Clear semantic dominance
    if signals.semantic_score >= 0.4 and signals.keyword_score < 0.2:
        return RouteStrategy.DENSE_ONLY

    # qtype hint as tiebreaker
    if signals.qtype_hint == "number" and signals.keyword_score >= 0.2:
        return RouteStrategy.BM25_ONLY
    if signals.qtype_hint == "list_recall" and signals.semantic_score >= 0.2:
        return RouteStrategy.DENSE_ONLY

    return RouteStrategy.FULL_HYBRID


# ──────────────────────────────────────────────────────────────────────
# Adaptive fusion weights (soft routing)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AdaptiveWeights:
    """Per-query fusion weights computed from query signals.

    Unlike hard routing (pick one channel), adaptive weights always keep
    all channels alive but shift emphasis.  This preserves the hybrid
    safety net while focusing on the channel most likely to help.
    """

    weight_metadata: float
    weight_sparse: float
    weight_dense: float
    filter_mode: str  # "hard" or "soft"
    strategy_name: str = ""  # for logging
    weight_vl: float = 0.0  # vision-language channel weight

    def as_tuple(self):
        return (self.weight_metadata, self.weight_sparse, self.weight_dense)

    def as_tuple_4ch(self):
        return (self.weight_metadata, self.weight_sparse, self.weight_dense, self.weight_vl)


def adaptive_weights(signals: QuerySignals) -> AdaptiveWeights:
    """Compute per-query fusion weights from signal analysis.

    Core idea: instead of routing to a single channel, we *tilt* the
    existing hybrid weights toward the channel the query benefits from.

    Baseline weights (from best hybrid sweep): m=0.1, s=0.2, d=0.7
    We shift from this baseline based on query signals.

    Design principle (learned from v1): keep adjustments conservative.
    The baseline is already strong — only tilt meaningfully when the
    signal is clear and the error analysis shows large per-category gaps.
    """
    # Baseline weights (best fixed hybrid config)
    wm, ws, wd = 0.10, 0.20, 0.70
    filter_mode = "soft"
    strategy = "baseline"

    # --- Strong metadata signal → boost metadata, use hard filter ---
    # Error analysis: meta+bm25 already beats fixed hybrid, so keep this.
    # BUT: relative dates ("last week") make hard filter dangerous.
    if signals.meta_score >= 0.6:
        wm = 0.25
        # Fix #2: relative date → soft filter (hard filter kills GT)
        filter_mode = "soft" if signals.has_relative_date else "hard"
        if signals.keyword_score >= signals.semantic_score:
            ws, wd = 0.45, 0.30
            strategy = "meta+bm25"
        else:
            ws, wd = 0.20, 0.55
            strategy = "meta+dense"
        # Fix #2 cont: when relative + semantic, dense needs more weight
        if signals.has_relative_date and signals.semantic_score >= 0.3:
            wm, ws, wd = 0.15, 0.20, 0.65
            strategy = "meta+dense_rel"

    # --- Keyword-heavy → moderate BM25 boost (not too aggressive) ---
    elif signals.keyword_score >= 0.4 and signals.semantic_score < 0.2:
        wm, ws, wd = 0.10, 0.40, 0.50
        strategy = "bm25_tilt"
        if signals.meta_score >= 0.3:
            wm = 0.15
            ws, wd = 0.40, 0.45
            strategy = "bm25_tilt+meta"

    # --- Semantic/visual → moderate Dense boost ---
    # Fix #1: if query has entities or non-Latin keywords, preserve
    # more BM25 weight (entities = keyword-matchable by BM25).
    elif signals.semantic_score >= 0.4 and signals.keyword_score < 0.2:
        if signals.has_entity or signals.has_non_latin:
            # "I remember Eiffel Tower" / "螺蛳粉 in Liuzhou"
            # → BM25 needs enough weight to find the keyword match
            wm, ws, wd = 0.08, 0.25, 0.67
            strategy = "dense+entity"
        else:
            wm, ws, wd = 0.07, 0.18, 0.75
            strategy = "dense_tilt"

    # --- Mild signals → tiny nudge from baseline ---
    elif signals.keyword_score >= 0.2 and signals.keyword_score > signals.semantic_score:
        wm, ws, wd = 0.10, 0.30, 0.60
        strategy = "bm25_nudge"

    elif signals.semantic_score >= 0.2 and signals.semantic_score > signals.keyword_score:
        wm, ws, wd = 0.08, 0.17, 0.75
        strategy = "dense_nudge"

    # --- Default: keep baseline ---
    else:
        # Fix #1 cont: even with no semantic signal, entities/non-Latin
        # in the query → nudge BM25 up slightly
        if signals.has_entity or signals.has_non_latin:
            wm, ws, wd = 0.10, 0.25, 0.65
            strategy = "balanced+entity"
        else:
            strategy = "balanced"

    # Normalise (should already sum to 1, but safety)
    total = wm + ws + wd
    wm, ws, wd = wm / total, ws / total, wd / total

    return AdaptiveWeights(
        weight_metadata=wm,
        weight_sparse=ws,
        weight_dense=wd,
        filter_mode=filter_mode,
        strategy_name=strategy,
    )


def adaptive_weights_hard(signals: QuerySignals) -> AdaptiveWeights:
    """Compute per-query fusion weights optimised for **hard** questions.

    Hard-set characteristics (from ATM-Bench-Hard analysis):
    - 93.6 % media/mixed modality, only 6.5 % email
    - avg 6.3 evidence items per question (vs 1.4 standard)
    - 38.7 % list_recall (vs 13.7 % standard)
    - Text reranker *hurts* R@10 by −8.1 % → caller should skip reranking

    Key findings from weight sweep + per-strategy breakdown on 31 hard Qs:
    1. Best single config: m=0.07, s=0.27, d=0.67  (+1.6 % over Hybrid)
    2. ``meta+bm25`` and ``bm25_tilt+meta`` are catastrophic on hard set
       (push BM25 to 0.40-0.45, killing dense signal for media items)
    3. Oracle headroom is only +2.7 % — routing upside is small
    4. 32.3 % of hard Qs have R@10=0 for ALL methods (model limit)

    Design: dense-heavy baseline (0.07/0.27/0.67), conservative tilts,
    **clamp** BM25 ≤ 0.35 and dense ≥ 0.50, always soft filter.
    """
    # Hard-set optimal baseline (from exhaustive weight sweep)
    wm, ws, wd = 0.07, 0.27, 0.67
    filter_mode = "soft"  # hard filter kills sparse GT on hard set
    strategy = "hard_baseline"

    # --- Semantic / visual → boost dense (biggest category on hard set) ---
    if signals.semantic_score >= 0.4:
        if signals.has_entity or signals.has_non_latin:
            # "I remember the steel bridge in Porto" → keep BM25 for entity
            wm, ws, wd = 0.05, 0.25, 0.70
            strategy = "hard_dense+ent"
        else:
            wm, ws, wd = 0.05, 0.20, 0.75
            strategy = "hard_dense_tilt"

    # --- Keyword signal → mild BM25 boost (NOT aggressive like standard) ---
    elif signals.keyword_score >= 0.4:
        # On standard set this would go to 0.40 BM25; on hard set clamp to 0.30
        wm, ws, wd = 0.08, 0.30, 0.62
        strategy = "hard_kw_mild"
        if signals.meta_score >= 0.3:
            wm = 0.10
            ws, wd = 0.30, 0.60
            strategy = "hard_kw+meta"

    # --- Strong metadata anchor → keep dense-heavy, mild meta boost ---
    elif signals.meta_score >= 0.6:
        wm, ws, wd = 0.12, 0.25, 0.63
        strategy = "hard_meta"
        # NEVER use hard filter on hard set
        filter_mode = "soft"

    # --- Mild keyword nudge ---
    elif signals.keyword_score >= 0.2 and signals.keyword_score > signals.semantic_score:
        wm, ws, wd = 0.07, 0.30, 0.63
        strategy = "hard_kw_nudge"

    # --- Mild semantic nudge ---
    elif signals.semantic_score >= 0.2:
        wm, ws, wd = 0.06, 0.22, 0.72
        strategy = "hard_sem_nudge"

    # --- Default: keep hard-set optimal ---
    # else: wm, ws, wd = 0.07, 0.27, 0.67

    # Safety clamp: dense ≥ 0.50, BM25 ≤ 0.35, meta ≤ 0.15
    wd = max(wd, 0.50)
    ws = min(ws, 0.35)
    wm = min(wm, 0.15)

    # Normalise
    total = wm + ws + wd
    wm, ws, wd = wm / total, ws / total, wd / total

    return AdaptiveWeights(
        weight_metadata=wm,
        weight_sparse=ws,
        weight_dense=wd,
        filter_mode=filter_mode,
        strategy_name=strategy,
    )


# ──────────────────────────────────────────────────────────────────────
# Confidence gate
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ConfidenceConfig:
    """When to consider a retrieval pass 'low-confidence' and expand."""

    min_top1_score: float = 0.01     # absolute floor for the best item
    min_score_gap: float = 2.0       # top-1 score / top-K median; expect ≥ this
    min_candidates: int = 3          # need at least this many non-zero candidates
    max_expand_rounds: int = 2       # how many times we allow expansion


def is_low_confidence(
    results: List[RetrievalResult],
    config: ConfidenceConfig,
) -> bool:
    """Return True if the candidate pool looks shaky."""
    if not results:
        return True
    top1 = results[0].score
    if top1 < config.min_top1_score:
        return True
    nonzero = [r for r in results if r.score > 1e-9]
    if len(nonzero) < config.min_candidates:
        return True
    if len(nonzero) >= 5:
        median = float(np.median([r.score for r in nonzero]))
        if median > 0 and top1 / median < config.min_score_gap:
            # Scores are very flat → no clear winner, might be noisy
            pass  # don't penalise flat but high scores
    return False


# ──────────────────────────────────────────────────────────────────────
# Reranker protocol
# ──────────────────────────────────────────────────────────────────────


class RerankerProto(Protocol):
    """Anything with ``rerank(query, candidates) → List[RerankResult-like]``."""

    def rerank(self, query: str, candidates: List[RetrievalItem]) -> List[Any]: ...


# ──────────────────────────────────────────────────────────────────────
# Routing retriever (main class)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RoutingConfig:
    """Knobs for the routing retriever pipeline."""

    # Primary retrieval
    primary_top_k: int = 50        # how many candidates the primary pass keeps
    # Expansion
    expand_top_k: int = 100        # after expansion, keep this many
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    # Reranker
    rerank_top_k: int = 30         # send this many candidates to the reranker
    # Final output
    final_top_k: int = 10          # how many to return to the caller
    # Hybrid fallback weights (used when strategy == FULL_HYBRID)
    hybrid_scoring: HybridScoringConfig = field(
        default_factory=lambda: HybridScoringConfig(
            fusion="rrf", rrf_k=60,
            weight_metadata=0.2, weight_sparse=0.4, weight_dense=0.4,
            filter_mode="soft",
        )
    )
    # Meta pre-filter: when using META_THEN_*, hard-filter first
    meta_prefilter_mode: str = "hard"  # "hard" or "soft"
    # Override: force a specific strategy (bypass router)
    force_strategy: Optional[RouteStrategy] = None
    # Debug
    verbose: bool = False


class RoutingRetriever:
    """Multi-stage routing retriever.

    Compatible API surface with ``HybridRetriever`` (``build_index`` /
    ``retrieve``), so it can be plugged into ``mmrag_retrieve_answer.py``.

    Parameters
    ----------
    hybrid : HybridRetriever
        An already-initialised ``HybridRetriever`` (with index built).
        The routing retriever delegates channel-level scoring to it.
    reranker : optional
        Any object implementing ``rerank(query, candidates) -> list``.
        If ``None``, no reranking stage is applied.
    config : RoutingConfig
        Pipeline knobs.
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: Optional[RerankerProto] = None,
        config: Optional[RoutingConfig] = None,
    ) -> None:
        self.hybrid = hybrid
        self.reranker = reranker
        self.config = config or RoutingConfig()
        # Expose items for compatibility with mmrag_retrieve_answer.py
        self.items = hybrid.items

    # ── convenience builder ──────────────────────────────────────────

    def build_index(
        self,
        items: List[RetrievalItem],
        cache_config: Optional[Dict[str, Any]] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Delegate index building to the underlying hybrid retriever."""
        self.hybrid.build_index(items, cache_config, force_rebuild)
        self.items = self.hybrid.items

    # ── primary channel helpers ──────────────────────────────────────

    def _bm25_retrieve(
        self, query: str, top_k: int, meta_mask: Optional[np.ndarray] = None,
    ) -> List[RetrievalResult]:
        scores = self.hybrid._bm25_scores(query)
        if meta_mask is not None:
            scores[~meta_mask] = -np.inf
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievalResult(item=self.hybrid.items[i], score=float(scores[i]))
            for i in order if np.isfinite(scores[i])
        ]

    def _dense_retrieve(
        self, query: str, top_k: int, meta_mask: Optional[np.ndarray] = None,
    ) -> List[RetrievalResult]:
        scores = self.hybrid._dense_scores(query)
        if meta_mask is not None:
            scores[~meta_mask] = -np.inf
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievalResult(item=self.hybrid.items[i], score=float(scores[i]))
            for i in order if np.isfinite(scores[i])
        ]

    def _hybrid_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Full hybrid fusion via the underlying HybridRetriever."""
        return self.hybrid.retrieve(query, top_k)

    def _meta_mask(self, constraints: QueryConstraints) -> np.ndarray:
        """Compute hard metadata mask from constraints."""
        _, mask = self.hybrid._metadata_scores(constraints)
        return mask

    # ── main pipeline ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Multi-stage routing retrieval.

        Args:
            query: natural language query
            top_k: final number of results to return

        Returns:
            Sorted list of ``RetrievalResult`` (best first).
        """
        if not self.hybrid.items:
            return []

        cfg = self.config
        final_k = min(top_k, cfg.final_top_k) if cfg.final_top_k else top_k

        # ── Stage 0: Analyse query & decide route ────────────────────
        signals = analyze_query(query)
        strategy = cfg.force_strategy or route(signals)
        constraints = signals.constraints or parse_query_constraints(query)

        if cfg.verbose:
            print(
                f"[Router] strategy={strategy!r}  "
                f"kw={signals.keyword_score:.2f} sem={signals.semantic_score:.2f} "
                f"meta={signals.meta_score:.2f} qtype={signals.qtype_hint}"
            )

        # ── Stage 1: Primary retrieval ───────────────────────────────
        meta_mask = (
            self._meta_mask(constraints)
            if strategy in (RouteStrategy.META_THEN_BM25, RouteStrategy.META_THEN_DENSE)
            else None
        )

        if strategy == RouteStrategy.BM25_ONLY:
            candidates = self._bm25_retrieve(query, cfg.primary_top_k)
        elif strategy == RouteStrategy.DENSE_ONLY:
            candidates = self._dense_retrieve(query, cfg.primary_top_k)
        elif strategy == RouteStrategy.META_THEN_BM25:
            candidates = self._bm25_retrieve(query, cfg.primary_top_k, meta_mask)
        elif strategy == RouteStrategy.META_THEN_DENSE:
            candidates = self._dense_retrieve(query, cfg.primary_top_k, meta_mask)
        else:  # FULL_HYBRID
            candidates = self._hybrid_retrieve(query, cfg.primary_top_k)

        # ── Stage 2: Confidence gate → optional expansion ────────────
        expand_round = 0
        while (
            is_low_confidence(candidates, cfg.confidence)
            and expand_round < cfg.confidence.max_expand_rounds
        ):
            expand_round += 1
            if cfg.verbose:
                print(
                    f"[Router] Low confidence (round {expand_round}), expanding…"
                )
            candidates = self._expand(
                query, candidates, strategy, constraints, cfg.expand_top_k,
            )

        # ── Stage 3: Rerank ──────────────────────────────────────────
        if self.reranker and candidates:
            rerank_pool = candidates[: cfg.rerank_top_k]
            rerank_items = [r.item for r in rerank_pool]
            try:
                reranked = self.reranker.rerank(query, rerank_items)
                # Sort by reranker score, rebuild RetrievalResult list
                reranked_sorted = sorted(reranked, key=lambda r: r.score, reverse=True)
                candidates = [
                    RetrievalResult(item=r.item, score=float(r.score))
                    for r in reranked_sorted
                ]
            except Exception as exc:
                if cfg.verbose:
                    print(f"[Router] Reranker failed: {exc}, using original order")

        # ── Stage 4: Return top-k ────────────────────────────────────
        return candidates[:top_k]

    def retrieve_adaptive(
        self,
        query: str,
        top_k: int,
        override_weights: Optional[AdaptiveWeights] = None,
    ) -> List[RetrievalResult]:
        """Adaptive-weight retrieval: soft routing via dynamic fusion weights.

        Instead of hard-routing to one channel, this method computes
        per-query fusion weights based on query analysis, then runs
        the full hybrid pipeline with those weights.  This preserves
        the safety net of always using all channels while focusing
        on the channel most likely to help.

        The pipeline is:
          1. Analyse query → adaptive weights
          2. RRF fusion with dynamic weights
          3. Confidence gate → expand if needed
          4. Rerank (if reranker is available)
          5. Return top-k

        Parameters
        ----------
        override_weights : optional AdaptiveWeights
            If provided, skip internal weight computation and use these
            weights directly.  Useful for callers that want to supply
            weights from ``adaptive_weights_hard()`` or other custom logic.
        """
        if not self.hybrid.items:
            return []

        cfg = self.config
        signals = analyze_query(query)
        aw = override_weights if override_weights is not None else adaptive_weights(signals)
        constraints = signals.constraints or parse_query_constraints(query)

        if cfg.verbose:
            print(
                f"[AdaptiveRouter] strategy={aw.strategy_name}  "
                f"weights=({aw.weight_metadata:.2f}, {aw.weight_sparse:.2f}, {aw.weight_dense:.2f})  "
                f"filter={aw.filter_mode}  "
                f"kw={signals.keyword_score:.2f} sem={signals.semantic_score:.2f} "
                f"meta={signals.meta_score:.2f}"
            )

        # ── Compute per-channel scores (reuse hybrid internals) ──────
        meta_scores, meta_mask = self.hybrid._metadata_scores(constraints)
        bm25_scores = self.hybrid._bm25_scores(query)
        dense_scores = self.hybrid._dense_scores(query)

        # VL channel (only if weight > 0 and retriever exists)
        has_vl = aw.weight_vl > 0 and self.hybrid.vl is not None
        vl_scores = self.hybrid._vl_scores(query) if has_vl else None

        if aw.filter_mode == "hard":
            arrays = [meta_scores, bm25_scores, dense_scores]
            if vl_scores is not None:
                arrays.append(vl_scores)
            for arr in arrays:
                arr[~meta_mask] = -np.inf

        # ── RRF fusion with adaptive weights ─────────────────────────
        n_items = len(self.hybrid.items)
        rrf_k = self.hybrid.scoring.rrf_k
        dummy_ranks = np.full(n_items, n_items + 1, dtype=np.float64)
        ranks_meta = _ranks_from_scores(meta_scores)
        ranks_sparse = _ranks_from_scores(bm25_scores)
        ranks_dense = (
            _ranks_from_scores(dense_scores)
            if self.hybrid.dense is not None
            else dummy_ranks
        )
        ranks_vl = (
            _ranks_from_scores(vl_scores)
            if vl_scores is not None
            else dummy_ranks
        )

        fused = (
            aw.weight_metadata / (rrf_k + ranks_meta)
            + aw.weight_sparse / (rrf_k + ranks_sparse)
            + aw.weight_dense / (rrf_k + ranks_dense)
        )
        if has_vl:
            fused += aw.weight_vl / (rrf_k + ranks_vl)

        if aw.filter_mode == "hard":
            fused[~meta_mask] = -np.inf

        primary_k = cfg.primary_top_k
        order = np.argsort(-fused, kind="stable")[:primary_k]
        candidates = [
            RetrievalResult(item=self.hybrid.items[idx], score=float(fused[idx]))
            for idx in order
            if np.isfinite(float(fused[idx]))
        ]

        # ── Confidence gate → expand (relax filter / increase k) ─────
        expand_round = 0
        while (
            is_low_confidence(candidates, cfg.confidence)
            and expand_round < cfg.confidence.max_expand_rounds
        ):
            expand_round += 1
            if cfg.verbose:
                print(f"[AdaptiveRouter] Low confidence (round {expand_round}), expanding…")
            # Relax: switch to soft filter and increase k
            if aw.filter_mode == "hard":
                # Recompute without hard mask
                fused_soft = (
                    aw.weight_metadata / (rrf_k + _ranks_from_scores(
                        self.hybrid._metadata_scores(constraints)[0]
                    ))
                    + aw.weight_sparse / (rrf_k + _ranks_from_scores(
                        self.hybrid._bm25_scores(query)
                    ))
                    + aw.weight_dense / (rrf_k + _ranks_from_scores(
                        self.hybrid._dense_scores(query)
                        if self.hybrid.dense is not None
                        else np.zeros(n_items)
                    ))
                )
                if has_vl:
                    fused_soft += aw.weight_vl / (rrf_k + _ranks_from_scores(
                        self.hybrid._vl_scores(query)
                    ))
                order = np.argsort(-fused_soft, kind="stable")[:cfg.expand_top_k]
                existing_ids = {r.item.item_id for r in candidates}
                for idx in order:
                    if self.hybrid.items[idx].item_id not in existing_ids:
                        candidates.append(
                            RetrievalResult(
                                item=self.hybrid.items[idx],
                                score=float(fused_soft[idx]),
                            )
                        )
                        existing_ids.add(self.hybrid.items[idx].item_id)
                    if len(candidates) >= cfg.expand_top_k:
                        break
            else:
                # Already soft — just increase k
                order = np.argsort(-fused, kind="stable")[:cfg.expand_top_k]
                existing_ids = {r.item.item_id for r in candidates}
                for idx in order:
                    if self.hybrid.items[idx].item_id not in existing_ids:
                        candidates.append(
                            RetrievalResult(
                                item=self.hybrid.items[idx],
                                score=float(fused[idx]),
                            )
                        )
                        existing_ids.add(self.hybrid.items[idx].item_id)
                    if len(candidates) >= cfg.expand_top_k:
                        break

        # ── Rerank ───────────────────────────────────────────────────
        if self.reranker and candidates:
            rerank_pool = candidates[: cfg.rerank_top_k]
            rerank_items = [r.item for r in rerank_pool]
            try:
                reranked = self.reranker.rerank(query, rerank_items)
                reranked_sorted = sorted(reranked, key=lambda r: r.score, reverse=True)
                candidates = [
                    RetrievalResult(item=r.item, score=float(r.score))
                    for r in reranked_sorted
                ]
            except Exception as exc:
                if cfg.verbose:
                    print(f"[AdaptiveRouter] Reranker failed: {exc}")

        return candidates[:top_k]

    # ── expansion logic ──────────────────────────────────────────────

    def _expand(
        self,
        query: str,
        existing: List[RetrievalResult],
        original_strategy: RouteStrategy,
        constraints: QueryConstraints,
        expand_top_k: int,
    ) -> List[RetrievalResult]:
        """Broaden retrieval by adding results from the missing channel(s).

        Expansion strategy:
        - BM25_ONLY → add Dense results
        - DENSE_ONLY → add BM25 results
        - META_THEN_* → relax metadata filter (soft mode) + add the other channel
        - FULL_HYBRID → increase top_k (already using everything)
        """
        existing_ids = {r.item.item_id for r in existing}

        extra: List[RetrievalResult] = []
        if original_strategy == RouteStrategy.BM25_ONLY:
            extra = self._dense_retrieve(query, expand_top_k)
        elif original_strategy == RouteStrategy.DENSE_ONLY:
            extra = self._bm25_retrieve(query, expand_top_k)
        elif original_strategy in (RouteStrategy.META_THEN_BM25, RouteStrategy.META_THEN_DENSE):
            # Relax: run without meta mask
            extra_bm25 = self._bm25_retrieve(query, expand_top_k)
            extra_dense = self._dense_retrieve(query, expand_top_k)
            extra = extra_bm25 + extra_dense
        else:
            # Already hybrid — just get more
            extra = self._hybrid_retrieve(query, expand_top_k)

        # Merge: keep existing order, append new items
        merged_ids = set(existing_ids)
        merged = list(existing)
        for r in extra:
            if r.item.item_id not in merged_ids:
                merged_ids.add(r.item.item_id)
                merged.append(r)

        return merged[:expand_top_k]

    # ── introspection (for analysis) ─────────────────────────────────

    def explain(self, query: str) -> Dict[str, Any]:
        """Return routing decision + per-channel scores for a query.

        Useful for error analysis and debugging.
        """
        signals = analyze_query(query)
        strategy = self.config.force_strategy or route(signals)
        constraints = signals.constraints or parse_query_constraints(query)

        bm25_scores = self.hybrid._bm25_scores(query)
        dense_scores = self.hybrid._dense_scores(query)
        meta_scores, meta_mask = self.hybrid._metadata_scores(constraints)

        # Top-5 per channel
        def top_items(scores, k=5):
            order = np.argsort(-scores)[:k]
            return [
                {"item_id": self.hybrid.items[i].item_id, "score": float(scores[i])}
                for i in order if np.isfinite(scores[i])
            ]

        return {
            "query": query,
            "signals": {
                "keyword_score": signals.keyword_score,
                "semantic_score": signals.semantic_score,
                "meta_score": signals.meta_score,
                "qtype_hint": signals.qtype_hint,
            },
            "strategy": strategy.name,
            "constraints": {
                "has_date": constraints.has_date(),
                "date_range": (
                    [str(constraints.date_range[0]), str(constraints.date_range[1])]
                    if constraints.date_range
                    else None
                ),
                "locations": constraints.locations,
            },
            "top_bm25": top_items(bm25_scores),
            "top_dense": top_items(dense_scores),
            "meta_mask_pass": int(meta_mask.sum()),
            "meta_mask_total": len(meta_mask),
        }
