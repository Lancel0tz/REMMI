"""Hybrid retrieval over SGM memory items.

Implements a `HybridRetriever` that combines:
  1. Metadata filtering / boosting over SGM fields (date, location).
  2. Sparse retrieval (BM25) over the rendered SGM text + selected metadata.
  3. Dense retrieval via any existing `BaseRetriever` instance (e.g.
     `SentenceTransformerRetriever` with all-MiniLM-L6-v2).

Score fusion supports both Reciprocal Rank Fusion (RRF) and weighted-sum
fusion over min-max normalized per-channel scores.

This is the core of the ChronicleMem method (reported as ChronicleMem), built on the
ATM-Bench benchmark. See the top-level README for the full contribution map.
"""

from chronicle.hybrid_retriever import (
    HybridRetriever,
    HybridScoringConfig,
    QueryConstraints,
    parse_query_constraints,
)
from chronicle.routing_retriever import (
    RoutingRetriever,
    RoutingConfig,
    ConfidenceConfig,
    RouteStrategy,
    QuerySignals,
    AdaptiveWeights,
    analyze_query,
    adaptive_weights,
    adaptive_weights_hard,
    route,
)

__all__ = [
    "HybridRetriever",
    "HybridScoringConfig",
    "QueryConstraints",
    "parse_query_constraints",
    "RoutingRetriever",
    "RoutingConfig",
    "ConfidenceConfig",
    "RouteStrategy",
    "QuerySignals",
    "AdaptiveWeights",
    "analyze_query",
    "adaptive_weights",
    "adaptive_weights_hard",
    "route",
]
