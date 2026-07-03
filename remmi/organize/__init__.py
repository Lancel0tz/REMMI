"""Event-based memory organization for REMMI (the *Episodic* pillar).

Organisers group long-term personal memory items (images, videos, emails)
into events/trips and emit a compact ``organized_memory.json`` index that a
QA agent can read *before* (or instead of) scanning the full per-item files.

Two organisers share one output schema:

- :mod:`remmi.organize.heuristic` — deterministic day-gap + city-change
  clustering. No LLM calls; CPU-only.
- :mod:`remmi.organize.agent_static` — an LLM agent performs a full-corpus
  organisation pass upfront (map-reduce over item batches) against any
  OpenAI-compatible endpoint.

The third strategy explored on top of these, *dynamic* organisation, needs no
offline organiser at all: the answering agent organises only the memory
relevant to the current question inside its own workspace (see
``agent_systems/prompts/system_prompt_org_dynamic.txt``).
"""

from remmi.organize.heuristic import (
    HeuristicConfig,
    MediaItem,
    build_organized_memory,
    cluster_events,
    load_media_items,
)

__all__ = [
    "HeuristicConfig",
    "MediaItem",
    "build_organized_memory",
    "cluster_events",
    "load_media_items",
]
