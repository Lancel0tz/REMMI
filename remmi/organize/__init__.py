"""Event-based memory organization for REMMI (the *Episodic* pillar).

Organisers group long-term personal memory items (images, videos, emails)
into events/trips and emit a compact ``organized_memory.json`` index that a
QA agent can read *before* (or instead of) scanning the full per-item files.

The organisers here all run OFFLINE, outside the answering agent's loop. They
differ in scope, and the first two share one output schema:

- :mod:`remmi.organize.heuristic` — deterministic day-gap + city-change
  clustering over the whole corpus. No LLM calls; CPU-only.
- :mod:`remmi.organize.agent_static` — an LLM organises the whole corpus once,
  upfront (map-reduce over item batches) against any OpenAI-compatible endpoint.
- :mod:`remmi.organize.dynamic` — per question instead of per corpus: retrieve
  the top-N relevant items, organise just those in ONE stateless LLM call, and
  inject the result at run time as ``memory/query_events.json``.

The contrasting family lives outside this package: the ``org_remmi*`` modes
(see ``agent_systems/prompts/``) have no offline organiser at all — the answering
agent organises in-session, writing ``timeline.md`` in its own workspace.
``org_remmi0`` does this with grep alone; ``org_remmi``..``org_remmi9`` add the
REMMI search tool (:mod:`remmi.organize.search_pack`).
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
