"""LLM query decomposer for multi-hop retrieval.

ATM-Bench-hard questions need many evidence items (median ~6, up to 17): a single
retrieval can't surface them all in the top-K. This decomposer asks an LLM to
break a question into atomic sub-queries; each is retrieved separately and the
rankings are fused (RRF), so evidence scattered across sub-aspects gets pulled
into the top-K.

Returns the list of sub-queries (always includes the original query as an anchor).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient

_SYSTEM_PROMPT = (
    "You are a query planner for a personal-memory retrieval system (photos, "
    "videos, emails). Given a question, output the minimal set of atomic SEARCH "
    "QUERIES whose retrieved evidence is needed to answer it.\n"
    "- For multi-part questions (list all X, count across several events/trips, "
    "sum costs over multiple items), produce ONE focused sub-query per part / per "
    "candidate event / per aspect (e.g. each city, each hotel, each leg of a trip).\n"
    "- Decompose by the entities/time-spans implied by the question.\n"
    "- For a simple single-fact question, return just one query.\n"
    "- Each sub-query: short, keyword/entity focused, retrieval-friendly (not a full sentence).\n"
    "Output STRICT JSON only: {\"subqueries\": [\"...\", \"...\"]}. 2-8 items."
)


def _extract_subqueries(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        subs = data.get("subqueries", [])
        return [str(s).strip() for s in subs if str(s).strip()]
    except Exception:
        return []


class QueryDecomposer:
    def __init__(self, llm_config: Dict[str, Any], max_subqueries: int = 8) -> None:
        self.llm = LLMClient(llm_config.get("provider", "vllm"), llm_config)
        self.max_subqueries = max_subqueries

    def decompose(self, query: str) -> Dict[str, Any]:
        """Return {'subqueries': [...], 'ok': bool, 'raw': str}.

        The list always starts with the original query (anchor), then the LLM's
        sub-queries (deduped). On failure → just [original].
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nReturn only the JSON. /no_think"},
        ]
        raw = ""
        subs: List[str] = []
        ok = False
        try:
            raw = self.llm.chat(messages)
            subs = _extract_subqueries(raw)
            ok = bool(subs)
        except Exception:
            subs = []

        # anchor original first, then unique sub-queries (case-insensitive dedupe)
        ordered = [query]
        seen = {query.strip().lower()}
        for s in subs:
            key = s.strip().lower()
            if key and key not in seen:
                ordered.append(s)
                seen.add(key)
            if len(ordered) >= self.max_subqueries + 1:
                break
        return {"subqueries": ordered, "ok": ok, "raw": raw, "n": len(ordered)}
