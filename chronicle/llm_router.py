"""LLM-based router for hybrid retrieval.

Instead of hand-coded threshold heuristics (see ``routing_retriever.adaptive_weights``),
an LLM looks at each query and decides per-query fusion weights for the four
hybrid channels. This is the "non-heuristic" alternative the routing module's
docstring anticipated ("An LLM-based router is pluggable as a future ...").

The router returns a dict:
  {weight_metadata, weight_sparse, weight_dense, weight_vl, filter_mode, raw}

Weights are normalised to sum to 1 over {metadata, sparse, dense}; vl is an
additive visual channel in [0, ~0.4]. On any parse/LLM failure it falls back to
the provided default weights, so retrieval never breaks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient

_SYSTEM_PROMPT = (
    "You are a retrieval router for a personal-memory search engine. The memory "
    "store contains EMAILS, PHOTO metadata (caption/OCR/tags/location/timestamp) and "
    "VIDEO metadata. A hybrid retriever fuses four scoring channels; your job is to "
    "decide how much to weight each channel FOR THIS QUERY.\n\n"
    "Channels:\n"
    "- metadata: matches explicit date ranges / locations mentioned in the query "
    "(e.g. 'last week', 'in Porto', '2024'). Use when the query has concrete time/place constraints.\n"
    "- sparse: BM25 keyword match. Use when the query has distinctive keywords, proper "
    "nouns, named entities, or non-English terms that should match document text literally.\n"
    "- dense: semantic embedding similarity. Use for paraphrastic / conceptual queries "
    "with little exact-keyword overlap.\n"
    "- vl: CLIP image/visual similarity. Use ONLY when the query is about visual content "
    "(photos/videos of objects, scenes, landmarks, food, animals, 'what did X look like').\n\n"
    "Output STRICT JSON only, no prose:\n"
    '{\"weight_metadata\": <0..1>, \"weight_sparse\": <0..1>, \"weight_dense\": <0..1>, '
    '\"weight_vl\": <0..0.4>, \"filter_mode\": \"soft\"|\"hard\"}\n'
    "Rules: metadata+sparse+dense should sum to ~1. vl is ADDITIVE (0 if non-visual). "
    "Use filter_mode 'hard' ONLY for an explicit absolute date/location with no vagueness; "
    "use 'soft' for relative dates ('last week') or fuzzy constraints."
)


@dataclass
class RouterWeights:
    weight_metadata: float
    weight_sparse: float
    weight_dense: float
    weight_vl: float
    filter_mode: str
    raw: str = ""
    ok: bool = True


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # strip any Qwen3 <think>...</think> reasoning block first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # find first {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class LLMRouter:
    def __init__(
        self,
        llm_config: Dict[str, Any],
        default_weights: Dict[str, float],
        vl_max: float = 0.4,
    ) -> None:
        self.llm = LLMClient(llm_config.get("provider", "vllm"), llm_config)
        self.default = default_weights
        self.vl_max = vl_max

    def route(self, query: str) -> RouterWeights:
        # "/no_think" disables Qwen3 thinking (otherwise <think> eats the token
        # budget and no JSON is produced). Harmless for non-Qwen models.
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nReturn only the JSON object. /no_think"},
        ]
        raw = ""
        try:
            raw = self.llm.chat(messages)
            data = _extract_json(raw)
            if not data:
                raise ValueError("no JSON in router output")
            return self._normalise(data, raw)
        except Exception:
            d = self.default
            return RouterWeights(
                weight_metadata=d["weight_metadata"],
                weight_sparse=d["weight_sparse"],
                weight_dense=d["weight_dense"],
                weight_vl=d.get("weight_vl", 0.0),
                filter_mode=d.get("filter_mode", "soft"),
                raw=raw,
                ok=False,
            )

    def _normalise(self, data: Dict[str, Any], raw: str) -> RouterWeights:
        def f(key: str, default: float = 0.0) -> float:
            try:
                return max(0.0, float(data.get(key, default)))
            except Exception:
                return default

        wm, ws, wd = f("weight_metadata"), f("weight_sparse"), f("weight_dense")
        wv = min(self.vl_max, f("weight_vl"))
        total = wm + ws + wd
        if total <= 0:  # degenerate → fall back
            d = self.default
            wm, ws, wd = d["weight_metadata"], d["weight_sparse"], d["weight_dense"]
            total = wm + ws + wd
        wm, ws, wd = wm / total, ws / total, wd / total
        fm = str(data.get("filter_mode", "soft")).lower()
        if fm not in ("soft", "hard"):
            fm = "soft"
        return RouterWeights(wm, ws, wd, wv, fm, raw=raw, ok=True)
