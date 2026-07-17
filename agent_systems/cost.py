#!/usr/bin/env python3
"""Price a run's token usage against Tokdash's pricing database.

Cost, not raw token count, is the comparable unit across our methods: an organise
pass on gpt-5-mini and an answer pass on Qwen3.6-27B are not the same token.

Token vocabulary (kept distinct because they bill at ~10x different rates):
  uncached  input tokens sent and actually processed        -> `input` rate
  cache_write  input tokens stored into the prompt cache    -> `cache_write` rate
  cache_read   input tokens served from the prompt cache    -> `cache_read` rate (~10% of input)
  output    generated tokens                                -> `output` rate
  billed    = uncached + cache_write + cache_read + output  (everything charged)

Caveat on our own numbers: our vLLM endpoint reports no cache tokens at all, so
every input token prices as full-rate uncached. A hosted API with prompt caching
would charge ~10% for the resent prefix, which matters most for the in-session
organisers (org_remmi*) that re-send a growing context every turn. So the costs
here are an UPPER bound for those methods, and near-exact for the offline ones
(one stateless call, nothing to cache). Never quote these as billing truth --
they are a reference conversion.

Usage:
    python3 scripts/cost.py --tokens-in 6960604 --tokens-out 130204 --model qwen3.6-27b
    python3 scripts/cost.py --usage-summary output/.../usage_summary.json --model qwen3.6-27b
    python3 scripts/cost.py --organise-usage output/events/dyn_events_orgd/_organise_usage.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

def _find_pricing_db() -> Path:
    """Locate tokdash's pricing_db.json without hardcoding one install path.

    Order: $TOKDASH_PRICING_DB, then any tokdash package importable/installed in
    the common venv/conda site-packages, then a last-resort known location.
    """
    env = os.environ.get("TOKDASH_PRICING_DB")
    if env:
        return Path(env)
    for base in (
        Path.home() / "opt/tokdash-venv/lib",
        Path.home() / ".local/lib",
        Path(os.environ.get("CONDA_PREFIX", "/nonexistent")) / "lib",
    ):
        for hit in base.glob("python3*/site-packages/tokdash/pricing_db.json"):
            return hit
    return Path(
        "/home/kz345/opt/tokdash-venv/lib/python3.11/site-packages/tokdash/pricing_db.json"
    )


PRICING_DB = _find_pricing_db()

# Our served model ids -> pricing_db keys. The vLLM tag carries a quantisation
# suffix and a vendor prefix that the price list does not use.
MODEL_ALIASES = {
    "Qwen/Qwen3.6-27B-FP8": "qwen3.6-27b",
    "openai-compatible_Qwen_Qwen3.6-27B-FP8": "qwen3.6-27b",
    "Qwen/Qwen3.5-9B": "qwen3.5-9b",
}


def load_prices(model: str, db_path: Path = PRICING_DB) -> dict:
    with open(db_path, "r", encoding="utf-8") as handle:
        db = json.load(handle)
    key = MODEL_ALIASES.get(model, model)
    key = db.get("aliases", {}).get(key, key)
    if key not in db["models"]:
        raise SystemExit(
            f"model {model!r} (-> {key!r}) not in pricing_db {db['version']}.\n"
            f"Add it to MODEL_ALIASES, or pass a key that exists."
        )
    return db["models"][key] | {"_key": key, "_db_version": db["version"]}


def price(
    *,
    uncached: int,
    output: int,
    cache_read: int = 0,
    cache_write: int = 0,
    prices: dict,
) -> dict:
    per = 1_000_000
    cost = {
        "uncached": uncached * prices["input"] / per,
        "cache_write": cache_write * prices["cache_write"] / per,
        "cache_read": cache_read * prices["cache_read"] / per,
        "output": output * prices["output"] / per,
    }
    billed_tokens = uncached + cache_write + cache_read + output
    return {
        "model": prices["_key"],
        "pricing_db": prices["_db_version"],
        "tokens": {
            "uncached": uncached,
            "cache_write": cache_write,
            "cache_read": cache_read,
            "output": output,
            "billed": billed_tokens,
        },
        "cost_usd": {k: round(v, 4) for k, v in cost.items()} | {
            "total": round(sum(cost.values()), 4)
        },
        "cache_hit_rate": round(cache_read / (uncached + cache_read), 4)
        if (uncached + cache_read)
        else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--usage-summary", help="pi harness usage_summary.json")
    src.add_argument("--organise-usage", help="_organise_usage.json from remmi.organize.dynamic")
    src.add_argument("--tokens-in", type=int, help="raw uncached input tokens")
    ap.add_argument("--tokens-out", type=int, default=0)
    ap.add_argument("--cache-read", type=int, default=0)
    ap.add_argument("--cache-write", type=int, default=0)
    ap.add_argument("--model", help="model id or pricing_db key (default: read from the file)")
    ap.add_argument("--pricing-db", help="path to tokdash pricing_db.json "
                    "(default: $TOKDASH_PRICING_DB or auto-discovered)")
    ap.add_argument("--label", default="", help="tag for the printed line")
    args = ap.parse_args()
    db_path = Path(args.pricing_db) if args.pricing_db else PRICING_DB

    if args.usage_summary:
        u = json.loads(Path(args.usage_summary).read_text())
        model = args.model or u.get("model_tag", "")
        fields = dict(
            uncached=u.get("sum_input_tokens_uncached", u.get("sum_input_tokens", 0)),
            output=u.get("sum_output_tokens", 0),
            cache_read=u.get("sum_cache_read_input_tokens", 0),
            cache_write=u.get("sum_cache_creation_input_tokens", 0),
        )
    elif args.organise_usage:
        u = json.loads(Path(args.organise_usage).read_text())
        model = args.model or u.get("model", "")
        fields = dict(uncached=u.get("input_tokens", 0), output=u.get("output_tokens", 0))
    else:
        model = args.model
        if not model:
            raise SystemExit("--model is required with --tokens-in")
        fields = dict(
            uncached=args.tokens_in, output=args.tokens_out,
            cache_read=args.cache_read, cache_write=args.cache_write,
        )

    result = price(prices=load_prices(model, db_path), **fields)
    if args.label:
        result["label"] = args.label
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
