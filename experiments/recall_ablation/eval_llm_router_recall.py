#!/usr/bin/env python3
"""Compare retrieval R@k under three routing schemes:

  1. fixed     — the Bayesian-optimized fixed weights (best_routing_config.json)
  2. heuristic — routing_retriever.adaptive_weights() hand-coded thresholds
  3. llm       — LLMRouter: an LLM picks per-query weights (the non-heuristic option)

Retrieval-only: no answer generation, so it's fast/cheap. Needs an LLM endpoint
(self-hosted vLLM) for the router. Outputs R@{1,5,10,25,50,100} for each scheme
plus per-query router decisions.
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltma import HybridRetriever, HybridScoringConfig
from ltma.llm_router import LLMRouter
from ltma.routing_retriever import (
    analyze_query,
    adaptive_weights,
    adaptive_weights_hard,
)
from memqa.retrieve.utils import (
    EmailTextConfig,
    MediaTextConfig,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever, VisionRetriever

EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"

RECALL_KS = [1, 5, 10, 25, 50, 100]


def build_llm_config(args) -> Dict[str, Any]:
    api_base = (args.api_base or "").rstrip("/")
    endpoint = api_base if api_base.endswith("/chat/completions") else f"{api_base}/chat/completions"
    return {
        "provider": "vllm",
        "model": args.model,
        "api_key": args.api_key or "",
        "endpoint": endpoint,
        "max_tokens": 512,
        "temperature": 0.0,
        "timeout": 90,
    }


def set_weights(retriever: HybridRetriever, w: Dict[str, float]) -> None:
    sc = retriever.scoring
    sc.weight_metadata = w["weight_metadata"]
    sc.weight_sparse = w["weight_sparse"]
    sc.weight_dense = w["weight_dense"]
    sc.weight_vl = w.get("weight_vl", 0.0)
    sc.vl_adaptive = False
    sc.filter_mode = w.get("filter_mode", "soft")


def recall_at_ks(retrieved: List[str], gold: set) -> Dict[int, float]:
    if not gold:
        return {k: 0.0 for k in RECALL_KS}
    return {k: len(gold & set(retrieved[:k])) / len(gold) for k in RECALL_KS}


def avg_recall(per_q: List[Dict[int, float]]) -> Dict[str, float]:
    out = {}
    n = max(len(per_q), 1)
    for k in RECALL_KS:
        out[f"R@{k}"] = round(sum(d[k] for d in per_q) / n * 100, 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-file", required=True)
    p.add_argument("--config-file", required=True, help="fixed-weight routing config json")
    p.add_argument("--dataset", choices=["standard", "hard"], default="hard",
                   help="which heuristic variant: adaptive_weights vs adaptive_weights_hard")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vl-embedding-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--retriever-batch-size", type=int, default=64)
    p.add_argument("--vl-batch-size", type=int, default=16)
    p.add_argument("--media-source", default="batch_results")
    p.add_argument("--force-rebuild", action="store_true")
    # router LLM
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--router-workers", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data + building retrievers ...")
    qa_list = load_json(Path(args.qa_file))
    config = load_json(Path(args.config_file))
    best = config["hybrid_scoring_config"]
    fixed_w = {
        "weight_metadata": best["weight_metadata"],
        "weight_sparse": best["weight_sparse"],
        "weight_dense": best["weight_dense"],
        "weight_vl": best.get("weight_vl", 0.0),
        "filter_mode": best.get("filter_mode", "soft"),
    }

    items = build_retrieval_items(
        email_entries=load_json(EMAIL_FILE),
        image_entries=load_json(IMAGE_BATCH),
        video_entries=load_json(VIDEO_BATCH),
        media_text_config=MediaTextConfig(),
        email_text_config=EmailTextConfig(),
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
    )
    text_retriever = SentenceTransformerRetriever(
        model_name=args.text_embedding_model, cache_dir=INDEX_CACHE,
        batch_size=args.retriever_batch_size, device=args.device,
    )
    text_retriever.build_index(items, {"retriever": "hybrid",
        "text_embedding_model": args.text_embedding_model,
        "media_source": args.media_source}, force_rebuild=args.force_rebuild)
    vl_retriever = VisionRetriever(
        model_name=args.vl_embedding_model, cache_dir=INDEX_CACHE,
        batch_size=args.vl_batch_size, device=args.device,
    )
    vl_retriever.build_index(items, {"retriever": "vision_vl_image_only",
        "vl_embedding_model": args.vl_embedding_model, "media_source": args.media_source,
        "embedding_space": "clip_image_text_shared"}, force_rebuild=args.force_rebuild)

    retriever = HybridRetriever(
        cache_dir=INDEX_CACHE, dense_retriever=text_retriever,
        vl_retriever=vl_retriever,
        scoring=HybridScoringConfig(**{k: fixed_w[k] for k in
            ["weight_metadata", "weight_sparse", "weight_dense", "weight_vl", "filter_mode"]},
            rrf_k=best.get("rrf_k", 60), fusion=best.get("fusion", "rrf")),
    )
    retriever.build_index(items)

    router = LLMRouter(build_llm_config(args), default_weights=fixed_w)
    hjf = adaptive_weights_hard if args.dataset == "hard" else adaptive_weights

    # ── 1) LLM router decisions (parallel network calls) ──
    print(f"Routing {len(qa_list)} queries via LLM ({args.router_workers} workers) ...")
    questions = [(q.get("id") or q.get("question_id"), q["question"], set(extract_evidence_ids(q))) for q in qa_list]
    llm_w: Dict[str, Any] = {}
    lock = threading.Lock()

    def do_route(item):
        qid, question, _ = item
        rw = router.route(question)
        with lock:
            llm_w[qid] = rw
    with ThreadPoolExecutor(max_workers=args.router_workers) as ex:
        list(tqdm(ex.map(do_route, questions), total=len(questions), desc="LLM routing"))

    router_fail = sum(1 for v in llm_w.values() if not v.ok)
    print(f"  router parse failures (fell back to fixed): {router_fail}/{len(questions)}")

    # ── 2) Retrieve under each scheme, compute R@k ──
    print("Retrieving under fixed / heuristic / llm ...")
    rec = {"fixed": [], "heuristic": [], "llm": []}
    decisions = []
    for qid, question, gold in tqdm(questions, desc="Retrieve"):
        sig = analyze_query(question)
        aw = hjf(sig)
        heur_w = {"weight_metadata": aw.weight_metadata, "weight_sparse": aw.weight_sparse,
                  "weight_dense": aw.weight_dense, "weight_vl": getattr(aw, "weight_vl", 0.0),
                  "filter_mode": aw.filter_mode}
        rw = llm_w[qid]
        llm_weights = {"weight_metadata": rw.weight_metadata, "weight_sparse": rw.weight_sparse,
                       "weight_dense": rw.weight_dense, "weight_vl": rw.weight_vl,
                       "filter_mode": rw.filter_mode}
        schemes = {"fixed": fixed_w, "heuristic": heur_w, "llm": llm_weights}
        row = {"id": qid, "question": question[:120]}
        for name, w in schemes.items():
            set_weights(retriever, w)
            ranked = [r.item.item_id for r in retriever.retrieve(question, top_k=200)]
            rec[name].append(recall_at_ks(ranked, gold))
            if name == "llm":
                row["llm_weights"] = {k: round(v, 3) for k, v in w.items() if isinstance(v, float)}
                row["llm_filter"] = w["filter_mode"]
                row["llm_ok"] = rw.ok
                row["llm_raw"] = (rw.raw or "")[:300]
                row["heur_strategy"] = aw.strategy_name
        decisions.append(row)

    summary = {name: avg_recall(rec[name]) for name in rec}
    summary["router_fail"] = router_fail
    summary["count"] = len(questions)

    (out_dir / "router_recall_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "router_decisions.jsonl", "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    print("\n===== R@k 对比 =====")
    print("  scheme       " + "  ".join(f"R@{k}" for k in RECALL_KS))
    for name in ["fixed", "heuristic", "llm"]:
        s = summary[name]
        print(f"  {name:10s} " + "  ".join(f"{s[f'R@{k}']:5.1f}" for k in RECALL_KS))
    print(f"\n  saved: {out_dir}/router_recall_summary.json + router_decisions.jsonl")


if __name__ == "__main__":
    main()
