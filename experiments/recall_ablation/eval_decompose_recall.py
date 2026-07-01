#!/usr/bin/env python3
"""Multi-hop retrieval via query decomposition — recall comparison.

  single     — single-shot retrieval on the original query (fixed weights)
  decomposed — LLM splits the query into sub-queries; each is retrieved (same
               fixed weights); rankings are fused with RRF.

Same fixed weights for both → isolates the effect of decomposition. Reports
R@k overall and split by number of gold-evidence items (multi-hop questions are
where decomposition should help).
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
from ltma.query_decomposer import QueryDecomposer
from memqa.retrieve.utils import (
    EmailTextConfig, MediaTextConfig, build_retrieval_items,
    extract_evidence_ids, load_json,
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever, VisionRetriever

EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"
RECALL_KS = [1, 5, 10, 25, 50, 100]


def build_llm_config(args):
    api_base = (args.api_base or "").rstrip("/")
    endpoint = api_base if api_base.endswith("/chat/completions") else f"{api_base}/chat/completions"
    return {"provider": "vllm", "model": args.model, "api_key": args.api_key or "",
            "endpoint": endpoint, "max_tokens": 512, "temperature": 0.0, "timeout": 90}


def rrf_merge(rankings: List[List[str]], k_rrf: int = 60, top: int = 200,
              anchor_weight: float = 1.0) -> List[str]:
    """RRF over rankings. rankings[0] is the original query (anchor); it gets
    `anchor_weight`x weight so its high-precision top-K isn't diluted by the
    sub-query results (sub-queries add tail coverage)."""
    score: Dict[str, float] = {}
    for i, ranking in enumerate(rankings):
        w = anchor_weight if i == 0 else 1.0
        for rank, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + w / (k_rrf + rank + 1)
    return [it for it, _ in sorted(score.items(), key=lambda x: -x[1])][:top]


def recall_at_ks(retrieved: List[str], gold: set) -> Dict[int, float]:
    if not gold:
        return {k: 0.0 for k in RECALL_KS}
    return {k: len(gold & set(retrieved[:k])) / len(gold) for k in RECALL_KS}


def avg(per_q):
    n = max(len(per_q), 1)
    return {f"R@{k}": round(sum(d[k] for d in per_q) / n * 100, 2) for k in RECALL_KS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-file", required=True)
    p.add_argument("--config-file", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vl-embedding-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--retriever-batch-size", type=int, default=64)
    p.add_argument("--vl-batch-size", type=int, default=16)
    p.add_argument("--media-source", default="batch_results")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retrieve-topk", type=int, default=200)
    p.add_argument("--anchor-weight", type=float, default=2.0,
                   help="RRF weight on the original query vs sub-queries (>1 protects R@10)")
    # match the SOTA base: VL-adaptive routing (best hard config), optional
    p.add_argument("--vl-adaptive", action="store_true",
                   help="use VL-adaptive base (visual queries boost VL from dense) — the best hard config")
    p.add_argument("--weight-vl-visual", type=float, default=0.25)
    p.add_argument("--weight-vl-base", type=float, default=0.0)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data + building retrievers ...")
    qa_list = load_json(Path(args.qa_file))
    config = load_json(Path(args.config_file))
    best = config["hybrid_scoring_config"]

    items = build_retrieval_items(
        email_entries=load_json(EMAIL_FILE), image_entries=load_json(IMAGE_BATCH),
        video_entries=load_json(VIDEO_BATCH), media_text_config=MediaTextConfig(),
        email_text_config=EmailTextConfig(), image_root=IMAGE_ROOT, video_root=VIDEO_ROOT)
    text_retriever = SentenceTransformerRetriever(
        model_name=args.text_embedding_model, cache_dir=INDEX_CACHE,
        batch_size=args.retriever_batch_size, device=args.device)
    text_retriever.build_index(items, {"retriever": "hybrid",
        "text_embedding_model": args.text_embedding_model, "media_source": args.media_source},
        force_rebuild=args.force_rebuild)
    vl_retriever = VisionRetriever(model_name=args.vl_embedding_model, cache_dir=INDEX_CACHE,
        batch_size=args.vl_batch_size, device=args.device)
    vl_retriever.build_index(items, {"retriever": "vision_vl_image_only",
        "vl_embedding_model": args.vl_embedding_model, "media_source": args.media_source,
        "embedding_space": "clip_image_text_shared"}, force_rebuild=args.force_rebuild)
    scoring = HybridScoringConfig(
        weight_metadata=best["weight_metadata"], weight_sparse=best["weight_sparse"],
        weight_dense=best["weight_dense"],
        weight_vl=(args.weight_vl_base if args.vl_adaptive else best.get("weight_vl", 0.0)),
        vl_adaptive=args.vl_adaptive, weight_vl_visual=args.weight_vl_visual,
        filter_mode=best.get("filter_mode", "soft"), rrf_k=best.get("rrf_k", 60),
        fusion=best.get("fusion", "rrf"))
    if args.vl_adaptive:
        print(f"  base = VL-adaptive (visual={args.weight_vl_visual}, non-visual={args.weight_vl_base})")
    retriever = HybridRetriever(cache_dir=INDEX_CACHE, dense_retriever=text_retriever,
        vl_retriever=vl_retriever, scoring=scoring)
    retriever.build_index(items)

    decomposer = QueryDecomposer(build_llm_config(args))
    questions = [(q.get("id") or q.get("question_id"), q["question"], set(extract_evidence_ids(q))) for q in qa_list]

    # 1) decompose (parallel LLM calls)
    print(f"Decomposing {len(questions)} queries ...")
    decomp: Dict[str, Any] = {}
    lock = threading.Lock()
    def do(item):
        qid, q, _ = item
        d = decomposer.decompose(q)
        with lock:
            decomp[qid] = d
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(do, questions), total=len(questions), desc="decompose"))
    fail = sum(1 for v in decomp.values() if not v["ok"])
    nsub = sum(v["n"] for v in decomp.values()) / max(len(decomp), 1)
    print(f"  decompose failures: {fail}/{len(questions)} | avg sub-queries: {nsub:.1f}")

    # 2) retrieve single vs decomposed, compute recall
    print("Retrieving (single vs decomposed) ...")
    rec = {"single": [], "decomposed": []}
    rec_multi = {"single": [], "decomposed": []}   # only questions with >3 gold items
    rows = []
    for qid, q, gold in tqdm(questions, desc="retrieve"):
        single = [r.item.item_id for r in retriever.retrieve(q, top_k=args.retrieve_topk)]
        subs = decomp[qid]["subqueries"]
        rankings = [single] + [[r.item.item_id for r in retriever.retrieve(s, top_k=args.retrieve_topk)]
                               for s in subs[1:]]
        merged = rrf_merge(rankings, k_rrf=best.get("rrf_k", 60), top=args.retrieve_topk,
                           anchor_weight=args.anchor_weight)
        rs, rd = recall_at_ks(single, gold), recall_at_ks(merged, gold)
        rec["single"].append(rs); rec["decomposed"].append(rd)
        if len(gold) > 3:
            rec_multi["single"].append(rs); rec_multi["decomposed"].append(rd)
        rows.append({"id": qid, "question": q[:120], "n_gold": len(gold),
                     "n_subq": len(subs), "subqueries": subs,
                     "R@10_single": round(rs[10]*100,1), "R@10_decomp": round(rd[10]*100,1)})

    summary = {
        "all": {"single": avg(rec["single"]), "decomposed": avg(rec["decomposed"]),
                "count": len(questions)},
        "multihop_gold>3": {"single": avg(rec_multi["single"]),
                            "decomposed": avg(rec_multi["decomposed"]),
                            "count": len(rec_multi["single"])},
        "decompose_fail": fail, "avg_subqueries": round(nsub, 2),
    }
    (out_dir / "decompose_recall_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "decompose_decisions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n===== R@k: single vs decomposed =====")
    for grp in ["all", "multihop_gold>3"]:
        g = summary[grp]
        print(f"\n  [{grp}]  (n={g['count']})")
        print("    scheme       " + "  ".join(f"R@{k}" for k in RECALL_KS))
        for n in ["single", "decomposed"]:
            s = g[n]
            print(f"    {n:11s} " + "  ".join(f"{s[f'R@{k}']:5.1f}" for k in RECALL_KS))
    print(f"\n  avg sub-queries/q: {summary['avg_subqueries']}  | decompose fails: {fail}")
    print(f"  saved: {out_dir}/")


if __name__ == "__main__":
    main()
