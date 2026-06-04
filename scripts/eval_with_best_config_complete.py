#!/usr/bin/env python3
"""Generate answers using best config + Reranker + LLM (完整流程).

完全对应原始 mmrag_retrieve_answer.py 的流程：
1. 检索（使用最优配置 + Reranker）
2. 构建 LLM 消息
3. 调用 LLM 生成答案
4. 输出结果

Usage:
    python scripts/eval_with_best_config_complete.py \
        --qa-file data/atm-bench/atm-bench.json \
        --config-file config/best_routing_config.json \
        --device cuda \
        --output-dir output/answers_with_best_config
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memqa.extensions.hybrid import HybridRetriever, HybridScoringConfig
from memqa.retrieve.utils import (
    RetrievalItem,
    EmailTextConfig,
    MediaTextConfig,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
    write_json,
)
from memqa.retrieve.retrievers import SentenceTransformerRetriever
from memqa.retrieve.rerankers import TextReranker
from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient

# ── Paths ────────────────────────────────────────────────────────────
EMAIL_FILE = ROOT / "data/raw_memory/email/emails.json"
IMAGE_BATCH = ROOT / "output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH = ROOT / "output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT = ROOT / "data/raw_memory/image"
VIDEO_ROOT = ROOT / "data/raw_memory/video"
INDEX_CACHE = ROOT / "output/retrieval/index_cache"

# Thread-local storage for LLM client
thread_local = threading.local()


def detect_dataset_type(qa_file: Path) -> str:
    """检测数据集类型 (standard 或 hard)."""
    qa_data = load_json(qa_file)
    num_questions = len(qa_data) if isinstance(qa_data, list) else len(qa_data.get("queries", []))

    # Hard set 通常有 31 个问题，Standard set 有 1013 个
    if num_questions <= 50:
        return "hard"
    else:
        return "standard"


def get_config_file(config_file: Optional[str], qa_file: Path) -> Path:
    """根据数据集自动选择正确的配置文件.

    如果 config_file 是 'auto'，则自动检测数据集类型并选择对应配置。
    """
    if config_file and config_file != "auto":
        return Path(config_file)

    # 自动检测数据集类型
    dataset_type = detect_dataset_type(qa_file)
    if dataset_type == "hard":
        default_config = ROOT / "config" / "best_routing_config_hard.json"
    else:
        default_config = ROOT / "config" / "best_routing_config.json"

    print(f"   📊 检测到数据集类型: {dataset_type}")
    print(f"   📄 自动选择配置: {default_config.name}")

    return default_config


def load_config(config_file: Path) -> Dict[str, Any]:
    """Load routing configuration."""
    with open(config_file) as f:
        return json.load(f)


def build_llm_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build LLM configuration from arguments."""
    return {
        "provider": args.provider,
        "model": args.model,
        "api_key": args.api_key,
        "api_base": args.api_base,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def get_llm(llm_config: Dict[str, Any]) -> LLMClient:
    """Get or create LLM client (thread-safe)."""
    if not hasattr(thread_local, "llm"):
        thread_local.llm = LLMClient(llm_config["provider"], llm_config)
    return thread_local.llm


def answer_question(
    qa_id: str,
    question: str,
    evidence_items: List[RetrievalItem],
    args: argparse.Namespace,
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate answer for a single question using LLM.

    完全对应原始脚本的 answer_single 函数。
    """
    try:
        # 构建 LLM 消息（对应原始脚本的 build_llm_messages）
        from memqa.qa_agent_baselines.MMRag.mmrag_retrieve_answer import build_llm_messages

        messages = build_llm_messages(question, evidence_items, args)

        # 调用 LLM 生成答案
        llm = get_llm(llm_config)
        answer, usage = llm.chat_with_usage(messages)

        result = {
            "id": qa_id,
            "answer": answer,
            "status": "ok",
        }
        if usage:
            result["prompt_tokens"] = usage.prompt_tokens
            result["completion_tokens"] = usage.completion_tokens
            result["total_tokens"] = usage.total_tokens

        return result

    except Exception as exc:
        print(f"❌ Failed to generate answer for {qa_id}: {exc}", file=sys.stderr)
        return {
            "id": qa_id,
            "answer": "",
            "status": "failed",
            "error": str(exc),
        }


def main():
    p = argparse.ArgumentParser(
        description="Generate answers with best config + Reranker + LLM (完整流程)"
    )
    p.add_argument("--qa-file", required=True, help="QA file path")
    p.add_argument(
        "--config-file",
        default="auto",
        help="Best routing config file (default: auto - automatically select based on dataset)",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use reranker to rerank top-K results",
    )
    p.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
        help="Reranker model name",
    )
    p.add_argument("--rerank-top-k", type=int, default=50)
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--max-workers", type=int, default=4, help="Max parallel workers for LLM")

    # LLM configuration
    p.add_argument("--provider", default="vllm", choices=["openai", "vllm"])
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    p.add_argument("--api-key", default=None, help="API key")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="API base URL")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)

    # MMRag configuration (for build_llm_messages)
    p.add_argument("--media-source", default="batch_results")
    p.add_argument("--no-evidence", action="store_true", default=False)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--max-total-frames", type=int, default=None)
    p.add_argument("--frame-strategy", default="uniform")
    p.add_argument("--insert-raw-images", action="store_true", default=False)
    p.add_argument("--vl-text-augment", action="store_true", default=True)

    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 ANSWER GENERATION WITH BEST CONFIG + RERANKER + LLM (完整流程)")
    print("=" * 80)

    # Auto-detect config file based on dataset
    print(f"\n📂 Loading data and detecting dataset type...")
    config_file = get_config_file(args.config_file, Path(args.qa_file))

    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    print(f"✅ Using config: {config_file}")
    config = load_config(config_file)

    # Load data
    print("\n📂 Loading data...")
    qa_list = load_json(Path(args.qa_file))
    email_entries = load_json(EMAIL_FILE)
    image_batch_data = load_json(IMAGE_BATCH)
    video_batch_data = load_json(VIDEO_BATCH)

    media_config = MediaTextConfig()
    email_config = EmailTextConfig()

    items = build_retrieval_items(
        email_entries=email_entries,
        image_entries=image_batch_data,
        video_entries=video_batch_data,
        media_text_config=media_config,
        email_text_config=email_config,
        image_root=IMAGE_ROOT,
        video_root=VIDEO_ROOT,
    )

    print(f"   ✓ Loaded {len(qa_list)} questions")
    print(f"   ✓ Loaded {len(items)} items")

    # Build retrievers
    print("\n🔧 Building retrievers...")
    text_retriever = SentenceTransformerRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=INDEX_CACHE,
        batch_size=64,
        device=args.device,
    )

    # Use best config
    best_config = config["hybrid_scoring_config"]
    scoring_config = HybridScoringConfig(
        filter_mode=best_config.get("filter_mode", "soft"),
        weight_metadata=best_config["weight_metadata"],
        weight_sparse=best_config["weight_sparse"],
        weight_dense=best_config["weight_dense"],
        weight_vl=best_config.get("weight_vl", 0.0),
        rrf_k=best_config.get("rrf_k", 60),
    )

    retriever = HybridRetriever(
        cache_dir=INDEX_CACHE,
        dense_retriever=text_retriever,
        scoring=scoring_config,
    )

    print("   ✓ Building index...")
    retriever.build_index(items)

    # Build reranker if needed
    reranker = None
    if args.use_reranker:
        print(f"   ✓ Loading reranker ({args.reranker_model})...")
        reranker = TextReranker(
            model_name=args.reranker_model,
            device=args.device,
            batch_size=8,
        )

    # Build LLM config
    llm_config = build_llm_config(args)

    # Prepare questions for answering
    print("\n📊 Preparing questions...")
    prepared_questions = []

    for qa in tqdm(qa_list, desc="Preparing"):
        qa_id = qa.get("id") or qa.get("question_id")
        question = qa["question"]

        # Retrieve
        ret_results = retriever.retrieve(question, top_k=200)
        retrieved_ids = [r.item.item_id for r in ret_results]

        # Rerank if needed
        if reranker is not None:
            top_k_for_rerank = min(args.rerank_top_k, len(ret_results))
            top_candidates = ret_results[:top_k_for_rerank]

            if top_candidates:
                items_for_rerank = [r.item for r in top_candidates]
                for item in items_for_rerank:
                    if hasattr(item, "text") and item.text:
                        item.text = item.text[:1024]

                try:
                    rerank_results = reranker.rerank(question, items_for_rerank)
                    if rerank_results and hasattr(rerank_results[0], "score"):
                        reranked_scores = np.array([r.score for r in rerank_results])
                    else:
                        reranked_scores = np.array(
                            rerank_results
                            if isinstance(rerank_results, (list, np.ndarray))
                            else [rerank_results]
                        )
                except Exception as e:
                    print(f"  ⚠️ Reranker error: {e}")
                    reranked_scores = np.array([0.5] * len(items_for_rerank))

                reranked_indices = np.argsort(-reranked_scores)
                reranked_ids = [
                    top_candidates[i].item.item_id for i in reranked_indices
                ]
                rest_ids = [r.item.item_id for r in ret_results[top_k_for_rerank:]]
                reranked_ids.extend(rest_ids)
                retrieved_ids = reranked_ids

        # Select evidence items for LLM
        if args.no_evidence:
            evidence_items = []
        else:
            # 只选择 top_k 的证据项（对应原始脚本的 retrieval_top_k）
            retrieval_top_k = getattr(args, "retrieval_top_k", 10)
            selected_results = ret_results[:retrieval_top_k]
            evidence_items = [r.item for r in selected_results]

        prepared_questions.append({
            "id": qa_id,
            "question": question,
            "evidence_items": evidence_items,
        })

    # Generate answers using LLM (对应原始脚本的 answer_single)
    print("\n📝 Generating answers with LLM...")
    answers = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                answer_question,
                entry["id"],
                entry["question"],
                entry["evidence_items"],
                args,
                llm_config,
            ): entry
            for entry in prepared_questions
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating answers"
        ):
            try:
                result = future.result()
                answers.append(result)
            except Exception as exc:
                entry = futures[future]
                print(f"❌ Error for {entry['id']}: {exc}", file=sys.stderr)
                answers.append({
                    "id": entry["id"],
                    "answer": "",
                    "status": "failed",
                    "error": str(exc),
                })

    # Save answers
    output_file = output_dir / "mmrag_answers.jsonl"
    print(f"\n💾 Saving answers to: {output_file}")
    with open(output_file, "w") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")

    # Save config info
    config_info = {
        "config_file": str(config_file),
        "scoring_config": {
            "filter_mode": scoring_config.filter_mode,
            "weight_metadata": scoring_config.weight_metadata,
            "weight_sparse": scoring_config.weight_sparse,
            "weight_dense": scoring_config.weight_dense,
            "weight_vl": scoring_config.weight_vl,
            "rrf_k": scoring_config.rrf_k,
        },
        "optimization_info": config.get("optimization", {}),
        "use_reranker": args.use_reranker,
        "llm_config": {
            "provider": args.provider,
            "model": args.model,
        },
        "total_questions": len(answers),
    }

    config_file_out = output_dir / "config_info.json"
    with open(config_file_out, "w") as f:
        json.dump(config_info, f, indent=2)

    # Statistics
    print(f"\n✅ Complete!")
    print("=" * 80)
    print(f"📄 Answers: {output_file}")
    print(f"📄 Config: {config_file_out}")
    print(f"📊 Total: {len(answers)} answers")

    # Token statistics
    total_prompt = sum(a.get("prompt_tokens", 0) for a in answers)
    total_completion = sum(a.get("completion_tokens", 0) for a in answers)
    total_tokens = sum(a.get("total_tokens", 0) for a in answers)

    if total_tokens > 0:
        print(f"\n📈 Token Statistics:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Prompt tokens: {total_prompt}")
        print(f"   Completion tokens: {total_completion}")
        print(f"   Avg tokens per Q: {total_tokens / len(answers):.0f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
