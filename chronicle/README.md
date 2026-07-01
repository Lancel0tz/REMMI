# `chronicle` — ChronicleMem method package

This package is the ChronicleMem contribution on top of [ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench):
a hybrid, query-adaptive retrieval method for long-term personalized memory QA
(the `ChronicleMem (Ours)` row in the top-level [README](../README.md#-results)).

It sits *alongside* the upstream baselines under `memqa/qa_agent_baselines/` and
plugs into the existing MMRAG pipeline, so the upstream code path is unchanged.

## Layout

```
chronicle/
├── __init__.py
├── hybrid_retriever.py       # HybridRetriever: metadata + BM25 + dense, RRF / weighted-sum fusion
├── routing_retriever.py      # RoutingRetriever: per-query adaptive channel weights (standard + -Hard)
├── llm_router.py             # LLM-driven route selection
├── query_decomposer.py       # multi-evidence query decomposition
├── _retrieval_item.py        # torch-free RetrievalItem shim (CPU sandbox fallback)
├── demo_cli.py               # python -m chronicle.demo_cli
└── test_hybrid_retriever.py  # unittest smoke tests (CPU-only)
```

## Retrieval channels

`HybridRetriever` fuses three channels with Reciprocal Rank Fusion (default) or
weighted-sum over min-max-normalized scores:

| Channel  | Source                                                    |
|----------|-----------------------------------------------------------|
| metadata | SGM `time` + `location` from `RetrievalItem.metadata`     |
| sparse   | BM25 over the SGM-rendered text + metadata fields         |
| dense    | any object exposing `retrieve(query, top_k)`              |

The dense channel is pluggable: pass any built `BaseRetriever`
(`SentenceTransformerRetriever`, the Qwen3-VL dual encoder, …) via
`dense_retriever=`. With no dense retriever it falls back to metadata + BM25 —
useful for fast CPU debugging and ablation.

`RoutingRetriever` wraps this: it analyzes per-query signals (metadata / keyword
/ semantic strength) and selects channel weights, with separate profiles for the
standard and `-Hard` splits (`adaptive_weights`, `adaptive_weights_hard`). The
frozen operating point lives in [`config/best_routing_config*.json`](../config).

## Try it (CPU, no data/models)

```bash
python -m unittest chronicle.test_hybrid_retriever -v
python -m chronicle.demo_cli
python -m chronicle.demo_cli "Where did I have ramen in Tokyo?"
```

## No torch import here

To stay testable on a CPU-only sandbox, this package depends only on `numpy` and
`rank_bm25`. Importing `memqa.retrieve.utils` directly would pull in
`memqa.retrieve.retrievers`, which imports `torch`; that import is guarded and a
tiny `_retrieval_item.py` shim mirrors the fields. In production runs (where
torch is present for the dense embedder) the real `RetrievalItem` is used.
