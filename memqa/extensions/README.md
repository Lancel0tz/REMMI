# Project Extensions

This subpackage hosts the experimental extensions proposed in
`Kuanyan Zhu kz345 project_proposal.pdf`
(Title: *Benchmarking Long-Term Agentic Multimodal Personal Memory*,
Supervisor: Bill Byrne, Co-Supervisor: Jingbiao Mei).

The extensions sit *alongside* the existing baselines under
`memqa/qa_agent_baselines/` so that nothing in the upstream codebase needs
to change while the project is iterating.

## Layout

```
memqa/extensions/
├── README.md
├── __init__.py
└── hybrid/                       # Direction 3: Hybrid Retrieval
    ├── __init__.py
    ├── hybrid_retriever.py       # main module: HybridRetriever
    ├── _retrieval_item.py        # sandbox fallback (avoids torch import)
    ├── demo_cli.py               # python -m memqa.extensions.hybrid.demo_cli
    └── test_hybrid_retriever.py  # unittest smoke tests (CPU-only)
```

Planned modules (not yet implemented):

```
linkage/   # Direction 1: SGM-aware adaptive memory linkage
events/    # Direction 2: event-based memory organization
```

## Direction 3 — Hybrid Retrieval (current)

`HybridRetriever` combines three channels and fuses them with either
Reciprocal Rank Fusion (default) or weighted-sum on min-max normalized
scores:

| Channel  | Source                                                   |
|----------|----------------------------------------------------------|
| metadata | SGM `timestamp` + `location` from `RetrievalItem.metadata`|
| sparse   | BM25 over the SGM-rendered text + metadata fields         |
| dense    | any object exposing `retrieve(query, top_k)`             |

The dense channel is pluggable: pass any built `BaseRetriever`
(`SentenceTransformerRetriever`, `Qwen3VLRetriever`, …) via
`dense_retriever=`. When no dense retriever is supplied, the hybrid
retriever falls back to metadata + BM25 only — useful for fast CPU
debugging and ablation.

### Run the smoke tests

```bash
python -m unittest memqa.extensions.hybrid.test_hybrid_retriever -v
```

### Interactive demo on a synthetic 5-item corpus

```bash
python -m memqa.extensions.hybrid.demo_cli
python -m memqa.extensions.hybrid.demo_cli "Where did I have ramen in Tokyo?"
```

### Wiring into MMRAG (next milestone)

`mmrag_retrieve_answer.py` currently dispatches retrievers via
`--retriever {qwen3_vl_embedding,vista,clip,text,sentence_transformer}`.
A small dispatch addition will register `--retriever hybrid` so the
hybrid retriever can plug in, with the dense channel filled by whichever
text/VL embedder the script otherwise uses.

## Why no torch / sentence-transformers imports here?

To keep the extension testable on a CPU-only sandbox (no CUDA, no torch),
the hybrid module only depends on `numpy` and `rank_bm25`. Importing
`memqa.retrieve.utils` directly would pull in `memqa.retrieve.retrievers`,
which unconditionally imports `torch`. We therefore guard that import
and provide a tiny `_retrieval_item.py` shim with the same fields. In
production runs (where the codebase already needs torch for the dense
embedder), the real `RetrievalItem` is used.
