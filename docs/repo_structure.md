# Repo Structure

CHRONICLE is organized around:

- `chronicle/`: the CHRONICLE method — hybrid + query-adaptive routing retrieval (our contribution)
- `config/`: frozen best routing/fusion configs used to reproduce the CHRONICLE results
- `experiments/`: exploratory sweeps/optimizers that produced the configs (not needed to reproduce)
- `memqa/`: ATM-Bench core library (processors, retrieval, baselines, evaluation), inherited from upstream plus our Qwen3-VL retriever and reranker additions
- `agent_systems/`: general-purpose agent benchmark harness
- `scripts/`: runnable workflows (data download, run, eval)
- `data/`: local inputs (gitignored)
- `output/`: generated artifacts/results (gitignored)

See the top-level [README](../README.md#-whats-new-in-chronicle) for the full
contribution map (what CHRONICLE adds on top of ATM-Bench).

## Key Directories

### `chronicle/`

The CHRONICLE method package (imported as `from chronicle import ...`):
- `hybrid_retriever.py`: metadata + BM25 + dense hybrid retrieval with RRF / weighted-sum fusion
- `routing_retriever.py`: query-adaptive channel routing (`RoutingRetriever`, standard + `-Hard` weight profiles)
- `llm_router.py`: LLM-driven route selection
- `query_decomposer.py`: multi-evidence query decomposition
- `demo_cli.py` / `test_hybrid_retriever.py`: CPU-only demo and smoke tests

### `memqa/mem_processor/`

Preprocessing pipelines for:
- `image/`
- `video/`
- `email/`

These scripts turn raw artifacts into normalized metadata (`batch_results.json`)
used by QA baselines.

### `memqa/retrieve/`

Retrieval + reranking utilities used by MMRAG and other baselines.

### `memqa/qa_agent_baselines/`

Baselines shipped with this repo:

- `MMRag/`: retrieval + evidence-grounded answering
- `oracle/`: upper bound (answers using GT evidence IDs)
- `NIAH/`: generation-only evaluation using fixed evidence pools
- `HippoRag2/`: HippoRAG 2 graph memory baseline
- `MemoryOS/`: tiered memory baseline (STM/MTM/LPM)
- `A-Mem/`: agentic memory baseline (two-stage cache)
- `mem0/`: mem0-backed memory baseline
- `SimpleMem/`: SimpleMem (LanceDB + FTS) baseline
- `Mempalace/`: MemPalace baseline

### `memqa/utils/evaluator/`

Evaluation tools for:
- QA metrics (`em`, `atm`, judge-based scoring)
- retrieval metrics
- joint metrics

### `scripts/QA_Agent/`

Public runnable scripts (repo-root execution):
- `MMRAG/`: main baseline scripts
- `Oracle/`: oracle + no-evidence scripts
- `NIAH/`: NIAH runs + utilities
- `HippoRag2/`: HippoRAG 2 runs
- `MemoryOS/`: MemoryOS runs
- `A-Mem/`: A‑Mem runs
- `mem0/`: Mem0 runs
- `SimpleMem/`: SimpleMem runs
- `Mempalace/`: MemPalace runs

### `data/`

Local-only inputs (see `docs/data.md`):
- `data/atm-bench/` (benchmark)
- `data/raw_memory/` (your artifacts + batch results)
- `data/processed_memory/` (optional)
