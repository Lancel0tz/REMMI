<div align="center">

# LTMA — Hybrid Adaptive-Routing Retrieval for Long-Term Personal Memory QA

**A hybrid, query-adaptive multimodal RAG method for long-term personalized referential memory QA, built on and evaluated with [ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench).**

[🇬🇧 English](README.md) • [🇨🇳 中文](README_zh.md)

<!-- TODO: add your own paper / project-page badges once public
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/XXXX.XXXXX)
-->
[![Benchmark: ATM-Bench](https://img.shields.io/badge/Benchmark-ATM--Bench-1f6feb.svg)](https://github.com/JingbiaoMei/ATM-Bench)
[![Live Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-Live-orange.svg)](https://atmbench.github.io/leaderboard.html)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Dataset-FFD21E.svg)](https://huggingface.co/datasets/Jingbiao/ATM-Bench)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[✨ What's New](#-whats-new-in-ltma) • [🧩 Method](#-method) • [📊 Results](#-results) • [🔁 Reproduce](#-reproduce) • [📁 Structure](#-repository-structure) • [📖 Citation](#-citation)

</div>

> **LTMA** is incremental work on top of the **ATM-Bench** benchmark. It does not
> re-release the benchmark; it contributes a retrieval method (reported as
> **ATM-RAG** in the results below) and the code to reproduce it. The benchmark,
> dataset, task definition, and the baselines LTMA compares against are the work
> of the ATM-Bench authors — see [Attribution](#-attribution--upstream).

---

## 📋 Table of Contents

- [✨ What's New in LTMA](#-whats-new-in-ltma)
- [🧩 Method](#-method)
- [📊 Results](#-results)
- [🔁 Reproduce](#-reproduce)
- [📁 Repository Structure](#-repository-structure)
- [🙏 Attribution & Upstream](#-attribution--upstream)
- [📖 Citation](#-citation)
- [📝 License](#-license)

<a id="whats-new-in-ltma"></a>
## ✨ What's New in LTMA

Everything below is added by LTMA on top of upstream ATM-Bench. The core method
lives in a single top-level package, [`ltma/`](ltma/), so the contribution
boundary is explicit.

| Contribution | Where | Notes |
|--------------|-------|-------|
| **Hybrid retriever** — metadata filtering + BM25 sparse + dense, fused via RRF or weighted-sum | [`ltma/hybrid_retriever.py`](ltma/hybrid_retriever.py) | Pluggable dense channel; CPU-testable without torch |
| **Query-adaptive routing** — per-query signal analysis → adaptive channel weights (easy/hard variants) | [`ltma/routing_retriever.py`](ltma/routing_retriever.py) | `RoutingRetriever`, `adaptive_weights`, `adaptive_weights_hard` |
| **LLM router** — LLM-driven route selection over retrieval channels | [`ltma/llm_router.py`](ltma/llm_router.py) | |
| **Query decomposition** — splits multi-evidence queries into sub-queries | [`ltma/query_decomposer.py`](ltma/query_decomposer.py) | |
| **Qwen3-VL dual-encoder retriever** for MMRAG dense retrieval | [`memqa/retrieve/retrievers.py`](memqa/retrieve/retrievers.py) | Added alongside upstream retrievers |
| **Qwen3-Reranker fix** | [`memqa/retrieve/rerankers.py`](memqa/retrieve/rerankers.py) | |
| **Frozen best configs** from Bayesian optimization | [`config/best_routing_config.json`](config/best_routing_config.json), [`config/best_routing_config_hard.json`](config/best_routing_config_hard.json) | The numbers below reproduce from these |
| **SimpleMem baseline port** | [`memqa/qa_agent_baselines/SimpleMem/`](memqa/qa_agent_baselines/SimpleMem/) | Added to the memory-system comparison |

The exploratory search/sweep code that *produced* the frozen configs (grid /
Bayesian / meta-strategy optimizers, recall ablations, retriever comparisons)
lives under [`experiments/`](experiments/README.md) and is **not** required to
reproduce the headline numbers.

<a id="method"></a>
## 🧩 Method

LTMA retrieves over ATM-Bench's Schema-Guided Memory (SGM) items with a hybrid,
query-adaptive pipeline:

1. **Three retrieval channels**
   - **Metadata** — soft filtering / boosting over SGM `time` + `location` fields.
   - **Sparse** — BM25 over the rendered SGM text and selected metadata fields.
   - **Dense** — any `BaseRetriever` (e.g. Qwen3 text embeddings, or the added
     Qwen3-VL dual encoder).
2. **Score fusion** — Reciprocal Rank Fusion (default) or weighted-sum over
   min-max-normalized per-channel scores.
3. **Query-adaptive routing** — `RoutingRetriever` analyzes per-query signals
   (metadata / keyword / semantic strength) and picks channel weights, with
   separate profiles for the standard and `-Hard` splits.
4. **Optional query decomposition and LLM routing** for multi-evidence queries.

The frozen operating point (from Bayesian optimization, `config/`) uses
`metadata:sparse:dense ≈ 0.19:0.39:0.56` with RRF, lifting Recall@10 on the
standard set from **0.732 → 0.775** (+4.3%).

<a id="results"></a>
## 📊 Results

> 🏆 The authoritative, up-to-date numbers live on the
> [ATM-Bench Live Leaderboard](https://atmbench.github.io/leaderboard.html).
> The snapshot below may lag behind new submissions.

**Memory-system comparison** — answerer `Qwen3-VL-8B-Instruct-FP8`, memory
processor `Qwen3-VL-2B-Instruct`, `-Hard` on the `atm-bench-hard` release set.
**LTMA is the `ATM-RAG (Ours)` row.**

| System | Index Time (hr) ↓ | ATM-Bench QS ↑ | ATM-Bench Recall@10 ↑ | ATM-Bench-Hard QS ↑ | ATM-Bench-Hard Recall@10 ↑ |
|--------|------------------:|---------------:|----------------------:|--------------------:|---------------------------:|
| [A-Mem](https://github.com/WujiangXu/A-mem) | 12.6 | 44.8 | 66.4 | 9.9 | 31.7 |
| [mem0](https://github.com/mem0ai/mem0) | 16.7 | 43.5 | 61.9 | 9.2 | 23.7 |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 36.6 | 47.2 | 59.2 | 13.7 | 32.7 |
| [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG) | 1.5 | 42.9 | 66.4 | 9.4 | 31.9 |
| [MemPalace](https://github.com/MemPalace/mempalace) | 0.5 | 56.8 | 76.4 | 9.7 | 28.3 |
| [SimpleMem](https://github.com/aiming-lab/SimpleMem) | 15.7 | 27.3 | 23.3 | 3.2 | 7.0 |
| **ATM-RAG / LTMA (Ours)** | **0.5** | **51.0** | **68.7** | **8.4** | **28.8** |

<!-- TODO: add ablations (routing on/off, per-channel, reranker) once finalized for the paper. -->

<a id="reproduce"></a>
## 🔁 Reproduce

### 1. Install

```bash
conda create -n ltma python=3.11 -y
conda activate ltma
pip install -r requirements.txt
pip install -e .
```

On macOS / Apple Silicon (GPU/video-serving packages like `vllm`/`decord` are
not reliable via pip):

```bash
bash scripts/setup_local_mac.sh
```

### 2. Data (ATM-Bench, ~3.3 GB from Hugging Face)

```bash
bash scripts/download_data.sh
```

This stages QA, NIAH pools, preprocessed memory, emails, raw media, and the
GPS reverse-geocoding cache under `data/` and `output/`. Full data layout:
[`docs/data.md`](docs/data.md).

### 3. API keys

```bash
export OPENAI_API_KEY="your-key"     # or api_keys/.openai_key
export VLLM_API_KEY="your-key"       # or api_keys/.vllm_key
```

### 4. Run LTMA (ATM-RAG)

Needs a vLLM endpoint serving `Qwen/Qwen3-VL-8B-Instruct-FP8` at
`http://127.0.0.1:8000/v1/...` (override with `VLLM_ENDPOINT` / `ANSWERER_MODEL`).

```bash
# Hybrid + adaptive-routing MMRAG on both the standard and -Hard splits,
# using the frozen config in config/best_routing_config*.json.
bash scripts/QA_Agent/MMRAG/run_routing_reranker.sh
```

The retriever-only variants (`scripts/QA_Agent/MMRAG/run_retrieval_only_cpu*.sh`)
evaluate Recall@k on CPU without an answerer endpoint. See
[`docs/reproducibility.md`](docs/reproducibility.md) for full settings.

### Smoke test the method (CPU, no data/models)

```bash
python -m unittest ltma.test_hybrid_retriever -v
python -m ltma.demo_cli "Where did I have ramen in Tokyo?"
```

<a id="repository-structure"></a>
## 📁 Repository Structure

```
LTMA/
├── ltma/               # ★ LTMA method: hybrid + adaptive-routing retrieval
├── config/             # Frozen best routing/fusion configs (reproduce these)
├── experiments/        # Research scaffolding (sweeps/optimizers) — not needed to reproduce
├── memqa/              # ATM-Bench core (baselines, retrievers, evaluation) + our retriever/reranker additions
├── agent_systems/      # General-purpose agent benchmark harness
├── scripts/            # Data download, run, and eval entry points
├── docs/               # Documentation
├── third_party/        # Vendored memory-system baselines
├── data/  output/      # User-provided data and outputs (gitignored)
└── LICENSE
```

<a id="attribution--upstream"></a>
## 🙏 Attribution & Upstream

LTMA is built on **ATM-Bench** and inherits its benchmark, dataset, task, core
`memqa/` code, and all comparison baselines.

- **Upstream:** [`JingbiaoMei/ATM-Bench`](https://github.com/JingbiaoMei/ATM-Bench)
- **Built on upstream commit:** [`d552cc5`](https://github.com/JingbiaoMei/ATM-Bench/commit/d552cc5b84f495ff173e7a9ddb598e9edfd2b539) *(2026-04-10)*
- **Dataset:** [`Jingbiao/ATM-Bench`](https://huggingface.co/datasets/Jingbiao/ATM-Bench) on Hugging Face
- **Leaderboard:** [atmbench.github.io/leaderboard.html](https://atmbench.github.io/leaderboard.html)

Vendored / ported baselines keep their own upstream licenses and pinned commits
(e.g. SimpleMem @ [`094027e`](https://github.com/aiming-lab/SimpleMem/commit/094027eca4c890dc9912be8cee1da04428de8076)); see
[`docs/baseline.md`](docs/baseline.md) and each baseline's README.

<a id="citation"></a>
## 📖 Citation

If you use LTMA, please also cite the ATM-Bench benchmark it is built on:

```bibtex
@article{mei2026atm,
  title={According to Me: Long-Term Personalized Referential Memory QA},
  author={Mei, Jingbiao and Chen, Jinghong and Yang, Guangyu and Hou, Xinyu and Li, Margaret and Byrne, Bill},
  journal={arXiv preprint arXiv:2603.01990},
  year={2026},
  url={https://arxiv.org/abs/2603.01990},
  doi={10.48550/arXiv.2603.01990}
}
```

<!-- TODO: add the LTMA citation once your paper is public.
@article{zhu2026ltma,
  title={<LTMA paper title>},
  author={Zhu, Kuanyan and ...},
  year={2026}
}
-->

<a id="license"></a>
## 📝 License

MIT — see [LICENSE](LICENSE). Portions are derived from
[ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench) (MIT). Vendored third-party
baselines under `third_party/` and `memqa/qa_agent_baselines/` retain their
respective upstream licenses.
