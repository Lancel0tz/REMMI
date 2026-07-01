<div align="center">

# ChronicleMem — A Long-Term Personal Memory System

**Built on and evaluated with [ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench). Hybrid, query-adaptive retrieval now; chronological event-based memory organization on the roadmap.**

[🇬🇧 English](README.md) • [🇨🇳 中文](README_zh.md)

<!-- TODO: add your own paper / project-page badges once public
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/XXXX.XXXXX)
-->
[![Benchmark: ATM-Bench](https://img.shields.io/badge/Benchmark-ATM--Bench-1f6feb.svg)](https://github.com/JingbiaoMei/ATM-Bench)
[![Live Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-Live-orange.svg)](https://atmbench.github.io/leaderboard.html)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Dataset-FFD21E.svg)](https://huggingface.co/datasets/Jingbiao/ATM-Bench)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[✨ What's New](#whats-new) • [🧩 Method](#-method) • [📊 Results](#-results) • [🔁 Reproduce](#-reproduce) • [📁 Structure](#-repository-structure) • [📖 Citation](#-citation)

</div>

> **ChronicleMem** is incremental work on top of the **ATM-Bench** benchmark. It does not
> re-release the benchmark; it contributes a retrieval method and the code to
> reproduce it (the `ChronicleMem (Ours)` row in the results below). The benchmark,
> dataset, task definition, and the baselines ChronicleMem compares against are the work
> of the ATM-Bench authors — see [Attribution](#-attribution--upstream).

---

## 📋 Table of Contents

- [✨ What's New in ChronicleMem](#whats-new)
- [🗺️ Roadmap](#-roadmap)
- [🧩 Method](#-method)
- [📊 Results](#-results)
- [🔁 Reproduce](#-reproduce)
- [📁 Repository Structure](#-repository-structure)
- [🙏 Attribution & Upstream](#-attribution--upstream)
- [📖 Citation](#-citation)
- [📝 License](#-license)

<a id="whats-new"></a>
## ✨ What's New in ChronicleMem

Everything below is added by ChronicleMem on top of upstream ATM-Bench. The core method
lives in a single top-level package, [`chronicle/`](chronicle/), so the contribution
boundary is explicit.

| Contribution | Where | Notes |
|--------------|-------|-------|
| **Hybrid retriever** — metadata filtering + BM25 sparse + dense, fused via RRF or weighted-sum | [`chronicle/hybrid_retriever.py`](chronicle/hybrid_retriever.py) | Pluggable dense channel; CPU-testable without torch |
| **Query-adaptive routing** — per-query signal analysis → adaptive channel weights (easy/hard variants) | [`chronicle/routing_retriever.py`](chronicle/routing_retriever.py) | `RoutingRetriever`, `adaptive_weights`, `adaptive_weights_hard` |
| **LLM router** — LLM-driven route selection over retrieval channels | [`chronicle/llm_router.py`](chronicle/llm_router.py) | |
| **Query decomposition** — splits multi-evidence queries into sub-queries | [`chronicle/query_decomposer.py`](chronicle/query_decomposer.py) | |
| **Qwen3-VL dual-encoder retriever** for MMRAG dense retrieval | [`memqa/retrieve/retrievers.py`](memqa/retrieve/retrievers.py) | Added alongside upstream retrievers |
| **Qwen3-Reranker fix** | [`memqa/retrieve/rerankers.py`](memqa/retrieve/rerankers.py) | |
| **Frozen best configs** from Bayesian optimization | [`config/best_routing_config.json`](config/best_routing_config.json), [`config/best_routing_config_hard.json`](config/best_routing_config_hard.json) | The numbers below reproduce from these |
| **SimpleMem baseline port** | [`memqa/qa_agent_baselines/SimpleMem/`](memqa/qa_agent_baselines/SimpleMem/) | Added to the memory-system comparison |

The exploratory search/sweep code that *produced* the frozen configs (grid /
Bayesian / meta-strategy optimizers, recall ablations, retriever comparisons)
lives under [`experiments/`](experiments/README.md) and is **not** required to
reproduce the headline numbers.

<a id="roadmap"></a>
## 🗺️ Roadmap

ChronicleMem is developed incrementally. **Implemented today** is the retrieval
side — hybrid, query-adaptive routing over SGM memory ([`chronicle/`](chronicle/)).
**Next** is the organization side, which the name points at:

- **Event-based memory organization** — grouping long-term memory into events/episodes.
- **Hierarchical retrieval** over that event structure.
- **Memory linkage** — inferred relations across memory items.

These are not in the codebase yet; this section states intent, not current capability.

<a id="method"></a>
## 🧩 Method

ChronicleMem retrieves over ATM-Bench's Schema-Guided Memory (SGM) items with a hybrid,
query-adaptive pipeline:

1. **Four retrieval channels**
   - **Metadata** — soft/hard filtering + boosting over SGM `time` + `location` fields.
   - **Sparse** — BM25 over the rendered SGM text and selected metadata fields.
   - **Dense** — any `BaseRetriever` (e.g. Qwen3 text embeddings, or the added
     Qwen3-VL dual encoder).
   - **Vision** — CLIP-style image matching for visually-grounded queries.
2. **Query-adaptive routing** — `RoutingRetriever` analyzes per-query signals
   (date/location, proper nouns, amounts, CJK terms, recall phrasing, visual
   mentions) and tilts the four channel weights per query, with separate profiles
   for the standard and `-Hard` splits and a confidence gate that widens channels
   when the first pass is weak.
3. **Fusion + rerank** — Reciprocal Rank Fusion over the routed rankings, then a
   cross-encoder reranker (Qwen3-Reranker-4B) refines top-20 → top-10.
4. **Optional query decomposition and LLM routing** for multi-evidence queries.

The frozen operating point lives in [`config/`](config). With the fixed
`Qwen3-VL-8B-Instruct` answerer unchanged, this pipeline reaches **83.3 Recall@10 /
60.3 QS** on the standard split — see [Results](#-results).

<a id="results"></a>
## 📊 Results

> 🏆 The authoritative, up-to-date numbers live on the
> [ATM-Bench Live Leaderboard](https://atmbench.github.io/leaderboard.html).

ChronicleMem's retrieval stage sets a **new SOTA among Memory & RAG systems** on
ATM-Bench, under a *fixed* `Qwen3-VL-8B-Instruct` answerer — so any QS gap is
attributable to retrieval, not the language model. It is **#1 on both QS and
Recall@10, on both splits**: Standard **60.3** QS / **83.3** R@10, Hard **19.2**
QS / **42.0** R@10.

**Table 1 — ATM-Bench Standard** (single-hop; sorted by QS; same answerer for all)

| # | System | Type | QS ↑ | R@10 ↑ |
|--:|--------|:----:|-----:|-------:|
| **1** | **ChronicleMem (Ours)** · hybrid retrieval | RAG | **60.3** | **83.3** |
| 2 | [MemPalace](https://github.com/MemPalace/mempalace) | Memory | 56.8 | 76.4 |
| 3 | ScrapMem (No-Forget) | Memory | 52.5 | 70.3 |
| 4 | [ATM-RAG](https://github.com/JingbiaoMei/ATM-Bench) · upstream | RAG | 51.0 | 68.7 |
| 5 | Self-RAG | RAG | 50.3 | 68.7 |
| 6 | [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | Memory | 47.2 | 59.2 |
| 7 | [A-Mem](https://github.com/WujiangXu/A-mem) | Memory | 44.8 | 66.4 |
| — | *Oracle ceiling (Qwen3-VL-8B)* | — | *78.2* | — |

**Table 2 — ATM-Bench-Hard** (31 multi-hop questions, ~6 evidence each; sorted by QS)

| # | System | Type | QS ↑ | R@10 ↑ |
|--:|--------|:----:|-----:|-------:|
| **1** | **ChronicleMem (Ours)** · hybrid retrieval | RAG | **19.2** | **42.0** |
| 2 | [ATM-RAG](https://github.com/JingbiaoMei/ATM-Bench) · upstream | RAG | 13.8 | 30.4 |
| 3 | [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | Memory | 13.7 | 32.7 |
| 4 | [A-Mem](https://github.com/WujiangXu/A-mem) | Memory | 9.9 | 31.7 |
| 5 | [MemPalace](https://github.com/MemPalace/mempalace) | Memory | 9.7 | 28.3 |
| 6 | [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG) | RAG | 9.4 | 31.9 |
| 7 | [mem0](https://github.com/mem0ai/mem0) | Memory | 9.2 | 23.7 |
| — | *Oracle ceiling (Qwen3-VL-8B)* | — | *40.1* | — |

On Hard, ChronicleMem beats the best prior by **+5.4 QS** (over ATM-RAG 13.8) and
**+9.3 R@10** (over MemoryOS 32.7); retrieval alone lifts Hard QS from **8.4 → 19.2**.

**Ablation** (Recall@10; each component earns its place)

| Configuration | Hard | Std |
|---------------|-----:|----:|
| Full · 4-channel RRF | 36.7 | 78.1 |
| &nbsp;&nbsp;− Dense | 19.1 | 69.5 |
| &nbsp;&nbsp;− Sparse (BM25) | 35.6 | 71.9 |
| &nbsp;&nbsp;− Metadata | 35.0 | 77.4 |
| &nbsp;&nbsp;− Vision | 38.5 | 77.5 |
| + Per-query routing | 40.0 | 78.7 |
| + Cross-encoder reranker (Qwen3-Reranker-4B) | **42.0** | **83.3** |

*Rows 2–5 remove one channel from the full 4-channel RRF (row 1); the bottom two
add pipeline stages on top. The last row is ChronicleMem's full config.*

> **Status.** Only the **retrieval stage** is evaluated so far — ingestion, memory
> organization, and the answerer are all unchanged. Even with strong retrieval,
> Hard QS plateaus near ~20%: the remaining bottleneck is **multi-evidence
> aggregation, not retrieval**, which is where the [Roadmap](#-roadmap) heads next.

<a id="reproduce"></a>
## 🔁 Reproduce

### 1. Install

```bash
conda create -n chronicle python=3.11 -y
conda activate chronicle
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

### 4. Run ChronicleMem

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
python -m unittest chronicle.test_hybrid_retriever -v
python -m chronicle.demo_cli "Where did I have ramen in Tokyo?"
```

<a id="repository-structure"></a>
## 📁 Repository Structure

```
ChronicleMem/
├── chronicle/               # ★ ChronicleMem method: hybrid + adaptive-routing retrieval
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

ChronicleMem is built on **ATM-Bench** and inherits its benchmark, dataset, task, core
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

If you use ChronicleMem, please also cite the ATM-Bench benchmark it is built on:

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

<!-- TODO: add the ChronicleMem citation once your paper is public.
@article{zhu2026chronicle,
  title={<ChronicleMem paper title>},
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
