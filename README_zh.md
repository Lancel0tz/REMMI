<div align="center">

# CHRONICLE — 长期个人记忆系统

**构建于并在 [ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench) 上评测。当前为混合、查询自适应的检索;按时间的事件化记忆组织在路线图上。**

[🇬🇧 English](README.md) • [🇨🇳 中文](README_zh.md)

<!-- TODO: 论文/项目主页公开后补上你自己的 badge
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/XXXX.XXXXX)
-->
[![Benchmark: ATM-Bench](https://img.shields.io/badge/Benchmark-ATM--Bench-1f6feb.svg)](https://github.com/JingbiaoMei/ATM-Bench)
[![Live Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-Live-orange.svg)](https://atmbench.github.io/leaderboard.html)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Dataset-FFD21E.svg)](https://huggingface.co/datasets/Jingbiao/ATM-Bench)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[✨ 新增内容](#-chronicle-新增内容) • [🧩 方法](#-方法) • [📊 结果](#-结果) • [🔁 复现](#-复现) • [📁 结构](#-仓库结构) • [📖 引用](#-引用)

</div>

> **CHRONICLE** 是在 **ATM-Bench** 基准之上的增量工作。它不重新发布该基准，而是贡献了
> 一个检索方法以及复现它的代码（即下方结果中的 `CHRONICLE (Ours)` 那一行）。基准、数据集、
> 任务定义以及 CHRONICLE 所对比的各 baseline 均为 ATM-Bench 作者的工作，详见
> [致谢与上游](#-致谢与上游)。

---

## 📋 目录

- [✨ CHRONICLE 新增内容](#-chronicle-新增内容)
- [🗺️ 路线图](#-路线图)
- [🧩 方法](#-方法)
- [📊 结果](#-结果)
- [🔁 复现](#-复现)
- [📁 仓库结构](#-仓库结构)
- [🙏 致谢与上游](#-致谢与上游)
- [📖 引用](#-引用)
- [📝 许可证](#-许可证)

<a id="chronicle-新增内容"></a>
## ✨ CHRONICLE 新增内容

以下全部是 CHRONICLE 在上游 ATM-Bench 之上新增的内容。核心方法集中在一个顶层包
[`chronicle/`](chronicle/) 中，贡献边界一目了然。

| 贡献 | 位置 | 说明 |
|------|------|------|
| **混合检索器** — 元数据过滤 + BM25 稀疏 + 稠密，通过 RRF 或加权和融合 | [`chronicle/hybrid_retriever.py`](chronicle/hybrid_retriever.py) | 稠密通道可插拔；无需 torch 即可在 CPU 上测试 |
| **查询自适应路由** — 逐查询信号分析 → 自适应通道权重（标准/困难两套） | [`chronicle/routing_retriever.py`](chronicle/routing_retriever.py) | `RoutingRetriever`、`adaptive_weights`、`adaptive_weights_hard` |
| **LLM 路由器** — 由 LLM 驱动的检索通道路由选择 | [`chronicle/llm_router.py`](chronicle/llm_router.py) | |
| **查询分解** — 将多证据查询拆成子查询 | [`chronicle/query_decomposer.py`](chronicle/query_decomposer.py) | |
| **Qwen3-VL 双编码器检索器**（MMRAG 稠密检索） | [`memqa/retrieve/retrievers.py`](memqa/retrieve/retrievers.py) | 与上游检索器并列新增 |
| **Qwen3-Reranker 修复** | [`memqa/retrieve/rerankers.py`](memqa/retrieve/rerankers.py) | |
| **冻结的最优配置**（贝叶斯优化产出） | [`config/best_routing_config.json`](config/best_routing_config.json)、[`config/best_routing_config_hard.json`](config/best_routing_config_hard.json) | 下方数字即由此复现 |
| **SimpleMem baseline 移植** | [`memqa/qa_agent_baselines/SimpleMem/`](memqa/qa_agent_baselines/SimpleMem/) | 加入记忆系统对比 |

*产出*这些冻结配置的探索性搜索/扫参代码（网格/贝叶斯/元策略优化器、召回消融、
检索器对比）位于 [`experiments/`](experiments/README.md)，**复现主结果并不需要**。

<a id="路线图"></a>
## 🗺️ 路线图

CHRONICLE 是增量式开发的。**当前已实现**的是检索这一侧——在 SGM 记忆上做混合、
查询自适应路由（[`chronicle/`](chronicle/)）。**下一步**是组织这一侧,也正是名字所指:

- **事件化记忆组织** — 把长期记忆按事件/情景分组。
- **分层检索** — 在该事件结构之上做层次化检索。
- **记忆链接** — 记忆项之间的推断关系。

以上尚未进入代码库;本节陈述的是意图,而非当前能力。

<a id="方法"></a>
## 🧩 方法

CHRONICLE 在 ATM-Bench 的 Schema-Guided Memory (SGM) 记忆项上做混合、查询自适应检索：

1. **三条检索通道**
   - **元数据** — 对 SGM `time` + `location` 字段做软过滤/加权。
   - **稀疏** — 在渲染后的 SGM 文本与所选元数据字段上做 BM25。
   - **稠密** — 任意 `BaseRetriever`（如 Qwen3 文本嵌入，或新增的 Qwen3-VL 双编码器）。
2. **分数融合** — RRF（默认）或对 min-max 归一化后的各通道分数做加权和。
3. **查询自适应路由** — `RoutingRetriever` 分析逐查询信号（元数据/关键词/语义强度）
   选择通道权重，对标准集与 `-Hard` 集分别采用不同权重档。
4. **可选的查询分解与 LLM 路由**，用于多证据查询。

冻结的工作点（贝叶斯优化，见 `config/`）采用
`元数据:稀疏:稠密 ≈ 0.19:0.39:0.56` + RRF，将标准集 Recall@10 从
**0.732 提升到 0.775**（+4.3%）。

<a id="结果"></a>
## 📊 结果

> 🏆 最权威、最新的数字见 [ATM-Bench 在线榜单](https://atmbench.github.io/leaderboard.html)，
> 下方快照可能滞后于新提交。

**记忆系统对比** — 回答模型 `Qwen3-VL-8B-Instruct-FP8`，记忆处理器
`Qwen3-VL-2B-Instruct`，`-Hard` 使用 `atm-bench-hard` 发布集。**CHRONICLE 即最下方
`CHRONICLE (Ours)` 那一行。**

| 系统 | 建索引时间 (hr) ↓ | ATM-Bench QS ↑ | ATM-Bench Recall@10 ↑ | ATM-Bench-Hard QS ↑ | ATM-Bench-Hard Recall@10 ↑ |
|------|------------------:|---------------:|----------------------:|--------------------:|---------------------------:|
| [A-Mem](https://github.com/WujiangXu/A-mem) | 12.6 | 44.8 | 66.4 | 9.9 | 31.7 |
| [mem0](https://github.com/mem0ai/mem0) | 16.7 | 43.5 | 61.9 | 9.2 | 23.7 |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 36.6 | 47.2 | 59.2 | 13.7 | 32.7 |
| [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG) | 1.5 | 42.9 | 66.4 | 9.4 | 31.9 |
| [MemPalace](https://github.com/MemPalace/mempalace) | 0.5 | 56.8 | 76.4 | 9.7 | 28.3 |
| [SimpleMem](https://github.com/aiming-lab/SimpleMem) | 15.7 | 27.3 | 23.3 | 3.2 | 7.0 |
| **CHRONICLE (Ours)** | **0.5** | **51.0** | **68.7** | **8.4** | **28.8** |

<!-- TODO: 论文定稿后补充消融（路由开关、各通道、reranker）。 -->

<a id="复现"></a>
## 🔁 复现

### 1. 安装

```bash
conda create -n chronicle python=3.11 -y
conda activate chronicle
pip install -r requirements.txt
pip install -e .
```

macOS / Apple Silicon（`vllm`/`decord` 等 GPU/视频服务包在 pip 上不稳定）：

```bash
bash scripts/setup_local_mac.sh
```

### 2. 数据（ATM-Bench，约 3.3 GB，来自 Hugging Face）

```bash
bash scripts/download_data.sh
```

会把 QA、NIAH 池、预处理记忆、邮件、原始媒体以及 GPS 反地理编码缓存
分别放到 `data/` 与 `output/`。完整数据布局见 [`docs/data.md`](docs/data.md)。

### 3. API 密钥

```bash
export OPENAI_API_KEY="your-key"     # 或 api_keys/.openai_key
export VLLM_API_KEY="your-key"       # 或 api_keys/.vllm_key
```

### 4. 运行 CHRONICLE

需要一个在 `http://127.0.0.1:8000/v1/...` 提供 `Qwen/Qwen3-VL-8B-Instruct-FP8`
的 vLLM 端点（可用 `VLLM_ENDPOINT` / `ANSWERER_MODEL` 覆盖）。

```bash
# 在标准集与 -Hard 集上运行混合 + 自适应路由的 MMRAG，
# 使用 config/best_routing_config*.json 中的冻结配置。
bash scripts/QA_Agent/MMRAG/run_routing_reranker.sh
```

仅检索变体（`scripts/QA_Agent/MMRAG/run_retrieval_only_cpu*.sh`）可在 CPU 上
评测 Recall@k，无需回答模型端点。完整设置见 [`docs/reproducibility.md`](docs/reproducibility.md)。

### 方法冒烟测试（CPU，无需数据/模型）

```bash
python -m unittest chronicle.test_hybrid_retriever -v
python -m chronicle.demo_cli "Where did I have ramen in Tokyo?"
```

<a id="仓库结构"></a>
## 📁 仓库结构

```
CHRONICLE/
├── chronicle/               # ★ CHRONICLE 方法：混合 + 自适应路由检索
├── config/             # 冻结的最优路由/融合配置（用于复现）
├── experiments/        # 研究脚手架（扫参/优化器）——复现主结果不需要
├── memqa/              # ATM-Bench 核心（baseline、检索、评测）+ 我们新增的检索器/reranker
├── agent_systems/      # 通用 agent 评测 harness
├── scripts/            # 数据下载、运行与评测入口
├── docs/               # 文档
├── third_party/        # 内置记忆系统 baseline
├── data/  output/      # 用户提供的数据与输出（已 gitignore）
└── LICENSE
```

<a id="致谢与上游"></a>
## 🙏 致谢与上游

CHRONICLE 构建于 **ATM-Bench**，继承其基准、数据集、任务、核心 `memqa/` 代码以及全部对比 baseline。

- **上游：** [`JingbiaoMei/ATM-Bench`](https://github.com/JingbiaoMei/ATM-Bench)
- **基于上游 commit：** [`d552cc5`](https://github.com/JingbiaoMei/ATM-Bench/commit/d552cc5b84f495ff173e7a9ddb598e9edfd2b539) *(2026-04-10)*
- **数据集：** Hugging Face 上的 [`Jingbiao/ATM-Bench`](https://huggingface.co/datasets/Jingbiao/ATM-Bench)
- **榜单：** [atmbench.github.io/leaderboard.html](https://atmbench.github.io/leaderboard.html)

内置/移植的 baseline 保留各自的上游许可证与固定 commit（例如 SimpleMem @
[`094027e`](https://github.com/aiming-lab/SimpleMem/commit/094027eca4c890dc9912be8cee1da04428de8076)），
详见 [`docs/baseline.md`](docs/baseline.md) 及各 baseline 的 README。

<a id="引用"></a>
## 📖 引用

若使用 CHRONICLE，请同时引用其所构建于的 ATM-Bench 基准：

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

<!-- TODO: 论文公开后补上 CHRONICLE 的引用。 -->

<a id="许可证"></a>
## 📝 许可证

MIT，见 [LICENSE](LICENSE)。部分代码派生自
[ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench)（MIT）。`third_party/` 与
`memqa/qa_agent_baselines/` 下内置的第三方 baseline 保留各自上游许可证。
