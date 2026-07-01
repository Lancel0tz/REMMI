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

1. **四条检索通道**
   - **元数据** — 对 SGM `time` + `location` 字段做软/硬过滤 + 加权。
   - **稀疏** — 在渲染后的 SGM 文本与所选元数据字段上做 BM25。
   - **稠密** — 任意 `BaseRetriever`（如 Qwen3 文本嵌入，或新增的 Qwen3-VL 双编码器）。
   - **视觉** — CLIP 式图像匹配，用于视觉相关的查询。
2. **查询自适应路由** — `RoutingRetriever` 分析逐查询信号（日期/地点、专有名词、
   金额、CJK 词、召回措辞、视觉提及）逐查询调整四条通道的权重，对标准集与 `-Hard`
   集分别采用不同权重档，并有置信门在首轮偏弱时放宽通道。
3. **融合 + 重排** — 对路由后的排名做 RRF，再由交叉编码器重排器
   （Qwen3-Reranker-4B）把 top-20 精修到 top-10。
4. **可选的查询分解与 LLM 路由**，用于多证据查询。

冻结的工作点见 [`config/`](config)。在固定 `Qwen3-VL-8B-Instruct` 回答模型不变的前提下，
该 pipeline 在标准集达到 **83.3 Recall@10 / 60.3 QS**——见 [结果](#-结果)。

<a id="结果"></a>
## 📊 结果

> 🏆 最权威、最新的数字见 [ATM-Bench 在线榜单](https://atmbench.github.io/leaderboard.html)。

在**固定** `Qwen3-VL-8B-Instruct` 回答模型下，CHRONICLE 的检索阶段在 ATM-Bench 上
**刷新了 Memory & RAG 系统的 SOTA**——因为回答模型不变，QS 的差距只能归因于检索、
而非语言模型。它在**两个 split 的 QS 和 Recall@10 上都排第一**：标准集 **60.3** QS /
**83.3** R@10，困难集 **19.2** QS / **42.0** R@10。

**表 1 — ATM-Bench 标准集**（单跳；按 QS 排序；所有系统同一回答模型）

| # | 系统 | 类型 | QS ↑ | R@10 ↑ |
|--:|------|:----:|-----:|-------:|
| **1** | **CHRONICLE (Ours)** · 混合检索 | RAG | **60.3** | **83.3** |
| 2 | [MemPalace](https://github.com/MemPalace/mempalace) | Memory | 56.8 | 76.4 |
| 3 | ScrapMem (No-Forget) | Memory | 52.5 | 70.3 |
| 4 | [ATM-RAG](https://github.com/JingbiaoMei/ATM-Bench) · 上游 | RAG | 51.0 | 68.7 |
| 5 | Self-RAG | RAG | 50.3 | 68.7 |
| 6 | [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | Memory | 47.2 | 59.2 |
| 7 | [A-Mem](https://github.com/WujiangXu/A-mem) | Memory | 44.8 | 66.4 |
| — | *Oracle 上限 (Qwen3-VL-8B)* | — | *78.2* | — |

**表 2 — ATM-Bench-Hard**（31 道多跳题，每题约 6 条证据；按 QS 排序）

| # | 系统 | 类型 | QS ↑ | R@10 ↑ |
|--:|------|:----:|-----:|-------:|
| **1** | **CHRONICLE (Ours)** · 混合检索 | RAG | **19.2** | **42.0** |
| 2 | [ATM-RAG](https://github.com/JingbiaoMei/ATM-Bench) · 上游 | RAG | 13.8 | 30.4 |
| 3 | [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | Memory | 13.7 | 32.7 |
| 4 | [A-Mem](https://github.com/WujiangXu/A-mem) | Memory | 9.9 | 31.7 |
| 5 | [MemPalace](https://github.com/MemPalace/mempalace) | Memory | 9.7 | 28.3 |
| 6 | [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG) | RAG | 9.4 | 31.9 |
| 7 | [mem0](https://github.com/mem0ai/mem0) | Memory | 9.2 | 23.7 |
| — | *Oracle 上限 (Qwen3-VL-8B)* | — | *40.1* | — |

在困难集上，CHRONICLE 比此前最好者高 **+5.4 QS**（对比 ATM-RAG 13.8）和
**+9.3 R@10**（对比 MemoryOS 32.7）；仅靠检索就把困难集 QS 从 **8.4 提升到 19.2**。

**消融**（Recall@10；每个组件都有其价值）

| 配置 | Hard | Std |
|------|-----:|----:|
| 完整 · 4 通道 RRF | 36.7 | 78.1 |
| &nbsp;&nbsp;− 稠密 | 19.1 | 69.5 |
| &nbsp;&nbsp;− 稀疏 (BM25) | 35.6 | 71.9 |
| &nbsp;&nbsp;− 元数据 | 35.0 | 77.4 |
| &nbsp;&nbsp;− 视觉 | 38.5 | 77.5 |
| + 逐查询路由 | 40.0 | 78.7 |
| + 交叉编码器重排 (Qwen3-Reranker-4B) | **42.0** | **83.3** |

*第 2–5 行是从完整 4 通道 RRF（第 1 行）中各去掉一条通道；最后两行是在其上叠加
pipeline 阶段，末行即 CHRONICLE 的完整配置。*

> **状态。** 目前只评测了**检索阶段**——摄取、记忆组织、回答模型全部不变。即便检索已很强，
> 困难集 QS 仍卡在 ~20%:剩余瓶颈是**多证据聚合,而非检索**,这正是 [路线图](#-路线图) 的下一步。

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
