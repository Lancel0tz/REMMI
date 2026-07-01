# Comprehensive Multi-Strategy Optimization Guide

你现在拥有一个**完整的贝叶斯优化系统**，可以自动探索和优化所有路由策略参数。

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│         Comprehensive Strategy Optimization System           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Basic Bayesian Optimization (初期探索)                  │
│     └─ optimize_routing_grid_search.py                      │
│        - 网格搜索 5×2×2 = 20 个配置                         │
│        - 耗时: ~3-4 小时                                     │
│        - 输出: 最优参数配置                                 │
│                                                             │
│  2. Comprehensive Strategy Optimization (全参数优化)        │
│     └─ comprehensive_strategy_optimizer.py                  │
│        - 优化 20+ 参数: 所有触发条件、通道权重              │
│        - 多轮级检索自动探索                                 │
│        - 贝叶斯优化: 30-50 次试验                           │
│        - 耗时: ~6-10 小时                                    │
│        - 输出: 全参数最优配置 + 多轮级策略                  │
│                                                             │
│  3. Meta-Learning Strategy Discovery (策略自动发现)         │
│     └─ meta_strategy_discovery.py                           │
│        - 演化算法: 5-10 个策略 × 20 代                      │
│        - 每个策略专门化处理特定查询类型                    │
│        - 学习查询→策略的映射                                │
│        - 耗时: ~4-6 小时                                     │
│        - 输出: 策略集合 + 触发条件                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 可优化的参数

### Tier 1: 基础参数 (5 个)
- `vl_weight`: 视觉-语言嵌入权重 (0.05-0.35)
- `filter_mode`: 元数据过滤策略 (soft/hard)
- `conditional_vl`: 条件性 VL 注入 (True/False)

### Tier 2: 通道参数 (10 个)
- `base_weight_*`: 各通道基础权重 (metadata, BM25, dense)
- `*_tilt_factor`: 通道倾斜因子 (如何从基线调整)
- `vl_conditional_threshold`: 何时注入 VL (0.2-0.8)

### Tier 3: 高级参数 (10+ 个)
- **信号阈值**: meta_score, keyword_score, semantic_score 触发点
- **硬模式配置**: BM25 cap, dense floor
- **多轮级参数**: coarse_top_k, dense_top_k, 重排序配置
- **查询特定调优**: 实体权重、相对日期处理等

## Quick Start: 3 步优化流程

### Step 1: 快速网格搜索 (2-3 小时)
了解参数空间的基本形态：

```bash
python scripts/QA_Agent/MMRAG/optimize_routing_grid_search.py \
  --qa-file data/atm-bench/atm-bench.json \
  --device cuda \
  --output-dir output/grid_opt
```

**查看结果**:
```bash
# 最佳配置
cat output/grid_opt/grid_search_results.json | jq '.[] | select(.recall["R@10"] > 0.72)'

# 按 R@10 排序
python -c "
import json
with open('output/grid_opt/grid_search_results.json') as f:
    results = json.load(f)
    sorted_results = sorted(results, key=lambda x: x['recall']['R@10'], reverse=True)
    for r in sorted_results[:5]:
        print(f\"{r['params']} → R@10={r['recall']['R@10']:.4f}\")
"
```

### Step 2: 全参数贝叶斯优化 (6-10 小时)

基于 Step 1 的洞察，进行深度优化：

```bash
python scripts/QA_Agent/MMRAG/comprehensive_strategy_optimizer.py \
  --n-trials 40 \
  --device cuda \
  --output-dir output/comprehensive_opt \
  --enable-multistage
```

**预期改进**: +0.5-2.0% R@10

**关键输出**:
- `trial_0000-0039/`: 每个试验的配置和结果
- `optimization_summary.json`: 最优参数 + 性能指标

### Step 3: 元学习策略发现 (4-6 小时)

自动发现针对不同查询类型优化的策略组合：

```bash
python scripts/QA_Agent/MMRAG/meta_strategy_discovery.py \
  --n-strategies 5 \
  --n-generations 20 \
  --device cuda \
  --output-dir output/meta_discovery
```

**输出**:
- `evolution_history.json`: 代际进化过程
- `final_strategies.json`: 5-10 个最优策略，各自专化于不同查询类型

## 参数解释和建议

### Signal Thresholds (触发条件)

| 参数 | 含义 | 建议范围 | 示例 |
|-----|------|---------|------|
| `meta_score_threshold` | 什么时候元数据信号足够强 | 0.3-0.8 | 0.6 = 60% 日期/位置关键词 |
| `keyword_score_threshold` | 何时优先 BM25 | 0.2-0.6 | 0.4 = 40% 普通关键词 |
| `semantic_score_threshold` | 何时优先 dense | 0.2-0.6 | 0.4 = 40% 语义词汇 |

**调优技巧**:
- 高阈值 (0.7+): 保守，只在信号极强时触发
- 中阈值 (0.4-0.6): 平衡，推荐
- 低阈值 (0.2-0.3): 激进，易误触发

### Channel Weights (通道权重)

```
基线配置 (best known):
  metadata: 0.10
  sparse:   0.20
  dense:    0.70
  vl:       0.20

Keyword-heavy queries (e.g., "what date..."):
  metadata: 0.20 ← boost metadata for date/location
  sparse:   0.50 ← boost BM25 for keyword matching
  dense:    0.25 ← reduce dense
  vl:       0.05

Visual/semantic queries (e.g., "describe the..."):
  metadata: 0.10
  sparse:   0.15
  dense:    0.60 ← boost dense for semantic understanding
  vl:       0.25 ← boost VL for images
```

### Multi-Stage Configuration (多轮级配置)

两阶段检索可显著提升效率同时保持精度：

```python
# Stage 1: BM25 粗筛 (快速)
coarse_k = 100  # 用 BM25 检索 top-100 (很快: ~10ms)

# Stage 2: Dense + VL 精排 (在 top-100 上)
dense_k = 10    # 从 100 个候选中精排出 top-10

# 效果: ~同样精度，查询时间减少 60-70%
```

## 性能基准

### 当前最佳 (Cond. Routing + VL w=0.20)
```
R@1:   0.388
R@5:   0.655
R@10:  0.732  ← 主要指标
R@100: 0.874
时间:  254ms/query
```

### 预期优化后
```
通过全参数贝叶斯优化:
  R@10: 0.745-0.760 (+1.5-4%)
  时间: 150-200ms (多轮级)

通过元学习策略:
  R@10: 0.750-0.770 (+2-5%)
  时间: 180-220ms (查询→策略路由开销)
```

## 监控优化进度

### 实时查看
```bash
# Watch best trial improving
watch -n 30 'tail -20 output/comprehensive_opt/optimization_summary.json | jq .best_params'

# Check trial outputs as they complete
ls -lrt output/comprehensive_opt/trial_* | tail -5
```

### 分析趋势
```python
import json
import matplotlib.pyplot as plt

# Load history
with open('output/comprehensive_opt/optimization_summary.json') as f:
    summary = json.load(f)

# Plot R@10 over trials
trials = range(len(summary['trials']))
r10_scores = [t['recall']['R@10'] for t in summary['trials']]

plt.plot(trials, r10_scores, marker='o')
plt.xlabel('Trial Number')
plt.ylabel('R@10')
plt.title('Optimization Progress')
plt.savefig('optimization_progress.png')
```

## 集成最优配置

### 在生产环境应用最优参数

```python
# 1. 从优化结果提取最优参数
import json
with open('output/comprehensive_opt/optimization_summary.json') as f:
    best_params = json.load(f)['best_params']

# 2. 创建自定义策略
from remmi import AdaptiveWeights

def get_best_weights(query: str) -> AdaptiveWeights:
    return AdaptiveWeights(
        weight_metadata=best_params['base_weight_meta'],
        weight_sparse=best_params['base_weight_sparse'],
        weight_dense=best_params['base_weight_dense'],
        vl_weight=best_params['vl_weight_base'],
    )

# 3. 应用到路由
router.retrieve_adaptive(query, override_weights=get_best_weights(query))
```

### 集成元学习策略
```python
# Load discovered strategies
with open('output/meta_discovery/final_strategies.json') as f:
    strategies = json.load(f)['best_strategies']

# 为每个策略创建应用逻辑
strategy_ensemble = {
    s['name']: {
        'weights': s['weights'],
        'triggers': s['triggers'],
    }
    for s in strategies
}

# 在推理时选择最佳策略
def route_query(query: str):
    signals = analyze_query(query)
    
    # 找到最匹配的策略
    best_strategy = None
    best_match_score = -1
    
    for name, config in strategy_ensemble.items():
        trigger_score = compute_trigger_match(signals, config['triggers'])
        if trigger_score > best_match_score:
            best_match_score = trigger_score
            best_strategy = config
    
    return retrieve_with_weights(query, best_strategy['weights'])
```

## 常见问题

### Q: 优化时间太长怎么办？
A: 按优先级：
1. 减少 trials: `--n-trials 20` (快速)
2. 用网格搜索代替贝叶斯优化 (5x速)
3. 只优化 3-5 个最关键参数

### Q: 优化没有改进怎么办？
A: 
1. 检查是否已在局部最优 (看 R@10 是否停滞)
2. 增加参数空间范围 (修改 `suggest_float` 的边界)
3. 尝试不同的初始配置种子 (`--seed` 参数)

### Q: 如何在硬集上优化？
A: 通过 `--qa-file data/atm-bench/atm-bench-hard.json` 运行，硬集优化参数会自动调整。

### Q: 多轮级会显著慢吗？
A: 不会。虽然有两阶段，但 Stage 1 (BM25) 很快，总时间通常减少 40-60%。

## 下一步

1. **现在运行** Step 1 (网格搜索) 了解参数空间
2. **然后运行** Step 2 (全参数优化) 获得最佳单一配置
3. **最后运行** Step 3 (元学习) 探索策略多样性

预期总收益: **R@10 +2-5%** (从 73.2% → 75-77%)

---

有任何问题或需要自定义优化？查看各脚本的 `--help` 获得完整参数列表！
