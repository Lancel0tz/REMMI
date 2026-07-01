# Routing Strategy Optimization Guide

This guide covers optimizing ATM-Bench routing retrieval using Bayesian optimization and grid search.

## Current Baseline

- **Method**: Conditional Routing + VL (w=0.20)
- **R@10**: 73.18%
- **R@100**: 87.42%
- **Query time**: ~254ms per question

## Available Optimization Scripts

### 1. Grid Search Over Parameter Combinations

**Purpose**: Exhaustively test all parameter combinations to find the best configuration.

**Parameters**:
- `vl_weight`: Vision-language embedding importance (0.05 → 0.30)
- `filter_mode`: Metadata filtering strategy (soft, hard)
- `conditional_vl`: Inject VL only for visual/list queries (True, False)

**Usage**:
```bash
python scripts/QA_Agent/MMRAG/optimize_routing_grid_search.py \
  --qa-file data/atm-bench/atm-bench.json \
  --device cuda \
  --output-dir output/routing_optimization
```

**Expected output**:
```
output/routing_optimization/
├── grid_search_results.json    # All trial results
└── best_config.json            # Best parameters found
```

### 2. Bayesian Optimization (Sequential)

**Purpose**: Use Gaussian process to intelligently sample parameter space.

**Advantages**:
- Fewer trials needed than grid search
- Automatically focuses on promising regions
- Better for continuous parameter spaces

**Parameters optimized**:
- `vl_weight`: 0.05 → 0.50 (finer granularity)
- `vl_conditional`: True/False

**Usage**:
```bash
python scripts/QA_Agent/MMRAG/bayesian_optimize_routing.py \
  --n-trials 20 \
  --device cuda \
  --output-dir output/bayesian_opt
```

**Expected output**:
```
output/bayesian_opt/
├── trial_000/
│   ├── result.json            # Single trial results
│   └── ...
├── optimization_history.csv    # All trials summary
├── best_result.json            # Best configuration
└── best_result_summary.json
```

### 3. Multi-Stage Cascading Routing

**Purpose**: Implement two-stage retrieval: BM25 coarse filter → dense re-ranking.

**Advantages**:
- Reduces computation (dense only on BM25 top-k)
- Often improves precision
- Can be combined with other strategies

**Parameters**:
- `coarse_top_k`: BM25 candidates to retrieve (50-200)
- `dense_top_k`: Final top-k to return (5-50)
- `vl_weight`: VL reranker weight (0.05-0.40)

**Usage**:
```bash
python scripts/QA_Agent/MMRAG/run_multistage_routing.py \
  --qa-file data/atm-bench/atm-bench.json \
  --device cuda \
  --coarse-top-k 100 \
  --dense-top-k 10 \
  --vl-weight 0.20
```

**Expected output**:
```
output/QA_Agent/MMRAG/multistage/
├── multistage_summary.json     # Overall results
└── multistage_details.json     # Per-question details
```

## Optimization Workflow

### Phase 1: Quick Baseline (Grid Search)
1. Run grid search to explore parameter space:
   ```bash
   python optimize_routing_grid_search.py --method grid
   ```
2. Identify promising regions (e.g., vl_weight=0.15-0.25, soft mode)

### Phase 2: Fine-Tuning (Bayesian Optimization)
1. Run Bayesian optimization with 15-20 trials:
   ```bash
   python bayesian_optimize_routing.py --n-trials 20
   ```
2. Focus on best parameters from Phase 1

### Phase 3: Advanced Strategies (Multi-Stage)
1. Test cascading retrieval with optimized parameters:
   ```bash
   # Vary coarse_top_k and dense_top_k
   for k_coarse in 50 100 150; do
     for k_dense in 10 25 50; do
       python run_multistage_routing.py \
         --coarse-top-k $k_coarse \
         --dense-top-k $k_dense
     done
   done
   ```

## Key Metrics to Track

| Metric | Meaning | Target |
|--------|---------|--------|
| **R@10** | Recall at rank 10 | > 73.5% |
| **R@100** | Recall at rank 100 | > 87.4% |
| **Query time** | Avg ms per question | < 500ms |
| **Precision@10** | Accuracy in top 10 | Maximize |

## Parameter Tuning Tips

### VL Weight (`vl_weight`)
- **0.05-0.10**: Conservative, minimal VL impact
- **0.15-0.20**: Balanced (current best)
- **0.25-0.35**: Aggressive VL injection
- **Too high (>0.40)**: Can hurt email/numeric queries

### Filter Mode (`filter_mode`)
- **soft**: Boost/penalize metadata (recommended for standard set)
- **hard**: Strict date/location filtering (better for hard set)

### Conditional VL (`conditional_vl`)
- **True**: Inject VL only for visual/list queries (recommended)
- **False**: Always use VL weight (can hurt email queries)

## Performance Comparisons

### Baseline Methods
```
Method                          R@10    R@100   Time(ms)
────────────────────────────────────────────────────────
Text-only (MiniLM)              68.1%   91.6%   ~50
Hybrid (soft, no VL)            68.5%   91.2%   ~80
Routing (adaptive fusion)        72.3%   87.9%   ~150
Routing + Conditional VL (0.20) 73.18%  87.42%  ~254
```

### Expected Improvements
- Grid search: +0.5-1.0% R@10
- Bayesian opt: +0.2-0.8% R@10
- Multi-stage: +0.3-1.5% R@10 + faster queries

## Monitoring Optimization Progress

### Check current best result:
```bash
cat output/bayesian_opt/best_result.json | jq .best_params
```

### View optimization history:
```bash
python -c "
import pandas as pd
df = pd.read_csv('output/bayesian_opt/optimization_history.csv')
print(df[['number', 'value', 'params_vl_weight']].tail(10))
"
```

### Compare against baseline:
```bash
python -c "
import json
with open('output/bayesian_opt/best_result.json') as f:
    best = json.load(f)
    improvement = best['improvement_vs_baseline']
    print(f'Improvement vs baseline: +{improvement*100:.2f}%')
"
```

## Cost vs Benefit

| Method | Trials | Time per trial | Total time | Expected gain |
|--------|--------|----------------|------------|---------------|
| Grid search (5×2×2) | 20 | 15 min | 5 hours | +0.8% |
| Bayesian (20 trials) | 20 | 15 min | 5 hours | +0.5% |
| Multi-stage (3×3) | 9 | 15 min | 2.25 hours | +0.5-1.5% |

## Recommended Next Steps

1. **Run grid search** (Phase 1) to get intuition
2. **Refine with Bayesian opt** (Phase 2) for fine-tuning
3. **Test multi-stage** (Phase 3) if computational budget allows
4. **Final validation** on held-out hard set

## Advanced Extensions

### Custom Strategies
Modify `routing_retriever.py` to implement:
- Knowledge-aware weighting (different weights for different query types)
- Time-sensitive boosting (recent documents ranked higher)
- Domain-specific scoring (emphasize certain media types)

### Hybrid Approaches
Combine multiple strategies:
```python
# Example: Ensemble multiple routing strategies
routes = [
  adaptive_weights(query),      # Adaptive fusion
  adaptive_weights_hard(query),  # Hard-optimized
  custom_weights(query),         # Domain-specific
]
final_weights = ensemble(routes)
```

### Real-time Adaptation
Learn routing parameters from user feedback:
```python
# Meta-learning on per-user performance
while True:
    for user_query, user_feedback in stream:
        # Update routing weights based on feedback
        adaptive_router.update(user_query, user_feedback)
```

## References

- Original routing implementation: `remmi/routing_retriever.py`
- Baseline comparison: `output/QA_Agent/MMRAG/main_table/`
- Paper: See project proposal for technical details
