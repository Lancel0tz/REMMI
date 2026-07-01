# Experiments — research scaffolding

This directory holds the exploratory scripts used while developing **LTMA**
(hyperparameter sweeps, routing-weight optimizers, recall ablations, retriever
comparisons, cluster launch scripts).

> **These are provided for transparency, not for reproduction.** You do **not**
> need anything in here to reproduce the headline ATM-RAG results — those run
> from the pinned config in [`config/`](../config) via the commands in the main
> [README](../README.md#-reproduce). The scripts here are kept as-is: they may
> reference machine-specific paths, SLURM partitions, or intermediate artifacts
> that are not shipped.

## Layout

| Path | What it is |
|------|------------|
| `routing_optimization/` | Grid / Bayesian / meta-strategy search over routing weights and fusion strategies, plus sweep analysis, error analysis (BM25 vs dense), and retriever-comparison configs (`Comparisons/`). The final weights these searches produced are frozen in `config/best_routing_config*.json`. |
| `recall_ablation/`      | Retrieval-recall ablations for query decomposition, the LLM router, and the reranker, plus their SLURM launchers and smoke tests. |

## Provenance

Everything here imports the LTMA method package the same way the production
scripts do (`from ltma import ...`), so it runs from the repo root against the
same code path — it is simply not part of the curated reproduce flow.
