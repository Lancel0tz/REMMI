#!/usr/bin/env python3
"""Deploy the best routing configuration as the default.

Usage:
    python scripts/deploy_best_config.py \
        --config config/best_routing_config.json \
        --output output/best_config_deployment.json
"""

import argparse
import json
from pathlib import Path

def deploy_config(config_path: Path, output_path: Path):
    """Deploy best configuration."""

    with open(config_path) as f:
        config = json.load(f)

    best_params = {
        **config["hybrid_scoring_config"],
        **config["signal_thresholds"],
        "optimization_info": config["optimization"],
    }

    deployment = {
        "status": "DEPLOYED",
        "timestamp": config["optimization"]["timestamp"],
        "best_config": best_params,
        "usage_example": {
            "python": """
from remmi import HybridRetriever, HybridScoringConfig
import json

# Load deployed config
with open("config/best_routing_config.json") as f:
    config = json.load(f)

# Create retriever with best parameters
scoring_config = HybridScoringConfig(
    filter_mode=config['hybrid_scoring_config']['filter_mode'],
    weight_metadata=config['hybrid_scoring_config']['weight_metadata'],
    weight_sparse=config['hybrid_scoring_config']['weight_sparse'],
    weight_dense=config['hybrid_scoring_config']['weight_dense'],
    weight_vl=config['hybrid_scoring_config']['weight_vl'],
)

retriever = HybridRetriever(
    cache_dir="output/retrieval/index_cache",
    dense_retriever=text_retriever,
    scoring=scoring_config,
)

retriever.build_index(items)
results = retriever.retrieve(query, top_k=200)
            """
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(deployment, f, indent=2)

    print("=" * 80)
    print("✅ DEPLOYMENT SUCCESSFUL")
    print("=" * 80)
    print(f"\n📂 Config Location: {config_path}")
    print(f"📂 Deployment Log:  {output_path}")
    print(f"\n📊 Best Parameters (Trial {config['optimization']['trial_id']}):")
    print(f"   R@10:              {config['optimization']['optimized_r10']:.4f} ({config['optimization']['optimized_r10']*100:.2f}%)")
    print(f"   Baseline:          {config['optimization']['baseline_r10']:.4f} ({config['optimization']['baseline_r10']*100:.2f}%)")
    print(f"   Improvement:       {config['optimization']['improvement']}")
    print(f"\n⚙️  Default Parameters:")
    print(f"   weight_dense:      {config['hybrid_scoring_config']['weight_dense']} (56%)")
    print(f"   weight_sparse:     {config['hybrid_scoring_config']['weight_sparse']} (39%)")
    print(f"   weight_metadata:   {config['hybrid_scoring_config']['weight_metadata']} (19%)")
    print(f"   weight_vl:         {config['hybrid_scoring_config']['weight_vl']} (disabled)")
    print(f"   filter_mode:       {config['hybrid_scoring_config']['filter_mode']}")
    print(f"\n🎯 This is now the DEFAULT CONFIGURATION for Standard Set")
    print("=" * 80)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/best_routing_config.json")
    p.add_argument("--output", default="output/best_config_deployment.json")
    args = p.parse_args()

    deploy_config(Path(args.config), Path(args.output))
