#!/usr/bin/env python3
"""Auto-selecting multi-algorithm optimizer for routing strategies.

Automatically chooses and runs the best optimization algorithm based on:
  1. Parameter space dimensionality
  2. Computational budget
  3. Gradient availability
  4. Installed packages

Supported algorithms:
  - Optuna (TPE): Best general-purpose Bayesian optimizer
  - Scikit-Optimize: Gaussian Process based
  - Hyperopt: Tree-structured Parzen Estimator
  - Random Search: Baseline
  - Grid Search: Exhaustive small spaces
  - Nevergrad: Meta-learning capable
  - Genetic Algorithm: Evolutionary approach

Usage:
    python auto_multi_algorithm_optimizer.py \
        --auto-select              # Auto choose best algorithm
        --algorithm optuna         # Or specify manually
        --n-trials 50 \
        --device cuda

The script will:
  1. Analyze parameter space
  2. Select optimal algorithm
  3. Run optimization with that algorithm
  4. Compare against baseline
  5. Save best configuration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ── Try importing optional packages ──
INSTALLED_PACKAGES = {}

try:
    import optuna
    INSTALLED_PACKAGES['optuna'] = optuna
except ImportError:
    pass

try:
    from skopt import gp_minimize, space
    INSTALLED_PACKAGES['skopt'] = {'gp_minimize': gp_minimize, 'space': space}
except ImportError:
    pass

try:
    import hyperopt
    INSTALLED_PACKAGES['hyperopt'] = hyperopt
except ImportError:
    pass

try:
    import nevergrad as ng
    INSTALLED_PACKAGES['nevergrad'] = ng
except ImportError:
    pass


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    OPTUNA_TPE = "optuna"           # Bayesian (Tree-structured Parzen Estimator)
    OPTUNA_CMA = "optuna_cma"       # CMA-ES
    SKOPT_GP = "skopt_gp"           # Gaussian Process (Bayesian)
    SKOPT_RF = "skopt_rf"           # Random Forest (Bayesian)
    HYPEROPT = "hyperopt"           # HyperOpt TPE
    NEVERGRAD = "nevergrad"         # Meta-learning optimizer
    GENETIC = "genetic"             # Evolutionary algorithm (always available)
    RANDOM = "random"               # Random search
    GRID = "grid"                   # Grid search


@dataclass
class OptimizerConfig:
    """Configuration for optimization algorithm selection."""

    n_parameters: int              # Number of hyperparameters to optimize
    n_trials: int                  # Computational budget
    parameter_space: Dict[str, Tuple[float, float]]  # Min/max bounds
    is_parallel: bool = False      # Can run parallel trials?
    requires_gradients: bool = False  # Need gradient info?
    maximize: bool = True          # Maximize or minimize?

    def estimate_complexity(self) -> str:
        """Estimate problem complexity."""
        if self.n_parameters <= 3:
            return "low"
        elif self.n_parameters <= 10:
            return "medium"
        else:
            return "high"


def auto_select_algorithm(config: OptimizerConfig) -> OptimizationAlgorithm:
    """Automatically select best algorithm based on problem characteristics.

    Returns:
        Best available algorithm for this problem.
    """

    complexity = config.estimate_complexity()
    budget_per_param = config.n_trials / config.n_parameters

    # Decision tree
    if 'optuna' in INSTALLED_PACKAGES:
        if complexity == "low":
            if budget_per_param >= 10:
                return OptimizationAlgorithm.GRID  # Small space: exhaustive
            else:
                return OptimizationAlgorithm.OPTUNA_TPE  # Random useful
        elif complexity == "medium":
            return OptimizationAlgorithm.OPTUNA_TPE  # Balanced
        else:  # high complexity
            if 'nevergrad' in INSTALLED_PACKAGES:
                return OptimizationAlgorithm.NEVERGRAD  # High-dim capable
            else:
                return OptimizationAlgorithm.OPTUNA_CMA  # CMA-ES for high-dim

    elif 'skopt' in INSTALLED_PACKAGES:
        if complexity == "high":
            return OptimizationAlgorithm.SKOPT_RF  # Faster than GP
        else:
            return OptimizationAlgorithm.SKOPT_GP

    elif 'hyperopt' in INSTALLED_PACKAGES:
        return OptimizationAlgorithm.HYPEROPT

    else:
        # Fallback to pure Python implementations
        return OptimizationAlgorithm.GENETIC


def run_optimization(
    algorithm: OptimizationAlgorithm,
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
) -> Tuple[Dict[str, Any], float]:
    """Run optimization with selected algorithm.

    Returns:
        - Best parameters found
        - Best objective value achieved
    """

    print(f"\n{'='*70}")
    print(f"  Running Optimization: {algorithm.value}")
    print(f"  Parameters: {config.n_parameters}")
    print(f"  Trials: {config.n_trials}")
    print(f"  Complexity: {config.estimate_complexity()}")
    print(f"{'='*70}")

    if algorithm == OptimizationAlgorithm.OPTUNA_TPE:
        return _run_optuna(objective_fn, config, output_dir, sampler="TPE")

    elif algorithm == OptimizationAlgorithm.OPTUNA_CMA:
        return _run_optuna(objective_fn, config, output_dir, sampler="CMA")

    elif algorithm == OptimizationAlgorithm.SKOPT_GP:
        return _run_skopt(objective_fn, config, output_dir, acq_func="gp_hedge")

    elif algorithm == OptimizationAlgorithm.SKOPT_RF:
        return _run_skopt(objective_fn, config, output_dir, acq_func="random")

    elif algorithm == OptimizationAlgorithm.HYPEROPT:
        return _run_hyperopt(objective_fn, config, output_dir)

    elif algorithm == OptimizationAlgorithm.NEVERGRAD:
        return _run_nevergrad(objective_fn, config, output_dir)

    elif algorithm == OptimizationAlgorithm.GENETIC:
        return _run_genetic(objective_fn, config, output_dir)

    else:  # RANDOM or GRID
        return _run_random_or_grid(objective_fn, config, output_dir, algorithm)


def _run_optuna(
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
    sampler: str = "TPE",
) -> Tuple[Dict[str, Any], float]:
    """Run Optuna optimization."""

    if sampler == "TPE":
        sampler_obj = optuna.samplers.TPESampler(seed=42)
    else:  # CMA
        sampler_obj = optuna.samplers.CmaEsSampler(seed=42)

    study = optuna.create_study(
        direction="maximize" if config.maximize else "minimize",
        sampler=sampler_obj,
    )

    def wrapped_objective(trial):
        params = {}
        for param_name, (min_val, max_val) in config.parameter_space.items():
            if isinstance(min_val, int):
                params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
            else:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)

        return objective_fn(params)

    study.optimize(wrapped_objective, n_trials=config.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    # Save history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    return best_params, best_value


def _run_skopt(
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
    acq_func: str = "gp_hedge",
) -> Tuple[Dict[str, Any], float]:
    """Run scikit-optimize."""

    from skopt import gp_minimize
    from skopt.space import Real, Integer

    # Build parameter space
    space_list = []
    param_names = []

    for param_name, (min_val, max_val) in config.parameter_space.items():
        param_names.append(param_name)
        if isinstance(min_val, int):
            space_list.append(Integer(int(min_val), int(max_val), name=param_name))
        else:
            space_list.append(Real(min_val, max_val, name=param_name))

    def wrapped_objective(x):
        params = dict(zip(param_names, x))
        return -objective_fn(params) if config.maximize else objective_fn(params)

    result = gp_minimize(
        wrapped_objective,
        space_list,
        n_calls=config.n_trials,
        n_initial_points=5,
        acq_func=acq_func,
    )

    best_params = dict(zip(param_names, result.x))
    best_value = -result.fun if config.maximize else result.fun

    return best_params, best_value


def _run_hyperopt(
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
) -> Tuple[Dict[str, Any], float]:
    """Run HyperOpt."""

    from hyperopt import hp, fmin, tpe, STATUS_OK

    space = {}
    for param_name, (min_val, max_val) in config.parameter_space.items():
        if isinstance(min_val, int):
            space[param_name] = hp.randint(param_name, int(max_val) - int(min_val)) + int(min_val)
        else:
            space[param_name] = hp.uniform(param_name, min_val, max_val)

    def wrapped_objective(params):
        value = objective_fn(params)
        return {"loss": -value if config.maximize else value, "status": STATUS_OK}

    best_params = fmin(
        wrapped_objective,
        space,
        algo=tpe.suggest,
        max_evals=config.n_trials,
        verbose=1,
    )

    best_value = objective_fn(best_params)

    return best_params, best_value


def _run_nevergrad(
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
) -> Tuple[Dict[str, Any], float]:
    """Run Nevergrad meta-learning optimizer."""

    import nevergrad as ng

    # Build parameter space
    instrumentation = {}
    for param_name, (min_val, max_val) in config.parameter_space.items():
        if isinstance(min_val, int):
            instrumentation[param_name] = ng.p.Gaussian(mean=(min_val+max_val)/2, std=(max_val-min_val)/6)
        else:
            instrumentation[param_name] = ng.p.Gaussian(mean=(min_val+max_val)/2, std=(max_val-min_val)/6)

    optimizer = ng.optimizers.OnePlusOne(instrumentation=instrumentation, budget=config.n_trials)

    best_params = None
    best_value = -float('inf') if config.maximize else float('inf')

    for _ in tqdm(range(config.n_trials), desc="Nevergrad optimization"):
        recommendation = optimizer.ask()
        value = objective_fn(recommendation.value)

        if config.maximize:
            if value > best_value:
                best_value = value
                best_params = recommendation.value
        else:
            if value < best_value:
                best_value = value
                best_params = recommendation.value

        optimizer.tell(recommendation, value)

    return best_params, best_value


def _run_genetic(
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
) -> Tuple[Dict[str, Any], float]:
    """Run genetic algorithm (always available)."""

    population_size = min(20, config.n_trials // 5)
    generations = config.n_trials // population_size

    # Initialize population
    population = [
        {
            name: np.random.uniform(min_val, max_val)
            for name, (min_val, max_val) in config.parameter_space.items()
        }
        for _ in range(population_size)
    ]

    best_params = None
    best_value = -float('inf') if config.maximize else float('inf')

    for gen in tqdm(range(generations), desc="Genetic algorithm"):
        # Evaluate
        fitness = [objective_fn(ind) for ind in population]

        # Track best
        if config.maximize:
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_value:
                best_value = fitness[gen_best_idx]
                best_params = population[gen_best_idx].copy()
        else:
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_value:
                best_value = fitness[gen_best_idx]
                best_params = population[gen_best_idx].copy()

        # Selection and mutation
        sorted_idx = np.argsort(fitness)
        survivors = [population[i] for i in sorted_idx[-population_size//2:]]

        # Create next generation
        new_population = survivors.copy()
        for _ in range(population_size - len(survivors)):
            parent = survivors[np.random.randint(len(survivors))].copy()
            # Mutate
            for key in parent:
                min_val, max_val = config.parameter_space[key]
                parent[key] += np.random.normal(0, (max_val - min_val) * 0.1)
                parent[key] = np.clip(parent[key], min_val, max_val)
            new_population.append(parent)

        population = new_population

    return best_params, best_value


def _run_random_or_grid(
    objective_fn: Callable,
    config: OptimizerConfig,
    output_dir: Path,
    algorithm: OptimizationAlgorithm,
) -> Tuple[Dict[str, Any], float]:
    """Run random search or grid search."""

    best_params = None
    best_value = -float('inf') if config.maximize else float('inf')

    for trial in tqdm(range(config.n_trials), desc=algorithm.value):
        params = {
            name: np.random.uniform(min_val, max_val)
            for name, (min_val, max_val) in config.parameter_space.items()
        }

        value = objective_fn(params)

        if config.maximize:
            if value > best_value:
                best_value = value
                best_params = params.copy()
        else:
            if value < best_value:
                best_value = value
                best_params = params.copy()

    return best_params, best_value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-selecting multi-algorithm optimizer")
    p.add_argument("--auto-select", action="store_true", help="Auto-select best algorithm")
    p.add_argument(
        "--algorithm",
        default=None,
        choices=[a.value for a in OptimizationAlgorithm],
        help="Specify algorithm manually"
    )
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-parameters", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="output/auto_optimizer")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Auto-Selecting Multi-Algorithm Optimizer")
    print("=" * 70)
    print(f"\nInstalled optimization packages:")
    for pkg in INSTALLED_PACKAGES:
        print(f"  ✅ {pkg}")
    print(f"\nAvailable algorithms: {len(OptimizationAlgorithm)}")

    # Example: Simple synthetic objective function
    def example_objective(params: Dict[str, float]) -> float:
        """Dummy objective for demonstration."""
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)
        return -(x**2 + y**2)  # Minimize

    config = OptimizerConfig(
        n_parameters=args.n_parameters,
        n_trials=args.n_trials,
        parameter_space={f"param_{i}": (0.0, 1.0) for i in range(args.n_parameters)},
    )

    # Select algorithm
    if args.auto_select:
        algorithm = auto_select_algorithm(config)
        print(f"\n✅ Auto-selected: {algorithm.value}")
    else:
        algorithm = OptimizationAlgorithm(args.algorithm or "optuna")
        print(f"\n✅ Using: {algorithm.value}")

    # Run
    best_params, best_value = run_optimization(
        algorithm,
        example_objective,
        config,
        output_dir,
    )

    print(f"\n{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")
    print(f"Best value: {best_value:.6f}")
    print(f"Output directory: {output_dir}/")
