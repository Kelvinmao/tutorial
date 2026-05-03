#!/usr/bin/env python3
"""
Chapter 14 — Evolutionary auto-tuner.

Usage:
    python auto_tuner.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Evolutionary Search (Genetic Algorithm) for Auto-Tuning
#
# Historical context: Genetic algorithms were introduced by Holland (1975).
# Their application to compiler optimization was explored by Cooper et al.
# (1999, "Optimizing for Reduced Code Space Using Genetic Algorithms").
# TVM's AutoTVM uses simulated annealing and XGBoost-guided search;
# Ansor uses beam search. The evolutionary approach here is simple and
# effective for the search space sizes encountered in practice.
#
# Problem solved: The search space has ~4,000 configurations (ch14
# search_space.py). Exhaustive search works but is slow for larger
# spaces. We need a smarter search that converges to good solutions
# in fewer evaluations.
#
# How it works (standard evolutionary/genetic algorithm):
#
# 1. INITIALIZATION: Generate a random population of `pop_size`
#    configurations by sampling the search space.
#
# 2. EVALUATION: Score each configuration using the cost model.
#    Lower cost = better fitness.
#
# 3. SELECTION: Sort by fitness. Keep the top 50% as survivors
#    (elitism — the best solutions always survive).
#
# 4. BREEDING: Fill the remaining slots by selecting pairs of
#    survivors and combining them:
#    a) CROSSOVER: For each parameter, randomly pick the value
#       from parent A or parent B (uniform crossover).
#    b) MUTATION: With probability `mutation_rate`, randomly
#       replace one parameter with a new random value.
#
#   Generation 0 (random):        Generation N (converged):
#
#   Cost
#   1000 │██                          │
#    800 │████                        │
#    600 │██████████                  │
#    400 │████████                    │
#    200 │████                        │████
#    100 │██                          │████████████████████
#        └────────────configs     └────────────configs
#
#   Crossover:   Parent A: {tile_m:32, tile_n:16}
#                Parent B: {tile_m:8,  tile_n:64}
#                Child:    {tile_m:32, tile_n:64} (mix of both)
#
#   Mutation:    Child: {tile_m:32, tile_n:64}
#                       → {tile_m:16, tile_n:64} (tile_m mutated)
#
# 5. REPEAT for `generations` generations. Track the best-ever
#    configuration and convergence history.
#
# Why it works: Good configurations tend to share good parameter
# values (e.g., tile_m=32 is often good). Crossover combines good
# values from different parents. Mutation prevents getting stuck
# in local optima. Elitism ensures we never lose the best solution.
#
# Complexity: O(generations × pop_size) cost model evaluations.
# Typically ~1,000 evaluations find near-optimal solutions.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import random
import json
from rich.console import Console
from rich.table import Table
from rich import box

from search_space import matmul_search_space, SearchSpace
from cost_model import estimate_cost

console = Console()


def mutate(config: dict[str, int], space: SearchSpace) -> dict[str, int]:
    """Mutate one random parameter."""
    new = dict(config)
    param = random.choice(space.params)
    new[param.name] = param.sample()
    return new


def crossover(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Uniform crossover of two configs."""
    child = {}
    for k in a:
        child[k] = a[k] if random.random() < 0.5 else b[k]
    return child


def evolutionary_search(M: int, N: int, K: int,
                        pop_size: int = 20,
                        generations: int = 50,
                        mutation_rate: float = 0.3) -> tuple:
    """
    Simple evolutionary search for best matmul configuration.

    Returns (best_config, best_cost, history).
    """
    space = matmul_search_space(M, N, K)
    console.print(f"Search space: {space.size()} configurations")

    # Initialize population
    population = [space.sample() for _ in range(pop_size)]
    history = []  # (generation, best_cost, avg_cost)

    best_ever = None
    best_cost_ever = float("inf")

    for gen in range(generations):
        # Evaluate
        costs = [(cfg, estimate_cost(M, N, K, cfg)) for cfg in population]
        costs.sort(key=lambda x: x[1])

        gen_best = costs[0][1]
        gen_avg = sum(c for _, c in costs) / len(costs)
        history.append((gen, gen_best, gen_avg))

        if gen_best < best_cost_ever:
            best_cost_ever = gen_best
            best_ever = costs[0][0]

        # Selection: keep top 50%
        survivors = [cfg for cfg, _ in costs[:pop_size // 2]]

        # Breed next generation
        next_gen = list(survivors)  # elitism
        while len(next_gen) < pop_size:
            a, b = random.sample(survivors, 2)
            child = crossover(a, b)
            if random.random() < mutation_rate:
                child = mutate(child, space)
            next_gen.append(child)

        population = next_gen

    return best_ever, best_cost_ever, history


def demo():
    console.print("\n[bold]═══ Auto-Tuning Search ═══[/]\n")

    M, N, K = 128, 128, 128
    best, cost, history = evolutionary_search(M, N, K)

    console.print(f"\n[bold green]Best configuration found:[/]")
    table = Table(box=box.ROUNDED)
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    for k, v in best.items():
        table.add_row(k, str(v))
    table.add_row("[bold]Cost[/]", f"[bold]{cost:.0f}[/]")
    console.print(table)

    # Save history for visualization
    with open("search_history.json", "w") as f:
        json.dump(history, f)
    console.print("\n[dim]Search history saved to search_history.json[/]")

    # Compare with baseline
    baseline = {"tile_m": 1, "tile_n": 1, "tile_k": 128,
                "unroll_factor": 1, "vectorize_width": 1}
    baseline_cost = estimate_cost(M, N, K, baseline)
    console.print(f"\nBaseline cost: {baseline_cost:.0f}")
    console.print(f"Tuned cost:    {cost:.0f}")
    console.print(f"Speedup:       {baseline_cost/cost:.1f}×")


if __name__ == "__main__":
    demo()
