#!/usr/bin/env python3
"""
Chapter 14 — Search space: define tunable optimization parameters.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CONCEPT: Search Space Definition for Auto-Tuning
#
# Historical context: The idea of defining a "search space" of compiler
# knobs and automatically searching for the best combination was
# pioneered by ATLAS (1998) for BLAS, and later by TVM's AutoTVM (2018)
# and Ansor (2020) for tensor programs. The observation: optimal tile
# sizes, unroll factors, and vectorization widths depend on the specific
# hardware (cache sizes, SIMD width, core count) and can't be determined
# analytically — they must be found empirically.
#
# Problem solved: For a 128×128 matmul, there are many possible
# configurations:
#   tile_m ∈ {1,2,4,8,16,32,64}    → 7 choices
#   tile_n ∈ {1,2,4,8,16,32,64}    → 7 choices
#   tile_k ∈ {1,2,4,8,16,32,64}    → 7 choices
#   unroll_factor ∈ {1,2,4,8}      → 4 choices
#   vectorize_width ∈ {1,4,8}      → 3 choices
#   Total: 7×7×7×4×3 = 4,116 configurations
#
# Exhaustive search is feasible for small spaces but explodes for real
# workloads. Hence the need for smart search algorithms (ch14 auto_tuner).
#
# How it works:
# - TunableParam: a named parameter with a list of discrete choices.
# - SearchSpace: a collection of TunableParams.
# - sample(): randomly pick one value for each parameter.
# - size(): compute the total number of configurations (product of choices).
#
#   Search space for matmul:
#
#   Parameter        Choices            Count
#   ┌───────────────┬─────────────────────┬─────┐
#   │ tile_m        │ [1,2,4,8,16,32,64] │   7 │
#   ├───────────────┼─────────────────────┼─────┤
#   │ tile_n        │ [1,2,4,8,16,32,64] │   7 │
#   ├───────────────┼─────────────────────┼─────┤
#   │ tile_k        │ [1,2,4,8,16,32,64] │   7 │
#   ├───────────────┼─────────────────────┼─────┤
#   │ unroll_factor │ [1,2,4,8]          │   4 │
#   ├───────────────┼─────────────────────┼─────┤
#   │ vec_width     │ [1,4,8]            │   3 │
#   └───────────────┴─────────────────────┴─────┘
#                    Total: 7×7×7×4×3 = 4,116 configurations
#
#   One sample: {tile_m:32, tile_n:16, tile_k:8, unroll:4, vec:4}
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass
class TunableParam:
    name: str
    choices: list[int]

    def sample(self) -> int:
        return random.choice(self.choices)


class SearchSpace:
    """A set of tunable parameters for a computation."""

    def __init__(self):
        self.params: list[TunableParam] = []

    def add(self, name: str, choices: list[int]) -> "SearchSpace":
        self.params.append(TunableParam(name, choices))
        return self

    def sample(self) -> dict[str, int]:
        return {p.name: p.sample() for p in self.params}

    def size(self) -> int:
        result = 1
        for p in self.params:
            result *= len(p.choices)
        return result


def matmul_search_space(M: int, N: int, K: int) -> SearchSpace:
    """Standard search space for matrix multiply tuning."""
    # Only include tile sizes that divide the dimension (or are smaller)
    tile_choices = [1, 2, 4, 8, 16, 32, 64]

    space = SearchSpace()
    space.add("tile_m", [t for t in tile_choices if t <= M])
    space.add("tile_n", [t for t in tile_choices if t <= N])
    space.add("tile_k", [t for t in tile_choices if t <= K])
    space.add("unroll_factor", [1, 2, 4, 8])
    space.add("vectorize_width", [1, 4, 8])
    return space


if __name__ == "__main__":
    space = matmul_search_space(128, 128, 128)
    print(f"Search space size: {space.size()} configurations")
    print(f"Sample: {space.sample()}")
    print(f"Sample: {space.sample()}")
