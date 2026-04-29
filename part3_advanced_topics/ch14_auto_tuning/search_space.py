#!/usr/bin/env python3
"""
Chapter 14 — Search space: define tunable optimization parameters.
"""

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
