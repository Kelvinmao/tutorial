#!/usr/bin/env python3
"""
Chapter 14 — Simple analytical cost model for matrix multiply.
Estimates execution time based on cache behavior and compute.
"""

from __future__ import annotations
import math


# Simple machine model
CACHE_LINE = 64        # bytes
L1_SIZE = 32 * 1024    # 32 KB
L2_SIZE = 256 * 1024   # 256 KB
FLOPS_PER_CYCLE = 8    # FMA units
CLOCK_GHZ = 3.0
ELEM_SIZE = 4          # float32


def estimate_cost(M: int, N: int, K: int, config: dict[str, int]) -> float:
    """
    Estimate relative cost (lower = better) for a matmul configuration.

    Factors:
    1. Compute: total FLOPs / throughput
    2. Cache misses: based on tile sizes and cache capacity
    3. Overhead: loop overhead from tiling
    """
    tile_m = config["tile_m"]
    tile_n = config["tile_n"]
    tile_k = config["tile_k"]
    unroll = config["unroll_factor"]
    vec_w = config["vectorize_width"]

    total_flops = 2.0 * M * N * K

    # Compute cost (cycles)
    effective_throughput = FLOPS_PER_CYCLE * min(vec_w, 8)
    compute_cycles = total_flops / effective_throughput

    # Cache model: working set of one tile
    tile_a_bytes = tile_m * tile_k * ELEM_SIZE
    tile_b_bytes = tile_k * tile_n * ELEM_SIZE
    tile_c_bytes = tile_m * tile_n * ELEM_SIZE
    working_set = tile_a_bytes + tile_b_bytes + tile_c_bytes

    # Cache miss penalty
    if working_set <= L1_SIZE:
        miss_penalty = 1.0      # all in L1
    elif working_set <= L2_SIZE:
        miss_penalty = 3.0      # L2 hit
    else:
        miss_penalty = 20.0     # main memory

    n_tiles = math.ceil(M / tile_m) * math.ceil(N / tile_n) * math.ceil(K / tile_k)
    memory_cycles = n_tiles * miss_penalty * (working_set / CACHE_LINE)

    # Loop overhead (more tiles = more overhead, unrolling reduces it)
    loop_overhead = n_tiles * (tile_m * tile_n * tile_k) / max(unroll, 1) * 0.01

    total = compute_cycles + memory_cycles + loop_overhead
    return total


if __name__ == "__main__":
    configs = [
        {"tile_m": 1, "tile_n": 1, "tile_k": 128, "unroll_factor": 1, "vectorize_width": 1},
        {"tile_m": 32, "tile_n": 32, "tile_k": 32, "unroll_factor": 4, "vectorize_width": 8},
        {"tile_m": 64, "tile_n": 64, "tile_k": 64, "unroll_factor": 8, "vectorize_width": 8},
    ]
    M, N, K = 128, 128, 128
    for cfg in configs:
        cost = estimate_cost(M, N, K, cfg)
        print(f"Config {cfg} → cost = {cost:.0f}")
