#!/usr/bin/env python3
"""
Chapter 11 — Loop Tiling: improve cache performance for matrix multiply.

Demonstrates the dramatic performance difference between naive and tiled
matrix multiplication.

Usage:
    python loop_tiling.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Loop Tiling (Cache Blocking / Strip Mining)
#
# Historical context: Loop tiling was formalized by Wolfe (1989) and
# Lam, Rothberg & Wolf (1991). It's the single most impactful loop
# optimization for dense linear algebra. The ATLAS library (1998)
# used auto-tuned tile sizes to match BLAS performance, and modern
# AI compilers (TVM, Halide) use tiling as a primary scheduling knob.
#
# Problem solved: Naive matrix multiply (ijk loops) has terrible cache
# behavior. When scanning column j of matrix B, each element is in a
# different cache line, causing an L1 miss per element. For large N,
# the working set far exceeds cache, so every access goes to main memory.
#
# How it works:
# 1. Split each loop dimension into an OUTER tile loop and an INNER
#    element loop:
#      for i in range(0, M, tile):      # outer tile
#        for j in range(0, N, tile):
#          for k in range(0, K, tile):
#            for ii in range(i, i+tile): # inner element
#              for jj in range(j, j+tile):
#                for kk in range(k, k+tile):
#                  C[ii,jj] += A[ii,kk] * B[kk,jj]
#
#   Naive access pattern:          Tiled access pattern:
#   (scanning B column-wise)        (B tile reused from cache)
#
#   B matrix:                       B matrix:
#   ┌───────────────┐               ┌───────────────┐
#   │ ↓ ↓ ↓ ↓ ↓ ↓ ↓ │               │████│           │
#   │ ↓ ↓ ↓ ↓ ↓ ↓ ↓ │  every access   │████│           │  tile stays
#   │ ↓ ↓ ↓ ↓ ↓ ↓ ↓ │  is a cache     │████│           │  in L1 cache
#   │ ↓ ↓ ↓ ↓ ↓ ↓ ↓ │  miss!          │████│           │  for tile_A
#   └───────────────┘               └───────────────┘  rows
#
#   Working set:                    Working set:
#   ≈ M×N×K (entire B)              ≈ tile×tile×3 (fits in cache!)
#
# 2. The tile×tile submatrices of A, B, and C fit in L1/L2 cache.
#    All arithmetic on a tile reuses data already in cache.
#
# 3. Cache miss analysis:
#    - Naive: B accessed column-wise → ~M*N*K/8 L1 misses
#    - Tiled: B tile accessed multiple times from cache → far fewer misses
#    - Typical improvement: 5–10× for matrices > 64×64
#
# Choosing tile size: tile should make the working set (tile_A + tile_B +
# tile_C = tile×K + K×tile + tile×tile) fit in L1 cache. For a 32KB L1
# with 8-byte doubles, tile=32 uses 32*32*3*8 = 24KB → fits.
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive triple-nested loop matrix multiply."""
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_tiled(A: np.ndarray, B: np.ndarray, tile: int = 32) -> np.ndarray:
    """Tiled matrix multiply — processes tile×tile blocks at a time."""
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N))

    for ii in range(0, M, tile):
        for jj in range(0, N, tile):
            for kk in range(0, K, tile):
                # Process a tile×tile block
                i_end = min(ii + tile, M)
                j_end = min(jj + tile, N)
                k_end = min(kk + tile, K)
                for i in range(ii, i_end):
                    for j in range(jj, j_end):
                        for k in range(kk, k_end):
                            C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NumPy reference (uses optimized BLAS)."""
    return A @ B


def count_cache_misses(M: int, N: int, K: int, cache_line: int = 64,
                       elem_size: int = 8) -> dict:
    """Estimate cache misses for naive vs tiled matmul."""
    elems_per_line = cache_line // elem_size  # 8 doubles per cache line

    # Naive: B is accessed column-wise → high miss rate
    # For each (i,j), we scan a full row of A (good) and a full column of B (bad)
    naive_misses = M * N * (K // elems_per_line)  # rough estimate

    # Tiled (tile=32): B columns accessed within tile → reused in cache
    tile = min(32, K)
    tiled_misses = (M * N * K) // (tile * elems_per_line)

    return {"naive": naive_misses, "tiled": tiled_misses}


def demo():
    console.print("\n[bold]═══ Loop Tiling for Matrix Multiply ═══[/]\n")

    # Use small size for Python loops
    N = 64
    console.print(f"Matrix size: {N}×{N}")

    A = np.random.randn(N, N)
    B = np.random.randn(N, N)

    # Benchmark
    results = {}

    console.print("\n[dim]Running benchmarks (Python loops, small matrices)...[/]")

    t0 = time.perf_counter()
    C_naive = matmul_naive(A, B)
    results["Naive (ijk)"] = time.perf_counter() - t0

    for tile_size in [8, 16, 32]:
        t0 = time.perf_counter()
        C_tiled = matmul_tiled(A, B, tile=tile_size)
        results[f"Tiled (tile={tile_size})"] = time.perf_counter() - t0
        # Verify correctness
        assert np.allclose(C_naive, C_tiled, atol=1e-10)

    t0 = time.perf_counter()
    C_numpy = matmul_numpy(A, B)
    results["NumPy (BLAS)"] = time.perf_counter() - t0
    assert np.allclose(C_naive, C_numpy, atol=1e-10)

    # Display results
    table = Table(title=f"MatMul Performance ({N}×{N})", box=box.ROUNDED)
    table.add_column("Method")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Speedup vs Naive", justify="right")

    naive_time = results["Naive (ijk)"]
    for method, t in results.items():
        speedup = naive_time / t if t > 0 else float('inf')
        table.add_row(method, f"{t*1000:.1f}", f"{speedup:.1f}x")

    console.print(table)

    # Cache miss analysis
    console.print("\n[bold cyan]Cache Miss Analysis:[/]")
    misses = count_cache_misses(N, N, N)
    console.print(f"  Estimated cache misses (naive):  {misses['naive']:,}")
    console.print(f"  Estimated cache misses (tiled):  {misses['tiled']:,}")
    console.print(f"  Reduction: {misses['naive'] / max(misses['tiled'], 1):.1f}x")

    console.print(
        "\n[dim]Key insight: tiling keeps data in L1/L2 cache between reuse,\n"
        "dramatically reducing the number of slow main memory accesses.[/]"
    )

    console.print("\n[bold green]✓ All results verified against NumPy.[/]")


if __name__ == "__main__":
    demo()
