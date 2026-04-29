#!/usr/bin/env python3
"""
Chapter 11 — Parallelization: distribute loop iterations across cores.

Usage:
    python parallelization.py
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def matmul_sequential(A, B):
    """Single-threaded matrix multiply using NumPy for each row."""
    M = A.shape[0]
    N = B.shape[1]
    C = np.zeros((M, N))
    for i in range(M):
        C[i, :] = A[i, :] @ B
    return C


def matmul_parallel(A, B, num_threads=4):
    """
    Parallel matrix multiply: distribute rows across threads.

    Each thread computes a contiguous block of output rows.
    """
    M = A.shape[0]
    C = np.zeros((M, B.shape[1]))

    def compute_rows(start, end):
        C[start:end, :] = A[start:end, :] @ B

    chunk = M // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = []
        for t in range(num_threads):
            start = t * chunk
            end = start + chunk if t < num_threads - 1 else M
            futures.append(pool.submit(compute_rows, start, end))
        for f in futures:
            f.result()

    return C


def demo():
    console.print("\n[bold]═══ Loop Parallelization ═══[/]\n")

    N = 512
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)

    console.print(f"Matrix size: {N}×{N}")
    console.print(f"Total work: {N}³ = {N**3:,} multiply-add operations\n")

    # Benchmark
    results = {}

    t0 = time.perf_counter()
    C_seq = matmul_sequential(A, B)
    results["Sequential"] = time.perf_counter() - t0

    for threads in [2, 4, 8]:
        t0 = time.perf_counter()
        C_par = matmul_parallel(A, B, num_threads=threads)
        results[f"Parallel ({threads} threads)"] = time.perf_counter() - t0
        assert np.allclose(C_seq, C_par, atol=1e-10)

    t0 = time.perf_counter()
    C_np = A @ B
    results["NumPy (BLAS, multi-threaded)"] = time.perf_counter() - t0

    table = Table(title="Parallel MatMul Performance", box=box.ROUNDED)
    table.add_column("Method")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Speedup", justify="right")

    baseline = results["Sequential"]
    for method, t in results.items():
        table.add_row(method, f"{t*1000:.1f}", f"{baseline/t:.1f}x")
    console.print(table)

    console.print("""
[bold cyan]Parallelization in AI Compilers:[/]

  ┌──────────────────────┐
  │ for i in 0..M:       │  ← parallelize outer loop
  │   for j in 0..N:     │     (each core gets M/num_cores rows)
  │     for k in 0..K:   │  ← keep inner loop sequential
  │       C[i,j] += ...  │     (good data locality)
  └──────────────────────┘

  Thread 0: rows [0, M/4)        Thread 2: rows [M/2, 3M/4)
  Thread 1: rows [M/4, M/2)     Thread 3: rows [3M/4, M)

[dim]AI compilers like TVM annotate loop axes with 'parallel', 'vectorize',
'unroll' directives to combine all these optimizations.[/]""")


if __name__ == "__main__":
    demo()
