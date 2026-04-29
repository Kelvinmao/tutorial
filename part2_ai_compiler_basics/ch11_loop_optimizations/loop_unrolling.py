#!/usr/bin/env python3
"""
Chapter 11 — Loop Unrolling: reduce loop overhead.

Usage:
    python loop_unrolling.py
"""

import numpy as np
import time
from rich.console import Console

console = Console()


def vector_add_normal(A, B, C, N):
    for i in range(N):
        C[i] = A[i] + B[i]


def vector_add_unrolled_4(A, B, C, N):
    """Unroll by factor 4: process 4 elements per iteration."""
    i = 0
    while i + 3 < N:
        C[i]     = A[i]     + B[i]
        C[i + 1] = A[i + 1] + B[i + 1]
        C[i + 2] = A[i + 2] + B[i + 2]
        C[i + 3] = A[i + 3] + B[i + 3]
        i += 4
    # Handle remainder
    while i < N:
        C[i] = A[i] + B[i]
        i += 1


def vector_add_unrolled_8(A, B, C, N):
    """Unroll by factor 8."""
    i = 0
    while i + 7 < N:
        C[i]     = A[i]     + B[i]
        C[i + 1] = A[i + 1] + B[i + 1]
        C[i + 2] = A[i + 2] + B[i + 2]
        C[i + 3] = A[i + 3] + B[i + 3]
        C[i + 4] = A[i + 4] + B[i + 4]
        C[i + 5] = A[i + 5] + B[i + 5]
        C[i + 6] = A[i + 6] + B[i + 6]
        C[i + 7] = A[i + 7] + B[i + 7]
        i += 8
    while i < N:
        C[i] = A[i] + B[i]
        i += 1


def demo():
    console.print("\n[bold]═══ Loop Unrolling ═══[/]\n")

    N = 10000
    A = np.random.randn(N)
    B = np.random.randn(N)
    C = np.zeros(N)

    benchmarks = {
        "Normal loop": vector_add_normal,
        "Unrolled x4": vector_add_unrolled_4,
        "Unrolled x8": vector_add_unrolled_8,
    }

    results = {}
    for name, func in benchmarks.items():
        t0 = time.perf_counter()
        for _ in range(10):
            func(A, B, C, N)
        results[name] = (time.perf_counter() - t0) / 10

    baseline = results["Normal loop"]
    for name, t in results.items():
        speedup = baseline / t
        console.print(f"  {name:20s}: {t*1000:.2f} ms  ({speedup:.2f}x)")

    console.print("\n[dim]In Python, unrolling has limited benefit due to interpreter overhead.")
    console.print("In C/LLVM, unrolling enables instruction-level parallelism on the CPU pipeline.[/]")

    # Show what unrolled code looks like in pseudo-C
    console.print("\n[bold cyan]What LLVM generates (conceptual):[/]")
    console.print("""
  // Original
  for (int i = 0; i < N; i++) {
      C[i] = A[i] + B[i];           // 1 add + 1 branch per iteration
  }

  // Unrolled x4
  for (int i = 0; i < N; i += 4) {
      C[i]   = A[i]   + B[i];       // 4 adds + 1 branch per 4 iterations
      C[i+1] = A[i+1] + B[i+1];     // → 75% fewer branches
      C[i+2] = A[i+2] + B[i+2];     // → better instruction pipelining
      C[i+3] = A[i+3] + B[i+3];
  }""")


if __name__ == "__main__":
    demo()
