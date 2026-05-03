#!/usr/bin/env python3
"""
Chapter 11 — Loop Unrolling: reduce loop overhead.

Usage:
    python loop_unrolling.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Loop Unrolling
#
# Historical context: Loop unrolling is one of the oldest compiler
# optimizations, used since the 1960s on early mainframes. Compilers
# like GCC and LLVM apply it automatically at -O2/-O3. The technique
# is simple: replicate the loop body N times and increment the counter
# by N, reducing the number of branch instructions.
#
# Problem solved: Each loop iteration has overhead:
# - Increment counter (add)
# - Compare against bound (cmp)
# - Conditional branch (branch, possibly mispredicted)
# For a tight inner loop (like vector add), this overhead can be 30–50%
# of total execution time.
#
# How it works:
# 1. Instead of processing 1 element per iteration, process N elements:
#      Normal:       for i in range(N): C[i] = A[i] + B[i]
#      Unrolled x4:  while i+3 < N:
#                      C[i]   = A[i]   + B[i]
#                      C[i+1] = A[i+1] + B[i+1]
#                      C[i+2] = A[i+2] + B[i+2]
#                      C[i+3] = A[i+3] + B[i+3]
#                      i += 4
#                    # remainder loop for leftover elements
#
#   Normal loop (1 element/iter):     Unrolled x4 (4 elements/iter):
#
#   ┌─────────────────────┐       ┌──────────────────────────────┐
#   │ iter 0: C[0]=A[0]+B[0]│       │ iter 0:                        │
#   │ i++ ; cmp ; branch   │       │   C[0]=A[0]+B[0]               │
#   │ iter 1: C[1]=A[1]+B[1]│       │   C[1]=A[1]+B[1]  ← no branch │
#   │ i++ ; cmp ; branch   │       │   C[2]=A[2]+B[2]  ← no branch │
#   │ iter 2: C[2]=A[2]+B[2]│       │   C[3]=A[3]+B[3]  ← no branch │
#   │ i++ ; cmp ; branch   │       │ i+=4 ; cmp ; branch            │
#   │ iter 3: C[3]=A[3]+B[3]│       └──────────────────────────────┘
#   │ i++ ; cmp ; branch   │       4 ops + 1 branch = 5 instructions
#   └─────────────────────┘       (75% fewer branches)
#   4 ops + 4 branches = 8 instr
#
# 2. Benefits:
#    - 75% fewer branches (1 branch per 4 iterations instead of 1 per 1)
#    - Better instruction-level parallelism (CPU can execute multiple
#      independent adds simultaneously in its pipeline)
#    - Enables vectorization (SIMD — see vectorization.py)
#
# 3. Tradeoff: unrolling increases code size. Too much unrolling can
#    hurt instruction cache performance. Typical unroll factors: 4–8.
#
# Note: In Python, unrolling has limited effect due to interpreter overhead.
# In compiled languages (C, LLVM), the speedup is dramatic.
# ═══════════════════════════════════════════════════════════════════════════

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
