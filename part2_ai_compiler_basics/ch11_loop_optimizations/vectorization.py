#!/usr/bin/env python3
"""
Chapter 11 — Vectorization: SIMD-style processing.

Demonstrates SIMD (Single Instruction, Multiple Data) concepts
using NumPy as a proxy for actual SIMD instructions.

Usage:
    python vectorization.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Vectorization (SIMD — Single Instruction, Multiple Data)
#
# Historical context: SIMD instructions first appeared in the Cray-1 (1976)
# with vector registers. Intel introduced MMX (1997), SSE (1999), and AVX
# (2011). Modern CPUs have 256-bit (AVX2) or 512-bit (AVX-512) registers
# that can process 4–16 floats in a single instruction. AI accelerators
# (TPUs, GPUs) are essentially massive SIMD engines.
#
# Problem solved: A scalar loop processes one element per instruction.
# Modern CPUs have wide vector units sitting idle. Vectorization loads
# multiple elements into a wide register and operates on all of them
# with a single instruction, giving 4–16× throughput.
#
# How it works on real hardware:
# 1. Load 4 doubles from A[i..i+3] into a 256-bit AVX register.
# 2. Load 4 doubles from B[i..i+3] into another register.
# 3. Execute vmulpd (vector multiply packed double) — 1 instruction
#    multiplies all 4 pairs simultaneously.
# 4. Execute vaddpd to accumulate the partial sums.
# 5. Repeat for the next 4 elements.
#
#   Scalar (1 element/cycle):        SIMD AVX2 (4 elements/cycle):
#
#   Register (64-bit):               Register (256-bit):
#   ┌─────────┐                        ┌──────┬──────┬──────┬──────┐
#   │  A[i]   │   ×                  │ A[i] │A[i+1]│A[i+2]│A[i+3]│
#   └─────────┘                        └──────┴──────┴──────┴──────┘
#   ┌─────────┐                            ×  vmulpd (1 instruction)
#   │  B[i]   │                        ┌──────┬──────┬──────┬──────┐
#   └─────────┘                        │ B[i] │B[i+1]│B[i+2]│B[i+3]│
#       │                              └──────┴──────┴──────┴──────┘
#       ▼                                  │
#   ┌─────────┐  1 result/cycle         ▼
#   │ A[i]·B[i]│                        ┌──────┬──────┬──────┬──────┐
#   └─────────┘                        │  p0  │  p1  │  p2  │  p3  │
#                                      └──────┴──────┴──────┴──────┘
#                                      4 results/cycle (4× throughput)
#
# In this file, we simulate SIMD by manually processing 4 elements
# per iteration in Python. NumPy's np.dot() uses real SIMD under the
# hood (via BLAS/OpenBLAS/MKL), so it's orders of magnitude faster.
#
# AI compiler integration: compilers like TVM annotate loop axes with
# "vectorize" to tell the code generator to use SIMD instructions.
# The unrolling (loop_unrolling.py) and vectorization optimizations
# work together: unrolling exposes independent operations, and
# vectorization packs them into SIMD instructions.
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def dot_product_scalar(A, B, N):
    """Scalar dot product: one element at a time."""
    result = 0.0
    for i in range(N):
        result += A[i] * B[i]
    return result


def dot_product_simd_4(A, B, N):
    """Simulated SIMD: process 4 elements at a time."""
    result = 0.0
    i = 0
    while i + 3 < N:
        # Simulated SIMD: 4 multiplies + 4 adds in parallel
        r0 = A[i]   * B[i]
        r1 = A[i+1] * B[i+1]
        r2 = A[i+2] * B[i+2]
        r3 = A[i+3] * B[i+3]
        result += r0 + r1 + r2 + r3
        i += 4
    while i < N:
        result += A[i] * B[i]
        i += 1
    return result


def dot_product_numpy(A, B):
    """NumPy: uses actual SIMD under the hood."""
    return np.dot(A, B)


def demo():
    console.print("\n[bold]═══ Vectorization (SIMD) ═══[/]\n")

    N = 10000
    A = np.random.randn(N)
    B = np.random.randn(N)

    # Verify correctness
    ref = dot_product_numpy(A, B)
    scalar = dot_product_scalar(A, B, N)
    simd4 = dot_product_simd_4(A, B, N)
    assert abs(scalar - ref) < 1e-6, f"Scalar mismatch: {scalar} vs {ref}"
    assert abs(simd4 - ref) < 1e-6, f"SIMD4 mismatch: {simd4} vs {ref}"

    # Benchmark
    results = {}
    t0 = time.perf_counter()
    for _ in range(10):
        dot_product_scalar(A, B, N)
    results["Scalar"] = (time.perf_counter() - t0) / 10

    t0 = time.perf_counter()
    for _ in range(10):
        dot_product_simd_4(A, B, N)
    results["SIMD x4 (simulated)"] = (time.perf_counter() - t0) / 10

    t0 = time.perf_counter()
    for _ in range(1000):
        dot_product_numpy(A, B)
    results["NumPy (real SIMD)"] = (time.perf_counter() - t0) / 1000

    table = Table(title=f"Dot Product ({N} elements)", box=box.ROUNDED)
    table.add_column("Method")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Speedup", justify="right")

    baseline = results["Scalar"]
    for method, t in results.items():
        table.add_row(method, f"{t*1000:.3f}", f"{baseline/t:.1f}x")
    console.print(table)

    console.print("\n[bold cyan]How SIMD works on real hardware:[/]")
    console.print("""
    CPU register (256-bit AVX):
    ┌────────┬────────┬────────┬────────┐
    │ A[0]   │ A[1]   │ A[2]   │ A[3]   │   ← 4 doubles loaded at once
    └────────┴────────┴────────┴────────┘
    ┌────────┬────────┬────────┬────────┐
    │ B[0]   │ B[1]   │ B[2]   │ B[3]   │   ← 4 doubles loaded at once
    └────────┴────────┴────────┴────────┘
    ─────── vmulpd (1 instruction) ──────
    ┌────────┬────────┬────────┬────────┐
    │ A[0]*  │ A[1]*  │ A[2]*  │ A[3]*  │   ← 4 multiplies in 1 cycle!
    │ B[0]   │ B[1]   │ B[2]   │ B[3]   │
    └────────┴────────┴────────┴────────┘
    """)
    console.print("[dim]AI compilers auto-vectorize tensor ops → massive speedups.[/]")


if __name__ == "__main__":
    demo()
