#!/usr/bin/env python3
"""
Chapter 13 — Simulate GPU thread/block mapping for matrix multiply.

We don't need a real GPU — this shows how loop iterations map to threads.

Usage:
    python gpu_mapping.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: GPU Thread/Block Mapping (CUDA Execution Model Simulation)
#
# Historical context: NVIDIA's CUDA (2007) introduced a hierarchical
# parallelism model: Grid → Blocks → Threads. Each thread computes one
# or a few output elements. AI compilers (TVM, Triton, XLA) must map
# loop iterations to this hierarchy. Understanding this mapping is
# essential for writing efficient GPU kernels.
#
# Problem solved: A CPU loop "for i, for j" is sequential. On a GPU,
# we need to assign each (i,j) output element to a specific thread in
# a specific block. The mapping determines memory access patterns,
# occupancy, and ultimately performance.
#
# How the CUDA execution model works:
# 1. GRID: The entire computation is divided into a 2D grid of blocks.
#    Grid dimensions: (ceil(M/block_y), ceil(N/block_x))
#
# 2. BLOCK: Each block is a 2D array of threads.
#    Block dimensions: (block_y, block_x) — typically 16×16 or 32×32.
#    All threads in a block share L1/shared memory and can synchronize.
#
# 3. THREAD: Each thread computes one element of the output matrix:
#    row = blockIdx.y * blockDim.y + threadIdx.y
#    col = blockIdx.x * blockDim.x + threadIdx.x
#    C[row, col] = sum_k(A[row, k] * B[k, col])
#
#   Output matrix C (8×8) with block_size = 4×4:
#
#   ┌───────────┬───────────┐
#   │ Block(0,0)│ Block(0,1)│    Grid: 2×2 blocks
#   │ 4×4 threads│ 4×4 threads│
#   │ C[0..3,   │ C[0..3,   │    Each block has 16 threads
#   │   0..3]   │   4..7]   │    Total: 64 threads for 64 elements
#   ├───────────┼───────────┤
#   │ Block(1,0)│ Block(1,1)│    Thread (1,2) in Block(0,0):
#   │ C[4..7,   │ C[4..7,   │      row = 0*4 + 1 = 1
#   │   0..3]   │   4..7]   │      col = 0*4 + 2 = 2
#   └───────────┴───────────┘      → computes C[1,2]
#
# This simulation maps the CPU loops to (block, thread) indices and
# produces heat maps showing which block/thread handles each output
# element. Real GPU kernels use shared memory tiling to reduce global
# memory access (combining the tiling concept from ch11 with the
# GPU mapping here).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.visualization import plot_heatmap

console = Console()


def simulate_gpu_matmul(A: np.ndarray, B: np.ndarray,
                        block_x: int = 4, block_y: int = 4):
    """
    Simulate GPU-style thread mapping for matrix multiply.

    Each thread computes one element C[row, col].
    Threads are organized into 2D blocks.
    """
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A is {A.shape}, B is {B.shape}")

    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y

    console.print(f"[bold]Grid dimensions:[/]  ({grid_y}, {grid_x}) blocks")
    console.print(f"[bold]Block dimensions:[/] ({block_y}, {block_x}) threads")
    console.print(f"[bold]Total threads:[/]    {grid_x * grid_y * block_x * block_y}")

    # Track which block handles which output element
    block_map = np.zeros((M, N), dtype=int)
    thread_map = np.zeros((M, N), dtype=int)

    C = np.zeros((M, N), dtype=np.float32)

    for by in range(grid_y):
        for bx in range(grid_x):
            for ty in range(block_y):
                for tx in range(block_x):
                    row = by * block_y + ty
                    col = bx * block_x + tx
                    if row < M and col < N:
                        # Each thread does the full dot product
                        val = 0.0
                        for k in range(K):
                            val += A[row, k] * B[k, col]
                        C[row, col] = val
                        block_map[row, col] = by * grid_x + bx
                        thread_map[row, col] = ty * block_x + tx

    return C, block_map, thread_map


def demo():
    console.print("\n[bold]═══ GPU Thread/Block Mapping ═══[/]\n")

    M, N, K = 16, 16, 8
    block_x, block_y = 4, 4
    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)

    C, block_map, thread_map = simulate_gpu_matmul(A, B, block_x, block_y)

    # Verify against NumPy using the same inputs as the simulated mapping.
    max_error = np.max(np.abs(C - (A @ B)))
    console.print(f"\n[bold]Output C shape:[/] {C.shape}")
    console.print(f"[bold]Max error vs NumPy:[/] {max_error:.6f}")

    # Visualize block assignments
    console.print("\n[bold cyan]Block Assignment Map[/]")
    console.print("Each cell shows which block computes that output element:")

    table = Table(box=box.SIMPLE, show_lines=False, padding=0)
    table.add_column("", style="dim")
    for j in range(min(N, 16)):
        table.add_column(str(j), justify="center", width=3)

    for i in range(min(M, 16)):
        row = [str(i)]
        for j in range(min(N, 16)):
            row.append(str(block_map[i, j]))
        table.add_row(*row)
    console.print(table)

    # Save visualization
    plot_heatmap(block_map.astype(float),
                title="Block ID per Output Element",
                xlabel="Column", ylabel="Row",
                filename="gpu_block_map.png")

    plot_heatmap(thread_map.astype(float),
                title="Thread ID within Block",
                xlabel="Column", ylabel="Row",
                filename="gpu_thread_map.png")

    console.print("\n[green]Saved gpu_block_map.png and gpu_thread_map.png[/]")


if __name__ == "__main__":
    demo()
