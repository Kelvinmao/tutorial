#!/usr/bin/env python3
"""
Chapter 13 — Simulate GPU thread/block mapping for matrix multiply.

We don't need a real GPU — this shows how loop iterations map to threads.

Usage:
    python gpu_mapping.py
"""

from __future__ import annotations
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.visualization import plot_heatmap

console = Console()


def simulate_gpu_matmul(M: int, N: int, K: int,
                        block_x: int = 4, block_y: int = 4):
    """
    Simulate GPU-style thread mapping for matrix multiply.

    Each thread computes one element C[row, col].
    Threads are organized into 2D blocks.
    """
    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y

    console.print(f"[bold]Grid dimensions:[/]  ({grid_y}, {grid_x}) blocks")
    console.print(f"[bold]Block dimensions:[/] ({block_y}, {block_x}) threads")
    console.print(f"[bold]Total threads:[/]    {grid_x * grid_y * block_x * block_y}")

    # Track which block handles which output element
    block_map = np.zeros((M, N), dtype=int)
    thread_map = np.zeros((M, N), dtype=int)

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
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

    C, block_map, thread_map = simulate_gpu_matmul(
        M, N, K, block_x, block_y
    )

    # Verify against numpy
    A = np.random.rand(M, K).astype(np.float32)
    console.print(f"\n[bold]Output C shape:[/] {C.shape}")

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
