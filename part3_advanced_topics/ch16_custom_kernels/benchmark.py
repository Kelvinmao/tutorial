#!/usr/bin/env python3
"""
Chapter 16 — Compile and benchmark C matrix multiply kernels.

Usage:
    python benchmark.py
"""

from __future__ import annotations
import subprocess
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rich.console import Console
from rich.table import Table
from rich import box
from utils.visualization import plot_bar_chart

console = Console()

KERNELS = [
    ("matmul_naive", "matmul_naive.c"),
    ("matmul_tiled", "matmul_tiled.c"),
]

SIZES = [128, 256, 512]


def compile_kernel(name: str, src: str) -> str | None:
    exe = f"./{name}"
    result = subprocess.run(
        ["gcc", "-O2", "-o", exe, src, "-lm"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        console.print(f"[red]Failed to compile {src}:[/] {result.stderr}")
        return None
    return exe


def run_kernel(exe: str, size: int) -> tuple[float, float] | None:
    result = subprocess.run(
        [exe, str(size)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None
    # Output format: name,size,time,gflops
    parts = result.stdout.strip().split(",")
    return float(parts[2]), float(parts[3])


def demo():
    console.print("\n[bold]═══ Kernel Benchmark ═══[/]\n")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Compile
    executables = {}
    for name, src in KERNELS:
        exe = compile_kernel(name, src)
        if exe:
            executables[name] = exe
            console.print(f"[green]✓[/] Compiled {name}")

    if not executables:
        console.print("[red]No kernels compiled successfully.[/]")
        return

    # Benchmark
    results = {}  # (kernel, size) → (time, gflops)
    for size in SIZES:
        console.print(f"\n[yellow]Benchmarking size {size}×{size}...[/]")
        for name, exe in executables.items():
            out = run_kernel(exe, size)
            if out:
                time_s, gflops = out
                results[(name, size)] = (time_s, gflops)
                console.print(f"  {name}: {time_s:.4f}s ({gflops:.2f} GFLOPS)")

    # Summary table
    console.print()
    table = Table(title="Results", box=box.ROUNDED)
    table.add_column("Size")
    for name, _ in KERNELS:
        if name in executables:
            table.add_column(f"{name}\n(GFLOPS)", justify="right")

    for size in SIZES:
        row = [str(size)]
        for name, _ in KERNELS:
            if name in executables:
                key = (name, size)
                if key in results:
                    row.append(f"{results[key][1]:.2f}")
                else:
                    row.append("—")
        table.add_row(*row)
    console.print(table)

    # Visualization
    if results:
        labels = []
        values = []
        for size in SIZES:
            for name in executables:
                key = (name, size)
                if key in results:
                    labels.append(f"{name}\n{size}")
                    values.append(results[key][1])
        plot_bar_chart(labels, values,
                      title="MatMul Performance (GFLOPS)",
                      ylabel="GFLOPS",
                      filename="benchmark_results.png")

    # Cleanup executables
    for exe in executables.values():
        if os.path.exists(exe):
            os.remove(exe)


if __name__ == "__main__":
    demo()
