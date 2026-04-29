#!/usr/bin/env python3
"""
Chapter 12 — Visualize memory usage over time.

Usage:
    python visualize_memory.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from memory_planner import MemOp, liveness_analysis, greedy_buffer_sharing
from utils.visualization import plot_timeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def waterfall_chart(buffers, mapping, physical, ops, filename="memory_waterfall.png"):
    """Draw a waterfall chart of memory usage over time."""
    n_steps = len(ops)

    # Naive: stack all live buffers per step
    naive_usage = []
    for t in range(n_steps):
        usage = sum(b.size for b in buffers.values() if b.birth <= t <= b.death)
        naive_usage.append(usage / 1024 / 1024)

    # Optimized: stack physical buffers that are in use
    opt_usage = []
    for t in range(n_steps):
        live_logical = {b.name for b in buffers.values() if b.birth <= t <= b.death}
        live_physical = set()
        for logical in live_logical:
            live_physical.add(mapping[logical])
        usage = sum(pb["size"] for pb in physical if pb["name"] in live_physical)
        opt_usage.append(usage / 1024 / 1024)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_steps)
    labels = [op.name for op in ops]

    ax.bar(x - 0.2, naive_usage, 0.35, label="Naive", color="#e74c3c", alpha=0.8)
    ax.bar(x + 0.2, opt_usage, 0.35, label="Optimized", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage: Naive vs Optimized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    print(f"Saved → {filename}")
    plt.close(fig)


def lifetime_gantt(buffers, mapping, filename="buffer_lifetimes.png"):
    """Gantt chart showing buffer lifetimes colored by physical allocation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(buffers.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    phys_names = sorted(set(mapping.values()))
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(phys_names)}

    for i, name in enumerate(names):
        buf = buffers[name]
        phys = mapping[name]
        ax.barh(i, buf.death - buf.birth + 0.8, left=buf.birth - 0.4,
                height=0.7, color=color_map[phys], edgecolor="black", linewidth=0.5)
        ax.text(buf.birth + (buf.death - buf.birth) / 2, i,
                f"{phys}", ha="center", va="center", fontsize=7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Execution Step")
    ax.set_title("Buffer Lifetimes (colored by physical buffer)")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    print(f"Saved → {filename}")
    plt.close(fig)


def main():
    ops = [
        MemOp("Conv1",   [],            ["conv1_out"],  [64*112*112*4]),
        MemOp("BN1",     ["conv1_out"],  ["bn1_out"],    [64*112*112*4]),
        MemOp("ReLU1",   ["bn1_out"],    ["relu1_out"],  [64*112*112*4]),
        MemOp("Conv2",   ["relu1_out"],  ["conv2_out"],  [128*56*56*4]),
        MemOp("BN2",     ["conv2_out"],  ["bn2_out"],    [128*56*56*4]),
        MemOp("ReLU2",   ["bn2_out"],    ["relu2_out"],  [128*56*56*4]),
        MemOp("Pool",    ["relu2_out"],  ["pool_out"],   [128*28*28*4]),
        MemOp("Flatten", ["pool_out"],   ["flat_out"],   [128*28*28*4]),
        MemOp("FC",      ["flat_out"],   ["fc_out"],     [1000*4]),
    ]

    buffers = liveness_analysis(ops)
    mapping, reuses, physical = greedy_buffer_sharing(buffers)

    waterfall_chart(buffers, mapping, physical, ops)
    lifetime_gantt(buffers, mapping)
    print("\nDone! Check the generated PNG files.")


if __name__ == "__main__":
    main()
