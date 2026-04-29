#!/usr/bin/env python3
"""
Chapter 12 — Memory planner: liveness analysis + buffer reuse.

Usage:
    python memory_planner.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


@dataclass
class MemBuffer:
    name: str
    size: int        # size in bytes
    birth: int       # time step when allocated
    death: int       # time step of last use
    alias: str = ""  # physical buffer this maps to (after planning)


@dataclass
class MemOp:
    """An operation in the execution sequence."""
    name: str
    inputs: list[str]
    outputs: list[str]
    output_sizes: list[int]


def liveness_analysis(ops: list[MemOp]) -> dict[str, MemBuffer]:
    """
    Determine the lifetime (birth, death) of each buffer.

    A buffer is "born" when the op that produces it executes,
    and "dies" at the last time step any op reads it.
    """
    buffers: dict[str, MemBuffer] = {}

    for t, op in enumerate(ops):
        # Birth: outputs are born at this step
        for out, size in zip(op.outputs, op.output_sizes):
            buffers[out] = MemBuffer(name=out, size=size, birth=t, death=t)
        # Death: inputs are last used at this step
        for inp in op.inputs:
            if inp in buffers:
                buffers[inp].death = max(buffers[inp].death, t)

    return buffers


def greedy_buffer_sharing(buffers: dict[str, MemBuffer]) -> dict[str, str]:
    """
    Assign physical buffers using a greedy algorithm.

    When a buffer's lifetime ends, its physical memory can be reused
    by a later buffer of the same or smaller size.

    Returns mapping: logical buffer name → physical buffer name.
    """
    sorted_bufs = sorted(buffers.values(), key=lambda b: b.birth)

    # Physical buffers: list of (name, size, available_after)
    physical: list[dict] = []
    mapping: dict[str, str] = {}
    reuses = 0

    for buf in sorted_bufs:
        # Try to find a reusable physical buffer
        best = None
        for pb in physical:
            if pb["available_after"] < buf.birth and pb["size"] >= buf.size:
                if best is None or pb["size"] < best["size"]:
                    best = pb  # prefer smallest fitting buffer

        if best is not None:
            mapping[buf.name] = best["name"]
            best["available_after"] = buf.death
            reuses += 1
        else:
            # Allocate new physical buffer
            pb_name = f"phys_{len(physical)}"
            physical.append({
                "name": pb_name,
                "size": buf.size,
                "available_after": buf.death,
            })
            mapping[buf.name] = pb_name

    return mapping, reuses, physical


def demo():
    console.print("\n[bold]═══ Memory Planning ═══[/]\n")

    # Simulate a simple CNN inference pipeline
    ops = [
        MemOp("Conv1",    [],         ["conv1_out"],  [64*112*112*4]),
        MemOp("BN1",      ["conv1_out"], ["bn1_out"],   [64*112*112*4]),
        MemOp("ReLU1",    ["bn1_out"],   ["relu1_out"], [64*112*112*4]),
        MemOp("Conv2",    ["relu1_out"], ["conv2_out"], [128*56*56*4]),
        MemOp("BN2",      ["conv2_out"], ["bn2_out"],   [128*56*56*4]),
        MemOp("ReLU2",    ["bn2_out"],   ["relu2_out"], [128*56*56*4]),
        MemOp("Pool",     ["relu2_out"], ["pool_out"],  [128*28*28*4]),
        MemOp("Flatten",  ["pool_out"],  ["flat_out"],  [128*28*28*4]),
        MemOp("FC",       ["flat_out"],  ["fc_out"],    [1000*4]),
    ]

    # Step 1: Liveness analysis
    buffers = liveness_analysis(ops)

    console.print("[bold cyan]Step 1: Liveness Analysis[/]")
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Buffer")
    table.add_column("Size (KB)", justify="right")
    table.add_column("Birth (step)")
    table.add_column("Death (step)")
    table.add_column("Lifetime")

    for name, buf in buffers.items():
        size_kb = buf.size / 1024
        lifetime = buf.death - buf.birth
        bar = "█" * (lifetime + 1)
        table.add_row(name, f"{size_kb:.0f}", str(buf.birth),
                     str(buf.death), bar)
    console.print(table)

    # Step 2: Buffer sharing
    console.print("\n[bold cyan]Step 2: Buffer Sharing[/]")
    mapping, reuses, physical = greedy_buffer_sharing(buffers)

    table2 = Table(title="Buffer Mapping", box=box.ROUNDED)
    table2.add_column("Logical Buffer")
    table2.add_column("→ Physical Buffer", style="green")
    for logical, physical_name in mapping.items():
        table2.add_row(logical, physical_name)
    console.print(table2)

    # Statistics
    total_naive = sum(b.size for b in buffers.values())
    total_optimized = sum(pb["size"] for pb in physical)

    console.print(f"\n[bold]Memory Statistics:[/]")
    console.print(f"  Logical buffers:  {len(buffers)}")
    console.print(f"  Physical buffers: {len(physical)} ({reuses} reuses)")
    console.print(f"  Naive memory:     {total_naive/1024/1024:.1f} MB")
    console.print(f"  Optimized memory: {total_optimized/1024/1024:.1f} MB")
    console.print(f"  Saved:            {(total_naive-total_optimized)/1024/1024:.1f} MB "
                  f"({(1-total_optimized/total_naive)*100:.0f}% reduction)")


if __name__ == "__main__":
    demo()
