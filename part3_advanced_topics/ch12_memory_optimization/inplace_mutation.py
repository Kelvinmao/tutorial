#!/usr/bin/env python3
"""
Chapter 12 — In-place mutation: detect ops that can write output
into an input buffer, saving an allocation.

Usage:
    python inplace_mutation.py
"""

from __future__ import annotations
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Ops that can safely overwrite their input
INPLACE_SAFE_OPS = {"relu", "sigmoid", "tanh", "add_scalar", "mul_scalar",
                    "batch_norm", "dropout", "leaky_relu"}


@dataclass
class GraphOp:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    output_sizes: list[int]


def find_inplace_candidates(ops: list[GraphOp]) -> list[tuple[str, str, str]]:
    """
    Detect ops that can write their output into an input buffer.

    Conditions for in-place:
    1. Op type is in INPLACE_SAFE_OPS (element-wise, same shape)
    2. The input buffer is not used by any later op (no alias conflict)

    Returns list of (op_name, input_to_overwrite, output_that_saves_alloc).
    """
    # Build a map: buffer → list of consumer op indices
    consumers: dict[str, list[int]] = {}
    for i, op in enumerate(ops):
        for inp in op.inputs:
            consumers.setdefault(inp, []).append(i)

    candidates = []
    for i, op in enumerate(ops):
        if op.op_type not in INPLACE_SAFE_OPS:
            continue
        if not op.inputs:
            continue

        # Check first input — the one we'd overwrite
        inp = op.inputs[0]
        # Safe if no consumer after this op reads the same buffer
        last_use = max(consumers.get(inp, [i]))
        if last_use <= i:
            candidates.append((op.name, inp, op.outputs[0]))

    return candidates


def apply_inplace(ops: list[GraphOp],
                  candidates: list[tuple[str, str, str]]) -> dict[str, str]:
    """
    Apply in-place mutations: remap output buffers to input buffers.
    Returns mapping of replaced output → reused input.
    """
    alias_map: dict[str, str] = {}
    for op_name, inp, out in candidates:
        alias_map[out] = inp
    return alias_map


def demo():
    console.print("\n[bold]═══ In-Place Mutation Detection ═══[/]\n")

    ops = [
        GraphOp("conv1",  "conv2d",     [],            ["t1"], [64*56*56*4]),
        GraphOp("bn1",    "batch_norm", ["t1"],        ["t2"], [64*56*56*4]),
        GraphOp("relu1",  "relu",       ["t2"],        ["t3"], [64*56*56*4]),
        GraphOp("conv2",  "conv2d",     ["t3"],        ["t4"], [128*28*28*4]),
        GraphOp("bn2",    "batch_norm", ["t4"],        ["t5"], [128*28*28*4]),
        GraphOp("relu2",  "relu",       ["t5"],        ["t6"], [128*28*28*4]),
        GraphOp("conv3",  "conv2d",     ["t6"],        ["t7"], [256*14*14*4]),
        GraphOp("relu3",  "relu",       ["t7"],        ["t8"], [256*14*14*4]),
    ]

    candidates = find_inplace_candidates(ops)
    alias_map = apply_inplace(ops, candidates)

    table = Table(title="In-Place Candidates", box=box.ROUNDED, show_lines=True)
    table.add_column("Op")
    table.add_column("Type")
    table.add_column("Input → Output")
    table.add_column("In-Place?", justify="center")

    for op in ops:
        is_inplace = op.outputs[0] in alias_map
        inp_out = f"{op.inputs[0] if op.inputs else '—'} → {op.outputs[0]}"
        style = "green" if is_inplace else ""
        marker = "✓ reuse" if is_inplace else ""
        table.add_row(op.name, op.op_type, inp_out, marker, style=style)

    console.print(table)

    saved = sum(
        next(op.output_sizes[0] for op in ops if op.outputs[0] == out)
        for out in alias_map
    )
    console.print(f"\n[bold]Savings:[/] {len(alias_map)} allocations eliminated "
                  f"({saved/1024/1024:.1f} MB saved)")


if __name__ == "__main__":
    demo()
