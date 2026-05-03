#!/usr/bin/env python3
"""
Chapter 9 — Graph-level constant folding.

Evaluate subgraphs with all-constant inputs at compile time.

Usage:
    python constant_folding_graph.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Graph-Level Constant Folding (Partial Evaluation)
#
# Historical context: This extends scalar constant folding (ch06) to the
# graph/tensor level. ONNX Runtime, TensorRT, and XLA all perform this.
# The key insight: many tensors in a neural network graph are constants
# (weights, batch-norm parameters). If all inputs to a node are constant,
# the node's output is also a constant that can be pre-computed.
#
# Problem solved: BatchNorm parameters (gamma, beta, mean, variance)
# are fixed after training. The formula:
#   output = gamma * (input - mean) / sqrt(var + eps) + beta
# contains subexpressions like gamma/sqrt(var+eps) that depend only on
# constants. These can be folded into a single scale+offset at compile
# time, reducing runtime computation.
#
# How it works:
# 1. Mark all nodes with no inputs (Const, Input) as "constant" or not.
# 2. Iterate: for each node, if ALL inputs are marked constant, mark
#    this node as constant too (and annotate it as "folded").
# 3. Repeat until no new nodes can be folded (fixpoint).
# 4. In a real compiler, folded nodes would be evaluated and replaced
#    with a single Const node holding the pre-computed result.
#
#   BatchNorm graph:               After constant folding:
#
#   gamma (Const) ┐                 scale (Const) ┐
#   var   (Const) ┤─► Div ┐          (= gamma /     ├─► Mul ┐
#   eps   (Const) ┘      ├─ Mul       sqrt(var+eps))  │       │
#   input (░░░░░) ──────┘    │     input ──────────┘       ├─ output
#   mean  (Const) ┐            ├─► Add  shift (Const) ────────┘
#   beta  (Const) ┘            │     (= beta - mean*scale)
#              Sub ─► offset ──┘
#
#   Before: 6 ops at runtime      After: 2 ops at runtime
#   (Sub, Div, Mul, Add, ...)      (Mul, Add) + pre-computed consts
#
# This is the tensor analog of the scalar constant folding in ch06.
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

from operator_fusion import OptGraph


def constant_fold(graph: OptGraph) -> int:
    """
    Fold constant subgraphs: if all inputs to a node are constants,
    replace the node with a constant.
    """
    constants: set[str] = set()
    folded = 0

    # Mark initial constants
    for name in graph.order:
        node = graph.nodes[name]
        if node.op in ("Const", "Input") and not node.inputs:
            constants.add(name)

    # Iteratively fold
    changed = True
    while changed:
        changed = False
        for name in list(graph.order):
            if name not in graph.nodes or name in constants:
                continue
            node = graph.nodes[name]
            if all(inp in constants for inp in node.inputs) and node.inputs:
                # All inputs are constant — fold this node
                node.op = f"Const({node.op})"
                node.fused_ops.append("folded")
                constants.add(name)
                folded += 1
                changed = True

    console.print(f"  [green]Graph constant folding: {folded} nodes folded[/]")
    return folded


def demo():
    console.print("\n[bold]═══ Graph-Level Constant Folding ═══[/]\n")

    graph = OptGraph()
    # BatchNorm parameters are often constant
    graph.add("input", "Input", [], (1, 64, 56, 56))
    graph.add("gamma", "Const", [], (64,))
    graph.add("beta", "Const", [], (64,))
    graph.add("mean", "Const", [], (64,))
    graph.add("var", "Const", [], (64,))
    # BN formula: gamma * (x - mean) / sqrt(var + eps) + beta
    # The (gamma / sqrt(var + eps)) and (beta - gamma*mean/sqrt(var+eps))
    # can be precomputed!
    graph.add("eps", "Const", [], (1,))
    graph.add("var_eps", "Add", ["var", "eps"], (64,))
    graph.add("sqrt_var", "Sqrt", ["var_eps"], (64,))
    graph.add("scale", "Div", ["gamma", "sqrt_var"], (64,))
    graph.add("mean_scaled", "Mul", ["mean", "scale"], (64,))
    graph.add("offset", "Sub", ["beta", "mean_scaled"], (64,))
    # Now apply to input
    graph.add("scaled_input", "Mul", ["input", "scale"], (1, 64, 56, 56))
    graph.add("output", "Add", ["scaled_input", "offset"], (1, 64, 56, 56))

    before = graph.clone()
    before.print_graph("Before Constant Folding")

    constant_fold(graph)
    console.print()
    graph.print_graph("After Constant Folding")

    console.print("\n[dim]Note: ops like var+eps, gamma/sqrt(var+eps) are all computed")
    console.print("from constants, so they can be evaluated at compile time![/]")


if __name__ == "__main__":
    demo()
