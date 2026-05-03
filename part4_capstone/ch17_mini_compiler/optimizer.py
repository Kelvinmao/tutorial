#!/usr/bin/env python3
"""
Chapter 17 — Graph + tensor optimizations for the mini compiler.
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHMS: Graph Optimization Passes for the Mini AI Compiler
#
# This module implements the optimization passes from chapter 9 in a
# production-style pipeline. The three passes are:
#
# 1. MATMUL + ADD FUSION (fuse_matmul_add):
#    Pattern: MatMul(x, w) → Add(result, bias)
#    Result:  MatMul(x, w, bias=bias)  [single fused node]
#    How: Find Add nodes whose input is a MatMul. Store the bias in
#    the MatMul's attrs["bias"]. Redirect all consumers of Add to
#    point to MatMul. Delete the Add node.
#    Benefit: Eliminates one kernel launch and one memory round-trip.
#
# 2. RELU FUSION (fuse_relu):
#    Pattern: MatMul/Add → ReLU
#    Result:  MatMul/Add with attrs["activation"] = "relu"
#    How: Find ReLU nodes whose input is a MatMul or Add. Mark the
#    producer's activation attribute. Redirect consumers and delete ReLU.
#    Benefit: ReLU is applied in-place within the fused kernel.
#
# 3. DEAD NODE ELIMINATION (eliminate_dead_nodes):
#    Same algorithm as DCE in ch06 but for graph nodes:
#    - Build a "used" set from all node inputs.
#    - Add the final output node to "used".
#    - Remove any non-input node not in the used set.
#    Benefit: After fusion, orphaned nodes are cleaned up.
#
# Combined effect on a typical MLP:
#   Before: x → MatMul → Add → ReLU → MatMul → Add → Softmax (7 nodes)
#   After:  x → MatMul+Add+ReLU → MatMul+Add → Softmax (3 compute nodes)
#
#   Before optimization (11 nodes):      After optimization (8 nodes):
#
#   x ─► MatMul ─► Add ─► ReLU          x ─► MatMul+Add+ReLU
#   w1 ─┘    b1 ─┘                       w1 ─┘    b1 ─┘
#                    │                               │
#               MatMul ─► Add ─► Softmax        MatMul+Add ─► Softmax
#          w2 ─┘    b2 ─┘                  w2 ─┘    b2 ─┘
#
#   Fusions applied:
#   1. MatMul + Add → fused (bias stored in MatMul attrs)  ×2
#   2. MatMul+Add + ReLU → fused (activation="relu")       ×1
#   3. Dead node elimination (orphaned Add/ReLU removed)   ×3
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from model_ir import ModelGraph, IRNode, OpType
from rich.console import Console

console = Console()


def fuse_matmul_add(graph: ModelGraph) -> int:
    """Fuse MatMul + Add into a single fused op."""
    fused = 0
    remove = set()

    for name, node in list(graph.nodes.items()):
        if node.op == OpType.ADD:
            a, b = node.inputs
            a_node = graph.nodes.get(a)
            if a_node and a_node.op == OpType.MATMUL:
                # Fuse: mark matmul as having bias
                a_node.attrs["bias"] = b
                a_node.shape = node.shape
                # Redirect consumers of add to matmul
                for n2 in graph.nodes.values():
                    n2.inputs = [a if x == name else x for x in n2.inputs]
                remove.add(name)
                fused += 1

    for name in remove:
        del graph.nodes[name]
    return fused


def fuse_relu(graph: ModelGraph) -> int:
    """Fuse ReLU into the preceding op."""
    fused = 0
    remove = set()

    for name, node in list(graph.nodes.items()):
        if node.op == OpType.RELU and len(node.inputs) == 1:
            prev = graph.nodes.get(node.inputs[0])
            if prev and prev.op in (OpType.MATMUL, OpType.ADD):
                prev.attrs["activation"] = "relu"
                for n2 in graph.nodes.values():
                    n2.inputs = [prev.name if x == name else x for x in n2.inputs]
                remove.add(name)
                fused += 1

    for name in remove:
        del graph.nodes[name]
    return fused


def eliminate_dead_nodes(graph: ModelGraph) -> int:
    """Remove nodes whose outputs are not used by anyone."""
    used = set()
    for node in graph.nodes.values():
        for inp in node.inputs:
            used.add(inp)
        # Fused nodes may keep extra dependencies in attrs rather than inputs.
        # Keep those dependency nodes alive so codegen does not reference a
        # deleted tensor such as a MatMul+Add fused bias.
        bias = node.attrs.get("bias")
        if isinstance(bias, str):
            used.add(bias)

    # Find output node (last in topo order)
    topo = graph.topo_order()
    if topo:
        used.add(topo[-1].name)

    removed = 0
    for name in list(graph.nodes.keys()):
        if name not in used and graph.nodes[name].op != OpType.INPUT:
            del graph.nodes[name]
            removed += 1
    return removed


def optimize(graph: ModelGraph, verbose: bool = True) -> ModelGraph:
    """Run all optimization passes."""
    if verbose:
        console.print("[bold cyan]Running optimizations...[/]")

    n = fuse_matmul_add(graph)
    if verbose and n:
        console.print(f"  Fused {n} MatMul+Add pairs")

    n = fuse_relu(graph)
    if verbose and n:
        console.print(f"  Fused {n} ReLU activations")

    n = eliminate_dead_nodes(graph)
    if verbose and n:
        console.print(f"  Eliminated {n} dead nodes")

    return graph
