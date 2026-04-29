#!/usr/bin/env python3
"""
Chapter 17 — Graph + tensor optimizations for the mini compiler.
"""

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
