#!/usr/bin/env python3
"""
Chapter 8 — Visualize computation graph with tensor shapes on edges.

Usage:
    python visualize_graph.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.dirname(__file__))

from visualization import render_dot
from comp_graph import CompGraph

OP_COLORS = {
    "Input": "#aed6f1",
    "Const": "#d5f5e3",
    "MatMul": "#f9e79f",
    "Add": "#fadbd8",
    "ReLU": "#d2b4de",
    "MSELoss": "#f5b7b1",
    "Mul": "#fdebd0",
    "Sum": "#a9dfbf",
}


def graph_to_dot(graph: CompGraph) -> str:
    lines = [
        'digraph CompGraph {',
        '  rankdir=TB;',
        '  node [shape=box fontname="Courier" fontsize=11 style=filled];',
        '  edge [fontname="Courier" fontsize=9];',
    ]

    topo = graph.topological_order()

    for node in topo:
        nid = f"n{node.node_id}"
        name = node.output.name if node.output and node.output.name else ""
        label = node.op
        if name:
            label += f"\\n'{name}'"
        if node.output:
            label += f"\\n{node.output.shape}"

        color = OP_COLORS.get(node.op, "#e8e8e8")
        lines.append(f'  {nid} [label="{label}" fillcolor="{color}"];')

    for node in topo:
        nid = f"n{node.node_id}"
        for inp in node.inputs:
            if inp.node:
                src_id = f"n{inp.node.node_id}"
                edge_label = str(inp.shape)
                lines.append(f'  {src_id} -> {nid} [label="{edge_label}"];')

    lines.append("}")
    return "\n".join(lines)


def visualize_computation_graph():
    graph = CompGraph()

    np.random.seed(42)
    X = graph.input(np.random.randn(2, 3), name="X")
    W1 = graph.input(np.random.randn(3, 4) * 0.1, name="W1")
    b1 = graph.input(np.zeros((1, 4)), name="b1")
    h = graph.matmul(X, W1)
    h = graph.add(h, b1)
    h = graph.relu(h)
    W2 = graph.input(np.random.randn(4, 2) * 0.1, name="W2")
    b2 = graph.input(np.zeros((1, 2)), name="b2")
    out = graph.matmul(h, W2)
    out = graph.add(out, b2)
    target = graph.input(np.array([[1, 0], [0, 1]], dtype=float), name="target")
    loss = graph.mse_loss(out, target)

    dot = graph_to_dot(graph)
    path = render_dot(dot, filename="computation_graph", output_dir="output")
    print(f"Done! Check: {path}")


if __name__ == "__main__":
    visualize_computation_graph()
