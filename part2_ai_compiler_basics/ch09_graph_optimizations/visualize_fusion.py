#!/usr/bin/env python3
"""
Chapter 9 — Visualize graph before and after fusion.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))

from visualization import render_dot
from operator_fusion import OptGraph, run_fusion_pipeline

OP_COLORS = {
    "Input": "#aed6f1", "Conv2D": "#f9e79f", "BatchNorm": "#fadbd8",
    "ReLU": "#d2b4de", "MatMul": "#fdebd0", "Add": "#a9dfbf",
    "Flatten": "#d5dbdb", "Const": "#d5f5e3",
}

def graph_to_dot(graph: OptGraph, title: str = "Graph") -> str:
    lines = [
        f'digraph "{title}" {{',
        '  rankdir=TB;',
        '  node [shape=box fontname="Courier" fontsize=10 style=filled];',
    ]
    for name in graph.order:
        node = graph.nodes[name]
        color = "#e8e8e8"
        for key in OP_COLORS:
            if key in node.op:
                color = OP_COLORS[key]
                break
        if node.fused_ops:
            color = "#82e0aa"
        label = f"{node.op}\\n{node.name}\\n{node.shape}"
        if node.fused_ops:
            label += f"\\n[{'+'.join(node.fused_ops)}]"
        label = label.replace('"', '\\"')
        lines.append(f'  "{name}" [label="{label}" fillcolor="{color}"];')
    for name in graph.order:
        for inp in graph.nodes[name].inputs:
            if inp in graph.nodes:
                lines.append(f'  "{inp}" -> "{name}";')
    lines.append("}")
    return "\n".join(lines)

def main():
    graph = OptGraph()
    graph.add("input", "Input", [], (1, 3, 224, 224))
    graph.add("conv1", "Conv2D", ["input"], (1, 64, 112, 112))
    graph.add("bn1", "BatchNorm", ["conv1"], (1, 64, 112, 112))
    graph.add("relu1", "ReLU", ["bn1"], (1, 64, 112, 112))
    graph.add("conv2", "Conv2D", ["relu1"], (1, 128, 56, 56))
    graph.add("bn2", "BatchNorm", ["conv2"], (1, 128, 56, 56))
    graph.add("relu2", "ReLU", ["bn2"], (1, 128, 56, 56))
    graph.add("flatten", "Flatten", ["relu2"], (1, 401408))
    graph.add("W_fc", "Input", [], (401408, 10))
    graph.add("fc", "MatMul", ["flatten", "W_fc"], (1, 10))
    graph.add("b_fc", "Input", [], (1, 10))
    graph.add("fc_add", "Add", ["fc", "b_fc"], (1, 10))

    before_dot = graph_to_dot(graph, "Before Fusion")
    render_dot(before_dot, "graph_before_fusion", output_dir="output")

    run_fusion_pipeline(graph)

    after_dot = graph_to_dot(graph, "After Fusion")
    render_dot(after_dot, "graph_after_fusion", output_dir="output")
    print("Done! Check output/ directory.")

if __name__ == "__main__":
    main()
