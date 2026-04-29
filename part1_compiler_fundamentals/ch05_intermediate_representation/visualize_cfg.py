#!/usr/bin/env python3
"""
Chapter 5 — Visualize the Control Flow Graph using graphviz.

Usage:
    python visualize_cfg.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))

from visualization import render_dot
from ir_builder import build_ir, Op
from cfg_builder import build_cfg


SAMPLE = """\
let x: int = 10
let y: int = 20
let z = x + y * 2

if z > 40:
    print(z)
else:
    print(x)
"""


def cfg_to_dot(cfg) -> str:
    lines = [
        'digraph CFG {',
        '  rankdir=TB;',
        '  node [shape=record fontname="Courier" fontsize=10 style=filled fillcolor="#e8f4fd"];',
    ]

    for label, block in cfg.blocks.items():
        # Build instruction list for the block label
        instrs = []
        for instr in block.instructions:
            s = str(instr).strip()
            # Escape special graphviz characters
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            s = s.replace("<", "\\<").replace(">", "\\>")
            s = s.replace("{", "\\{").replace("}", "\\}")
            s = s.replace("|", "\\|")
            instrs.append(s)

        instr_text = "\\l".join(instrs) + "\\l"
        safe_label = label.replace("<", "").replace(">", "")
        node_label = f"{{ {safe_label} | {instr_text} }}"
        lines.append(f'  "{label}" [label="{node_label}"];')

    # Edges
    for label, block in cfg.blocks.items():
        for succ in block.successors:
            lines.append(f'  "{label}" -> "{succ}";')

    lines.append("}")
    return "\n".join(lines)


def visualize_cfg(source: str, filename: str = "cfg"):
    instructions = build_ir(source)
    cfg = build_cfg(instructions)
    dot = cfg_to_dot(cfg)
    return render_dot(dot, filename=filename, output_dir="output")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    print("Building CFG visualization...")
    path = visualize_cfg(source)
    print(f"Done! Check: {path}")
