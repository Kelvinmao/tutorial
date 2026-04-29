#!/usr/bin/env python3
"""
Chapter 4 — Visualize the symbol table as a scope tree using graphviz.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))

from visualization import render_dot
from parser import parse_source
from type_checker import TypeChecker

SAMPLE = """\
let x: int = 10

def add(a: int, b: int) -> int:
    let result = a + b
    return result

def main() -> int:
    let y = add(x, 20)
    if y > 25:
        let z = y * 2
        print(z)
    return y
"""

def build_scope_dot(source: str) -> str:
    ast = parse_source(source)
    checker = TypeChecker()
    sym_table, errors = checker.check(ast)

    lines = [
        'digraph scopes {',
        '  rankdir=TB;',
        '  node [shape=record fontname="Courier" fontsize=10];',
        '  compound=true;',
    ]

    # Group symbols by scope
    scope_map: dict[str, list] = {}
    for sym in sym_table.all_symbols:
        key = f"{sym.scope_name}_{sym.scope_level}"
        scope_map.setdefault(key, []).append(sym)

    # Create subgraph for each scope
    for i, (scope_key, symbols) in enumerate(scope_map.items()):
        scope_name = symbols[0].scope_name
        level = symbols[0].scope_level
        color = ["#d4e6f1", "#d5f5e3", "#fdebd0", "#e8daef", "#fadbd8"][level % 5]

        sym_rows = "|".join(f"{s.name}: {s.sym_type}" for s in symbols)
        label = f"{{ {scope_name} (level {level}) | {sym_rows} }}"
        label = label.replace('"', '\\"')
        lines.append(f'  scope_{i} [label="{label}" style=filled fillcolor="{color}"];')

    # Add edges: global → function scopes, function → block scopes
    scope_keys = list(scope_map.keys())
    for i, key in enumerate(scope_keys):
        level = scope_map[key][0].scope_level
        if level > 0:
            # Find parent scope
            for j in range(i - 1, -1, -1):
                parent_level = scope_map[scope_keys[j]][0].scope_level
                if parent_level == level - 1:
                    lines.append(f"  scope_{j} -> scope_{i};")
                    break

    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    print("Building scope tree visualization...")
    dot = build_scope_dot(source)
    render_dot(dot, filename="symbol_table", output_dir="output")
