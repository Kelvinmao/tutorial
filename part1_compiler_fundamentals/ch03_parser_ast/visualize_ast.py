#!/usr/bin/env python3
"""
Chapter 3 — Visualize the AST as a graphviz tree.

Usage:
    python visualize_ast.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))

from visualization import render_dot
from parser import parse_source
from ast_nodes import *

# Color scheme for different node types
NODE_STYLES = {
    "Program": 'fillcolor="#d4e6f1" style=filled',
    "LetDecl": 'fillcolor="#d5f5e3" style=filled',
    "AssignStmt": 'fillcolor="#d5f5e3" style=filled',
    "IfStmt": 'fillcolor="#fdebd0" style=filled',
    "WhileStmt": 'fillcolor="#fdebd0" style=filled',
    "PrintStmt": 'fillcolor="#e8daef" style=filled',
    "FuncDef": 'fillcolor="#fadbd8" style=filled',
    "ReturnStmt": 'fillcolor="#fadbd8" style=filled',
    "BinOp": 'fillcolor="#fcf3cf" style=filled shape=circle',
    "UnaryOp": 'fillcolor="#fcf3cf" style=filled shape=circle',
    "Comparison": 'fillcolor="#fcf3cf" style=filled shape=diamond',
    "IntLit": 'fillcolor="#abebc6" style=filled shape=ellipse',
    "FloatLit": 'fillcolor="#abebc6" style=filled shape=ellipse',
    "BoolLit": 'fillcolor="#abebc6" style=filled shape=ellipse',
    "StringLit": 'fillcolor="#abebc6" style=filled shape=ellipse',
    "Identifier": 'fillcolor="#aed6f1" style=filled shape=ellipse',
    "FuncCall": 'fillcolor="#f9e79f" style=filled shape=hexagon',
}


def ast_to_dot(node, parent_id=None, counter=None, lines=None):
    """Recursively convert an AST node to DOT language lines."""
    if counter is None:
        counter = [0]
    if lines is None:
        lines = []

    nid = f"n{counter[0]}"
    counter[0] += 1

    # Determine label and style
    node_type = type(node).__name__
    style = NODE_STYLES.get(node_type, 'style=filled fillcolor="#ffffff"')

    if isinstance(node, Program):
        label = "Program"
    elif isinstance(node, LetDecl):
        typ = f"\\n: {node.type_annotation}" if node.type_annotation else ""
        label = f"let {node.name}{typ}"
    elif isinstance(node, AssignStmt):
        label = f"{node.name} ="
    elif isinstance(node, IfStmt):
        label = "if"
    elif isinstance(node, WhileStmt):
        label = "while"
    elif isinstance(node, PrintStmt):
        label = "print"
    elif isinstance(node, FuncDef):
        params_str = ", ".join(f"{n}:{t}" for n, t in node.params)
        ret = f"→{node.return_type}" if node.return_type else ""
        label = f"def {node.name}({params_str}){ret}"
    elif isinstance(node, ReturnStmt):
        label = "return"
    elif isinstance(node, ExprStmt):
        label = "expr"
    elif isinstance(node, BinOp):
        label = node.op
    elif isinstance(node, UnaryOp):
        label = node.op
    elif isinstance(node, Comparison):
        label = node.op
    elif isinstance(node, IntLit):
        label = str(node.value)
    elif isinstance(node, FloatLit):
        label = str(node.value)
    elif isinstance(node, BoolLit):
        label = str(node.value)
    elif isinstance(node, StringLit):
        label = f'"{node.value}"'
    elif isinstance(node, Identifier):
        label = node.name
    elif isinstance(node, FuncCall):
        label = f"{node.name}()"
    else:
        label = node_type

    label = label.replace('"', '\\"')
    lines.append(f'  {nid} [label="{label}" {style}];')

    if parent_id is not None:
        lines.append(f"  {parent_id} -> {nid};")

    # Recurse into children
    children = _get_children(node)
    for child in children:
        ast_to_dot(child, nid, counter, lines)

    return lines


def _get_children(node):
    """Return the child AST nodes of a given node."""
    if isinstance(node, Program):
        return node.body
    elif isinstance(node, LetDecl):
        return [node.value]
    elif isinstance(node, AssignStmt):
        return [node.value]
    elif isinstance(node, IfStmt):
        children = [node.condition] + node.then_body
        if node.else_body:
            children += node.else_body
        return children
    elif isinstance(node, WhileStmt):
        return [node.condition] + node.body
    elif isinstance(node, PrintStmt):
        return [node.value]
    elif isinstance(node, FuncDef):
        return node.body
    elif isinstance(node, ReturnStmt):
        return [node.value] if node.value else []
    elif isinstance(node, ExprStmt):
        return [node.expr]
    elif isinstance(node, BinOp):
        return [node.left, node.right]
    elif isinstance(node, UnaryOp):
        return [node.operand]
    elif isinstance(node, Comparison):
        return [node.left, node.right]
    elif isinstance(node, FuncCall):
        return node.args
    else:
        return []


SAMPLE = """\
let x: int = 10
let y = x * 2 + 3
if y > 20:
    print(y)
"""


def visualize_ast(source: str, filename: str = "ast"):
    """Parse source and render AST to graphviz image."""
    ast = parse_source(source)
    lines = ['digraph AST {',
             '  rankdir=TB;',
             '  node [fontname="Courier" fontsize=11];']
    ast_to_dot(ast, lines=lines)
    lines.append("}")
    dot_source = "\n".join(lines)

    out = render_dot(dot_source, filename=filename, output_dir="output")
    return out


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    print("Parsing and visualizing AST...")
    print(f"Source:\n{source}")
    path = visualize_ast(source)
    print(f"Done! Check: {path}")
