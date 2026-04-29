#!/usr/bin/env python3
"""
Chapter 10 — Loop nest representation and manipulation.

Usage:
    python loop_nest.py
"""

from rich.console import Console
console = Console()

from tensor_expression import (placeholder, reduce_axis, compute,
                                reduce_sum, lower_to_loops, print_loop_nest,
                                Loop, Statement, IterVar)


def demo():
    console.print("\n[bold]═══ Loop Nest Structure ═══[/]\n")

    M, K, N = 16, 32, 16
    A = placeholder((M, K), "A")
    B = placeholder((K, N), "B")
    k = reduce_axis(K, "k")
    C = compute((M, N), lambda i, j: reduce_sum(A[i, k] * B[k, j], k), "C")

    loops = lower_to_loops(C)
    console.print("[bold]Original loop nest:[/]")
    print_loop_nest(loops)

    # Show loop nest metadata
    console.print("\n[bold]Loop nest analysis:[/]")
    _analyze_loop(loops, depth=0)


def _analyze_loop(node, depth=0):
    if isinstance(node, Loop):
        kind = "REDUCE" if node.var.is_reduce else "SPATIAL"
        console.print(f"  {'  '*depth}Level {depth}: {node.var.name} "
                      f"[0..{node.var.extent}) [{kind}]")
        for child in node.body:
            _analyze_loop(child, depth + 1)
    elif isinstance(node, Statement):
        console.print(f"  {'  '*depth}Assignment: {node.target} {node.op} {node.expr}")
    elif isinstance(node, list):
        for child in node:
            _analyze_loop(child, depth)


if __name__ == "__main__":
    demo()
