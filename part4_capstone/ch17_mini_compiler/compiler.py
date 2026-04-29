#!/usr/bin/env python3
"""
Chapter 17 — Main compiler driver: Model → Optimize → Codegen → Run.

Usage:
    python compiler.py
"""

from __future__ import annotations
import subprocess
import tempfile
import os

from model_ir import ModelGraph
from optimizer import optimize
from codegen import emit_c
from rich.console import Console
from rich.syntax import Syntax

console = Console()


def build_mlp() -> ModelGraph:
    """Build a simple MLP: input(1,784) → FC(128) → ReLU → FC(10) → Softmax."""
    g = ModelGraph("MLP")

    x = g.input("x", [1, 784])
    w1 = g.const("w1", [784, 128])
    b1 = g.const("b1", [1, 128])
    w2 = g.const("w2", [128, 10])
    b2 = g.const("b2", [1, 10])

    h = g.matmul(x, w1, "fc1")
    h = g.add(h, b1, "add_bias1")
    h = g.relu(h, "relu1")
    h = g.matmul(h, w2, "fc2")
    h = g.add(h, b2, "add_bias2")
    out = g.softmax(h, "output")

    return g


def compile_and_run(graph: ModelGraph) -> str:
    """Full compilation pipeline."""

    console.print(f"\n[bold]═══ Mini AI Compiler ═══[/]\n")

    # Step 1: Show original graph
    console.print("[bold cyan]1. Original Graph:[/]")
    console.print(graph.summary())
    original_count = len(graph.nodes)

    # Step 2: Optimize
    console.print(f"\n[bold cyan]2. Optimization:[/]")
    optimize(graph)
    console.print(f"   Nodes: {original_count} → {len(graph.nodes)}")
    console.print(f"\n[bold cyan]   Optimized Graph:[/]")
    console.print(graph.summary())

    # Step 3: Code generation
    console.print(f"\n[bold cyan]3. Code Generation:[/]")
    c_code = emit_c(graph)

    # Show first 40 lines
    preview = "\n".join(c_code.split("\n")[:40]) + "\n..."
    console.print(Syntax(preview, "c", theme="monokai"))

    # Save generated code
    gen_file = "generated_model.c"
    with open(gen_file, "w") as f:
        f.write(c_code)
    console.print(f"\n   Full C code saved to {gen_file}")

    # Step 4: Compile and run
    console.print(f"\n[bold cyan]4. Compile & Execute:[/]")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "model.c")
        exe = os.path.join(tmpdir, "model")

        with open(src, "w") as f:
            f.write(c_code)

        result = subprocess.run(
            ["gcc", "-O2", "-o", exe, src, "-lm"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            console.print(f"[red]Compilation failed:[/]\n{result.stderr}")
            return ""

        console.print("[green]   Compiled successfully![/]")

        result = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            console.print(f"\n[bold green]Output:[/]\n{result.stdout}")
            return result.stdout

    return ""


if __name__ == "__main__":
    graph = build_mlp()
    compile_and_run(graph)
