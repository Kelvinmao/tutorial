#!/usr/bin/env python3
"""
Chapter 17 — Main compiler driver: Model → Optimize → Codegen → Run.

Usage:
    python compiler.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# CAPSTONE: End-to-End Mini AI Compiler Pipeline
#
# This is the capstone that integrates all concepts from the tutorial:
#
# Phase 1: MODEL DEFINITION (model_ir.py)
#   Build an MLP graph: Input(784) → FC(128) → ReLU → FC(10) → Softmax
#   This is analogous to defining a model in PyTorch/TensorFlow.
#   Creates ~11 nodes: input, weights, biases, matmuls, adds, relu, softmax.
#
# Phase 2: GRAPH OPTIMIZATION (optimizer.py)
#   Run the same passes as ch09 on our graph:
#   - Fuse MatMul+Add → Linear (2 fusions)
#   - Fuse Linear+ReLU → Linear_ReLU (1 fusion)
#   - Eliminate dead nodes (orphaned Add nodes)
#   Result: 11 nodes → ~8 nodes (fewer kernel launches)
#
# Phase 3: CODE GENERATION (codegen.py)
#   Emit a complete C program with:
#   - Stack-allocated tensor buffers
#   - Calls to matmul, add_bias, relu, softmax helpers
#   - Timing instrumentation
#
# Phase 4: COMPILE & EXECUTE
#   - Write C code to temp file
#   - Invoke gcc -O2 to compile
#   - Run the binary and capture output
#   - Display inference results and timing
#
# This pipeline mirrors production AI compilers:
#   PyTorch model → TorchScript/ONNX → TVM/XLA graph opts → codegen → run
#
#   End-to-end flow:
#
#   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐
#   │  Model   │──►│ Optimize │──►│ CodeGen  │──►│ gcc -O2  │──►│ Run  │
#   │ (Python) │   │ (fusion, │   │ (emit C) │   │ (compile)│   │      │
#   │          │   │  DCE)    │   │          │   │          │   │      │
#   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────┘
#   model_ir.py    optimizer.py    codegen.py      subprocess    output
#
#   MLP(784→128→10)  11→8 nodes    ~100 lines C   native binary  inference
#                    3 fusions      + helper fns   x86_64         result +
#                    3 DCE                                        timing
# ═══════════════════════════════════════════════════════════════════════════

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
