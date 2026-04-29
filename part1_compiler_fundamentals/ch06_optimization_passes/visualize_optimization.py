#!/usr/bin/env python3
"""
Chapter 6 — Visualize optimization passes: side-by-side before/after.

Usage:
    python visualize_optimization.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from visualization import plot_comparison, plot_bar_chart
from ir_builder import build_ir
from constant_folding import constant_folding
from dead_code_elimination import dead_code_elimination
from common_subexpr_elim import common_subexpr_elimination
from rich.console import Console

console = Console()

SAMPLE = """\
let a = 2 + 3
let b = a * 4
let c = 10 - 2
let d = a + c
let e = a + c
print(d)
"""


def run_optimization_pipeline(source: str):
    """Run all optimizations and visualize the results."""
    console.print("\n[bold]═══ Optimization Pipeline Visualization ═══[/]\n")
    console.print(f"[bold]Source:[/]\n{source}\n")

    ir = build_ir(source)
    original = [str(i).strip() for i in ir]

    console.print("[bold cyan]Step 1: Constant Folding[/]")
    ir = constant_folding(ir)
    after_cf = [str(i).strip() for i in ir]

    console.print("[bold cyan]Step 2: Common Subexpression Elimination[/]")
    ir = common_subexpr_elimination(ir)
    after_cse = [str(i).strip() for i in ir]

    console.print("[bold cyan]Step 3: Dead Code Elimination[/]")
    ir = dead_code_elimination(ir)
    after_dce = [str(i).strip() for i in ir]

    console.print()

    # Generate side-by-side comparison images
    plot_comparison(original, after_cf,
                    "Original IR", "After Constant Folding",
                    filename="opt_1_const_fold", output_dir="output")

    plot_comparison(after_cf, after_cse,
                    "After Const Fold", "After CSE",
                    filename="opt_2_cse", output_dir="output")

    plot_comparison(after_cse, after_dce,
                    "After CSE", "After DCE",
                    filename="opt_3_dce", output_dir="output")

    plot_comparison(original, after_dce,
                    "Original", "Fully Optimized",
                    filename="opt_full", output_dir="output")

    # Instruction count bar chart
    plot_bar_chart(
        ["Original", "After CF", "After CSE", "After DCE"],
        [len(original), len(after_cf), len(after_cse), len(after_dce)],
        title="Instruction Count Through Optimization Pipeline",
        ylabel="Number of Instructions",
        filename="opt_instruction_count",
        output_dir="output",
        colors=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"],
    )

    console.print("\n[bold green]✓ Visualizations saved to output/ directory[/]")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    run_optimization_pipeline(source)
