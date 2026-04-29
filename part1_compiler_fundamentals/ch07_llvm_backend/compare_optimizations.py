#!/usr/bin/env python3
"""
Chapter 7 — Compare your hand-written optimizations vs LLVM's optimizer.

Shows side-by-side the LLVM IR before/after your passes, and before/after
LLVM's own optimization passes.

Usage:
    python compare_optimizations.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch06_optimization_passes"))

from ir_builder import build_ir
from constant_folding import constant_folding
from dead_code_elimination import dead_code_elimination
from common_subexpr_elim import common_subexpr_elimination
from llvm_emitter import emit_llvm_ir
from llvm_optimizer import optimize_llvm_ir
from visualization import plot_comparison, plot_bar_chart

from rich.console import Console
from rich.panel import Panel

console = Console()

try:
    from llvmlite import binding
    HAS_LLVMLITE = True
except ImportError:
    HAS_LLVMLITE = False


SAMPLE = """\
let a = 2 + 3
let b = a * 4
let c = 10 - 2
let d = a + c
let e = a + c
let f = d + e
"""


def compare_optimizations(source: str):
    console.print("\n[bold]═══ Optimization Comparison: Yours vs LLVM ═══[/]\n")
    console.print(f"[bold]Source:[/]\n{source}")

    # === Path 1: No custom optimization → LLVM ===
    ir_raw = build_ir(source)
    llvm_raw = emit_llvm_ir(ir_raw)
    raw_lines = llvm_raw.strip().split('\n')

    # === Path 2: Custom optimizations → LLVM ===
    ir_opt = build_ir(source)
    console.print("[bold cyan]Your custom optimizations:[/]")
    ir_opt = constant_folding(ir_opt)
    ir_opt = common_subexpr_elimination(ir_opt)
    ir_opt = dead_code_elimination(ir_opt)
    llvm_custom = emit_llvm_ir(ir_opt)
    custom_lines = llvm_custom.strip().split('\n')

    console.print(Panel(llvm_raw, title="[yellow]LLVM IR (no custom opt)[/]",
                        border_style="yellow"))
    console.print(Panel(llvm_custom, title="[cyan]LLVM IR (after custom opts)[/]",
                        border_style="cyan"))

    # === Path 3: LLVM's own optimizations ===
    if HAS_LLVMLITE:
        llvm_o2_from_raw = optimize_llvm_ir(llvm_raw, opt_level=2)
        llvm_o2_from_custom = optimize_llvm_ir(llvm_custom, opt_level=2)

        o2_raw_lines = llvm_o2_from_raw.strip().split('\n')
        o2_custom_lines = llvm_o2_from_custom.strip().split('\n')

        console.print(Panel(llvm_o2_from_raw,
                            title="[green]After LLVM -O2 (no custom opt)[/]",
                            border_style="green"))
        console.print(Panel(llvm_o2_from_custom,
                            title="[green]After LLVM -O2 (with custom opts first)[/]",
                            border_style="green"))

        # Visualization
        plot_comparison(raw_lines, o2_raw_lines,
                        "Without Optimization", "After LLVM -O2",
                        filename="compare_llvm_o2",
                        output_dir="output")

        plot_comparison(custom_lines, o2_custom_lines,
                        "After Custom Opts", "Custom + LLVM -O2",
                        filename="compare_custom_plus_llvm",
                        output_dir="output")

        plot_bar_chart(
            ["No opt", "Custom only", "LLVM -O2 only", "Custom + LLVM -O2"],
            [len(raw_lines), len(custom_lines),
             len(o2_raw_lines), len(o2_custom_lines)],
            title="LLVM IR Line Count Comparison",
            ylabel="Lines of LLVM IR",
            filename="compare_line_counts",
            output_dir="output",
            colors=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"],
        )
        console.print("\n[bold green]Visualizations saved to output/[/]")
    else:
        console.print("\n[yellow]Install llvmlite to see LLVM optimization comparison.[/]")
        plot_comparison(raw_lines, custom_lines,
                        "No Custom Opt", "After Custom Opts",
                        filename="compare_custom_only",
                        output_dir="output")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    compare_optimizations(source)
