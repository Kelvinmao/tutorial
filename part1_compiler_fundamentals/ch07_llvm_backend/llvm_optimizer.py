#!/usr/bin/env python3
"""
Chapter 7 — Run LLVM optimization passes on generated LLVM IR.

Demonstrates LLVM's powerful optimization pipeline.

Usage:
    python llvm_optimizer.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: LLVM Optimization Pass Pipeline
#
# Historical context: LLVM's pass infrastructure was designed from day one
# to be modular: each optimization is a self-contained "pass" that reads
# IR, transforms it, and writes IR back. Passes compose into pipelines.
# At -O2, LLVM runs ~100 passes including constant folding, SROA (scalar
# replacement of aggregates), GVN (global value numbering), loop
# unrolling, vectorization, and dead code elimination.
#
# Problem solved: Our hand-written passes (ch06) cover the basics, but
# production-quality optimization requires hundreds of interacting passes.
# LLVM provides these for free once we emit LLVM IR.
#
# How it works here:
# 1. Parse the LLVM IR text into an LLVM module object.
# 2. Verify the module (check for malformed IR).
# 3. Create a PassManagerBuilder with the desired opt_level (0–3).
# 4. Populate a ModulePassManager from the builder.
# 5. Run the pass manager — it transforms the IR in-place.
# 6. Return the optimized IR as text.
#
#    LLVM IR (unoptimized)          LLVM IR (O2)
#    ┌───────────────────────┐    ┌───────────────────────┐
#    │ %t0 = alloca i64    │    │ %1 = mul i64 4, %x  │
#    │ store i64 4, %t0    │    │ %2 = add i64 %1, 1  │
#    │ %1 = load %t0       │    │ ret i64 %2          │
#    │ %2 = load %x        │    └───────────────────────┘
#    │ %3 = mul %1, %2     │     ▲
#    │ store %3, %t1       │     │ mem2reg: alloca → regs
#    │ ...                 │     │ const fold: 4 inlined
#    │ (20+ instructions)  │     │ DCE: dead stores removed
#    └───────────────────────┘     │ (3 instructions)
#
# Optimization levels:
#   -O0: No optimization (debug builds)
#   -O1: Basic: mem2reg, constant folding, DCE
#   -O2: Standard: + GVN, loop opts, vectorization (default for release)
#   -O3: Aggressive: + loop unrolling, more inlining (may increase size)
# ═══════════════════════════════════════════════════════════════════════════

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from ir_builder import build_ir
from llvm_emitter import emit_llvm_ir

from rich.console import Console
from rich.panel import Panel

console = Console()

try:
    from llvmlite import binding as llvm
    HAS_LLVMLITE = True
except ImportError:
    HAS_LLVMLITE = False


def optimize_llvm_ir(llvm_ir_str: str, opt_level: int = 2) -> str:
    """
    Run LLVM optimization passes on LLVM IR text.

    Parameters
    ----------
    llvm_ir_str : str
        LLVM IR text (as produced by llvm_emitter.py)
    opt_level : int
        Optimization level: 0 (none), 1 (basic), 2 (standard), 3 (aggressive)

    Returns
    -------
    str
        Optimized LLVM IR text
    """
    if not HAS_LLVMLITE:
        return llvm_ir_str + "\n; (optimization skipped — llvmlite not installed)"

    # Initialize LLVM
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # Parse the LLVM IR
    mod = llvm.parse_assembly(llvm_ir_str)
    mod.verify()

    # Create and run pass manager
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = opt_level
    pm = llvm.create_module_pass_manager()
    pmb.populate(pm)
    pm.run(mod)

    return str(mod)


SAMPLE = """\
let a = 2 + 3
let b = a * 4
let c = b - 10
let d = c + a
"""

if __name__ == "__main__":
    if not HAS_LLVMLITE:
        console.print("[red]llvmlite is required for this demo.[/]")
        console.print("[yellow]Install with: pip install llvmlite[/]")
        sys.exit(0)

    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE

    console.print("\n[bold]═══ LLVM Optimization Passes ═══[/]\n")
    console.print(f"[bold]Source:[/]\n{source}")

    ir = build_ir(source)
    llvm_ir_str = emit_llvm_ir(ir)

    console.print(Panel(llvm_ir_str, title="[yellow]Unoptimized LLVM IR[/]",
                        border_style="yellow"))

    for level in [1, 2, 3]:
        optimized = optimize_llvm_ir(llvm_ir_str, opt_level=level)
        console.print(Panel(optimized,
                            title=f"[cyan]Optimized LLVM IR (-O{level})[/]",
                            border_style="cyan"))
