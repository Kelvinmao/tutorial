#!/usr/bin/env python3
"""
Chapter 6 — Dead Code Elimination optimization pass.

Removes instructions whose results are never used.

Usage:
    python dead_code_elimination.py
"""

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from ir_builder import build_ir, IRInstr, Op
from rich.console import Console

console = Console()

# Instructions with side effects should never be removed
SIDE_EFFECT_OPS = {Op.PRINT, Op.CALL, Op.RETURN, Op.JUMP, Op.BRANCH,
                   Op.LABEL, Op.FUNC_BEGIN, Op.FUNC_END, Op.PARAM}


def dead_code_elimination(instructions: list[IRInstr]) -> list[IRInstr]:
    """
    Remove instructions whose destination register is never read.

    Algorithm:
    1. Collect all registers that are *used* (appear as src1 or src2)
    2. Keep instructions that:
       a. Have side effects (print, call, branch, etc.)
       b. Write to a used register
       c. Write to a non-temporary (user variable) — always keep
    3. Repeat until no more changes (fixpoint)
    """
    changed = True
    result = list(instructions)
    total_removed = 0

    while changed:
        changed = False

        # Collect used registers
        used: set[str] = set()
        for instr in result:
            if instr.src1 and isinstance(instr.src1, str):
                used.add(instr.src1)
            if instr.src2 and isinstance(instr.src2, str):
                used.add(instr.src2)

        new_result = []
        for instr in result:
            # Always keep side-effect instructions
            if instr.op in SIDE_EFFECT_OPS:
                new_result.append(instr)
                continue

            # Always keep assignments to non-temporary variables
            if instr.dst and not instr.dst.startswith("t"):
                new_result.append(instr)
                continue

            # Keep if the destination is used
            if instr.dst and instr.dst in used:
                new_result.append(instr)
                continue

            # This instruction is dead — remove it
            changed = True
            total_removed += 1

        result = new_result

    console.print(f"  [green]Dead code elimination: {total_removed} instructions removed[/]")
    return result


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let a = 10
let b = 20
let c = a + b
let d = a * b
print(c)
"""
# Note: 'd' is never used, so its computation should be eliminated

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    ir = build_ir(source)

    console.print("\n[bold]Before DCE:[/]")
    for i in ir:
        console.print(f"  {i}")

    console.print("\n[bold]Applying dead code elimination...[/]")
    optimized = dead_code_elimination(ir)

    console.print("\n[bold]After DCE:[/]")
    for i in optimized:
        console.print(f"  {i}")
    console.print(f"\nReduced from {len(ir)} to {len(optimized)} instructions.")
