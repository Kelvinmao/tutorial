#!/usr/bin/env python3
"""
Chapter 6 — Common Subexpression Elimination (CSE) optimization pass.

Detects duplicate computations and reuses the first result.

Usage:
    python common_subexpr_elim.py
"""

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from ir_builder import build_ir, IRInstr, Op
from rich.console import Console

console = Console()

# Commutative ops where a op b == b op a
COMMUTATIVE = {Op.ADD, Op.MUL, Op.EQ, Op.NE}


def common_subexpr_elimination(instructions: list[IRInstr]) -> list[IRInstr]:
    """
    Detect and eliminate common subexpressions.

    For each arithmetic/comparison instruction, create a canonical key
    (op, src1, src2). If we've seen the same key before, replace the
    instruction with a COPY from the previous result.
    """
    # Map from (op, src1, src2) → result register
    expr_map: dict[tuple, str] = {}
    result: list[IRInstr] = []
    eliminated = 0

    for instr in instructions:
        if instr.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD,
                        Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE):
            # Create canonical key (normalize commutative ops)
            if instr.op in COMMUTATIVE:
                key = (instr.op, *sorted([instr.src1, instr.src2]))
            else:
                key = (instr.op, instr.src1, instr.src2)

            if key in expr_map:
                # Reuse previous result
                result.append(IRInstr(Op.COPY, instr.dst, expr_map[key]))
                eliminated += 1
            else:
                expr_map[key] = instr.dst
                result.append(instr)
        else:
            # For assignments, invalidate any expressions involving this variable
            if instr.dst:
                keys_to_remove = [k for k in expr_map
                                  if instr.dst in k]
                for k in keys_to_remove:
                    del expr_map[k]
            result.append(instr)

    console.print(f"  [green]CSE: {eliminated} expressions eliminated[/]")
    return result


# ── Demo ─────────────────────────────────────────────────────────────────────

# This manually creates IR to demonstrate CSE clearly
if __name__ == "__main__":
    # Simulate IR for: let y = (a+b) * (a+b)
    # which generates: t0 = a+b; t1 = a+b; t2 = t0*t1
    ir = [
        IRInstr(Op.LOAD_CONST, "a", "5"),
        IRInstr(Op.LOAD_CONST, "b", "3"),
        IRInstr(Op.ADD, "t0", "a", "b"),      # t0 = a + b
        IRInstr(Op.ADD, "t1", "a", "b"),      # t1 = a + b  (same!)
        IRInstr(Op.MUL, "t2", "t0", "t1"),    # t2 = t0 * t1
        IRInstr(Op.COPY, "y", "t2"),
        IRInstr(Op.SUB, "t3", "a", "b"),      # t3 = a - b
        IRInstr(Op.SUB, "t4", "a", "b"),      # t4 = a - b  (same!)
        IRInstr(Op.ADD, "t5", "t3", "t4"),
        IRInstr(Op.COPY, "z", "t5"),
    ]

    console.print("\n[bold]Before CSE:[/]")
    for i in ir:
        console.print(f"  {i}")

    console.print("\n[bold]Applying CSE...[/]")
    optimized = common_subexpr_elimination(ir)

    console.print("\n[bold]After CSE:[/]")
    for i in optimized:
        console.print(f"  {i}")
