#!/usr/bin/env python3
"""
Chapter 6 — Common Subexpression Elimination (CSE) optimization pass.

Detects duplicate computations and reuses the first result.

Usage:
    python common_subexpr_elim.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Common Subexpression Elimination (CSE)
#
# Historical context: CSE was one of the optimizations identified by
# Cocke and Schwartz at IBM (1970). The idea: if the compiler computes
# the same expression twice with the same operands and nothing has
# changed in between, the second computation is redundant.
#
# Problem solved: Code often computes the same thing multiple times.
# Example: "let y = (a+b) * (a+b)" generates:
#   t0 = a + b
#   t1 = a + b    ← same computation as t0!
#   t2 = t0 * t1
# CSE detects this and replaces t1 with a copy of t0:
#   t0 = a + b
#   t1 = t0       ← reuse (copy instead of recompute)
#   t2 = t0 * t1
# DCE can then remove the copy if t1 is only used in t2.
#
# How it works:
# 1. Maintain a dict "expr_map" mapping canonical expression keys to
#    the register that holds the first result.
#    Key format: (op, src1, src2)
#
# 2. For commutative ops (ADD, MUL, EQ, NE), normalize the key by
#    sorting operands: (ADD, "a", "b") == (ADD, "b", "a").
#    This catches commutativity: "a+b" matches "b+a".
#
#   Example:
#
#   t0 = a + b          expr_map: {(ADD,a,b): t0}
#   t1 = b + a          key (ADD,a,b) → commutative sort → (ADD,a,b)
#     └─► t1 = t0         HIT! replace with COPY from t0
#   t2 = t0 * t1        key (MUL,t0,t0) since t1=t0 now
#   t3 = c - d          expr_map: {(ADD,a,b): t0, (SUB,c,d): t3}
#   t4 = c - d          key (SUB,c,d) already in map!
#     └─► t4 = t3         HIT! replace with COPY
#
#   Before:              After:
#   t0 = a + b           t0 = a + b
#   t1 = b + a           t1 = t0      ← reused (commutative match)
#   t2 = t0 * t1         t2 = t0 * t0
#   t3 = c - d           t3 = c - d
#   t4 = c - d           t4 = t3      ← reused
#
# 3. For each arithmetic/comparison instruction:
#    - Compute its canonical key.
#    - If the key is already in expr_map, replace the instruction
#      with a COPY from the first result.
#    - Otherwise, record the mapping for future lookups.
#
# 4. Invalidation: When a variable is reassigned (dst is not a temp),
#    remove any expr_map entries that reference it as an operand,
#    because the expression is no longer valid.
#
# Complexity: O(n) single pass (with O(1) dict lookups).
# ═══════════════════════════════════════════════════════════════════════════

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
