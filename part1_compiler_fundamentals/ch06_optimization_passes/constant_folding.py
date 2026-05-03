#!/usr/bin/env python3
"""
Chapter 6 — Constant Folding optimization pass.

Evaluates operations with constant operands at compile time.

Usage:
    python constant_folding.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Constant Folding (with Constant Propagation)
#
# Historical context: Constant folding is one of the oldest compiler
# optimizations, present even in early Fortran compilers (1957). The
# idea is simple but powerful: if the compiler knows the values of both
# operands at compile time, it can compute the result immediately.
#
# Problem solved: Programs often contain expressions like "2 + 3" or
# "width * 2" where width was assigned a constant. Without this pass,
# the CPU would evaluate these at runtime even though the answer is
# known at compile time.
#
# How it works:
# 1. Maintain a dictionary "known" mapping register names to their
#    known constant values.
# 2. Scan instructions sequentially:
#    - LOAD_CONST dst, value: Record known[dst] = value.
#    - COPY dst, src: If src is in known, propagate: known[dst] = known[src].
#      (This is the "constant propagation" part.)
#    - ADD/SUB/MUL/...: Look up both src1 and src2 in known.
#      If both are constants, compute the result, store it in known[dst],
#      and replace the instruction with a LOAD_CONST.
#    - Comparisons (EQ, LT, etc.): Same approach, fold to 0 or 1.
#    - NEG: If operand is known, negate it.
# 3. Folded instructions are replaced in-place in the output list.
#
#   Example trace:              known dict:
#
#   t0 = 2                      {t0: 2}
#   t1 = 3                      {t0: 2, t1: 3}
#   t2 = t0 + t1                both known! → fold:
#     └─► t2 = 5                 {t0: 2, t1: 3, t2: 5}
#   x = t2                      propagate:
#     └─► x = 5                  {t0: 2, t1: 3, t2: 5, x: 5}
#   t3 = x * 10                 x known! 10 is literal! → fold:
#     └─► t3 = 50                {... t3: 50}
#
#   Before:           After:
#   t0 = 2            t0 = 2        ← can be removed by DCE
#   t1 = 3            t1 = 3        ← can be removed by DCE
#   t2 = t0 + t1      t2 = 5        ← folded
#   x  = t2           x  = 5        ← propagated
#   t3 = x * 10       t3 = 50       ← folded
#
# Complexity: O(n) single pass over the instruction list.
# Limitation: This is intra-procedural and doesn't handle control flow
# (a value assigned in one branch may not be constant in another).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from ir_builder import build_ir, IRInstr, Op
from rich.console import Console
from rich.panel import Panel

console = Console()

ARITH_OPS = {
    Op.ADD: lambda a, b: a + b,
    Op.SUB: lambda a, b: a - b,
    Op.MUL: lambda a, b: a * b,
    Op.DIV: lambda a, b: a / b if b != 0 else None,
    Op.MOD: lambda a, b: a % b if b != 0 else None,
}

COMPARE_OPS = {
    Op.EQ: lambda a, b: 1 if a == b else 0,
    Op.NE: lambda a, b: 1 if a != b else 0,
    Op.LT: lambda a, b: 1 if a < b else 0,
    Op.GT: lambda a, b: 1 if a > b else 0,
    Op.LE: lambda a, b: 1 if a <= b else 0,
    Op.GE: lambda a, b: 1 if a >= b else 0,
}


def _try_parse_number(s: str):
    """Try to parse a string as int or float."""
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def constant_folding(instructions: list[IRInstr]) -> list[IRInstr]:
    """
    Perform constant folding on IR instructions.

    Maintains a map of registers to known constant values.
    When both operands of an operation are known, computes the result.
    """
    known: dict[str, float | int] = {}
    result: list[IRInstr] = []
    folded_count = 0

    for instr in instructions:
        if instr.op == Op.LOAD_CONST:
            val = _try_parse_number(instr.src1)
            if val is not None:
                known[instr.dst] = val
            result.append(instr)

        elif instr.op == Op.COPY:
            if instr.src1 in known:
                known[instr.dst] = known[instr.src1]
            result.append(instr)

        elif instr.op in ARITH_OPS:
            v1 = known.get(instr.src1)
            v2 = known.get(instr.src2)
            if v1 is not None and v2 is not None:
                fn = ARITH_OPS[instr.op]
                val = fn(v1, v2)
                if val is not None:
                    known[instr.dst] = val
                    result.append(IRInstr(Op.LOAD_CONST, instr.dst, str(val)))
                    folded_count += 1
                    continue
            result.append(instr)

        elif instr.op in COMPARE_OPS:
            v1 = known.get(instr.src1)
            v2 = known.get(instr.src2)
            if v1 is not None and v2 is not None:
                fn = COMPARE_OPS[instr.op]
                val = fn(v1, v2)
                known[instr.dst] = val
                result.append(IRInstr(Op.LOAD_CONST, instr.dst, str(val)))
                folded_count += 1
                continue
            result.append(instr)

        elif instr.op == Op.NEG:
            v = known.get(instr.src1)
            if v is not None:
                known[instr.dst] = -v
                result.append(IRInstr(Op.LOAD_CONST, instr.dst, str(-v)))
                folded_count += 1
                continue
            result.append(instr)

        else:
            result.append(instr)

    console.print(f"  [green]Constant folding: {folded_count} operations folded[/]")
    return result


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let a = 2 + 3
let b = a * 4
let c = 10 - 2
"""

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    ir = build_ir(source)

    console.print("\n[bold]Before constant folding:[/]")
    for i in ir:
        console.print(f"  {i}")

    console.print("\n[bold]Applying constant folding...[/]")
    optimized = constant_folding(ir)

    console.print("\n[bold]After constant folding:[/]")
    for i in optimized:
        console.print(f"  {i}")
