#!/usr/bin/env python3
"""
Chapter 5 — Three-Address Code IR builder for MiniLang.

Translates an AST into a flat list of IR instructions (three-address code).

Usage:
    python ir_builder.py
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))

from parser import parse_source
from ast_nodes import *
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── IR Opcodes ───────────────────────────────────────────────────────────────

class Op(Enum):
    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"
    NEG = "neg"
    # Comparison
    EQ  = "eq"
    NE  = "ne"
    LT  = "lt"
    GT  = "gt"
    LE  = "le"
    GE  = "ge"
    # Data movement
    LOAD_CONST = "load_const"
    COPY = "copy"
    # Control flow
    LABEL = "label"
    JUMP = "jump"
    BRANCH = "branch"       # conditional branch
    # Function
    CALL = "call"
    PARAM = "param"
    RETURN = "return"
    FUNC_BEGIN = "func_begin"
    FUNC_END = "func_end"
    # I/O
    PRINT = "print"
    # SSA
    PHI = "phi"


OP_MAP = {
    "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
    "==": Op.EQ, "!=": Op.NE, "<": Op.LT, ">": Op.GT, "<=": Op.LE, ">=": Op.GE,
}


# ── IR Instruction ───────────────────────────────────────────────────────────

@dataclass
class IRInstr:
    op: Op
    dst: Optional[str] = None
    src1: Optional[str] = None
    src2: Optional[str] = None

    def __repr__(self):
        if self.op == Op.LOAD_CONST:
            return f"  {self.dst} = {self.src1}"
        elif self.op == Op.COPY:
            return f"  {self.dst} = {self.src1}"
        elif self.op == Op.NEG:
            return f"  {self.dst} = -{self.src1}"
        elif self.op == Op.LABEL:
            return f"{self.dst}:"
        elif self.op == Op.JUMP:
            return f"  jump {self.dst}"
        elif self.op == Op.BRANCH:
            return f"  branch {self.src1}, {self.dst}, {self.src2}"
        elif self.op == Op.CALL:
            return f"  {self.dst} = call {self.src1}"
        elif self.op == Op.PARAM:
            return f"  param {self.src1}"
        elif self.op == Op.RETURN:
            return f"  return {self.src1 or ''}"
        elif self.op == Op.FUNC_BEGIN:
            return f"func {self.dst}:"
        elif self.op == Op.FUNC_END:
            return f"end func {self.dst}"
        elif self.op == Op.PRINT:
            return f"  print {self.src1}"
        elif self.op == Op.PHI:
            return f"  {self.dst} = φ({self.src1}, {self.src2})"
        else:
            return f"  {self.dst} = {self.src1} {self.op.value} {self.src2}"


# ── IR Builder ───────────────────────────────────────────────────────────────

class IRBuilder:
    """Translate AST → three-address code IR."""

    def __init__(self):
        self.instructions: list[IRInstr] = []
        self._temp_counter = 0
        self._label_counter = 0

    def _new_temp(self) -> str:
        name = f"t{self._temp_counter}"
        self._temp_counter += 1
        return name

    def _new_label(self, prefix: str = "L") -> str:
        name = f"{prefix}{self._label_counter}"
        self._label_counter += 1
        return name

    def _emit(self, op: Op, dst=None, src1=None, src2=None):
        instr = IRInstr(op, dst, src1, src2)
        self.instructions.append(instr)
        return instr

    def build(self, program: Program) -> list[IRInstr]:
        for stmt in program.body:
            self._build_stmt(stmt)
        return self.instructions

    # ── Statements ───────────────────────────────────────────────────────

    def _build_stmt(self, stmt: Statement):
        if isinstance(stmt, LetDecl):
            val = self._build_expr(stmt.value)
            self._emit(Op.COPY, stmt.name, val)
        elif isinstance(stmt, AssignStmt):
            val = self._build_expr(stmt.value)
            self._emit(Op.COPY, stmt.name, val)
        elif isinstance(stmt, IfStmt):
            self._build_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._build_while(stmt)
        elif isinstance(stmt, PrintStmt):
            val = self._build_expr(stmt.value)
            self._emit(Op.PRINT, src1=val)
        elif isinstance(stmt, FuncDef):
            self._build_funcdef(stmt)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                val = self._build_expr(stmt.value)
                self._emit(Op.RETURN, src1=val)
            else:
                self._emit(Op.RETURN)
        elif isinstance(stmt, ExprStmt):
            self._build_expr(stmt.expr)

    def _build_if(self, stmt: IfStmt):
        cond = self._build_expr(stmt.condition)
        then_label = self._new_label("then")
        else_label = self._new_label("else")
        end_label = self._new_label("endif")

        self._emit(Op.BRANCH, dst=then_label, src1=cond, src2=else_label)

        self._emit(Op.LABEL, dst=then_label)
        for s in stmt.then_body:
            self._build_stmt(s)
        self._emit(Op.JUMP, dst=end_label)

        self._emit(Op.LABEL, dst=else_label)
        if stmt.else_body:
            for s in stmt.else_body:
                self._build_stmt(s)
        self._emit(Op.JUMP, dst=end_label)

        self._emit(Op.LABEL, dst=end_label)

    def _build_while(self, stmt: WhileStmt):
        cond_label = self._new_label("while_cond")
        body_label = self._new_label("while_body")
        end_label = self._new_label("while_end")

        self._emit(Op.LABEL, dst=cond_label)
        cond = self._build_expr(stmt.condition)
        self._emit(Op.BRANCH, dst=body_label, src1=cond, src2=end_label)

        self._emit(Op.LABEL, dst=body_label)
        for s in stmt.body:
            self._build_stmt(s)
        self._emit(Op.JUMP, dst=cond_label)

        self._emit(Op.LABEL, dst=end_label)

    def _build_funcdef(self, stmt: FuncDef):
        self._emit(Op.FUNC_BEGIN, dst=stmt.name)
        for pname, _ in stmt.params:
            self._emit(Op.PARAM, src1=pname)
        for s in stmt.body:
            self._build_stmt(s)
        self._emit(Op.FUNC_END, dst=stmt.name)

    # ── Expressions ──────────────────────────────────────────────────────

    def _build_expr(self, expr: Expr) -> str:
        """Build IR for an expression, return the result register name."""
        if isinstance(expr, IntLit):
            t = self._new_temp()
            self._emit(Op.LOAD_CONST, t, str(expr.value))
            return t
        elif isinstance(expr, FloatLit):
            t = self._new_temp()
            self._emit(Op.LOAD_CONST, t, str(expr.value))
            return t
        elif isinstance(expr, BoolLit):
            t = self._new_temp()
            self._emit(Op.LOAD_CONST, t, "1" if expr.value else "0")
            return t
        elif isinstance(expr, StringLit):
            t = self._new_temp()
            self._emit(Op.LOAD_CONST, t, repr(expr.value))
            return t
        elif isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, BinOp):
            left = self._build_expr(expr.left)
            right = self._build_expr(expr.right)
            t = self._new_temp()
            op = OP_MAP.get(expr.op)
            if op:
                self._emit(op, t, left, right)
            return t
        elif isinstance(expr, UnaryOp):
            operand = self._build_expr(expr.operand)
            t = self._new_temp()
            self._emit(Op.NEG, t, operand)
            return t
        elif isinstance(expr, Comparison):
            left = self._build_expr(expr.left)
            right = self._build_expr(expr.right)
            t = self._new_temp()
            op = OP_MAP.get(expr.op)
            if op:
                self._emit(op, t, left, right)
            return t
        elif isinstance(expr, FuncCall):
            for arg in expr.args:
                val = self._build_expr(arg)
                self._emit(Op.PARAM, src1=val)
            t = self._new_temp()
            self._emit(Op.CALL, t, expr.name)
            return t
        return "?"


def build_ir(source: str) -> list[IRInstr]:
    """Parse and build IR from source code."""
    ast = parse_source(source)
    builder = IRBuilder()
    return builder.build(ast)


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let x: int = 10
let y: int = 20
let z = x + y * 2

if z > 40:
    print(z)
"""

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    console.print("\n" + "═" * 60)
    console.print("  MiniLang IR Builder — Three-Address Code")
    console.print("═" * 60)
    console.print(f"\n[bold]Source:[/]\n{source}")
    console.print("─" * 60)

    instructions = build_ir(source)
    ir_text = "\n".join(str(i) for i in instructions)
    console.print(Panel(ir_text, title="[cyan]IR Output[/]", border_style="cyan"))
    console.print(f"Total instructions: {len(instructions)}")
