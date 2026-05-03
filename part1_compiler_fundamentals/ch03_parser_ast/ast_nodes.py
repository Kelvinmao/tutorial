"""
Chapter 3 — AST Node definitions for MiniLang.

Every node type in the Abstract Syntax Tree is a simple dataclass.

Design rationale: The AST is the central data structure of the compiler
frontend. It represents the *structure* of the program stripped of syntactic
noise (semicolons, parentheses, indentation). Each node corresponds to a
semantic concept: declarations, control flow, expressions.

Historical context: ASTs became standard in the 1970s as compilers moved
from ad-hoc translation to structured multi-pass architectures. Earlier
compilers used parse trees (concrete syntax trees), but ASTs are more
compact because they omit tokens that only serve parsing (like parentheses).

Organization: The hierarchy follows the grammar:
  Program → list of Statement
  Statement → LetDecl | IfStmt | WhileStmt | FuncDef | ...
  Expr → BinOp | UnaryOp | IntLit | Identifier | FuncCall | ...

  Source code:                 AST:

    let x = 2 + 3 * 4               Program
    if x > 10:                       │
        return x                     ├── LetDecl(x)
                                     │     └── BinOp(+)
                                     │          ├── IntLit(2)
                                     │          └── BinOp(*)
                                     │               ├── IntLit(3)
                                     │               └── IntLit(4)
                                     └── IfStmt
                                           ├── cond: BinOp(>)
                                           │        ├── Ident(x)
                                           │        └── IntLit(10)
                                           └── then: [ReturnStmt]
                                                      └── Ident(x)

  Note: parentheses from source don't appear in the AST -- they only
  guide the parser in deciding tree structure (precedence).

Each downstream phase (type checker ch04, IR builder ch05) walks these
nodes via isinstance() dispatch. This is the "Interpreter" pattern—
simple and sufficient for a small language.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Top Level ────────────────────────────────────────────────────────────────

@dataclass
class Program:
    """Root of the AST — a sequence of statements."""
    body: list[Statement] = field(default_factory=list)


# ── Statements ───────────────────────────────────────────────────────────────

@dataclass
class Statement:
    """Base class for all statements (just for type hints)."""
    pass


@dataclass
class LetDecl(Statement):
    """Variable declaration: let x: int = expr"""
    name: str = ""
    type_annotation: Optional[str] = None
    value: Expr = None  # type: ignore


@dataclass
class IfStmt(Statement):
    """If statement with optional else block."""
    condition: Expr = None  # type: ignore
    then_body: list[Statement] = field(default_factory=list)
    else_body: list[Statement] = field(default_factory=list)


@dataclass
class WhileStmt(Statement):
    """While loop."""
    condition: Expr = None  # type: ignore
    body: list[Statement] = field(default_factory=list)


@dataclass
class PrintStmt(Statement):
    """Print statement: print(expr)"""
    value: Expr = None  # type: ignore


@dataclass
class FuncDef(Statement):
    """Function definition: def f(x: int) -> int: ..."""
    name: str = ""
    params: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    return_type: Optional[str] = None
    body: list[Statement] = field(default_factory=list)


@dataclass
class ReturnStmt(Statement):
    """Return statement: return expr"""
    value: Optional[Expr] = None


@dataclass
class ExprStmt(Statement):
    """Expression used as a statement (e.g., function call)."""
    expr: Expr = None  # type: ignore


@dataclass
class AssignStmt(Statement):
    """Assignment: x = expr"""
    name: str = ""
    value: Expr = None  # type: ignore


# ── Expressions ──────────────────────────────────────────────────────────────

@dataclass
class Expr:
    """Base class for all expressions."""
    pass


@dataclass
class BinOp(Expr):
    """Binary operation: left op right"""
    op: str = ""
    left: Expr = None  # type: ignore
    right: Expr = None  # type: ignore


@dataclass
class UnaryOp(Expr):
    """Unary operation: -expr"""
    op: str = ""
    operand: Expr = None  # type: ignore


@dataclass
class IntLit(Expr):
    """Integer literal."""
    value: int = 0


@dataclass
class FloatLit(Expr):
    """Float literal."""
    value: float = 0.0


@dataclass
class BoolLit(Expr):
    """Boolean literal."""
    value: bool = False


@dataclass
class StringLit(Expr):
    """String literal."""
    value: str = ""


@dataclass
class Identifier(Expr):
    """Variable reference."""
    name: str = ""


@dataclass
class FuncCall(Expr):
    """Function call: f(a, b, c)"""
    name: str = ""
    args: list[Expr] = field(default_factory=list)


@dataclass
class Comparison(Expr):
    """Comparison: left op right (==, !=, <, >, <=, >=)"""
    op: str = ""
    left: Expr = None  # type: ignore
    right: Expr = None  # type: ignore


# ── Pretty Printer ───────────────────────────────────────────────────────────

def pretty_print(node, indent: int = 0) -> str:
    """Return a human-readable indented string of the AST."""
    pad = "  " * indent

    if isinstance(node, Program):
        lines = [f"{pad}Program"]
        for stmt in node.body:
            lines.append(pretty_print(stmt, indent + 1))
        return "\n".join(lines)

    if isinstance(node, LetDecl):
        typ = f": {node.type_annotation}" if node.type_annotation else ""
        return (f"{pad}LetDecl '{node.name}'{typ}\n"
                f"{pretty_print(node.value, indent + 1)}")

    if isinstance(node, AssignStmt):
        return (f"{pad}Assign '{node.name}'\n"
                f"{pretty_print(node.value, indent + 1)}")

    if isinstance(node, IfStmt):
        lines = [f"{pad}If"]
        lines.append(f"{pad}  condition:")
        lines.append(pretty_print(node.condition, indent + 2))
        lines.append(f"{pad}  then:")
        for s in node.then_body:
            lines.append(pretty_print(s, indent + 2))
        if node.else_body:
            lines.append(f"{pad}  else:")
            for s in node.else_body:
                lines.append(pretty_print(s, indent + 2))
        return "\n".join(lines)

    if isinstance(node, WhileStmt):
        lines = [f"{pad}While"]
        lines.append(pretty_print(node.condition, indent + 1))
        for s in node.body:
            lines.append(pretty_print(s, indent + 1))
        return "\n".join(lines)

    if isinstance(node, PrintStmt):
        return f"{pad}Print\n{pretty_print(node.value, indent + 1)}"

    if isinstance(node, FuncDef):
        params = ", ".join(f"{n}: {t}" for n, t in node.params)
        ret = f" -> {node.return_type}" if node.return_type else ""
        lines = [f"{pad}FuncDef '{node.name}'({params}){ret}"]
        for s in node.body:
            lines.append(pretty_print(s, indent + 1))
        return "\n".join(lines)

    if isinstance(node, ReturnStmt):
        if node.value:
            return f"{pad}Return\n{pretty_print(node.value, indent + 1)}"
        return f"{pad}Return"

    if isinstance(node, ExprStmt):
        return f"{pad}ExprStmt\n{pretty_print(node.expr, indent + 1)}"

    if isinstance(node, BinOp):
        return (f"{pad}BinOp '{node.op}'\n"
                f"{pretty_print(node.left, indent + 1)}\n"
                f"{pretty_print(node.right, indent + 1)}")

    if isinstance(node, UnaryOp):
        return (f"{pad}UnaryOp '{node.op}'\n"
                f"{pretty_print(node.operand, indent + 1)}")

    if isinstance(node, Comparison):
        return (f"{pad}Compare '{node.op}'\n"
                f"{pretty_print(node.left, indent + 1)}\n"
                f"{pretty_print(node.right, indent + 1)}")

    if isinstance(node, IntLit):
        return f"{pad}Int({node.value})"
    if isinstance(node, FloatLit):
        return f"{pad}Float({node.value})"
    if isinstance(node, BoolLit):
        return f"{pad}Bool({node.value})"
    if isinstance(node, StringLit):
        return f"{pad}Str({node.value!r})"
    if isinstance(node, Identifier):
        return f"{pad}Id({node.name})"
    if isinstance(node, FuncCall):
        lines = [f"{pad}Call '{node.name}'"]
        for a in node.args:
            lines.append(pretty_print(a, indent + 1))
        return "\n".join(lines)

    return f"{pad}??? {type(node).__name__}"
