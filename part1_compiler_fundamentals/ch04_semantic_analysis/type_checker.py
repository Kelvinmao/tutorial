#!/usr/bin/env python3
"""
Chapter 4 — Type checker and symbol table for MiniLang.

Performs semantic analysis on an AST:
- Builds a symbol table with scoped variable declarations
- Type-checks all expressions
- Reports errors for undeclared variables, type mismatches, etc.

Usage:
    python type_checker.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Semantic Analysis — Scoped Symbol Table + Type Checking
#
# Historical context: Semantic analysis was formalized alongside type theory
# in the 1960s–70s (Algol 68 had one of the first rigorous type systems).
# The parser validates *syntax* (is the program grammatically correct?),
# but semantic analysis validates *meaning* (are operations type-safe?
# are variables declared before use? do function signatures match calls?).
#
# Problem solved: Catch errors that are syntactically valid but
# semantically wrong. For example:
#   let x: int = "hello"    → Type mismatch (int vs string)
#   print(y)                → Undeclared variable 'y'
#   if 42: ...              → Condition should be bool, not int
#
# ALGORITHM 1 — Scoped Symbol Table:
# - A stack of dictionaries, one per scope (global, function, if/while block).
# - declare() adds a variable to the current (innermost) scope.
# - lookup() searches from innermost to outermost — this gives lexical scoping.
# - enter_scope() pushes a new dict; exit_scope() pops it.
# This is the standard approach used by most compilers (GCC, Clang, javac).
#
#   let x: int = 1
#   def foo(y: int) -> int:       Scope stack during foo body:
#       let z = x + y
#       return z                  ┌────────────────────┐  ← innermost (foo)
#                                 │ z: int, y: int   │
#                                 └────────────────────┘
#                                 ┌────────────────────┐  ← outermost (global)
#                                 │ x: int, foo: fn  │
#                                 └────────────────────┘
#
#   lookup("z") → found in scope[1] (foo)      ✓
#   lookup("x") → not in scope[1], found in scope[0] (global) ✓
#   lookup("w") → not in any scope → ERROR: undeclared
#
# ALGORITHM 2 — Type Checking via AST Walk:
# - Recursively visit every AST node.
# - For each expression, infer its type bottom-up:
#     IntLit → "int", FloatLit → "float", BoolLit → "bool"
#     Identifier → look up in symbol table
#     BinOp → check operands, apply promotion (int + float → float)
#     Comparison → always returns "bool"
#     FuncCall → extract return type from function signature
# - For each statement, check type constraints:
#     LetDecl → value type must match declared type
#     IfStmt → condition must be bool
#     ReturnStmt → value type must match function return type
# - Collect errors rather than aborting on the first one.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))

from parser import parse_source
from ast_nodes import *
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ── Symbol Table ─────────────────────────────────────────────────────────────

class Symbol:
    """Represents a declared name in the symbol table."""
    def __init__(self, name: str, sym_type: str, scope_level: int,
                 scope_name: str = "global"):
        self.name = name
        self.sym_type = sym_type          # "int", "float", "bool", "fn→int", etc.
        self.scope_level = scope_level
        self.scope_name = scope_name

    def __repr__(self):
        return f"Symbol({self.name}: {self.sym_type}, scope={self.scope_name}[{self.scope_level}])"


class SymbolTable:
    """
    Scoped symbol table using a stack of dictionaries.

    Each scope is a dict mapping variable names to Symbol objects.
    Lookup walks from the innermost scope outward.

    This implements lexical (static) scoping: a variable reference resolves
    to the nearest enclosing declaration. The stack grows when we enter
    a block (function body, if/while body) and shrinks when we leave.

    Example scope stack for a function call:
      [global_scope, function_scope, if_body_scope]
       │               │                └─ local vars in if
       │               └─ function params + locals
       └─ top-level declarations
    """
    def __init__(self):
        self.scopes: list[dict[str, Symbol]] = [{}]  # start with global scope
        self.scope_names: list[str] = ["global"]
        self.level = 0
        self.all_symbols: list[Symbol] = []  # flat list for visualization

    def enter_scope(self, name: str = "block"):
        self.level += 1
        self.scopes.append({})
        self.scope_names.append(name)

    def exit_scope(self):
        self.scopes.pop()
        self.scope_names.pop()
        self.level -= 1

    def declare(self, name: str, sym_type: str) -> Symbol:
        """Declare a variable in the current scope."""
        scope_name = self.scope_names[-1]
        sym = Symbol(name, sym_type, self.level, scope_name)
        self.scopes[-1][name] = sym
        self.all_symbols.append(sym)
        return sym

    def lookup(self, name: str) -> Symbol | None:
        """Look up a variable, searching from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def current_scope_name(self) -> str:
        return self.scope_names[-1]


# ── Type Checker ─────────────────────────────────────────────────────────────

class SemanticError:
    def __init__(self, message: str, node=None):
        self.message = message
        self.node = node

    def __repr__(self):
        return f"SemanticError: {self.message}"


class TypeChecker:
    """
    Walk the AST, build the symbol table, and type-check everything.
    """
    def __init__(self):
        self.symbols = SymbolTable()
        self.errors: list[SemanticError] = []
        self.current_func_return_type: str | None = None

    def check(self, program: Program) -> tuple[SymbolTable, list[SemanticError]]:
        for stmt in program.body:
            self._check_stmt(stmt)
        return self.symbols, self.errors

    def _error(self, msg: str, node=None):
        self.errors.append(SemanticError(msg, node))

    # ── Statements ───────────────────────────────────────────────────────

    def _check_stmt(self, stmt: Statement):
        if isinstance(stmt, LetDecl):
            self._check_let(stmt)
        elif isinstance(stmt, AssignStmt):
            self._check_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self._check_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._check_while(stmt)
        elif isinstance(stmt, PrintStmt):
            self._check_expr(stmt.value)
        elif isinstance(stmt, FuncDef):
            self._check_funcdef(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._check_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr)

    def _check_let(self, node: LetDecl):
        val_type = self._check_expr(node.value)
        declared_type = node.type_annotation
        if declared_type and val_type and declared_type != val_type:
            # Allow int → float promotion
            if not (declared_type == "float" and val_type == "int"):
                self._error(
                    f"Type mismatch in let '{node.name}': "
                    f"declared {declared_type}, got {val_type}", node)
        final_type = declared_type or val_type or "unknown"
        self.symbols.declare(node.name, final_type)

    def _check_assign(self, node: AssignStmt):
        sym = self.symbols.lookup(node.name)
        val_type = self._check_expr(node.value)
        if sym is None:
            self._error(f"Assignment to undeclared variable '{node.name}'", node)
            self.symbols.declare(node.name, val_type or "unknown")
        elif val_type and sym.sym_type != val_type:
            if not (sym.sym_type == "float" and val_type == "int"):
                self._error(
                    f"Type mismatch in assignment to '{node.name}': "
                    f"expected {sym.sym_type}, got {val_type}", node)

    def _check_if(self, node: IfStmt):
        cond_type = self._check_expr(node.condition)
        if cond_type and cond_type != "bool":
            self._error(f"If condition should be bool, got {cond_type}", node)
        self.symbols.enter_scope("if-then")
        for s in node.then_body:
            self._check_stmt(s)
        self.symbols.exit_scope()
        if node.else_body:
            self.symbols.enter_scope("if-else")
            for s in node.else_body:
                self._check_stmt(s)
            self.symbols.exit_scope()

    def _check_while(self, node: WhileStmt):
        cond_type = self._check_expr(node.condition)
        if cond_type and cond_type != "bool":
            self._error(f"While condition should be bool, got {cond_type}", node)
        self.symbols.enter_scope("while")
        for s in node.body:
            self._check_stmt(s)
        self.symbols.exit_scope()

    def _check_funcdef(self, node: FuncDef):
        # Build function type signature
        param_types = [t for _, t in node.params]
        ret = node.return_type or "void"
        fn_type = f"fn({','.join(param_types)})→{ret}"
        self.symbols.declare(node.name, fn_type)

        self.symbols.enter_scope(node.name)
        old_return = self.current_func_return_type
        self.current_func_return_type = node.return_type

        for pname, ptype in node.params:
            self.symbols.declare(pname, ptype)

        for s in node.body:
            self._check_stmt(s)

        self.current_func_return_type = old_return
        self.symbols.exit_scope()

    def _check_return(self, node: ReturnStmt):
        if node.value:
            val_type = self._check_expr(node.value)
            if (self.current_func_return_type and val_type and
                    val_type != self.current_func_return_type):
                if not (self.current_func_return_type == "float" and val_type == "int"):
                    self._error(
                        f"Return type mismatch: expected {self.current_func_return_type}, "
                        f"got {val_type}", node)

    # ── Expressions ──────────────────────────────────────────────────────

    def _check_expr(self, expr: Expr) -> str | None:
        """Type-check an expression and return its type."""
        if isinstance(expr, IntLit):
            return "int"
        elif isinstance(expr, FloatLit):
            return "float"
        elif isinstance(expr, BoolLit):
            return "bool"
        elif isinstance(expr, StringLit):
            return "string"
        elif isinstance(expr, Identifier):
            sym = self.symbols.lookup(expr.name)
            if sym is None:
                self._error(f"Undeclared variable '{expr.name}'", expr)
                return None
            return sym.sym_type
        elif isinstance(expr, BinOp):
            lt = self._check_expr(expr.left)
            rt = self._check_expr(expr.right)
            if lt is None or rt is None:
                return None
            # String concatenation
            if lt == "string" and rt == "string" and expr.op == "+":
                return "string"
            # Numeric operations
            if lt in ("int", "float") and rt in ("int", "float"):
                return "float" if "float" in (lt, rt) else "int"
            self._error(f"Cannot apply '{expr.op}' to {lt} and {rt}", expr)
            return None
        elif isinstance(expr, UnaryOp):
            t = self._check_expr(expr.operand)
            if t and t not in ("int", "float"):
                self._error(f"Cannot negate type {t}", expr)
            return t
        elif isinstance(expr, Comparison):
            lt = self._check_expr(expr.left)
            rt = self._check_expr(expr.right)
            # Comparisons always produce bool
            return "bool"
        elif isinstance(expr, FuncCall):
            sym = self.symbols.lookup(expr.name)
            if sym is None:
                self._error(f"Undeclared function '{expr.name}'", expr)
                return None
            # Extract return type from fn signature
            if sym.sym_type.startswith("fn("):
                ret = sym.sym_type.split("→")[-1]
                return ret
            return None
        return None


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let width: int = 10
let height: float = 20.5
let area = width * height

def square(x: int) -> int:
    return x * x

if area > 100:
    print(square(5))
"""

SAMPLE_ERRORS = """\
let x: int = "hello"
let y = x + z
"""


def run_checker(source: str, label: str = ""):
    ast = parse_source(source)
    checker = TypeChecker()
    sym_table, errors = checker.check(ast)

    if label:
        console.rule(f"[bold]{label}[/]")

    # Display symbol table
    table = Table(title="Symbol Table", box=box.ROUNDED, show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Scope", style="green")
    table.add_column("Level", justify="center")

    for sym in sym_table.all_symbols:
        table.add_row(sym.name, sym.sym_type, sym.scope_name, str(sym.scope_level))
    console.print(table)

    # Display errors
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)}):[/]")
        for err in errors:
            console.print(f"  [red]✗[/] {err.message}")
    else:
        console.print("\n[bold green]✓ No semantic errors found.[/]")
    console.print()


if __name__ == "__main__":
    console.print("\n" + "═" * 60)
    console.print("  MiniLang Semantic Analysis")
    console.print("═" * 60 + "\n")

    run_checker(SAMPLE, "Valid Program")
    run_checker(SAMPLE_ERRORS, "Program with Errors")
