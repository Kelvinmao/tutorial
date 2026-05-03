#!/usr/bin/env python3
"""
Chapter 3 — Recursive descent parser for MiniLang.

Consumes tokens from the Chapter 2 lexer and builds an AST.

Usage:
    python parser.py                      # parse sample program
    python -c "from parser import parse_source; ..."
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Recursive Descent Parser with Precedence Climbing
#
# Historical context: Recursive descent parsing was described by Lucas (1961)
# and Hoare (1962). It became the most popular hand-written parsing technique
# because each grammar rule maps directly to a function, making the code
# easy to read, debug, and extend. GCC, Clang, Go, Rust, and CPython all
# use hand-written recursive descent parsers.
#
# Problem solved: Given a flat sequence of tokens, build a hierarchical
# Abstract Syntax Tree (AST) that captures the program's structure.
# The parser must enforce:
#   - Operator precedence (* before +)
#   - Associativity (left-to-right for arithmetic)
#   - Statement structure (let, if, while, def, return)
#   - Nested blocks (via INDENT/DEDENT tokens)
#
# How it works:
# 1. GRAMMAR → FUNCTIONS: Each grammar rule becomes a method.
#      program → statement*
#      statement → let | if | while | def | return | assign | expr_stmt
#      expr → comparison
#      comparison → additive (('==' | '<' | ...) additive)?
#      additive → multiplicative (('+' | '-') multiplicative)*
#      multiplicative → unary (('*' | '/') unary)*
#      unary → '-' unary | primary
#      primary → INT | FLOAT | BOOL | STRING | ID | ID '(' args ')' | '(' expr ')'
#
# 2. PRECEDENCE BY NESTING: Lower-precedence rules call higher-precedence
#    rules for their operands. parse_additive() calls parse_multiplicative()
#    as its "atom", so * binds tighter than +.
#
#    Parsing "2 + 3 * 4":
#
#    parse_additive()
#    │  calls parse_multiplicative()  → returns Num(2)
#    │  sees '+'
#    │  calls parse_multiplicative()  → enters loop:
#    │     │  calls parse_primary()    → returns Num(3)
#    │     │  sees '*'
#    │     │  calls parse_primary()    → returns Num(4)
#    │     └─ returns BinOp(*, 3, 4)
#    └─ returns BinOp(+, 2, BinOp(*, 3, 4))
#
#    Result:     (+)
#               /   \
#              2    (*)
#                  /   \
#                 3     4
#
# 3. LOOKAHEAD: The parser peeks at the current token to decide which rule
#    to apply. Assignment vs. expression is decided by checking if the next
#    two tokens are ID EQUALS. Function call is detected by seeing ID LPAREN.
#
# 4. INDENTATION BLOCKS: The lexer emits INDENT/DEDENT tokens, so the
#    parser handles blocks just like any other token — _parse_block() eats
#    INDENT, parses statements, then eats DEDENT.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import sys
import os

# Add parent dir so we can import from ch02
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))

from lexer import tokenize, Token, TokenType
from ast_nodes import *


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        loc = f"L{token.line}:{token.col}"
        super().__init__(f"Parse error at {loc}: {message} (got {token.type.name} {token.value!r})")
        self.token = token


class Parser:
    """
    Recursive descent parser for MiniLang.

    Grammar precedence (lowest → highest):
        comparison → additive → multiplicative → unary → primary
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # ── Token helpers ────────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _at(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _eat(self, expected: TokenType | None = None) -> Token:
        tok = self.tokens[self.pos]
        if expected is not None and tok.type != expected:
            raise ParseError(f"Expected {expected.name}", tok)
        self.pos += 1
        return tok

    def _skip_newlines(self):
        while self._at(TokenType.NEWLINE):
            self._eat()

    # ── Program ──────────────────────────────────────────────────────────

    def parse_program(self) -> Program:
        prog = Program()
        self._skip_newlines()
        while not self._at(TokenType.EOF):
            prog.body.append(self._parse_statement())
            self._skip_newlines()
        return prog

    # ── Statements ───────────────────────────────────────────────────────

    def _parse_statement(self) -> Statement:
        tok = self._peek()

        if tok.type == TokenType.LET:
            return self._parse_let()
        elif tok.type == TokenType.IF:
            return self._parse_if()
        elif tok.type == TokenType.WHILE:
            return self._parse_while()
        elif tok.type == TokenType.PRINT:
            return self._parse_print()
        elif tok.type == TokenType.DEF:
            return self._parse_funcdef()
        elif tok.type == TokenType.RETURN:
            return self._parse_return()
        # Assignment: ID = expr  (lookahead for EQUALS)
        elif tok.type == TokenType.ID and self.pos + 1 < len(self.tokens) and \
                self.tokens[self.pos + 1].type == TokenType.EQUALS:
            return self._parse_assign()
        else:
            return self._parse_expr_stmt()

    def _parse_let(self) -> LetDecl:
        self._eat(TokenType.LET)
        name = self._eat(TokenType.ID).value
        type_ann = None
        if self._at(TokenType.COLON):
            self._eat(TokenType.COLON)
            type_ann = self._eat().value  # TYPE_INT, TYPE_FLOAT, etc.
        self._eat(TokenType.EQUALS)
        value = self._parse_expr()
        if self._at(TokenType.NEWLINE):
            self._eat()
        return LetDecl(name=name, type_annotation=type_ann, value=value)

    def _parse_assign(self) -> AssignStmt:
        name = self._eat(TokenType.ID).value
        self._eat(TokenType.EQUALS)
        value = self._parse_expr()
        if self._at(TokenType.NEWLINE):
            self._eat()
        return AssignStmt(name=name, value=value)

    def _parse_if(self) -> IfStmt:
        self._eat(TokenType.IF)
        condition = self._parse_expr()
        self._eat(TokenType.COLON)
        if self._at(TokenType.NEWLINE):
            self._eat()
        then_body = self._parse_block()
        else_body = []
        if self._at(TokenType.ELSE):
            self._eat(TokenType.ELSE)
            self._eat(TokenType.COLON)
            if self._at(TokenType.NEWLINE):
                self._eat()
            else_body = self._parse_block()
        return IfStmt(condition=condition, then_body=then_body, else_body=else_body)

    def _parse_while(self) -> WhileStmt:
        self._eat(TokenType.WHILE)
        condition = self._parse_expr()
        self._eat(TokenType.COLON)
        if self._at(TokenType.NEWLINE):
            self._eat()
        body = self._parse_block()
        return WhileStmt(condition=condition, body=body)

    def _parse_print(self) -> PrintStmt:
        self._eat(TokenType.PRINT)
        self._eat(TokenType.LPAREN)
        value = self._parse_expr()
        self._eat(TokenType.RPAREN)
        if self._at(TokenType.NEWLINE):
            self._eat()
        return PrintStmt(value=value)

    def _parse_funcdef(self) -> FuncDef:
        self._eat(TokenType.DEF)
        name = self._eat(TokenType.ID).value
        self._eat(TokenType.LPAREN)
        params = []
        if not self._at(TokenType.RPAREN):
            params = self._parse_params()
        self._eat(TokenType.RPAREN)
        ret_type = None
        if self._at(TokenType.ARROW):
            self._eat(TokenType.ARROW)
            ret_type = self._eat().value
        self._eat(TokenType.COLON)
        if self._at(TokenType.NEWLINE):
            self._eat()
        body = self._parse_block()
        return FuncDef(name=name, params=params, return_type=ret_type, body=body)

    def _parse_params(self) -> list[tuple[str, str]]:
        params = []
        name = self._eat(TokenType.ID).value
        self._eat(TokenType.COLON)
        ptype = self._eat().value
        params.append((name, ptype))
        while self._at(TokenType.COMMA):
            self._eat(TokenType.COMMA)
            name = self._eat(TokenType.ID).value
            self._eat(TokenType.COLON)
            ptype = self._eat().value
            params.append((name, ptype))
        return params

    def _parse_return(self) -> ReturnStmt:
        self._eat(TokenType.RETURN)
        value = None
        if not self._at(TokenType.NEWLINE, TokenType.EOF):
            value = self._parse_expr()
        if self._at(TokenType.NEWLINE):
            self._eat()
        return ReturnStmt(value=value)

    def _parse_expr_stmt(self) -> ExprStmt:
        expr = self._parse_expr()
        if self._at(TokenType.NEWLINE):
            self._eat()
        return ExprStmt(expr=expr)

    def _parse_block(self) -> list[Statement]:
        """Parse an indented block of statements."""
        stmts = []
        if self._at(TokenType.INDENT):
            self._eat(TokenType.INDENT)
            self._skip_newlines()
            while not self._at(TokenType.DEDENT, TokenType.EOF):
                stmts.append(self._parse_statement())
                self._skip_newlines()
            if self._at(TokenType.DEDENT):
                self._eat(TokenType.DEDENT)
        else:
            # Single-line block (no indent)
            stmts.append(self._parse_statement())
        return stmts

    # ── Expressions ──────────────────────────────────────────────────────

    def _parse_expr(self) -> Expr:
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        left = self._parse_additive()
        if self._at(TokenType.EQEQ, TokenType.NEQ, TokenType.LT,
                    TokenType.GT, TokenType.LEQ, TokenType.GEQ):
            op = self._eat().value
            right = self._parse_additive()
            return Comparison(op=op, left=left, right=right)
        return left

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self._at(TokenType.PLUS, TokenType.MINUS):
            op = self._eat().value
            right = self._parse_multiplicative()
            left = BinOp(op=op, left=left, right=right)
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_unary()
        while self._at(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self._eat().value
            right = self._parse_unary()
            left = BinOp(op=op, left=left, right=right)
        return left

    def _parse_unary(self) -> Expr:
        if self._at(TokenType.MINUS):
            op = self._eat().value
            operand = self._parse_unary()
            return UnaryOp(op=op, operand=operand)
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        tok = self._peek()

        if tok.type == TokenType.INT:
            self._eat()
            return IntLit(value=tok.value)
        elif tok.type == TokenType.FLOAT:
            self._eat()
            return FloatLit(value=tok.value)
        elif tok.type == TokenType.BOOL:
            self._eat()
            return BoolLit(value=tok.value)
        elif tok.type == TokenType.STRING:
            self._eat()
            return StringLit(value=tok.value)
        elif tok.type == TokenType.ID:
            self._eat()
            # Check for function call: ID "("
            if self._at(TokenType.LPAREN):
                self._eat(TokenType.LPAREN)
                args = []
                if not self._at(TokenType.RPAREN):
                    args.append(self._parse_expr())
                    while self._at(TokenType.COMMA):
                        self._eat(TokenType.COMMA)
                        args.append(self._parse_expr())
                self._eat(TokenType.RPAREN)
                return FuncCall(name=tok.value, args=args)
            return Identifier(name=tok.value)
        elif tok.type == TokenType.LPAREN:
            self._eat(TokenType.LPAREN)
            expr = self._parse_expr()
            self._eat(TokenType.RPAREN)
            return expr
        else:
            raise ParseError("Expected expression", tok)


# ── Public API ───────────────────────────────────────────────────────────────

def parse_source(source: str) -> Program:
    """Parse a MiniLang source string and return the AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()


def parse_tokens(tokens: list[Token]) -> Program:
    """Parse a token list and return the AST."""
    parser = Parser(tokens)
    return parser.parse_program()


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let width: int = 10
let height: int = 20
let area = width * height

def square(x: int) -> int:
    return x * x

if area > 100:
    print(area)
    print(square(5))
"""

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    print("═" * 60)
    print("  MiniLang Parser — AST Output")
    print("═" * 60)
    print(f"\nSource:\n{source}")
    print("─" * 60)
    ast = parse_source(source)
    print(pretty_print(ast))
