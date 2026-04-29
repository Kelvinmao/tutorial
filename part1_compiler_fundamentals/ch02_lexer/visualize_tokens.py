#!/usr/bin/env python3
"""
Chapter 2 — Visualize the token stream with color-coded output.

Usage:
    python visualize_tokens.py
    python visualize_tokens.py "let x = 42 + 3"
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from lexer import tokenize, TokenType

console = Console()

# Color scheme for different token categories
TOKEN_COLORS = {
    # Keywords
    TokenType.LET: "bold magenta",
    TokenType.IF: "bold magenta",
    TokenType.ELSE: "bold magenta",
    TokenType.WHILE: "bold magenta",
    TokenType.PRINT: "bold magenta",
    TokenType.DEF: "bold magenta",
    TokenType.RETURN: "bold magenta",
    # Types
    TokenType.TYPE_INT: "bold cyan",
    TokenType.TYPE_FLOAT: "bold cyan",
    TokenType.TYPE_BOOL: "bold cyan",
    # Literals
    TokenType.INT: "yellow",
    TokenType.FLOAT: "yellow",
    TokenType.BOOL: "green",
    TokenType.STRING: "green",
    # Identifiers
    TokenType.ID: "white",
    # Operators
    TokenType.PLUS: "bold red",
    TokenType.MINUS: "bold red",
    TokenType.STAR: "bold red",
    TokenType.SLASH: "bold red",
    TokenType.PERCENT: "bold red",
    TokenType.EQUALS: "bold blue",
    TokenType.EQEQ: "bold blue",
    TokenType.NEQ: "bold blue",
    TokenType.LT: "bold blue",
    TokenType.GT: "bold blue",
    TokenType.LEQ: "bold blue",
    TokenType.GEQ: "bold blue",
    # Delimiters
    TokenType.LPAREN: "dim",
    TokenType.RPAREN: "dim",
    TokenType.COLON: "dim",
    TokenType.COMMA: "dim",
    TokenType.ARROW: "dim",
    # Whitespace
    TokenType.NEWLINE: "dim italic",
    TokenType.INDENT: "dim italic",
    TokenType.DEDENT: "dim italic",
    TokenType.EOF: "dim",
}

TOKEN_CATEGORIES = {
    "Keyword": {TokenType.LET, TokenType.IF, TokenType.ELSE, TokenType.WHILE,
                TokenType.PRINT, TokenType.DEF, TokenType.RETURN},
    "Type": {TokenType.TYPE_INT, TokenType.TYPE_FLOAT, TokenType.TYPE_BOOL},
    "Literal": {TokenType.INT, TokenType.FLOAT, TokenType.BOOL, TokenType.STRING},
    "Identifier": {TokenType.ID},
    "Operator": {TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
                 TokenType.PERCENT, TokenType.EQUALS, TokenType.EQEQ, TokenType.NEQ,
                 TokenType.LT, TokenType.GT, TokenType.LEQ, TokenType.GEQ},
    "Delimiter": {TokenType.LPAREN, TokenType.RPAREN, TokenType.COLON,
                  TokenType.COMMA, TokenType.ARROW},
    "Structure": {TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EOF},
}

def get_category(ttype: TokenType) -> str:
    for cat, types in TOKEN_CATEGORIES.items():
        if ttype in types:
            return cat
    return "Unknown"

SAMPLE = """\
# MiniLang sample
let width: int = 10
let height: float = 20.5
let area = width * height
if area > 100:
    print(area)
"""

def visualize(source: str):
    tokens = tokenize(source)

    # ── Token table ──
    table = Table(title="Token Stream", box=box.HEAVY_EDGE,
                  show_lines=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", width=12)
    table.add_column("Value", width=20)
    table.add_column("Category", width=12)
    table.add_column("Location", style="dim", width=8)

    for i, tok in enumerate(tokens):
        color = TOKEN_COLORS.get(tok.type, "white")
        cat = get_category(tok.type)
        val_repr = repr(tok.value) if isinstance(tok.value, str) else str(tok.value)
        table.add_row(
            str(i),
            f"[{color}]{tok.type.name}[/]",
            f"[{color}]{val_repr}[/]",
            cat,
            f"L{tok.line}:{tok.col}",
        )

    console.print(f"\n[bold]Source code:[/]\n{source}")
    console.print(table)

    # ── Statistics ──
    stats = Table(title="Token Statistics", box=box.ROUNDED)
    stats.add_column("Category", style="bold")
    stats.add_column("Count", justify="right")
    for cat in TOKEN_CATEGORIES:
        count = sum(1 for t in tokens if t.type in TOKEN_CATEGORIES[cat])
        if count > 0:
            stats.add_row(cat, str(count))
    stats.add_row("[bold]Total[/]", f"[bold]{len(tokens)}[/]")
    console.print(stats)

    # ── Colored source reconstruction ──
    console.print("\n[bold]Color-coded source reconstruction:[/]")
    text = Text()
    for tok in tokens:
        color = TOKEN_COLORS.get(tok.type, "white")
        if tok.type == TokenType.NEWLINE:
            text.append("\n")
        elif tok.type in (TokenType.INDENT, TokenType.DEDENT, TokenType.EOF):
            continue
        else:
            text.append(str(tok.value), style=color)
            text.append(" ")
    console.print(text)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    visualize(source)
