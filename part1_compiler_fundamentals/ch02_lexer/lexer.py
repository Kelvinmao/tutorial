#!/usr/bin/env python3
"""
Chapter 2 — A hand-written lexer for MiniLang.

This is the foundation tokenizer reused in chapters 3–7.

Usage:
    python lexer.py                         # tokenize sample code
    python -c "from lexer import tokenize; print(tokenize('let x = 1 + 2'))"
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


# ── Token Types ──────────────────────────────────────────────────────────────

class TokenType(Enum):
    # Literals
    INT      = auto()
    FLOAT    = auto()
    BOOL     = auto()
    STRING   = auto()

    # Identifiers & keywords
    ID       = auto()
    LET      = auto()
    IF       = auto()
    ELSE     = auto()
    WHILE    = auto()
    PRINT    = auto()
    DEF      = auto()
    RETURN   = auto()

    # Type names (used as keywords)
    TYPE_INT   = auto()
    TYPE_FLOAT = auto()
    TYPE_BOOL  = auto()

    # Operators
    PLUS     = auto()   # +
    MINUS    = auto()   # -
    STAR     = auto()   # *
    SLASH    = auto()   # /
    PERCENT  = auto()   # %
    EQUALS   = auto()   # =
    EQEQ     = auto()   # ==
    NEQ      = auto()   # !=
    LT       = auto()   # <
    GT       = auto()   # >
    LEQ      = auto()   # <=
    GEQ      = auto()   # >=

    # Delimiters
    LPAREN   = auto()   # (
    RPAREN   = auto()   # )
    COLON    = auto()   # :
    COMMA    = auto()   # ,
    ARROW    = auto()   # ->
    NEWLINE  = auto()
    INDENT   = auto()
    DEDENT   = auto()

    # Special
    EOF      = auto()
    UNKNOWN  = auto()


# Keywords lookup
KEYWORDS = {
    "let": TokenType.LET,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "print": TokenType.PRINT,
    "def": TokenType.DEF,
    "return": TokenType.RETURN,
    "int": TokenType.TYPE_INT,
    "float": TokenType.TYPE_FLOAT,
    "bool": TokenType.TYPE_BOOL,
    "true": TokenType.BOOL,
    "false": TokenType.BOOL,
}


# ── Token ────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    type: TokenType
    value: str | int | float | bool
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


# ── Lexer ────────────────────────────────────────────────────────────────────

class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"Lexer error at L{line}:{col}: {message}")
        self.line = line
        self.col = col


class Lexer:
    """
    Hand-written scanner for MiniLang.

    Consumes raw source text and yields Token objects.  Tracks line/column
    for error reporting.  Handles indentation-based blocks (simplified).
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []
        self.indent_stack = [0]  # track indentation levels

    # ── Character helpers ────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return "\0"

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _match(self, expected: str) -> bool:
        if self._peek() == expected:
            self._advance()
            return True
        return False

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _emit(self, ttype: TokenType, value, line: int, col: int):
        self.tokens.append(Token(ttype, value, line, col))

    # ── Scanners for multi-character tokens ──────────────────────────────

    def _scan_number(self) -> Token:
        start = self.pos
        line, col = self.line, self.col
        while not self._at_end() and self._peek().isdigit():
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            self._advance()  # consume '.'
            while not self._at_end() and self._peek().isdigit():
                self._advance()
            text = self.source[start:self.pos]
            return Token(TokenType.FLOAT, float(text), line, col)
        text = self.source[start:self.pos]
        return Token(TokenType.INT, int(text), line, col)

    def _scan_word(self) -> Token:
        start = self.pos
        line, col = self.line, self.col
        while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        text = self.source[start:self.pos]
        ttype = KEYWORDS.get(text, TokenType.ID)
        value: str | bool = text
        if text == "true":
            value = True
        elif text == "false":
            value = False
        return Token(ttype, value, line, col)

    def _scan_string(self) -> Token:
        line, col = self.line, self.col
        quote = self._advance()  # consume opening quote
        chars = []
        while not self._at_end() and self._peek() != quote:
            if self._peek() == "\\":
                self._advance()
                esc = self._advance()
                escape_map = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
                chars.append(escape_map.get(esc, esc))
            else:
                chars.append(self._advance())
        if self._at_end():
            raise LexerError("Unterminated string", line, col)
        self._advance()  # consume closing quote
        return Token(TokenType.STRING, "".join(chars), line, col)

    def _skip_comment(self):
        while not self._at_end() and self._peek() != "\n":
            self._advance()

    # ── Handle indentation ───────────────────────────────────────────────

    def _handle_indentation(self):
        """Count leading spaces at the start of a line and emit INDENT/DEDENT."""
        spaces = 0
        while not self._at_end() and self._peek() == " ":
            self._advance()
            spaces += 1
        # Skip blank lines and comment-only lines
        if self._at_end() or self._peek() == "\n" or self._peek() == "#":
            return
        current = self.indent_stack[-1]
        if spaces > current:
            self.indent_stack.append(spaces)
            self._emit(TokenType.INDENT, spaces, self.line, 1)
        else:
            while spaces < self.indent_stack[-1]:
                self.indent_stack.pop()
                self._emit(TokenType.DEDENT, spaces, self.line, 1)

    # ── Main tokenize loop ──────────────────────────────────────────────

    def tokenize(self) -> list[Token]:
        """Scan the entire source and return a list of tokens."""
        at_line_start = True

        while not self._at_end():
            if at_line_start:
                self._handle_indentation()
                at_line_start = False
                if self._at_end():
                    break

            ch = self._peek()
            line, col = self.line, self.col

            # Whitespace (non-newline)
            if ch == " " or ch == "\t":
                self._advance()
                continue

            # Newline
            if ch == "\n":
                self._advance()
                self._emit(TokenType.NEWLINE, "\\n", line, col)
                at_line_start = True
                continue

            # Comments
            if ch == "#":
                self._skip_comment()
                continue

            # Numbers
            if ch.isdigit():
                self.tokens.append(self._scan_number())
                continue

            # Words (identifiers and keywords)
            if ch.isalpha() or ch == "_":
                self.tokens.append(self._scan_word())
                continue

            # Strings
            if ch in ('"', "'"):
                self.tokens.append(self._scan_string())
                continue

            # Two-character operators
            self._advance()
            if ch == "=" and self._match("="):
                self._emit(TokenType.EQEQ, "==", line, col)
            elif ch == "!" and self._match("="):
                self._emit(TokenType.NEQ, "!=", line, col)
            elif ch == "<" and self._match("="):
                self._emit(TokenType.LEQ, "<=", line, col)
            elif ch == ">" and self._match("="):
                self._emit(TokenType.GEQ, ">=", line, col)
            elif ch == "-" and self._match(">"):
                self._emit(TokenType.ARROW, "->", line, col)
            # Single-character operators and delimiters
            elif ch == "+": self._emit(TokenType.PLUS, "+", line, col)
            elif ch == "-": self._emit(TokenType.MINUS, "-", line, col)
            elif ch == "*": self._emit(TokenType.STAR, "*", line, col)
            elif ch == "/": self._emit(TokenType.SLASH, "/", line, col)
            elif ch == "%": self._emit(TokenType.PERCENT, "%", line, col)
            elif ch == "=": self._emit(TokenType.EQUALS, "=", line, col)
            elif ch == "<": self._emit(TokenType.LT, "<", line, col)
            elif ch == ">": self._emit(TokenType.GT, ">", line, col)
            elif ch == "(": self._emit(TokenType.LPAREN, "(", line, col)
            elif ch == ")": self._emit(TokenType.RPAREN, ")", line, col)
            elif ch == ":": self._emit(TokenType.COLON, ":", line, col)
            elif ch == ",": self._emit(TokenType.COMMA, ",", line, col)
            else:
                self._emit(TokenType.UNKNOWN, ch, line, col)

        # Emit remaining DEDENTs
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self._emit(TokenType.DEDENT, 0, self.line, self.col)

        self._emit(TokenType.EOF, "", self.line, self.col)
        return self.tokens


# ── Convenience function ────────────────────────────────────────────────────

def tokenize(source: str) -> list[Token]:
    """Tokenize a source string and return the token list."""
    return Lexer(source).tokenize()


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE_PROGRAM = """\
# Compute the area of a rectangle
let width: int = 10
let height: int = 20
let area = width * height
if area > 100:
    print(area)
"""

if __name__ == "__main__":
    print("═" * 60)
    print("  MiniLang Lexer — Token Stream")
    print("═" * 60)
    print(f"\nSource:\n{SAMPLE_PROGRAM}")
    print("─" * 60)
    tokens = tokenize(SAMPLE_PROGRAM)
    for tok in tokens:
        print(f"  {tok}")
    print(f"\nTotal tokens: {len(tokens)}")
