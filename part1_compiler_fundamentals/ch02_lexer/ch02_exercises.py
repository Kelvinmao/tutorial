"""
Chapter 2 — Exercise stubs.

Fill in each function, then run:
    pytest -m exercise -k ch02 -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from lexer import Token, TokenType, KEYWORDS, Lexer  # noqa: E402


def tokenize_with_logic_ops(source: str) -> list[tuple[str, str]]:
    """Exercise 2.1 — Extend the lexer to recognise ``and``, ``or``, ``not``.

    Tokenize *source* and return a list of ``(type_name, value)`` pairs.
    The three new keywords must produce token types named ``"AND"``,
    ``"OR"``, and ``"NOT"``.  All other tokens use the standard
    ``TokenType`` member names (e.g. ``"LET"``, ``"INT"``, ``"ID"``).
    Include the final ``"EOF"`` token.

    Hint: you can temporarily patch ``KEYWORDS`` and post-process the
    token list, or subclass ``Lexer`` — whatever you like.

    >>> tokenize_with_logic_ops("true and false")  # doctest: +SKIP
    [('BOOL', 'true'), ('AND', 'and'), ('BOOL', 'false'), ..., ('EOF', '')]
    """
    raise NotImplementedError("Exercise 2.1: implement tokenize_with_logic_ops")
