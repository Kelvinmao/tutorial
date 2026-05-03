"""
Chapter 6 — Exercise stubs.

Fill in each function, then run:
    pytest -m exercise -k ch06 -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))
sys.path.insert(0, os.path.dirname(__file__))

from ir_builder import IRInstr, Op  # noqa: E402


def algebraic_simplification(instructions: list[IRInstr]) -> list[IRInstr]:
    """Exercise 6.1 — Algebraic simplification pass.

    Scan *instructions* and apply these identity rules:

    * ``x + 0``  →  ``x``   (replace ADD with COPY)
    * ``x * 1``  →  ``x``   (replace MUL with COPY)
    * ``x * 0``  →  ``0``   (replace MUL with LOAD_CONST "0")
    * ``x - x``  →  ``0``   (replace SUB with LOAD_CONST "0")

    You will need a ``known_constants`` map (like ``constant_folding``
    does) so you can tell whether an operand is the literal ``0`` or ``1``.

    Return the simplified instruction list (same length is fine — just
    swap the op/src fields for matched instructions).
    """
    raise NotImplementedError("Exercise 6.1: implement algebraic_simplification")
