"""
Chapter 12 — Exercise stubs.

Fill in each function, then run:
    pytest -m exercise -k ch12 -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from memory_planner import MemBuffer  # noqa: E402


def first_fit_buffer_sharing(
    buffers: dict[str, MemBuffer],
) -> tuple[dict[str, str], int, list[dict]]:
    """Exercise 12.1 — First-fit buffer allocation.

    Same contract as ``greedy_buffer_sharing`` in ``memory_planner.py``,
    but when looking for a reusable physical buffer, pick the **first**
    one whose lifetime has ended and whose size is large enough —
    instead of the *smallest* fitting one (best-fit).

    Returns
    -------
    mapping : dict[str, str]
        logical buffer name → physical buffer name
    reuses : int
        number of times an existing physical buffer was reused
    physical : list[dict]
        list of ``{"name": str, "size": int, "available_after": int}``

    Hint: the reference ``greedy_buffer_sharing`` is only ~20 lines.
    Copy it and change the selection criterion.
    """
    raise NotImplementedError("Exercise 12.1: implement first_fit_buffer_sharing")
