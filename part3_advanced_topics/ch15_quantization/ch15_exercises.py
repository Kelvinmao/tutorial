"""
Chapter 15 — Exercise stubs.

Fill in each function, then run:
    pytest -m exercise -k ch15 -v
"""

from __future__ import annotations

import numpy as np


def quantize_int4(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Exercise 15.1a — Symmetric 4-bit quantization.

    Quantize *x* to 4-bit signed integers (range −7 … +7).

    Returns ``(x_q, scale)`` where::

        scale = max(|x|) / 7
        x_q   = clip(round(x / scale), -7, 7)   # stored as int8

    The returned ``x_q`` should have dtype ``np.int8``.
    """
    raise NotImplementedError("Exercise 15.1a: implement quantize_int4")


def dequantize_int4(x_q: np.ndarray, scale: float) -> np.ndarray:
    """Exercise 15.1b — Dequantize INT4 values back to float32.

    Returns ``x_q.astype(float32) * scale``.
    """
    raise NotImplementedError("Exercise 15.1b: implement dequantize_int4")
