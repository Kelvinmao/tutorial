"""
Chapter 8 — Exercise stubs.

Fill in each function, then run:
    pytest -m exercise -k ch08 -v
"""

from __future__ import annotations

import numpy as np


def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """Exercise 8.1a — Sigmoid activation (forward).

    Compute  σ(x) = 1 / (1 + exp(-x)).

    Must be numerically stable for large positive *and* negative values.
    """
    raise NotImplementedError("Exercise 8.1a: implement sigmoid_forward")


def sigmoid_backward(
    sigmoid_output: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Exercise 8.1b — Sigmoid activation (backward / gradient).

    Given the *already-computed* sigmoid output σ(x) and the upstream
    gradient, return the gradient with respect to x:

        dL/dx = σ(x) · (1 − σ(x)) · grad_output
    """
    raise NotImplementedError("Exercise 8.1b: implement sigmoid_backward")
