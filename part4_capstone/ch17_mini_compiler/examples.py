#!/usr/bin/env python3
"""
Chapter 17 — Example models for the mini compiler.

Usage:
    python examples.py
"""

from model_ir import ModelGraph
from compiler import compile_and_run
from rich.console import Console

console = Console()


def two_layer_mlp():
    """Two-layer MLP for MNIST-like classification."""
    g = ModelGraph("TwoLayerMLP")
    x = g.input("x", [1, 784])
    w1 = g.const("w1", [784, 256])
    b1 = g.const("b1", [1, 256])
    w2 = g.const("w2", [256, 128])
    b2 = g.const("b2", [1, 128])
    w3 = g.const("w3", [128, 10])
    b3 = g.const("b3", [1, 10])

    h = g.matmul(x, w1, "fc1")
    h = g.add(h, b1, "bias1")
    h = g.relu(h, "relu1")
    h = g.matmul(h, w2, "fc2")
    h = g.add(h, b2, "bias2")
    h = g.relu(h, "relu2")
    h = g.matmul(h, w3, "fc3")
    h = g.add(h, b3, "bias3")
    g.softmax(h, "output")
    return g


def simple_regression():
    """Simple linear regression: y = Wx + b."""
    g = ModelGraph("LinearRegression")
    x = g.input("x", [1, 10])
    w = g.const("w", [10, 1])
    b = g.const("b", [1, 1])

    h = g.matmul(x, w, "linear")
    g.add(h, b, "output")
    return g


if __name__ == "__main__":
    console.print("\n[bold]===== Example 1: Two-Layer MLP =====[/]")
    compile_and_run(two_layer_mlp())

    console.print("\n\n[bold]===== Example 2: Linear Regression =====[/]")
    compile_and_run(simple_regression())
