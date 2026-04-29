#!/usr/bin/env python3
"""
Chapter 8 — Reverse-mode automatic differentiation on a computation graph.

This is how PyTorch, TensorFlow, and JAX compute gradients!

Usage:
    python autodiff.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from comp_graph import CompGraph, Tensor, GraphNode

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def backward(loss_tensor: Tensor, graph: CompGraph):
    """
    Reverse-mode automatic differentiation.

    Starting from the loss, propagate gradients backward through the graph
    using the chain rule.

    Algorithm:
    1. Get nodes in reverse topological order
    2. Start with grad_loss = 1.0
    3. For each node, call its backward_fn to get input gradients
    4. Accumulate gradients for each tensor
    """
    # Initialize loss gradient
    loss_tensor.grad = np.ones_like(loss_tensor.data)

    # Traverse in reverse topological order
    topo = graph.topological_order()

    for node in reversed(topo):
        if node.output is None or node.output.grad is None:
            continue
        if node.backward_fn is None:
            continue

        # Compute gradients for this node's inputs
        grad_inputs = node.backward_fn(node.output.grad)

        # Accumulate gradients on input tensors
        for inp, grad in zip(node.inputs, grad_inputs):
            if inp.grad is None:
                inp.grad = np.zeros_like(inp.data)
            # Handle shape mismatch from broadcasting
            if inp.grad.shape != grad.shape:
                # Simple reshape/sum to match
                try:
                    inp.grad = inp.grad + grad.reshape(inp.grad.shape)
                except ValueError:
                    inp.grad = inp.grad + np.sum(grad).reshape(inp.grad.shape)
            else:
                inp.grad = inp.grad + grad


def numerical_gradient(func, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient for verification."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old_val = x[idx]
        x[idx] = old_val + eps
        f_plus = func()
        x[idx] = old_val - eps
        f_minus = func()
        x[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo_autodiff():
    console.print("\n[bold]═══ Reverse-Mode Automatic Differentiation ═══[/]\n")

    graph = CompGraph()

    # Simple computation: loss = sum((X @ W + b - target)^2) / n
    np.random.seed(42)
    X = graph.input(np.array([[1.0, 2.0], [3.0, 4.0]]), name="X")
    W = graph.input(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), name="W")
    b = graph.input(np.array([[0.1, 0.2, 0.3]]), name="b")
    target = graph.input(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]), name="target")

    # Forward pass
    h = graph.matmul(X, W)       # [2,2] @ [2,3] → [2,3]
    h = graph.add(h, b)          # [2,3] + [1,3] → [2,3]
    h = graph.relu(h)            # [2,3]
    loss = graph.mse_loss(h, target)  # scalar

    console.print(f"  Forward pass:")
    console.print(f"    X shape: {X.shape}")
    console.print(f"    W shape: {W.shape}")
    console.print(f"    b shape: {b.shape}")
    console.print(f"    Output shape: {h.shape}")
    console.print(f"    Loss: {loss.data:.6f}")

    # Backward pass
    backward(loss, graph)

    console.print(f"\n  Backward pass (gradients):")
    console.print(f"    ∂loss/∂W shape: {W.grad.shape}")
    console.print(f"    ∂loss/∂W:\n{W.grad}")
    console.print(f"\n    ∂loss/∂b shape: {b.grad.shape}")
    console.print(f"    ∂loss/∂b: {b.grad}")

    # Verify with numerical gradient
    console.print(f"\n  [dim]Verifying with numerical gradients...[/]")

    def compute_loss_for_W():
        h_val = X.data @ W.data + b.data
        h_val = np.maximum(0, h_val)
        return np.mean((h_val - target.data) ** 2)

    num_grad_W = numerical_gradient(compute_loss_for_W, W.data)
    max_diff = np.max(np.abs(W.grad - num_grad_W))
    console.print(f"    Max difference (autodiff vs numerical): {max_diff:.2e}")

    if max_diff < 1e-4:
        console.print(f"    [bold green]✓ Gradients match![/]")
    else:
        console.print(f"    [bold red]✗ Gradient mismatch![/]")

    # Show gradient flow table
    console.print()
    table = Table(title="Gradient Summary", box=box.ROUNDED)
    table.add_column("Tensor")
    table.add_column("Shape")
    table.add_column("Grad Norm")

    for name, tensor in [("X", X), ("W", W), ("b", b)]:
        if tensor.grad is not None:
            norm = np.linalg.norm(tensor.grad)
            table.add_row(name, str(tensor.shape), f"{norm:.6f}")
    console.print(table)


if __name__ == "__main__":
    demo_autodiff()
