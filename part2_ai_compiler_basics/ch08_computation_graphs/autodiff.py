#!/usr/bin/env python3
"""
Chapter 8 — Reverse-mode automatic differentiation on a computation graph.

This is how PyTorch, TensorFlow, and JAX compute gradients!

Usage:
    python autodiff.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Reverse-Mode Automatic Differentiation (Backpropagation)
#
# Historical context: Backpropagation was independently discovered multiple
# times — Linnainmaa (1970) described reverse accumulation, Werbos (1974)
# applied it to neural networks, and Rumelhart, Hinton & Williams (1986)
# popularized it. Every modern deep learning framework implements it.
# The algorithm is simply the chain rule applied in reverse topological
# order over a computation graph.
#
# Problem solved: Given a scalar loss L computed from parameters W through
# a chain of operations, compute ∂L/∂W for every parameter W. This is
# needed for gradient descent optimization of neural networks.
#
# Why reverse mode: For a function f: R^n → R (many inputs, one output),
# reverse mode computes ALL n partial derivatives in one backward pass.
# Forward mode would require n separate passes. Since neural networks
# have millions of parameters but one scalar loss, reverse mode wins.
#
# How it works:
# 1. Get nodes in topological order (inputs first, loss last).
# 2. Set the loss gradient to 1.0 (∂L/∂L = 1).
# 3. Walk nodes in REVERSE topological order (loss → inputs).
# 4. For each node, call its backward_fn(grad_output) to compute
#    the gradient of the loss w.r.t. each input:
#    - MatMul backward: ∂L/∂A = grad @ Bᵀ, ∂L/∂B = Aᵀ @ grad
#    - Add backward:    ∂L/∂A = grad, ∂L/∂B = grad
#    - ReLU backward:   ∂L/∂X = grad * (X > 0)
#    - MSE backward:    ∂L/∂pred = 2*(pred - target)/n
# 5. ACCUMULATE gradients on each input tensor (not replace — a tensor
#    may be used by multiple downstream ops).
#
#   Forward pass (left to right):        Backward pass (right to left):
#
#   x ──► MatMul ──► Add ──► ReLU ──► MSE ─► loss
#   w ──┘         b ──┘                    │
#                                          │ ∂L/∂L = 1.0
#                                          ▼
#   ∂L     ∂L        ∂L        ∂L        ∂L
#   ── ── ── ──── ── ──── ── ──── ── ──── ──
#   ∂x     ∂w        ∂b        ∂relu     ∂mse
#    │      │         │         │          │
#    ◄──────┼─MatMul──◄──Add───◄─ReLU────◄─MSE
#          grad@Bᵀ   grad      grad*(x>0)  2(p-t)/n
#
#   Key: each backward_fn receives ∂L/∂output, returns ∂L/∂inputs.
#   Gradients ACCUMULATE (+=) because one tensor may feed multiple ops.
#
# This implementation also includes numerical_gradient() for verification:
# it computes ∂f/∂x[i] ≈ (f(x+ε) - f(x-ε)) / 2ε. If the autodiff
# gradient matches the numerical gradient, the implementation is correct.
# ═══════════════════════════════════════════════════════════════════════════

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
