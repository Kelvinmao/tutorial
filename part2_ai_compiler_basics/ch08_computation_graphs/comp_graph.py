#!/usr/bin/env python3
"""
Chapter 8 — Computation Graph: build a DAG of tensor operations.

This is the core data structure reused in chapters 9, 12, and 17.

Usage:
    python comp_graph.py
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ── Tensor wrapper ───────────────────────────────────────────────────────────

@dataclass
class Tensor:
    """A tensor value with its computation graph node."""
    data: np.ndarray
    node: Optional[GraphNode] = None  # the node that produced this tensor
    grad: Optional[np.ndarray] = None  # gradient (filled during backward)
    name: str = ""

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        name = f" '{self.name}'" if self.name else ""
        return f"Tensor{name}(shape={self.shape})"


# ── Graph Node ───────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node in the computation graph."""
    op: str                                 # operation name
    inputs: list[Tensor] = field(default_factory=list)
    output: Optional[Tensor] = None
    # backward function: receives grad_output, returns list of grad_inputs
    backward_fn: Optional[Callable] = None
    attrs: dict = field(default_factory=dict)  # extra attributes
    node_id: int = 0

    def __repr__(self):
        input_shapes = [str(t.shape) for t in self.inputs]
        out_shape = str(self.output.shape) if self.output else "?"
        return f"Node({self.op}, inputs={input_shapes}, output={out_shape})"


# ── Computation Graph ────────────────────────────────────────────────────────

class CompGraph:
    """
    A computation graph that records tensor operations as a DAG.

    Supports:
    - Forward evaluation
    - Reverse-mode automatic differentiation
    - Topological ordering
    """

    def __init__(self):
        self.nodes: list[GraphNode] = []
        self._counter = 0

    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def _add_node(self, op: str, inputs: list[Tensor],
                  output_data: np.ndarray,
                  backward_fn=None, **attrs) -> Tensor:
        node = GraphNode(
            op=op,
            inputs=inputs,
            backward_fn=backward_fn,
            attrs=attrs,
            node_id=self._next_id(),
        )
        output = Tensor(data=output_data, node=node)
        node.output = output
        self.nodes.append(node)
        return output

    # ── Tensor creation ──────────────────────────────────────────────────

    def input(self, data: np.ndarray, name: str = "") -> Tensor:
        """Create an input tensor (leaf node)."""
        t = Tensor(data=np.array(data, dtype=np.float64), name=name)
        node = GraphNode(op="Input", output=t, node_id=self._next_id())
        node.output = t
        t.node = node
        self.nodes.append(node)
        return t

    def constant(self, data: np.ndarray, name: str = "") -> Tensor:
        """Create a constant tensor."""
        t = Tensor(data=np.array(data, dtype=np.float64), name=name)
        node = GraphNode(op="Const", output=t, node_id=self._next_id())
        t.node = node
        self.nodes.append(node)
        return t

    # ── Operations ───────────────────────────────────────────────────────

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication: C = A @ B"""
        result = a.data @ b.data

        def backward(grad_out):
            grad_a = grad_out @ b.data.T
            grad_b = a.data.T @ grad_out
            return [grad_a, grad_b]

        return self._add_node("MatMul", [a, b], result, backward)

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition with broadcasting."""
        result = a.data + b.data

        def backward(grad_out):
            grad_a = grad_out
            grad_b = grad_out
            # Handle broadcasting
            if a.data.shape != grad_out.shape:
                grad_a = np.sum(grad_out, axis=tuple(
                    range(len(grad_out.shape) - len(a.data.shape))))
            if b.data.shape != grad_out.shape:
                axes = []
                for i, (s1, s2) in enumerate(
                        zip(b.data.shape, grad_out.shape[-len(b.data.shape):])):
                    if s1 == 1 and s2 != 1:
                        axes.append(i)
                if axes:
                    grad_b = np.sum(grad_out, axis=tuple(axes), keepdims=True)
                else:
                    grad_b = grad_out
                if len(b.data.shape) < len(grad_out.shape):
                    extra = len(grad_out.shape) - len(b.data.shape)
                    grad_b = np.sum(grad_out, axis=tuple(range(extra)))
            return [grad_a, grad_b]

        return self._add_node("Add", [a, b], result, backward)

    def relu(self, x: Tensor) -> Tensor:
        """ReLU activation: max(0, x)"""
        result = np.maximum(0, x.data)

        def backward(grad_out):
            return [grad_out * (x.data > 0).astype(float)]

        return self._add_node("ReLU", [x], result, backward)

    def mse_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Mean Squared Error loss."""
        diff = pred.data - target.data
        result = np.array(np.mean(diff ** 2))

        def backward(grad_out):
            n = np.prod(pred.data.shape)
            grad = 2.0 * diff / n * grad_out
            return [grad, -grad]

        return self._add_node("MSELoss", [pred, target], result, backward)

    def multiply(self, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication."""
        result = a.data * b.data

        def backward(grad_out):
            return [grad_out * b.data, grad_out * a.data]

        return self._add_node("Mul", [a, b], result, backward)

    def sum(self, x: Tensor) -> Tensor:
        """Sum all elements."""
        result = np.array(np.sum(x.data))

        def backward(grad_out):
            return [np.ones_like(x.data) * grad_out]

        return self._add_node("Sum", [x], result, backward)

    # ── Graph utilities ──────────────────────────────────────────────────

    def topological_order(self) -> list[GraphNode]:
        """Return nodes in topological order (inputs first)."""
        visited = set()
        order = []

        def dfs(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for inp in node.inputs:
                if inp.node:
                    dfs(inp.node)
            order.append(node)

        for node in self.nodes:
            dfs(node)
        return order

    def print_graph(self):
        """Display the computation graph."""
        table = Table(title="Computation Graph", box=box.ROUNDED, show_lines=True)
        table.add_column("ID", style="dim", width=4)
        table.add_column("Op", style="bold cyan", width=12)
        table.add_column("Input Shapes", width=25)
        table.add_column("Output Shape", width=15)

        for node in self.topological_order():
            inp_shapes = ", ".join(str(t.shape) for t in node.inputs)
            out_shape = str(node.output.shape) if node.output else "?"
            name = node.output.name if node.output and node.output.name else ""
            op_str = f"{node.op}" + (f" '{name}'" if name else "")
            table.add_row(str(node.node_id), op_str, inp_shapes, out_shape)

        console.print(table)


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo_simple_mlp():
    """Build a computation graph for a simple MLP forward pass."""
    console.print("\n[bold]═══ Computation Graph: Simple MLP ═══[/]\n")

    graph = CompGraph()

    # Input: batch of 2 samples, 3 features each
    X = graph.input(np.random.randn(2, 3), name="X")
    console.print(f"  Input X: shape {X.shape}")

    # Layer 1: Linear (3 → 4)
    W1 = graph.input(np.random.randn(3, 4) * 0.1, name="W1")
    b1 = graph.input(np.zeros((1, 4)), name="b1")
    h1 = graph.matmul(X, W1)
    h1 = graph.add(h1, b1)
    h1 = graph.relu(h1)
    console.print(f"  After Layer 1: shape {h1.shape}")

    # Layer 2: Linear (4 → 2)
    W2 = graph.input(np.random.randn(4, 2) * 0.1, name="W2")
    b2 = graph.input(np.zeros((1, 2)), name="b2")
    out = graph.matmul(h1, W2)
    out = graph.add(out, b2)
    console.print(f"  Output: shape {out.shape}")

    # Loss
    target = graph.input(np.array([[1.0, 0.0], [0.0, 1.0]]), name="target")
    loss = graph.mse_loss(out, target)
    console.print(f"  Loss: {loss.data:.6f}")

    console.print()
    graph.print_graph()

    return graph, loss


if __name__ == "__main__":
    demo_simple_mlp()
