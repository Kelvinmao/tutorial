#!/usr/bin/env python3
"""
Chapter 10 — Mini Tensor Expression DSL.

A simplified version of TVM's tensor expression language. Lets you
define tensor computations declaratively, then lower them to loop nests.

Usage:
    python tensor_expression.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Tensor Expression DSL + Lowering to Loop Nests
#
# Historical context: Halide (2012, MIT) introduced the idea of separating
# the *algorithm* (what to compute) from the *schedule* (how to compute
# it — loop order, tiling, parallelism). TVM (2018, Chen et al.) brought
# this to deep learning. The tensor expression DSL lets you write:
#   C[i,j] = sum_k(A[i,k] * B[k,j])
# and the compiler decides how to implement the loops, tiling, etc.
#
# Problem solved: Tensor operations like matmul are defined mathematically
# as element-wise formulas with reductions. The compiler needs to lower
# these to concrete loop nests that can run on hardware. The loop order,
# nesting, and bounds must be correct.
#
# How it works:
# 1. DEFINE: User writes a declarative computation using the DSL:
#    - placeholder(shape, name) creates input tensor references
#    - reduce_axis(extent, name) creates a reduction loop variable
#    - compute(shape, lambda, name) defines the output formula
#    - reduce_sum(expr, axis) marks an accumulation over a reduction axis
#
#    Example: C[i,j] = sum_k(A[i,k] * B[k,j])
#      A = placeholder((M,K), "A")
#      B = placeholder((K,N), "B")
#      k = reduce_axis(K, "k")
#      C = compute((M,N), lambda i,j: reduce_sum(A[i,k]*B[k,j], k), "C")
#
# 2. LOWER TO LOOPS (lower_to_loops):
#    a) Create spatial loops for each output dimension (i, j).
#    b) If there's a reduction (sum over k), add an initialization
#       statement (C[i][j] = 0) and an inner reduction loop.
#    c) The innermost body is an accumulation: C[i][j] += A[i][k] * B[k][j]
#    d) Spatial loops wrap the reduction loops:
#         for i:       (spatial)
#           for j:     (spatial)
#             C = 0    (init)
#             for k:   (reduction)
#               C += A*B
#
#   Tensor expression:                Loop nest (lowered):
#
#   C[i,j] = ∑ₖ A[i,k] * B[k,j]      for i in 0..M:        # spatial
#                                       for j in 0..N:      # spatial
#   ┌─────────────────────┐              C[i][j] = 0     # init
#   │  Declarative       │              for k in 0..K:  # reduce
#   │  "what to compute" │                C[i][j] += A[i][k] * B[k][j]
#   └──────────┬──────────┘
#              │ lower_to_loops()    ┌─────────────────────┐
#              └─────────────►    │  Imperative          │
#                                   │  "how to compute"    │
#                                   └─────────────────────┘
#
#   This is the Halide/TVM separation of algorithm vs. schedule.
#   The loop nest can then be tiled, unrolled, vectorized (ch11).
#
# 3. The resulting loop nest can then be transformed by ch11 optimizations
#    (tiling, unrolling, vectorization, parallelization).
#
# This is the bridge between the high-level graph IR (ch08-09) and the
# low-level loop IR that maps to hardware (ch11-13).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Tensor Expression Nodes ─────────────────────────────────────────────────

@dataclass
class IterVar:
    """An iteration variable: represents a loop dimension."""
    name: str
    extent: int      # loop bound: for name in range(extent)
    is_reduce: bool = False  # True for reduction axes (e.g., k in matmul)

    def __repr__(self):
        kind = "reduce" if self.is_reduce else "spatial"
        return f"IterVar({self.name}, 0..{self.extent}, {kind})"


@dataclass
class TensorRef:
    """Reference to a tensor buffer."""
    name: str
    shape: tuple[int, ...]
    data: np.ndarray | None = None

    def __getitem__(self, indices):
        return TensorAccess(self, indices if isinstance(indices, tuple) else (indices,))

    def __repr__(self):
        return f"Tensor({self.name}, shape={self.shape})"


@dataclass
class TensorAccess:
    """An indexed access into a tensor: A[i, j]."""
    tensor: TensorRef
    indices: tuple

    def __mul__(self, other):
        return BinExpr("*", self, other)

    def __add__(self, other):
        return BinExpr("+", self, other)

    def __sub__(self, other):
        return BinExpr("-", self, other)

    def __repr__(self):
        idx_str = ", ".join(v.name if isinstance(v, IterVar) else str(v)
                            for v in self.indices)
        return f"{self.tensor.name}[{idx_str}]"


@dataclass
class BinExpr:
    """Binary expression in the tensor computation."""
    op: str
    left: object
    right: object

    def __mul__(self, other):
        return BinExpr("*", self, other)

    def __add__(self, other):
        return BinExpr("+", self, other)

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


@dataclass
class ReduceSum:
    """Sum reduction over a reduction axis."""
    expr: object           # expression to sum
    reduce_axis: IterVar   # the reduction iteration variable

    def __repr__(self):
        return f"sum({self.expr}, axis={self.reduce_axis.name})"


# ── Tensor Expression API ───────────────────────────────────────────────────

def placeholder(shape: tuple[int, ...], name: str = "tensor") -> TensorRef:
    """Create an input tensor placeholder."""
    return TensorRef(name=name, shape=shape)


def reduce_axis(extent: int, name: str = "k") -> IterVar:
    """Create a reduction axis."""
    return IterVar(name=name, extent=extent, is_reduce=True)


@dataclass
class ComputeOp:
    """A tensor computation defined by shape + lambda."""
    name: str
    shape: tuple[int, ...]
    iter_vars: list[IterVar]
    reduce_vars: list[IterVar]
    body: object  # the expression tree
    output: TensorRef = None

    def __repr__(self):
        axes = ", ".join(v.name for v in self.iter_vars)
        return f"Compute {self.name}[{axes}] = {self.body}"


def compute(shape: tuple[int, ...], fcompute: Callable, name: str = "out") -> ComputeOp:
    """
    Define a tensor computation.

    Parameters
    ----------
    shape : tuple
        Output tensor shape.
    fcompute : callable
        A function that takes iteration variables and returns an expression.
    name : str
        Name of the output tensor.

    Example
    -------
        k = reduce_axis(K, "k")
        C = compute((M, N), lambda i, j: reduce_sum(A[i,k] * B[k,j], k), "C")
    """
    # Create spatial iteration variables
    dim_names = ["i", "j", "l", "m"][:len(shape)]
    iter_vars = [IterVar(name=n, extent=s) for n, s in zip(dim_names, shape)]

    # Call fcompute with IterVar objects
    body = fcompute(*iter_vars)

    # Collect reduction variables
    reduce_vars = []
    _collect_reduce_vars(body, reduce_vars)

    output = TensorRef(name=name, shape=shape)
    op = ComputeOp(name=name, shape=shape, iter_vars=iter_vars,
                   reduce_vars=reduce_vars, body=body, output=output)
    return op


def reduce_sum(expr, axis: IterVar) -> ReduceSum:
    """Define a sum reduction."""
    return ReduceSum(expr=expr, reduce_axis=axis)


def _collect_reduce_vars(expr, result: list):
    """Walk expression to find reduction variables."""
    if isinstance(expr, ReduceSum):
        if expr.reduce_axis not in result:
            result.append(expr.reduce_axis)
        _collect_reduce_vars(expr.expr, result)
    elif isinstance(expr, BinExpr):
        _collect_reduce_vars(expr.left, result)
        _collect_reduce_vars(expr.right, result)


# ── Lowering: Tensor Expression → Loop Nest ────────────────────────────────

@dataclass
class Loop:
    """A loop in the loop nest."""
    var: IterVar
    body: list  # list of Loop or Statement


@dataclass
class Statement:
    """An assignment statement in the loop body."""
    target: str       # e.g., "C[i][j]"
    op: str           # "=" or "+="
    expr: str         # right-hand side as string


def lower_to_loops(compute_op: ComputeOp) -> Loop:
    """
    Lower a ComputeOp to an explicit loop nest.

    Spatial loops come first, then reduction loops (if any).
    """
    def expr_to_str(expr) -> str:
        if isinstance(expr, TensorAccess):
            idx = "][".join(v.name if isinstance(v, IterVar) else str(v)
                           for v in expr.indices)
            return f"{expr.tensor.name}[{idx}]"
        elif isinstance(expr, BinExpr):
            return f"({expr_to_str(expr.left)} {expr.op} {expr_to_str(expr.right)})"
        elif isinstance(expr, ReduceSum):
            return expr_to_str(expr.expr)
        elif isinstance(expr, (int, float)):
            return str(expr)
        elif isinstance(expr, IterVar):
            return expr.name
        return str(expr)

    # Build target string
    idx = "][".join(v.name for v in compute_op.iter_vars)
    target = f"{compute_op.name}[{idx}]"

    # Innermost body
    if compute_op.reduce_vars:
        # Has reduction: init to 0, then accumulate
        init_stmt = Statement(target, "=", "0.0")
        accum_stmt = Statement(target, "+=", expr_to_str(compute_op.body))

        # Build reduction loops
        inner = [accum_stmt]
        for rv in reversed(compute_op.reduce_vars):
            inner = [Loop(rv, inner)]

        body = [init_stmt] + inner
    else:
        body = [Statement(target, "=", expr_to_str(compute_op.body))]

    # Wrap in spatial loops
    result = body
    for sv in reversed(compute_op.iter_vars):
        result = [Loop(sv, result)]

    # Return outermost loop
    return result[0] if len(result) == 1 else result


def print_loop_nest(node, indent: int = 0):
    """Pretty-print a loop nest."""
    pad = "  " * indent
    if isinstance(node, Loop):
        kind = "reduce" if node.var.is_reduce else "spatial"
        console.print(f"{pad}[cyan]for[/] {node.var.name} in [0, {node.var.extent}):  "
                      f"[dim]# {kind}[/]")
        for child in node.body:
            print_loop_nest(child, indent + 1)
    elif isinstance(node, Statement):
        console.print(f"{pad}[green]{node.target} {node.op} {node.expr}[/]")
    elif isinstance(node, list):
        for child in node:
            print_loop_nest(child, indent)


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo():
    console.print("\n[bold]═══ Tensor Expression DSL ═══[/]\n")

    # === Example 1: Element-wise Add ===
    console.print("[bold cyan]Example 1: Element-wise Add[/]")
    M, N = 4, 4
    A = placeholder((M, N), "A")
    B = placeholder((M, N), "B")
    C_add = compute((M, N), lambda i, j: A[i, j] + B[i, j], "C")
    console.print(f"  Definition: {C_add}")
    loops = lower_to_loops(C_add)
    print_loop_nest(loops)

    # === Example 2: Matrix Multiply ===
    console.print(f"\n[bold cyan]Example 2: Matrix Multiply[/]")
    M, K, N = 4, 8, 4
    A = placeholder((M, K), "A")
    B = placeholder((K, N), "B")
    k = reduce_axis(K, "k")
    C_mm = compute((M, N), lambda i, j: reduce_sum(A[i, k] * B[k, j], k), "C")
    console.print(f"  Definition: {C_mm}")
    loops = lower_to_loops(C_mm)
    print_loop_nest(loops)

    # === Example 3: ReLU ===
    console.print(f"\n[bold cyan]Example 3: ReLU (conceptual)[/]")
    X = placeholder((M, N), "X")
    # Note: max/conditional not implemented in mini DSL, shown conceptually
    console.print("  for i in [0, M):")
    console.print("    for j in [0, N):")
    console.print("      Y[i][j] = max(0, X[i][j])")

    # === Verify with NumPy ===
    console.print(f"\n[bold cyan]Verification: MatMul via loop nest[/]")
    a_data = np.random.randn(4, 8)
    b_data = np.random.randn(8, 4)
    c_result = np.zeros((4, 4))

    # Execute the loop nest manually
    for i in range(4):
        for j in range(4):
            c_result[i][j] = 0.0
            for k_idx in range(8):
                c_result[i][j] += a_data[i][k_idx] * b_data[k_idx][j]

    np_result = a_data @ b_data
    max_diff = np.max(np.abs(c_result - np_result))
    console.print(f"  Max diff vs NumPy: {max_diff:.2e}")
    if max_diff < 1e-10:
        console.print(f"  [bold green]✓ Loop nest produces correct result![/]")


if __name__ == "__main__":
    demo()
