#!/usr/bin/env python3
"""
Chapter 17 — Model IR: a simple DSL for describing neural networks
and a graph intermediate representation.
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Neural Network Graph IR (High-Level Model Representation)
#
# Historical context: This is a simplified version of the graph IRs used
# by ONNX (2017), TensorFlow's graph_def, and TVM's Relay IR. The core
# idea: represent a neural network as a DAG of typed tensor operations
# with shape inference.
#
# Problem solved: Provide a structured representation for neural networks
# that can be:
# 1. Inspected (summary, visualization)
# 2. Optimized (operator fusion, dead node elimination in optimizer.py)
# 3. Lowered to executable code (codegen.py)
#
# How it works:
# - OpType enum defines supported operations (INPUT, MATMUL, ADD, RELU,
#   SOFTMAX, etc.).
# - IRNode stores: name, op type, input references (by name), output shape,
#   and extra attributes (like fused bias, activation type).
# - ModelGraph provides a builder API:
#     x = g.input("x", [1, 784])
#     h = g.matmul(x, w, "fc1")
#     h = g.relu(h, "relu1")
#   Each method creates an IRNode, infers the output shape from inputs,
#   and records it in the graph.
# - topo_order() returns nodes in dependency order (topological sort)
#   using DFS. This ensures every node is visited after its inputs.
#
#   MLP model definition:           Graph IR (DAG):
#
#   x = input([1,784])              x [1,784]
#   w1 = const([784,128])           │
#   b1 = const([1,128])             w1──►┌──────┐
#   h = matmul(x, w1)                    │MatMul│──┐
#   h = add(h, b1)               b1──►┌──┴──────┘  │ [1,128]
#   h = relu(h)                       │  Add  │────┘
#   w2 = const([128,10])              └───┬────┘
#   b2 = const([1,10])                ┌───┴───┐
#   h = matmul(h, w2)                 │ ReLU  │     [1,128]
#   h = add(h, b2)                    └───┬───┘
#   out = softmax(h)           w2──►┌─────┴────┐
#                                   │  MatMul  │──┐
#                            b2──►┌─┴──────────┘  │ [1,10]
#                                 │    Add    │───┘
#                                 └────┬──────┘
#                                 ┌────┴──────┐
#                                 │  Softmax  │   [1,10]
#                                 └───────────┘
#
#   topo_order(): [x, w1, b1, fc1, add1, relu1, w2, b2, fc2, add2, softmax]
#
# The graph is the input to the optimization pipeline (optimizer.py)
# and then to code generation (codegen.py).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class OpType(Enum):
    INPUT = auto()
    MATMUL = auto()
    ADD = auto()
    RELU = auto()
    SIGMOID = auto()
    SOFTMAX = auto()
    CONV2D = auto()
    MAXPOOL = auto()
    FLATTEN = auto()
    BATCHNORM = auto()
    CONST = auto()


@dataclass
class TensorShape:
    dims: list[int]

    @property
    def size(self) -> int:
        result = 1
        for d in self.dims:
            result *= d
        return result

    def __repr__(self):
        return f"({', '.join(map(str, self.dims))})"


@dataclass
class IRNode:
    name: str
    op: OpType
    inputs: list[str] = field(default_factory=list)
    shape: Optional[TensorShape] = None
    attrs: dict = field(default_factory=dict)


class ModelGraph:
    """A computation graph for a neural network model."""

    def __init__(self, name: str = "model"):
        self.name = name
        self.nodes: dict[str, IRNode] = {}
        self._counter = 0

    def _next_name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def input(self, name: str, shape: list[int]) -> str:
        node = IRNode(name, OpType.INPUT, shape=TensorShape(shape))
        self.nodes[name] = node
        return name

    def const(self, name: str, shape: list[int]) -> str:
        node = IRNode(name, OpType.CONST, shape=TensorShape(shape))
        self.nodes[name] = node
        return name

    def matmul(self, a: str, b: str, name: str = "") -> str:
        name = name or self._next_name("matmul")
        sa = self.nodes[a].shape
        sb = self.nodes[b].shape
        out_shape = TensorShape([sa.dims[0], sb.dims[1]])
        node = IRNode(name, OpType.MATMUL, [a, b], out_shape)
        self.nodes[name] = node
        return name

    def add(self, a: str, b: str, name: str = "") -> str:
        name = name or self._next_name("add")
        node = IRNode(name, OpType.ADD, [a, b], self.nodes[a].shape)
        self.nodes[name] = node
        return name

    def relu(self, x: str, name: str = "") -> str:
        name = name or self._next_name("relu")
        node = IRNode(name, OpType.RELU, [x], self.nodes[x].shape)
        self.nodes[name] = node
        return name

    def softmax(self, x: str, name: str = "") -> str:
        name = name or self._next_name("softmax")
        node = IRNode(name, OpType.SOFTMAX, [x], self.nodes[x].shape)
        self.nodes[name] = node
        return name

    def topo_order(self) -> list[IRNode]:
        """Topological sort of the graph."""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for inp in self.nodes[name].inputs:
                visit(inp)
            order.append(self.nodes[name])

        for name in self.nodes:
            visit(name)
        return order

    def summary(self) -> str:
        lines = [f"Model: {self.name}", "=" * 40]
        for node in self.topo_order():
            inputs = ", ".join(node.inputs) if node.inputs else "—"
            lines.append(f"  {node.name:20s}  {node.op.name:12s}  "
                        f"{str(node.shape):16s}  inputs=[{inputs}]")
        return "\n".join(lines)
