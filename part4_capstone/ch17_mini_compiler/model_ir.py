#!/usr/bin/env python3
"""
Chapter 17 — Model IR: a simple DSL for describing neural networks
and a graph intermediate representation.
"""

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
