#!/usr/bin/env python3
"""
Chapter 15 — Mixed precision: assign different precisions per layer.

Usage:
    python mixed_precision.py
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

from quantize import quantize_symmetric, dequantize_symmetric

console = Console()


@dataclass
class Layer:
    name: str
    weights: np.ndarray
    sensitivity: float = 0.0  # how much accuracy drops with INT8


def measure_sensitivity(layer: Layer) -> float:
    """
    Measure quantization sensitivity by comparing FP32 vs INT8 output
    on random input.
    """
    np.random.seed(42)
    x = np.random.randn(1, layer.weights.shape[0]).astype(np.float32)

    # FP32 output
    y_fp = x @ layer.weights

    # INT8 output
    w_q, scale = quantize_symmetric(layer.weights)
    w_deq = dequantize_symmetric(w_q, scale)
    y_q = x @ w_deq

    # Relative error
    return float(np.mean(np.abs(y_fp - y_q)) / (np.mean(np.abs(y_fp)) + 1e-8))


def assign_precision(layers: list[Layer], threshold: float = 0.05) -> dict[str, str]:
    """
    Assign FP32 to sensitive layers, INT8 to the rest.
    """
    assignments = {}
    for layer in layers:
        layer.sensitivity = measure_sensitivity(layer)
        if layer.sensitivity > threshold:
            assignments[layer.name] = "FP32"
        else:
            assignments[layer.name] = "INT8"
    return assignments


def demo():
    console.print("\n[bold]═══ Mixed Precision Assignment ═══[/]\n")
    np.random.seed(42)

    # Simulate layers with different weight distributions
    layers = [
        Layer("embed",  np.random.randn(128, 64).astype(np.float32) * 0.01),
        Layer("attn_q", np.random.randn(64, 64).astype(np.float32) * 0.1),
        Layer("attn_k", np.random.randn(64, 64).astype(np.float32) * 0.1),
        Layer("attn_v", np.random.randn(64, 64).astype(np.float32) * 0.1),
        Layer("ffn1",   np.random.randn(64, 256).astype(np.float32) * 0.3),
        Layer("ffn2",   np.random.randn(256, 64).astype(np.float32) * 0.3),
        Layer("head",   np.random.randn(64, 10).astype(np.float32) * 1.0),
    ]

    assignments = assign_precision(layers, threshold=0.02)

    table = Table(title="Mixed Precision Assignments", box=box.ROUNDED)
    table.add_column("Layer")
    table.add_column("Shape")
    table.add_column("Weight Range")
    table.add_column("Sensitivity")
    table.add_column("Precision")

    total_fp32 = 0
    total_int8 = 0
    for layer in layers:
        prec = assignments[layer.name]
        style = "red" if prec == "FP32" else "green"
        if prec == "FP32":
            total_fp32 += layer.weights.nbytes
        else:
            total_int8 += layer.weights.size  # 1 byte per element
        table.add_row(
            layer.name,
            str(layer.weights.shape),
            f"[{layer.weights.min():.2f}, {layer.weights.max():.2f}]",
            f"{layer.sensitivity:.4f}",
            f"[{style}]{prec}[/]",
        )
    console.print(table)

    original = sum(l.weights.nbytes for l in layers)
    mixed = total_fp32 + total_int8
    console.print(f"\nOriginal size (all FP32): {original/1024:.1f} KB")
    console.print(f"Mixed precision size:     {mixed/1024:.1f} KB")
    console.print(f"Compression:              {original/mixed:.1f}×")


if __name__ == "__main__":
    demo()
