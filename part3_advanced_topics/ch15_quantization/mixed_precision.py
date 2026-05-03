#!/usr/bin/env python3
"""
Chapter 15 — Mixed precision: assign different precisions per layer.

Usage:
    python mixed_precision.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Mixed Precision Assignment via Sensitivity Analysis
#
# Historical context: Mixed precision was popularized by Micikevicius
# et al. (NVIDIA, 2018, "Mixed Precision Training") for training, and
# Hawq (Dong et al., 2019) for inference. The insight: not all layers
# are equally sensitive to quantization. Some (like the first and last
# layers) lose significant accuracy when quantized, while most middle
# layers are robust.
#
# Problem solved: Quantizing ALL layers to INT8 may cause unacceptable
# accuracy loss. Keeping ALL layers at FP32 gives no speedup. Mixed
# precision finds the optimal per-layer precision that maximizes
# compression while keeping accuracy loss below a threshold.
#
# How it works:
# 1. MEASURE SENSITIVITY: For each layer, compare FP32 output to
#    quantized (INT8 + dequantized) output on random input.
#    Sensitivity = mean(|y_fp32 - y_int8|) / mean(|y_fp32|)
#    High sensitivity = quantization changes the output a lot.
#
# 2. ASSIGN PRECISION: Compare each layer's sensitivity to a threshold.
#    - sensitivity > threshold → FP32 (keep full precision)
#    - sensitivity ≤ threshold → INT8 (safe to quantize)
#
#   Layer sensitivity analysis:
#
#   Layer     Sensitivity  Decision
#   ┌────────┬──────────────────────────────┬───────┐
#   │ embed  │ ██████████████████  0.12   │ FP32  │  ← sensitive!
#   ├────────┼──────────────────────────────┼───────┤
#   │ attn_q │ █████              0.03   │ INT8  │
#   ├────────┼──────────────────────────────┼───────┤
#   │ ffn1   │ ████              0.02   │ INT8  │
#   ├────────┼──────────────────────────────┼───────┤
#   │ ffn2   │ ██████             0.04   │ INT8  │
#   ├────────┼──────────────────────────────┼───────┤
#   │ head   │ ███████████████████ 0.15   │ FP32  │  ← sensitive!
#   └────────┴──────────────────────────────┴───────┘
#                               threshold=0.05 ──▲
#
#   Result: 60% INT8 + 40% FP32 → ~3× compression, <0.5% accuracy loss
#
# 3. RESULT: A per-layer precision map:
#    embed: FP32 (tiny weights, sensitive to quantization noise)
#    attn_q: INT8 (moderate weights, robust)
#    ffn1:   INT8 (large weights, robust)
#    head:   FP32 (large weights, sensitive — final classifier)
#
# Typical result: 70-80% of layers can be INT8, giving ~3× compression
# with <0.5% accuracy loss.
# ═══════════════════════════════════════════════════════════════════════════

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
