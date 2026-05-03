#!/usr/bin/env python3
"""
Chapter 15 — INT8 quantization: symmetric and asymmetric.

Usage:
    python quantize.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: INT8 Quantization (Symmetric + Asymmetric)
#
# Historical context: Quantization for neural networks was popularized
# by Jacob et al. (Google, 2018, "Quantization and Training of Neural
# Networks for Efficient Integer-Arithmetic-Only Inference"). It enables
# running models on edge devices (phones, microcontrollers) with limited
# compute and memory. TensorRT, TFLite, and ONNX Runtime all support it.
#
# Problem solved: Neural network weights and activations are stored as
# 32-bit floats (4 bytes each). INT8 quantization reduces this to 1 byte,
# giving 4× compression and enabling faster integer arithmetic.
#
# ALGORITHM 1 — Symmetric Quantization:
#   scale = max(|x|) / 127
#   x_q = clamp(round(x / scale), -128, 127)  → int8
#   x_deq = x_q * scale                        → float32
#
# Key property: zero maps to zero (zero_point = 0). Simple but wastes
# range when the data isn't centered around zero.
#
#   Float range:    [-2.0 ......... 0 ......... +2.0]
#   Symmetric INT8: [-128 ........ 0 ........ +127]
#   scale = 2.0/127 = 0.0157
#
#   float:  -1.5  →  round(-1.5/0.0157) = round(-95.5) = -96  → int8
#   deq:    -96 * 0.0157 = -1.507  (error: 0.007)
#
# ALGORITHM 2 — Asymmetric Quantization:
#   scale = (x_max - x_min) / 255
#   zero_point = round(-x_min / scale)
#   x_q = clamp(round(x / scale + zero_point), 0, 255)  → uint8
#   x_deq = (x_q - zero_point) * scale                   → float32
#
# Key property: Uses the full [0, 255] range even if x_min ≠ -x_max.
# Better accuracy for ReLU outputs (which are all ≥0).
#
#   ReLU output:     [0.0 .................. +6.0]
#   Asymmetric:      [0 .................... 255]
#   scale = 6.0/255 = 0.0235,  zero_point = 0
#
#   vs Symmetric:    [-128 ... 0 .......... 127]
#   scale = 6.0/127 = 0.0472   (2× coarser! half range wasted)
#
# Quantized matmul: C_float = (A_q @ B_q) * (scale_a * scale_b)
# The integer matmul uses int32 accumulation to avoid overflow,
# then rescales the result to float32.
#
# Tradeoff: 4× smaller, ~2× faster inference, but some accuracy loss
# (typically <1% top-1 accuracy on ImageNet with proper calibration).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def quantize_symmetric(x: np.ndarray, bits: int = 8) -> tuple:
    """Symmetric quantization: zero_point = 0, scale from max abs."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / qmax
    x_q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
    return x_q, scale


def dequantize_symmetric(x_q: np.ndarray, scale: float) -> np.ndarray:
    return x_q.astype(np.float32) * scale


def quantize_asymmetric(x: np.ndarray, bits: int = 8) -> tuple:
    """Asymmetric quantization: uses both zero_point and scale."""
    qmin, qmax = 0, 2 ** bits - 1
    x_min, x_max = np.min(x), np.max(x)
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = int(np.round(qmin - x_min / scale))
    zero_point = np.clip(zero_point, qmin, qmax)
    x_q = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint8)
    return x_q, scale, zero_point


def dequantize_asymmetric(x_q: np.ndarray, scale: float,
                          zero_point: int) -> np.ndarray:
    return (x_q.astype(np.float32) - zero_point) * scale


def quantized_matmul(A_q: np.ndarray, B_q: np.ndarray,
                     scale_a: float, scale_b: float) -> np.ndarray:
    """Integer matmul with scale recovery."""
    # Compute in int32 to avoid overflow
    C_int = A_q.astype(np.int32) @ B_q.astype(np.int32)
    return C_int.astype(np.float32) * (scale_a * scale_b)


def demo():
    console.print("\n[bold]═══ INT8 Quantization ═══[/]\n")
    np.random.seed(42)

    # Simulate a weight tensor
    weights = np.random.randn(4, 4).astype(np.float32) * 0.5

    console.print("[bold cyan]Original (FP32):[/]")
    console.print(weights.round(3))

    # Symmetric
    w_q, scale = quantize_symmetric(weights)
    w_deq = dequantize_symmetric(w_q, scale)
    err_sym = np.mean(np.abs(weights - w_deq))

    console.print(f"\n[bold cyan]Symmetric Quantized (INT8):[/]")
    console.print(f"  Scale = {scale:.6f}")
    console.print(f"  Quantized:\n{w_q}")
    console.print(f"  Dequantized:\n{w_deq.round(3)}")
    console.print(f"  Mean abs error: {err_sym:.6f}")

    # Asymmetric
    w_q2, scale2, zp = quantize_asymmetric(weights)
    w_deq2 = dequantize_asymmetric(w_q2, scale2, zp)
    err_asym = np.mean(np.abs(weights - w_deq2))

    console.print(f"\n[bold cyan]Asymmetric Quantized (UINT8):[/]")
    console.print(f"  Scale = {scale2:.6f}, Zero point = {zp}")
    console.print(f"  Mean abs error: {err_asym:.6f}")

    # Quantized matmul
    console.print(f"\n[bold cyan]Quantized MatMul:[/]")
    A = np.random.randn(4, 4).astype(np.float32) * 0.3
    B = np.random.randn(4, 4).astype(np.float32) * 0.5
    C_fp = A @ B

    A_q, sa = quantize_symmetric(A)
    B_q, sb = quantize_symmetric(B)
    C_q = quantized_matmul(A_q, B_q, sa, sb)

    err_matmul = np.mean(np.abs(C_fp - C_q))
    console.print(f"  FP32 result:\n{C_fp.round(3)}")
    console.print(f"  INT8 result:\n{C_q.round(3)}")
    console.print(f"  Mean abs error: {err_matmul:.6f}")

    # Summary
    table = Table(title="Comparison", box=box.ROUNDED)
    table.add_column("Method")
    table.add_column("Size", justify="right")
    table.add_column("Error", justify="right")
    table.add_row("FP32", f"{weights.nbytes} B", "0")
    table.add_row("INT8 symmetric", f"{w_q.nbytes} B", f"{err_sym:.6f}")
    table.add_row("UINT8 asymmetric", f"{w_q2.nbytes} B", f"{err_asym:.6f}")
    console.print(table)
    console.print(f"  Compression: {weights.nbytes / w_q.nbytes:.0f}× smaller")


if __name__ == "__main__":
    demo()
