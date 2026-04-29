#!/usr/bin/env python3
"""
Chapter 15 — Visualize quantization effects.

Usage:
    python visualize_quantization.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantize import quantize_symmetric, dequantize_symmetric


def main():
    np.random.seed(42)
    weights = np.random.randn(1000).astype(np.float32) * 0.5

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Distribution before/after
    w_q, scale = quantize_symmetric(weights)
    w_deq = dequantize_symmetric(w_q, scale)

    axes[0, 0].hist(weights, bins=50, alpha=0.7, label="FP32", color="blue")
    axes[0, 0].hist(w_deq, bins=50, alpha=0.7, label="INT8", color="red")
    axes[0, 0].set_title("Weight Distribution: FP32 vs INT8")
    axes[0, 0].legend()

    # 2. Quantization error
    error = weights - w_deq
    axes[0, 1].hist(error, bins=50, color="orange")
    axes[0, 1].set_title(f"Quantization Error (MAE={np.mean(np.abs(error)):.4f})")

    # 3. Scatter: original vs quantized
    axes[1, 0].scatter(weights, w_deq, s=1, alpha=0.5)
    axes[1, 0].plot([-2, 2], [-2, 2], "r--", linewidth=1)
    axes[1, 0].set_xlabel("FP32")
    axes[1, 0].set_ylabel("INT8 (dequantized)")
    axes[1, 0].set_title("FP32 vs INT8 Values")

    # 4. Error vs bit width
    bit_widths = [2, 3, 4, 5, 6, 7, 8]
    errors = []
    for bits in bit_widths:
        wq, s = quantize_symmetric(weights, bits=bits)
        wd = dequantize_symmetric(wq, s)
        errors.append(np.mean(np.abs(weights - wd)))
    axes[1, 1].bar(bit_widths, errors, color="green", alpha=0.8)
    axes[1, 1].set_xlabel("Bit Width")
    axes[1, 1].set_ylabel("Mean Abs Error")
    axes[1, 1].set_title("Error vs Bit Width")

    fig.tight_layout()
    fig.savefig("quantization_analysis.png", dpi=120)
    print("Saved → quantization_analysis.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
