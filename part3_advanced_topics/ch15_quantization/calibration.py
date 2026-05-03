#!/usr/bin/env python3
"""
Chapter 15 — Calibration: find optimal quantization parameters
using representative data.

Usage:
    python calibration.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Quantization Calibration (Scale Selection)
#
# Historical context: NVIDIA introduced calibration as part of TensorRT
# (2017). The key insight: using max(|x|) for the quantization scale
# is suboptimal because outliers waste the limited integer range. Better
# methods use the actual data distribution to pick a scale that minimizes
# quantization error for the typical values.
#
# Problem solved: The quantization scale determines how floating-point
# values map to integers. A bad scale either clips too many values
# (losing large activations) or wastes resolution (too coarse for
# small values). Calibration finds the sweet spot using real data.
#
# Three calibration methods:
#
# 1. MinMax (simplest):
#    scale = max(|x|) / 127 over all calibration batches.
#    Pro: no clipping. Con: a single outlier wastes the entire range.
#
# 2. Percentile (99.9%):
#    scale = percentile(|x|, 99.9) / 127
#    Clips the top 0.1% as outliers. Pro: robust to outliers.
#    Con: fixed percentile may not be optimal.
#
# 3. MSE-optimal (best accuracy):
#    Search over candidate clip values and pick the one that minimizes
#    mean squared error between original and dequantized values.
#
#   Data distribution with outlier:
#
#   Freq
#    │██
#    │████                            MinMax clips here
#    │██████                                  │
#    │████████                                ▼
#    │████████████                   •      │
#    └───────────────────────────────► value
#                       ▲              outlier
#                 Percentile clips here
#                 (uses 99.9% of range,
#                  ignores outlier)
#
#   MSE-optimal: tries 100 clip values, picks the one
#   that minimizes ∑(x - dequant(quant(x)))²
#    Algorithm:
#      for clip_val in [max*1/100, max*2/100, ..., max]:
#        quantize with scale = clip_val / 127
#        compute MSE between original and dequantized
#      return scale with lowest MSE
#    Pro: directly minimizes reconstruction error.
#    Con: slower (100 evaluations per tensor).
#
# In practice, TensorRT uses KL-divergence calibration (similar to MSE
# but measures distribution distance), and INT8 calibration is run once
# on a representative dataset before deployment.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import numpy as np
from rich.console import Console

from quantize import quantize_symmetric, dequantize_symmetric

console = Console()


def calibrate_minmax(data_batches: list[np.ndarray]) -> float:
    """MinMax calibration: use global min/max across batches."""
    global_max = max(np.max(np.abs(batch)) for batch in data_batches)
    scale = global_max / 127
    return scale


def calibrate_percentile(data_batches: list[np.ndarray],
                         percentile: float = 99.9) -> float:
    """Percentile calibration: clip outliers before computing scale."""
    all_data = np.concatenate([b.flatten() for b in data_batches])
    clip_val = np.percentile(np.abs(all_data), percentile)
    scale = clip_val / 127
    return scale


def calibrate_mse(data_batches: list[np.ndarray],
                  n_steps: int = 100) -> float:
    """MSE calibration: find scale that minimizes reconstruction error."""
    all_data = np.concatenate([b.flatten() for b in data_batches])
    abs_max = np.max(np.abs(all_data))

    best_scale = abs_max / 127
    best_mse = float("inf")

    for i in range(1, n_steps + 1):
        clip_val = abs_max * i / n_steps
        scale = clip_val / 127
        clipped = np.clip(all_data, -clip_val, clip_val)
        q = np.clip(np.round(clipped / scale), -127, 127)
        deq = q * scale
        mse = np.mean((all_data - deq) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = scale

    return best_scale


def demo():
    console.print("\n[bold]═══ Quantization Calibration ═══[/]\n")
    np.random.seed(42)

    # Simulate activation data with some outliers
    batches = []
    for _ in range(10):
        batch = np.random.randn(64, 128).astype(np.float32) * 0.5
        # Add a few outliers
        outlier_idx = np.random.choice(batch.size, size=5)
        batch.flat[outlier_idx] = np.random.randn(5) * 5.0
        batches.append(batch)

    methods = {
        "MinMax": calibrate_minmax(batches),
        "Percentile (99.9%)": calibrate_percentile(batches, 99.9),
        "MSE-optimal": calibrate_mse(batches),
    }

    # Evaluate each
    all_data = np.concatenate([b.flatten() for b in batches])
    console.print(f"Data range: [{all_data.min():.2f}, {all_data.max():.2f}]")
    console.print(f"Data std:   {all_data.std():.4f}\n")

    for name, scale in methods.items():
        clipped = np.clip(all_data, -127 * scale, 127 * scale)
        q = np.clip(np.round(clipped / scale), -127, 127).astype(np.int8)
        deq = q.astype(np.float32) * scale
        mse = np.mean((all_data - deq) ** 2)
        mae = np.mean(np.abs(all_data - deq))
        console.print(f"[bold]{name}[/]")
        console.print(f"  Scale = {scale:.6f}, MSE = {mse:.6f}, MAE = {mae:.6f}")


if __name__ == "__main__":
    demo()
