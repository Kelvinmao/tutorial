#!/usr/bin/env python3
"""
Chapter 15 — Calibration: find optimal quantization parameters
using representative data.

Usage:
    python calibration.py
"""

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
