#!/usr/bin/env python3
"""
Chapter 9 — Layout transformation: convert between NCHW and NHWC.

Different hardware prefers different memory layouts for tensors.
This pass automatically inserts layout conversions.

Usage:
    python layout_transform.py
"""

import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    """Convert [N, C, H, W] → [N, H, W, C]"""
    return np.transpose(tensor, (0, 2, 3, 1))


def nhwc_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """Convert [N, H, W, C] → [N, C, H, W]"""
    return np.transpose(tensor, (0, 3, 1, 2))


def demo_layout_transform():
    console.print("\n[bold]═══ Layout Transformation: NCHW ↔ NHWC ═══[/]\n")

    # Create sample tensor
    N, C, H, W = 1, 3, 4, 4
    nchw = np.arange(N * C * H * W, dtype=np.float32).reshape(N, C, H, W)

    console.print("[bold]NCHW layout[/] (channels first — PyTorch, NVIDIA GPUs):")
    console.print(f"  Shape: {nchw.shape}  (N={N}, C={C}, H={H}, W={W})")
    console.print(f"  Memory layout: channels are contiguous in memory")
    console.print(f"  Pixel (0,0) across channels: {nchw[0, :, 0, 0]}")
    console.print()

    nhwc = nchw_to_nhwc(nchw)
    console.print("[bold]NHWC layout[/] (channels last — TensorFlow, CPUs):")
    console.print(f"  Shape: {nhwc.shape}  (N={N}, H={H}, W={W}, C={C})")
    console.print(f"  Memory layout: spatial positions are contiguous")
    console.print(f"  Pixel (0,0) across channels: {nhwc[0, 0, 0, :]}")

    # Verify roundtrip
    roundtrip = nhwc_to_nchw(nhwc)
    assert np.array_equal(nchw, roundtrip), "Roundtrip failed!"
    console.print(f"\n[green]✓ Roundtrip NCHW → NHWC → NCHW: correct[/]")

    # === Performance comparison ===
    console.print("\n[bold cyan]Performance: Conv2D in NCHW vs NHWC[/]\n")

    # Simulate 3x3 convolution access patterns
    N, C, H, W = 1, 64, 56, 56
    nchw_data = np.random.randn(N, C, H, W).astype(np.float32)
    nhwc_data = nchw_to_nhwc(nchw_data)

    # NCHW: for each output pixel, access input across channels (strided access)
    t0 = time.perf_counter()
    for _ in range(10):
        result = np.sum(nchw_data[:, :, 1:H-1, 1:W-1], axis=1)
    nchw_time = time.perf_counter() - t0

    # NHWC: for each output pixel, channels are contiguous (better cache)
    t0 = time.perf_counter()
    for _ in range(10):
        result = np.sum(nhwc_data[:, 1:H-1, 1:W-1, :], axis=-1)
    nhwc_time = time.perf_counter() - t0

    table = Table(title="Layout Performance (simulated conv reduction)",
                  box=box.ROUNDED)
    table.add_column("Layout")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Notes")
    table.add_row("NCHW", f"{nchw_time*1000:.1f}",
                  "Better for GPU (coalesced memory)")
    table.add_row("NHWC", f"{nhwc_time*1000:.1f}",
                  "Better for CPU (cache-friendly)")
    console.print(table)

    console.print("\n[dim]AI compilers automatically choose the best layout for each target")
    console.print("and insert minimal conversions at graph boundaries.[/]")


if __name__ == "__main__":
    demo_layout_transform()
