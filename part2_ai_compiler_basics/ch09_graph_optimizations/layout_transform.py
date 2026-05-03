#!/usr/bin/env python3
"""
Chapter 9 — Layout transformation: convert between NCHW and NHWC.

Different hardware prefers different memory layouts for tensors.
This pass automatically inserts layout conversions.

Usage:
    python layout_transform.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Tensor Layout Transformation (NCHW ↔ NHWC)
#
# Historical context: The "data layout problem" emerged when different
# hardware targets favored different memory orderings:
# - NCHW (batch, channels, height, width): Used by NVIDIA cuDNN, PyTorch.
#   Channels are contiguous → good for GPU coalesced memory access.
# - NHWC (batch, height, width, channels): Used by TensorFlow, ARM CPUs.
#   Spatial positions are contiguous → good for CPU cache locality.
# AI compilers like TVM and XLA automatically pick the best layout per
# operator and insert minimal conversions at boundaries.
#
# Problem solved: A model trained in PyTorch (NCHW) may run faster on
# a CPU in NHWC layout. The compiler needs to:
# 1. Decide the optimal layout for each operator on the target hardware.
# 2. Insert layout conversion (transpose) ops at graph boundaries.
# 3. Minimize the number of conversions (each is expensive).
#
# How it works:
# - nchw_to_nhwc: numpy transpose with axes (0, 2, 3, 1)
#   Moves channels from position 1 to position 3.
# - nhwc_to_nchw: transpose with axes (0, 3, 1, 2)
#   Moves channels from position 3 to position 1.
#
#   NCHW memory layout:          NHWC memory layout:
#   (Batch, Channels, H, W)      (Batch, H, W, Channels)
#
#   For a 1×3×2×2 tensor (1 image, 3 channels, 2×2):
#
#   NCHW (channel-first):        NHWC (channel-last):
#   ┌─────────────────┐          ┌─────────────────┐
#   │ R R R R             │          │ R G B R G B       │
#   │ G G G G             │          │ R G B R G B       │
#   │ B B B B             │          └─────────────────┘
#   └─────────────────┘
#   All R pixels together         Pixel (R,G,B) together
#   → GPU coalesced access        → CPU spatial locality
#
#   transpose(0,2,3,1)            transpose(0,3,1,2)
#   NCHW ───────────► NHWC          NHWC ───────────► NCHW
#
# In a real AI compiler, a graph pass would:
# 1. Annotate each op with its preferred layout.
# 2. Find boundaries where adjacent ops disagree.
# 3. Insert transpose nodes at those boundaries.
# 4. Propagate layouts to minimize total conversions.
# ═══════════════════════════════════════════════════════════════════════════

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
