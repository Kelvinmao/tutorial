#!/usr/bin/env python3
"""
Chapter 11 — Visualize memory access patterns before/after tiling.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from visualization import plot_heatmap

def demo():
    N = 32

    # === Naive access pattern for B matrix during matmul ===
    # For each (i,j), we access B[0..K, j] — column-wise access
    naive_access = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                naive_access[k, j] += 1  # B[k,j] access count
    plot_heatmap(naive_access, title="B access pattern: Naive MatMul",
                 xlabel="Column (j)", ylabel="Row (k)",
                 filename="access_naive", output_dir="output", cmap="YlOrRd")

    # === Tiled access pattern ===
    tile = 8
    tiled_access_steps = []
    for ii in range(0, N, tile):
        for jj in range(0, N, tile):
            step_access = np.zeros((N, N))
            for kk in range(0, N, tile):
                for i in range(ii, min(ii + tile, N)):
                    for j in range(jj, min(jj + tile, N)):
                        for k in range(kk, min(kk + tile, N)):
                            step_access[k, j] += 1
            tiled_access_steps.append(step_access)

    # Show first tile's access pattern
    plot_heatmap(tiled_access_steps[0],
                 title=f"B access pattern: Tiled (tile={tile}, first block)",
                 xlabel="Column (j)", ylabel="Row (k)",
                 filename="access_tiled_first", output_dir="output", cmap="YlOrRd")

    # Show total tiled access (should match naive)
    total_tiled = sum(tiled_access_steps)
    plot_heatmap(total_tiled,
                 title=f"B access pattern: Tiled (total, tile={tile})",
                 xlabel="Column (j)", ylabel="Row (k)",
                 filename="access_tiled_total", output_dir="output", cmap="YlOrRd")

    # === Cache simulation ===
    console_msg = f"""
Memory Access Patterns saved to output/:
  - access_naive.png          Naive matmul: scattered column access on B
  - access_tiled_first.png    First tile block: localized access
  - access_tiled_total.png    All tiles combined: same total work

Key insight: tiling changes ACCESS ORDER, not total computation.
The tiled pattern keeps a small {tile}x{tile} block of B in cache,
reusing it across multiple accesses before moving to the next block.
"""
    print(console_msg)


if __name__ == "__main__":
    demo()
