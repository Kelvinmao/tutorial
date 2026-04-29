#!/usr/bin/env python3
"""
Chapter 10 — Visualize iteration domains and loop nests.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))

from visualization import plot_heatmap, render_dot


def visualize_iteration_domain():
    """Visualize the 2D iteration domain of a matrix multiply."""
    M, N = 16, 16

    # Create iteration domain heatmap
    # Each cell (i,j) shows how many times it's computed
    domain = np.ones((M, N))
    plot_heatmap(domain, title="Spatial Iteration Domain (M=16, N=16)",
                 xlabel="j (columns)", ylabel="i (rows)",
                 filename="iteration_domain", output_dir="output",
                 cmap="Blues")

    # Show access pattern: which elements of A are accessed for C[0,:]
    K = 8
    access_A = np.zeros((M, K))
    for j in range(N):  # computing C[0, 0..N]
        for k in range(K):
            access_A[0][k] += 1
    plot_heatmap(access_A, title="A access pattern for C[row=0, :]",
                 xlabel="k", ylabel="i (row of A)",
                 filename="access_pattern_A", output_dir="output",
                 cmap="Oranges")


def visualize_loop_nest_as_tree():
    """Render a loop nest as a tree using graphviz."""
    dot = '''digraph LoopNest {
  rankdir=TB;
  node [shape=box fontname="Courier" fontsize=11 style=filled];

  root [label="MatMul C[M,N] = A[M,K] @ B[K,N]" fillcolor="#d4e6f1"];
  loop_i [label="for i in [0, M)\\nspatial" fillcolor="#d5f5e3"];
  loop_j [label="for j in [0, N)\\nspatial" fillcolor="#d5f5e3"];
  init [label="C[i][j] = 0" fillcolor="#fadbd8"];
  loop_k [label="for k in [0, K)\\nreduce" fillcolor="#fdebd0"];
  body [label="C[i][j] += A[i][k] * B[k][j]" fillcolor="#fadbd8"];

  root -> loop_i;
  loop_i -> loop_j;
  loop_j -> init;
  loop_j -> loop_k;
  loop_k -> body;
}'''
    render_dot(dot, filename="loop_nest_tree", output_dir="output")
    print("Done! Check output/ directory.")


if __name__ == "__main__":
    visualize_iteration_domain()
    visualize_loop_nest_as_tree()
