#!/usr/bin/env python3
"""
Chapter 13 — Generate C code from loop nests, compile, and run.

Usage:
    python cpu_codegen.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: CPU Code Generation (Loop Nest → C Code Emission)
#
# Historical context: Code generation is the final compiler phase,
# translating the optimized IR into executable code. TVM generates C
# code for CPU targets, which is then compiled by GCC/Clang. This
# approach leverages decades of C compiler optimization and avoids
# the complexity of directly emitting machine code.
#
# Problem solved: We have an optimized loop nest (from ch10-11) and
# need to turn it into something that runs on a real CPU. Generating
# C code is practical because:
# 1. C compilers (GCC, Clang) handle register allocation, instruction
#    scheduling, and SIMD auto-vectorization.
# 2. C is portable across architectures.
# 3. The generated code is readable and debuggable.
#
# How it works:
# 1. gen_matmul_c() generates a complete C program as a string:
#    - Allocates matrices A, B, C with malloc/calloc
#    - Initializes A, B with random values
#    - Emits either naive (ijk) or tiled (blocked) loop nests
#    - Times the computation with clock_gettime
#    - Computes and prints GFLOPS
#
#   Loop nest → C code emission:
#
#   IR loop nest:                    Generated C code:
#
#   for i in 0..M (spatial)          for (int i=0; i<M; i++) {
#     for j in 0..N (spatial)          for (int j=0; j<N; j++) {
#       C[i][j] = 0  (init)              C[i*N+j] = 0.0;
#       for k in 0..K (reduce)           for (int k=0; k<K; k++) {
#         C[i][j] += A*B                   C[i*N+j] += A[i*K+k]
#                                                     * B[k*N+j];
#                                        }
#                                      }
#                                    }
#
#   Tiled variant adds 3 outer loops (ii,jj,kk) with stride=tile_size
#   and clamps inner bounds: min(ii+tile, M)
#
# 2. compile_and_run() writes the C code to a temp file, invokes
#    gcc -O2, runs the binary, and captures stdout.
#
# 3. The tiled version uses 6 nested loops (ii, jj, kk, i, j, k)
#    with bounds clamping (min(ii+tile, M)) to handle edge tiles.
#
# This is a simplified version of what TVM, Halide, and XLA do when
# targeting CPU backends.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import subprocess
import tempfile
import os
import textwrap
from rich.console import Console
from rich.syntax import Syntax

console = Console()


def gen_matmul_c(M: int, N: int, K: int, tiled: bool = False,
                 tile_size: int = 32) -> str:
    """Generate C code for matrix multiplication."""
    if tiled:
        loop_body = textwrap.dedent(f"""\
        // Tiled matrix multiply
        for (int ii = 0; ii < {M}; ii += {tile_size})
          for (int jj = 0; jj < {N}; jj += {tile_size})
            for (int kk = 0; kk < {K}; kk += {tile_size})
              for (int i = ii; i < ii+{tile_size} && i < {M}; i++)
                for (int j = jj; j < jj+{tile_size} && j < {N}; j++)
                  for (int k = kk; k < kk+{tile_size} && k < {K}; k++)
                    C[i*{N}+j] += A[i*{K}+k] * B[k*{N}+j];
        """)
    else:
        loop_body = textwrap.dedent(f"""\
        // Naive matrix multiply
        for (int i = 0; i < {M}; i++)
          for (int j = 0; j < {N}; j++)
            for (int k = 0; k < {K}; k++)
              C[i*{N}+j] += A[i*{K}+k] * B[k*{N}+j];
        """)

    return textwrap.dedent(f"""\
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <string.h>

    int main() {{
        const int M = {M}, N = {N}, K = {K};
        float *A = malloc(M * K * sizeof(float));
        float *B = malloc(K * N * sizeof(float));
        float *C = calloc(M * N, sizeof(float));

        // Initialize with small values
        srand(42);
        for (int i = 0; i < M*K; i++) A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K*N; i++) B[i] = (float)(rand() % 100) / 100.0f;

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

    {textwrap.indent(loop_body, '    ')}

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;

        printf("C[0][0] = %.4f\\n", C[0]);
        printf("Time: %.4f seconds\\n", elapsed);
        printf("GFLOPS: %.2f\\n", (2.0*M*N*K) / elapsed / 1e9);

        free(A); free(B); free(C);
        return 0;
    }}
    """)


def compile_and_run(c_code: str, label: str) -> str:
    """Compile C code and run it, returning stdout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "kernel.c")
        exe = os.path.join(tmpdir, "kernel")

        with open(src, "w") as f:
            f.write(c_code)

        result = subprocess.run(
            ["gcc", "-O2", "-o", exe, src, "-lm"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            console.print(f"[red]Compilation failed:[/]\n{result.stderr}")
            return ""

        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        return result.stdout


def demo():
    console.print("\n[bold]═══ CPU Code Generation ═══[/]\n")
    M, N, K = 256, 256, 256

    # Naive version
    console.print("[bold cyan]1. Naive MatMul[/]")
    naive_c = gen_matmul_c(M, N, K, tiled=False)
    console.print(Syntax(naive_c[:600] + "\n...", "c", theme="monokai"))

    console.print("\n[yellow]Compiling & running...[/]")
    out = compile_and_run(naive_c, "naive")
    if out:
        console.print(out)

    # Tiled version
    console.print("\n[bold cyan]2. Tiled MatMul (tile=32)[/]")
    tiled_c = gen_matmul_c(M, N, K, tiled=True, tile_size=32)

    console.print("[yellow]Compiling & running...[/]")
    out = compile_and_run(tiled_c, "tiled")
    if out:
        console.print(out)

    # Save generated code for inspection
    with open("generated_matmul_naive.c", "w") as f:
        f.write(naive_c)
    with open("generated_matmul_tiled.c", "w") as f:
        f.write(tiled_c)
    console.print("\n[green]Generated C files saved for inspection.[/]")


if __name__ == "__main__":
    demo()
