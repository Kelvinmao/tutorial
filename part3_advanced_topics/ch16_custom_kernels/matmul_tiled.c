/*
 * Chapter 16 — Tiled (cache-blocked) matrix multiply.
 * Compile: gcc -O2 -o matmul_tiled matmul_tiled.c -lm
 *
 * ALGORITHM: Cache-Blocked (Tiled) Matrix Multiply
 *
 * Historical context: Cache blocking was a key technique in ATLAS
 * (1998, Whaley & Dongarra), which auto-tuned tile sizes to match
 * the CPU's cache hierarchy. Modern BLAS libraries (OpenBLAS, MKL)
 * use multi-level tiling with hand-tuned micro-kernels.
 *
 * Problem solved: The naive version (matmul_naive.c) accesses B in
 * column order, causing a cache miss per element. Tiling restructures
 * the loops so that a TILE×TILE sub-block of B stays in L1 cache
 * while it's reused across rows of A.
 *
 * Implementation: Six nested loops:
 *   ii, jj, kk: outer tile loops (stride by TILE)
 *   i, j, k:    inner element loops (within tile bounds)
 * The working set per tile is:
 *   A_tile: TILE×TILE = 32×32 = 4KB
 *   B_tile: TILE×TILE = 4KB
 *   C_tile: TILE×TILE = 4KB
 *   Total: 12KB → fits comfortably in 32KB L1 cache
 *
 * Edge handling: min(ii+TILE, M) clamps the inner bounds when the
 * matrix dimensions aren't multiples of TILE.
 *
 *   Tiled memory access pattern:
 *
 *   Matrix B:                    Inner loops access a TILE×TILE block:
 *   ┌─────────────────────┐
 *   │ ████ │              │      B tile (32×32 = 4KB):
 *   │ ████ │              │      [b00 b01 ... b31]  ← all 32 used!
 *   │ ████ │              │      [b32 b33 ... b63]  ← all 32 used!
 *   │ ████ │              │      ...
 *   │      │              │      Cache utilization: 32/16 = 2 lines,
 *   │      │              │      both fully used = 100% utilization
 *   └─────────────────────┘
 *
 *   Tile reuse: same B tile is accessed TILE times (once per A row
 *   in the tile), so each cache line is reused 32× instead of 1×.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE 32

void matmul_tiled(const float *A, const float *B, float *C,
                  int M, int N, int K) {
    for (int ii = 0; ii < M; ii += TILE)
        for (int jj = 0; jj < N; jj += TILE)
            for (int kk = 0; kk < K; kk += TILE) {
                int i_end = ii + TILE < M ? ii + TILE : M;
                int j_end = jj + TILE < N ? jj + TILE : N;
                int k_end = kk + TILE < K ? kk + TILE : K;
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++) {
                        float sum = C[i*N + j];
                        for (int k = kk; k < k_end; k++)
                            sum += A[i*K + k] * B[k*N + j];
                        C[i*N + j] = sum;
                    }
            }
}

int main(int argc, char **argv) {
    int size = 512;
    if (argc > 1) size = atoi(argv[1]);
    int M = size, N = size, K = size;

    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C = calloc(M * N, sizeof(float));

    srand(42);
    for (int i = 0; i < M*K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K*N; i++) B[i] = (float)(rand() % 100) / 100.0f;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matmul_tiled(A, B, C, M, N, K);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("tiled,%d,%.6f,%.2f\n", size, elapsed,
           (2.0*M*N*K) / elapsed / 1e9);

    free(A); free(B); free(C);
    return 0;
}
