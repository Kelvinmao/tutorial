/*
 * Chapter 16 — Naive matrix multiply (baseline).
 * Compile: gcc -O2 -o matmul_naive matmul_naive.c -lm
 *
 * ALGORITHM: Naive Triple-Nested Loop Matrix Multiply (ijk order)
 *
 * Historical context: This is the textbook O(n³) matrix multiply from
 * every linear algebra course. It's correct but has terrible cache
 * performance: the inner loop scans column j of B, but B is stored
 * row-major, so each B[k*N+j] access hits a different cache line.
 * For large N, every B access is a cache miss.
 *
 * Problem: Serves as the baseline to show how much tiling (ch16
 * matmul_tiled.c) and compiler optimizations (gcc -O2) can help.
 *
 * Implementation: Three nested loops (i, j, k) with row-major indexing.
 * A[i*K+k] is sequential (good cache behavior).
 * B[k*N+j] is strided by N (poor cache behavior).
 * C[i*N+j] is a single element (good, stays in register).
 *
 *   Memory access pattern for B (row-major storage):
 *
 *   B in memory: [b00 b01 b02 b03 | b10 b11 b12 b13 | ...]
 *                 ^^^              ^^^              stride = N
 *                 k=0              k=1
 *
 *   Inner loop (k): accesses B[0*N+j], B[1*N+j], B[2*N+j], ...
 *   Each access jumps N elements (N*4 bytes) — different cache line!
 *
 *   Cache line (64 bytes = 16 floats):
 *   [b00 b01 b02 ... b15]  ← only b00 used, 15 elements wasted
 *   [b16 b17 b18 ... b31]  ← only b16 used, 15 elements wasted
 *                              = 1/16 cache utilization!
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul_naive(const float *A, const float *B, float *C,
                  int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
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

    matmul_naive(A, B, C, M, N, K);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("naive,%d,%.6f,%.2f\n", size, elapsed,
           (2.0*M*N*K) / elapsed / 1e9);

    free(A); free(B); free(C);
    return 0;
}
