/*
 * Chapter 16 — Tiled (cache-blocked) matrix multiply.
 * Compile: gcc -O2 -o matmul_tiled matmul_tiled.c -lm
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
