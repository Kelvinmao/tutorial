/*
 * Chapter 16 — Naive matrix multiply (baseline).
 * Compile: gcc -O2 -o matmul_naive matmul_naive.c -lm
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
