#!/usr/bin/env python3
"""
Chapter 17 — Code generator: emit C code from optimized graph.
"""

from __future__ import annotations
import textwrap
from model_ir import ModelGraph, IRNode, OpType


def emit_c(graph: ModelGraph) -> str:
    """Generate C code for the model graph."""
    lines = [
        '#include <stdio.h>',
        '#include <stdlib.h>',
        '#include <math.h>',
        '#include <string.h>',
        '#include <time.h>',
        '',
    ]

    # Emit helper functions
    lines.append(textwrap.dedent("""\
    static void matmul(const float *A, const float *B, float *C,
                       int M, int N, int K) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += A[i*K+k] * B[k*N+j];
                C[i*N+j] = sum;
            }
    }

    static void add_bias(float *out, const float *bias, int rows, int cols) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out[i*cols+j] += bias[j];
    }

    static void relu_inplace(float *x, int n) {
        for (int i = 0; i < n; i++)
            if (x[i] < 0) x[i] = 0;
    }

    static void softmax(const float *x, float *out, int n) {
        float max_val = x[0];
        for (int i = 1; i < n; i++)
            if (x[i] > max_val) max_val = x[i];
        float sum = 0;
        for (int i = 0; i < n; i++) {
            out[i] = expf(x[i] - max_val);
            sum += out[i];
        }
        for (int i = 0; i < n; i++) out[i] /= sum;
    }
    """))

    # Main function
    lines.append('int main() {')
    lines.append('    srand(42);')

    topo = graph.topo_order()

    # Allocate buffers
    for node in topo:
        if node.shape:
            size = node.shape.size
            if node.op in (OpType.INPUT, OpType.CONST):
                lines.append(f'    float {node.name}[{size}];')
                lines.append(f'    for (int i = 0; i < {size}; i++) '
                           f'{node.name}[i] = (float)(rand() % 100) / 100.0f;')
            else:
                lines.append(f'    float {node.name}[{size}];')
                lines.append(f'    memset({node.name}, 0, sizeof({node.name}));')

    lines.append('')
    lines.append('    struct timespec start, end;')
    lines.append('    clock_gettime(CLOCK_MONOTONIC, &start);')
    lines.append('')

    # Emit ops
    for node in topo:
        if node.op in (OpType.INPUT, OpType.CONST):
            continue

        if node.op == OpType.MATMUL:
            a, b = node.inputs[0], node.inputs[1]
            sa = graph.nodes[a].shape
            sb = graph.nodes[b].shape
            M, K, N = sa.dims[0], sa.dims[1], sb.dims[1]
            lines.append(f'    matmul({a}, {b}, {node.name}, {M}, {N}, {K});')

            if "bias" in node.attrs:
                bias = node.attrs["bias"]
                lines.append(f'    add_bias({node.name}, {bias}, {M}, {N});')

            if node.attrs.get("activation") == "relu":
                lines.append(f'    relu_inplace({node.name}, {node.shape.size});')

        elif node.op == OpType.ADD:
            a, b = node.inputs
            size = node.shape.size
            lines.append(f'    for (int i = 0; i < {size}; i++) '
                       f'{node.name}[i] = {a}[i] + {b}[i];')

        elif node.op == OpType.RELU:
            x = node.inputs[0]
            size = node.shape.size
            lines.append(f'    for (int i = 0; i < {size}; i++) '
                       f'{node.name}[i] = {x}[i] > 0 ? {x}[i] : 0;')

        elif node.op == OpType.SOFTMAX:
            x = node.inputs[0]
            lines.append(f'    softmax({x}, {node.name}, {node.shape.size});')

    lines.append('')
    lines.append('    clock_gettime(CLOCK_MONOTONIC, &end);')
    lines.append('    double elapsed = (end.tv_sec - start.tv_sec) + '
                 '(end.tv_nsec - start.tv_nsec) / 1e9;')

    # Print output
    output = topo[-1]
    n = min(output.shape.size, 10)
    lines.append(f'    printf("Output ({output.name}): ");')
    lines.append(f'    for (int i = 0; i < {n}; i++) '
                f'printf("%.4f ", {output.name}[i]);')
    lines.append('    printf("\\n");')
    lines.append('    printf("Inference time: %.6f seconds\\n", elapsed);')
    lines.append('    return 0;')
    lines.append('}')

    return '\n'.join(lines)
