#!/usr/bin/env python3
"""
Chapter 17 — Real hardware experiment: compile a tiny MLP to native C.

This experiment closes the loop between the Python reference model and a
generated native executable:

    deterministic weights/input → NumPy reference → generated C → local compiler

The model is intentionally small so it can run in CI and on laptops while still
exercising the same kernels as a larger inference graph: MatMul, AddBias, ReLU,
and Softmax.

Usage:
    python real_mlp_experiment.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from rich.console import Console
from rich.table import Table


console = Console()

INPUT_SIZE = 16
HIDDEN_SIZE = 8
OUTPUT_SIZE = 4


@dataclass
class MLPWeights:
    x: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass
class ExperimentResult:
    numpy_output: np.ndarray
    native_output: np.ndarray
    max_abs_diff: float
    numpy_seconds: float
    native_seconds: float
    c_source: str


def build_deterministic_mlp() -> MLPWeights:
    """Create a tiny fixed MLP without storing binary weight fixtures."""
    rng = np.random.default_rng(17)
    return MLPWeights(
        x=rng.normal(0.0, 0.5, size=(1, INPUT_SIZE)).astype(np.float32),
        w1=rng.normal(0.0, 0.2, size=(INPUT_SIZE, HIDDEN_SIZE)).astype(np.float32),
        b1=rng.normal(0.0, 0.05, size=(1, HIDDEN_SIZE)).astype(np.float32),
        w2=rng.normal(0.0, 0.2, size=(HIDDEN_SIZE, OUTPUT_SIZE)).astype(np.float32),
        b2=rng.normal(0.0, 0.05, size=(1, OUTPUT_SIZE)).astype(np.float32),
    )


def numpy_reference(weights: MLPWeights, iterations: int = 1) -> tuple[np.ndarray, float]:
    """Run the model in NumPy and return output plus elapsed seconds."""
    start = time.perf_counter()
    out = None
    for _ in range(iterations):
        hidden = weights.x @ weights.w1 + weights.b1
        hidden = np.maximum(hidden, 0.0)
        logits = hidden @ weights.w2 + weights.b2
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        out = exp / np.sum(exp, axis=1, keepdims=True)
    elapsed = time.perf_counter() - start
    return out.astype(np.float32), elapsed


def _format_c_array(name: str, array: np.ndarray) -> str:
    values = ", ".join(f"{float(v):.9g}f" for v in array.reshape(-1))
    return f"static const float {name}[{array.size}] = {{ {values} }};"


def emit_native_mlp_c(weights: MLPWeights, iterations: int = 1000) -> str:
    """Emit standalone C for the deterministic MLP inference workload."""
    arrays = "\n".join(
        [
            _format_c_array("x", weights.x),
            _format_c_array("w1", weights.w1),
            _format_c_array("b1", weights.b1),
            _format_c_array("w2", weights.w2),
            _format_c_array("b2", weights.b2),
        ]
    )
    return f"""#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE {INPUT_SIZE}
#define HIDDEN_SIZE {HIDDEN_SIZE}
#define OUTPUT_SIZE {OUTPUT_SIZE}
#define ITERATIONS {iterations}

{arrays}

static void matmul(const float *a, const float *b, float *out,
                   int rows, int inner, int cols) {{
    for (int i = 0; i < rows; i++) {{
        for (int j = 0; j < cols; j++) {{
            float sum = 0.0f;
            for (int k = 0; k < inner; k++) {{
                sum += a[i * inner + k] * b[k * cols + j];
            }}
            out[i * cols + j] = sum;
        }}
    }}
}}

static void add_bias(float *x, const float *bias, int n) {{
    for (int i = 0; i < n; i++) {{
        x[i] += bias[i];
    }}
}}

static void relu(float *x, int n) {{
    for (int i = 0; i < n; i++) {{
        if (x[i] < 0.0f) x[i] = 0.0f;
    }}
}}

static void softmax(const float *x, float *out, int n) {{
    float max_value = x[0];
    for (int i = 1; i < n; i++) {{
        if (x[i] > max_value) max_value = x[i];
    }}

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {{
        out[i] = expf(x[i] - max_value);
        sum += out[i];
    }}

    for (int i = 0; i < n; i++) {{
        out[i] /= sum;
    }}
}}

static void inference(float *output) {{
    float hidden[HIDDEN_SIZE];
    float logits[OUTPUT_SIZE];

    matmul(x, w1, hidden, 1, INPUT_SIZE, HIDDEN_SIZE);
    add_bias(hidden, b1, HIDDEN_SIZE);
    relu(hidden, HIDDEN_SIZE);

    matmul(hidden, w2, logits, 1, HIDDEN_SIZE, OUTPUT_SIZE);
    add_bias(logits, b2, OUTPUT_SIZE);
    softmax(logits, output, OUTPUT_SIZE);
}}

static double seconds_between(struct timespec start, struct timespec end) {{
    return (double)(end.tv_sec - start.tv_sec)
        + (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
}}

int main(void) {{
    float output[OUTPUT_SIZE];
    struct timespec start;
    struct timespec end;

    timespec_get(&start, TIME_UTC);
    for (int i = 0; i < ITERATIONS; i++) {{
        inference(output);
    }}
    timespec_get(&end, TIME_UTC);

    printf("OUTPUT");
    for (int i = 0; i < OUTPUT_SIZE; i++) {{
        printf(" %.9g", output[i]);
    }}
    printf("\\n");
    printf("SECONDS %.9f\\n", seconds_between(start, end));
    return 0;
}}
"""


def find_c_compiler() -> str:
    """Find a local C compiler that works on Linux or macOS."""
    cc = os.environ.get("CC")
    if cc and shutil.which(cc):
        return cc
    for candidate in ("cc", "gcc", "clang"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("No C compiler found. Install gcc or clang to run this experiment.")


def compile_and_run_c(c_source: str) -> tuple[np.ndarray, float]:
    """Compile generated C and parse its output vector and elapsed seconds."""
    compiler = find_c_compiler()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "mlp_native.c"
        exe = tmp / "mlp_native"
        src.write_text(c_source)

        compile_result = subprocess.run(
            [compiler, "-O3", "-std=c11", str(src), "-lm", "-o", str(exe)],
            capture_output=True,
            text=True,
            check=False,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                "C compilation failed:\n"
                f"STDOUT:\n{compile_result.stdout}\n"
                f"STDERR:\n{compile_result.stderr}"
            )

        run_result = subprocess.run(
            [str(exe)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

    output = None
    elapsed = None
    for line in run_result.stdout.splitlines():
        if line.startswith("OUTPUT "):
            output = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
        elif line.startswith("SECONDS "):
            elapsed = float(line.split()[1])

    if output is None or elapsed is None:
        raise RuntimeError(f"Unexpected native output:\n{run_result.stdout}")
    return output, elapsed


def run_experiment(iterations: int = 1000) -> ExperimentResult:
    """Run NumPy and generated native C, then compare outputs."""
    weights = build_deterministic_mlp()
    numpy_output, numpy_seconds = numpy_reference(weights, iterations)
    c_source = emit_native_mlp_c(weights, iterations)
    native_output, native_seconds = compile_and_run_c(c_source)
    max_abs_diff = float(np.max(np.abs(numpy_output.reshape(-1) - native_output)))

    return ExperimentResult(
        numpy_output=numpy_output.reshape(-1),
        native_output=native_output,
        max_abs_diff=max_abs_diff,
        numpy_seconds=numpy_seconds,
        native_seconds=native_seconds,
        c_source=c_source,
    )


def main() -> None:
    result = run_experiment()

    table = Table(title="Tiny MLP Native Compilation Experiment")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Output size", str(OUTPUT_SIZE))
    table.add_row("Max abs diff", f"{result.max_abs_diff:.8f}")
    table.add_row("NumPy time", f"{result.numpy_seconds:.6f}s")
    table.add_row("Native C time", f"{result.native_seconds:.6f}s")
    table.add_row("NumPy output", np.array2string(result.numpy_output, precision=6))
    table.add_row("Native output", np.array2string(result.native_output, precision=6))

    console.print(table)
    if result.max_abs_diff > 1e-5:
        raise SystemExit("Generated C output differs from NumPy reference.")


if __name__ == "__main__":
    main()
