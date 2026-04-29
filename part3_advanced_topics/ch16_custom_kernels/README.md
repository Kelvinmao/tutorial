# Chapter 16 — Custom Kernels & Performance

Write hand-optimized C kernels and benchmark them against naive versions.

## What You'll Learn
- Writing naive vs tiled vs SIMD matrix multiply in C
- Benchmarking methodology
- Understanding performance bottlenecks

## Files
| File | Description |
|------|-------------|
| `matmul_naive.c` | Baseline triple-loop matmul |
| `matmul_tiled.c` | Cache-friendly tiled version |
| `benchmark.py` | Compile and benchmark all versions |
| `exercises.md` | Practice problems |

## Run
```bash
python benchmark.py    # compiles C kernels and benchmarks
```

## Key Idea

Custom kernels matter when a generic implementation leaves hardware
performance on the table. Matrix multiplication is a useful example because it
is simple to specify but sensitive to memory locality, loop order, cache reuse,
vectorization, and compiler flags.

This chapter compares:

- a naive triple-loop implementation,
- a tiled implementation that improves cache locality,
- and the benchmarking harness that compiles and times both versions.

The goal is not to beat vendor libraries. The goal is to see why compiler
scheduling decisions become performance decisions.

## Benchmarking Notes

Treat single-run timings as noisy. For more reliable numbers, add warmup runs,
repeat each size several times, report median/min timings, pin CPU frequency if
possible, and compare against a tuned library such as BLAS. Also inspect the
generated assembly when a change appears faster than expected.
