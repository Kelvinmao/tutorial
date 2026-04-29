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
