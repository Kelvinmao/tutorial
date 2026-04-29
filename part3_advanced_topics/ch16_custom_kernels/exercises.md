# Chapter 16 — Exercises

## Exercise 1: Loop Reordering
Create `matmul_ikj.c` with the loop order i→k→j (instead of i→j→k).
Benchmark and explain why it's faster for row-major storage.

## Exercise 2: Register Blocking
Write a version that computes a 4×4 block of C in registers before
writing back. This is called "register tiling" or "micro-kernel."

## Exercise 3: Compiler Flags
Benchmark the naive version with different flags: `-O0`, `-O1`, `-O2`,
`-O3`, `-Ofast`, `-march=native`. Plot the results.

## Exercise 4: NumPy Comparison
Add NumPy's `np.dot()` to the benchmark. How close do your C kernels
get to BLAS performance?

## Exercise 5: Memory Bandwidth
Instrument the kernels to count memory accesses (loads/stores).
Compute the arithmetic intensity (FLOPs/byte) for each version
and plot on a roofline chart.
