# Chapter 11 — Exercises

## Exercise 11.1: Optimal Tile Size (Easy)
Run `loop_tiling.py` with different tile sizes (4, 8, 16, 32, 64).
Plot performance vs tile size. What's the optimal tile size for your CPU?
Why does performance degrade for very large tiles?

## Exercise 11.2: 2D Tiling (Medium)
The current tiling only tiles the i and j loops. Add tiling for the k loop
too (3D tiling). Does this improve performance further? When?

## Exercise 11.3: Combined Optimization (Hard)
Apply tiling + unrolling + vectorization together:
```python
for ii in range(0, M, tile):       # tiled
  for jj in range(0, N, tile):     # tiled
    for i in range(ii, ii+tile):   # unrolled x4
      C[i, jj:jj+tile] += A[i,k] * B[k, jj:jj+tile]  # vectorized
```
Benchmark against individual optimizations.

## Exercise 11.4: Auto-Vectorization Check (Easy)
Write a simple C loop (`for(i=0;i<N;i++) c[i]=a[i]+b[i]`) and compile with
`gcc -O2 -ftree-vectorize -fopt-info-vec`. Does the compiler auto-vectorize it?
What happens with `gcc -O0`?

## Exercise 11.5: Roofline Model (Hard)
Implement a roofline model analysis for matrix multiply:
- Measure FLOPS and memory bandwidth
- Plot the roofline diagram
- Determine if your implementation is compute-bound or memory-bound
