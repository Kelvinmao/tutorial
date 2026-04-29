# Chapter 11: Loop Optimizations

## Learning Objectives

After this chapter you will:
- Understand loop tiling and why it's essential for cache performance
- Implement loop unrolling and vectorization (SIMD)
- Apply parallelization to multi-core CPUs
- Visualize memory access patterns before and after optimization

## Why Loop Optimization Matters

A naive MatMul has terrible cache behavior. With 1024×1024 matrices:
```
Naive:       ~10 seconds
Tiled:       ~1 second     (10x faster)
Tiled+SIMD:  ~0.1 seconds  (100x faster)
```

The algorithm is identical — only the **loop structure** changes!

## Key Optimizations

### 1. Loop Tiling (Blocking)
```
Before:                         After (tile size = 4):
for i in 0..M:                 for ii in 0..M step 4:
  for j in 0..N:                 for jj in 0..N step 4:
    for k in 0..K:                 for i in ii..ii+4:
      C[i,j] += A[i,k]*B[k,j]       for j in jj..jj+4:
                                       for k in 0..K:
                                         C[i,j] += A[i,k]*B[k,j]
```
Tiling ensures data fits in cache, reducing memory latency dramatically.

### 2. Loop Unrolling
Process multiple iterations per loop step to reduce branch overhead:
```
for i in 0..N step 4:
  C[i]   = A[i]   + B[i]
  C[i+1] = A[i+1] + B[i+1]
  C[i+2] = A[i+2] + B[i+2]
  C[i+3] = A[i+3] + B[i+3]
```

### 3. Vectorization (SIMD)
Use SIMD instructions to process multiple elements simultaneously:
```
C[i:i+4] = A[i:i+4] + B[i:i+4]   // 4 adds in 1 instruction!
```

### 4. Parallelization
Distribute independent iterations across CPU cores:
```
parallel for i in 0..M:    // each core handles a range of i
  for j in 0..N:
    ...
```

## Try It

```bash
python loop_tiling.py          # see tiling in action with benchmarks
python loop_unrolling.py       # unrolling effects
python vectorization.py        # SIMD simulation
python parallelization.py      # multi-core parallel loops
python visualize_tiling.py     # memory access pattern heatmaps
```

## Next Chapter

→ [Chapter 12: Memory Optimization](../../part3_advanced_topics/ch12_memory_optimization/README.md)
