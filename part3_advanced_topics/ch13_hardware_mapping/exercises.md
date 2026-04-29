# Chapter 13 — Exercises

## Exercise 1: Shared Memory Simulation
Modify `gpu_mapping.py` to simulate **shared memory tiling**: each block
loads its tile of A and B into a local array, then computes from that.
Count the number of global memory accesses saved.

## Exercise 2: Generate Vector Code
Extend `cpu_codegen.py` to emit AVX intrinsics (`_mm256_*`) for the
innermost loop. Compare performance with the scalar version.

## Exercise 3: Thread Coarsening
Instead of 1 output element per thread, have each thread compute a
2×2 tile of output. Modify the mapping and count total threads needed.

## Exercise 4: Occupancy Calculator
Write a function `compute_occupancy(threads_per_block, regs_per_thread,
shared_mem_per_block)` that estimates GPU occupancy given hardware limits
(e.g., max 1024 threads/block, 65536 registers/SM, 48KB shared mem/SM).

## Exercise 5: CUDA-style Pseudocode
Write a codegen that emits CUDA-like pseudocode (with `__global__`,
`threadIdx`, `blockIdx`) from the tensor IR in Chapter 10.
