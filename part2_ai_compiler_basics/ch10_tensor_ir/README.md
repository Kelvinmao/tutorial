# Chapter 10: Tensor IR — A DSL for Tensor Computations

## Learning Objectives

After this chapter you will:
- Understand the gap between graph-level and loop-level representations
- Build a mini tensor expression DSL (like TVM's `te.compute`)
- Lower tensor expressions to explicit loop nests
- Visualize the iteration domain

## Graph IR vs Tensor IR

Graph IR says **what** to compute (MatMul, ReLU, Add), but not **how**.
Tensor IR specifies **how** each operation executes as nested loops:

```
  Graph IR:   C = MatMul(A, B)        # what
  Tensor IR:  for i in 0..M:          # how
                for j in 0..N:
                  C[i,j] = 0
                  for k in 0..K:
                    C[i,j] += A[i,k] * B[k,j]
```

This is the level where AI compilers apply their most powerful optimizations:
tiling, vectorization, parallelization, etc.

## Tensor Expressions

We define computations declaratively:

```python
# "C[i,j] = sum over k of A[i,k] * B[k,j]"
C = te.compute((M, N), lambda i, j:
    te.sum(A[i, k] * B[k, j], axis=k))
```

This is then **lowered** to explicit loops:

```
for (i, 0, M):
  for (j, 0, N):
    C[i][j] = 0.0
    for (k, 0, K):
      C[i][j] = C[i][j] + A[i][k] * B[k][j]
```

## Try It

```bash
python tensor_expression.py     # define and lower tensor computations
python loop_nest.py             # explicit loop nest representation
python visualize_loops.py       # iteration domain visualization
```

## Next Chapter

→ [Chapter 11: Loop Optimizations](../ch11_loop_optimizations/README.md)
