# Chapter 6: Optimization Passes

## Learning Objectives

After this chapter you will:
- Understand the most common compiler optimizations
- Implement constant folding, dead code elimination, and common subexpression elimination
- See how each optimization transforms the IR
- Visualize before/after comparisons

## Why Optimize?

The IR from Chapter 5 is correct but inefficient. Optimizations make it faster
and smaller **without changing the program's behavior**.

```
Before (7 instructions):           After (2 instructions):
  t0 = 2                             t4 = 14
  t1 = 3                             x = t4
  t2 = t0 + t1       ─ constant ─▶
  t3 = t2 * 2           folding
  t4 = t3 + t0 + t1     + DCE
  x = t4
  print(t0)           ── DCE ──▶  (removed — t0 unused after folding)
```

## The Three Classic Optimizations

### 1. Constant Folding
If both operands of an operation are known constants, compute the result at
compile time:
```
t0 = 3           →  t0 = 3
t1 = 4           →  t1 = 4
t2 = t0 + t1     →  t2 = 7      ← folded!
```

### 2. Dead Code Elimination (DCE)
Remove instructions whose results are never used:
```
t0 = 5           →  (removed — t0 never used)
t1 = 10          →  t1 = 10
x = t1           →  x = t1
```

### 3. Common Subexpression Elimination (CSE)
If the same expression is computed twice, reuse the first result:
```
t0 = a + b       →  t0 = a + b
t1 = a + b       →  t1 = t0     ← reuse!
t2 = t0 * t1     →  t2 = t0 * t0
```

## Try It

```bash
python constant_folding.py           # watch constant folding in action
python dead_code_elimination.py      # see dead code removed
python common_subexpr_elim.py        # see CSE at work
python visualize_optimization.py     # side-by-side before/after
```

## Next Chapter

→ [Chapter 7: LLVM Backend](../ch07_llvm_backend/README.md)
