# Chapter 10 — Exercises

## Exercise 10.1: Convolution Expression (Medium)
Write a tensor expression for 2D convolution:
```
out[n, oc, h, w] = sum over (ic, kh, kw) of
    input[n, ic, h+kh, w+kw] * weight[oc, ic, kh, kw]
```
Lower it to a 7-level loop nest and verify against `np.correlate`.

## Exercise 10.2: Multiple Outputs (Medium)
Extend the tensor expression system to support operations that produce
multiple outputs, like batch normalization which computes both the
normalized output and running statistics.

## Exercise 10.3: Symbolic Shapes (Hard)
Currently shapes are concrete integers. Add support for symbolic shapes
(`M`, `N`, `K` as variables) so the loop nest can be generated before
knowing actual tensor sizes.

## Exercise 10.4: Expression Simplification (Easy)
Add algebraic simplification to tensor expressions:
- `x * 1` → `x`, `x + 0` → `x`, `x * 0` → `0`
Apply during lowering to reduce the loop body.

## Exercise 10.5: Loop Nest Interpreter (Medium)
Write an interpreter that takes a loop nest representation and executes
it with actual NumPy arrays. Verify correctness against NumPy operations.
