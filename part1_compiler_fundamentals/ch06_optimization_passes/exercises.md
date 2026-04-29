# Chapter 6 — Exercises

## Exercise 6.1: Algebraic Simplification (Medium)
Implement an algebraic simplification pass that handles:
- `x + 0` → `x` and `x * 1` → `x`
- `x * 0` → `0` and `x - x` → `0`
- `x * 2` → `x + x` (cheaper on some architectures)

## Exercise 6.2: Copy Propagation (Medium)
Implement copy propagation: when `x = y`, replace all subsequent uses
of `x` with `y` (if `y` hasn't been reassigned). This enables more DCE.

## Exercise 6.3: Pass Ordering (Easy)
Run the three passes in different orders and compare results:
- CF → CSE → DCE
- CSE → CF → DCE
- DCE → CF → CSE
Which ordering gives the best result? Why does pass ordering matter?

## Exercise 6.4: Strength Reduction (Hard)
Implement strength reduction for loop induction variables:
- Replace `i * 4` inside a loop with `i_scaled += 4` (addition is cheaper)
- Replace `x ** 2` with `x * x`
This requires loop detection in the CFG.

## Exercise 6.5: Optimization Statistics (Easy)
Add detailed statistics to each optimization pass:
- How many operations were folded/eliminated?
- What percentage reduction in IR size?
- Visualize these statistics as a bar chart.
