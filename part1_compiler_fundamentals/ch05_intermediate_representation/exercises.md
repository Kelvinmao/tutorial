# Chapter 5 — Exercises

## Exercise 5.1: Trace the IR (Easy)
Trace the IR generation for `let x = (1 + 2) * (3 + 4)` by hand.
How many temporary variables are created? Run `python ir_builder.py` to verify.

## Exercise 5.2: Build SSA (Medium)
Implement SSA (Static Single Assignment) conversion for straight-line code:
- Every assignment creates a new versioned variable (`x_1`, `x_2`, ...)
- Uses always refer to the most recently defined version
Test with: `let x = 1; x = x + 1; x = x * 2`

## Exercise 5.3: While Loop CFG (Easy)
Draw the CFG for this program by hand, then verify with `visualize_cfg.py`:
```
let i = 0
while i < 10:
    i = i + 1
print(i)
```
How many basic blocks are there? What are the back edges?

## Exercise 5.4: Dominance Tree (Hard)
Implement a dominance analysis on the CFG:
- Block A **dominates** block B if every path from entry to B goes through A
- Build the dominator tree and visualize it
This is essential for SSA construction and many optimizations.

## Exercise 5.5: IR Interpreter (Medium)
Write an interpreter that executes the IR directly (without generating
machine code). It should:
- Maintain a map of variable → value
- Execute each instruction by updating the map
- Handle jumps and branches by moving the instruction pointer
Test with the sample programs and verify correct output.
