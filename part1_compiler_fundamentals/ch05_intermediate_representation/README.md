# Chapter 5: Intermediate Representation (IR)

## Learning Objectives

After this chapter you will:
- Understand why compilers use IR instead of directly generating machine code
- Build a three-address code IR from an AST
- Construct basic blocks and a Control Flow Graph (CFG)
- Understand SSA (Static Single Assignment) form
- Visualize CFGs using graphviz

## Why Intermediate Representation?

```
    Source A ─┐              ┌─ Target X (x86)
    Source B ─┼──▶  IR  ──▶──┼─ Target Y (ARM)
    Source C ─┘              └─ Target Z (LLVM)
```

Without IR, you'd need M×N compiler implementations (M source languages × N
target architectures). With IR, you only need M frontends + N backends.

## Three-Address Code

The simplest IR form. Each instruction has at most **three operands**:
```
result = operand1 op operand2
```

Example — `let y = (a + b) * (c - d)`:
```
t0 = a + b
t1 = c - d
t2 = t0 * t1
y  = t2
```

## Basic Blocks and Control Flow Graph

A **basic block** is a sequence of instructions with:
- One entry point (top)
- One exit point (bottom)
- No branches in the middle

A **Control Flow Graph (CFG)** connects basic blocks with edges showing
possible execution paths:

```
    ┌──────────┐
    │ Block 0  │
    │ t0 = ... │
    │ if t0    │
    └──┬───┬───┘
       │   │
   ┌───▼┐ ┌▼───┐
   │ B1 │ │ B2 │
   │then│ │else│
   └───┬┘ └┬───┘
       │   │
    ┌──▼───▼──┐
    │  Block 3 │
    │  merge   │
    └──────────┘
```

## SSA Form

In **Static Single Assignment** form, every variable is assigned exactly once.
When a variable could come from two paths, we use a **φ (phi) function**:

```
Before SSA:                  After SSA:
  x = 1                       x1 = 1
  if cond:                     if cond:
    x = 2                       x2 = 2
  print(x)                    x3 = φ(x1, x2)
                               print(x3)
```

## Try It

```bash
python ir_builder.py        # generate IR from a sample program
python cfg_builder.py       # build and display CFG
python visualize_cfg.py     # render CFG as graphviz image
```

## Next Chapter

→ [Chapter 6: Optimization Passes](../ch06_optimization_passes/README.md)
