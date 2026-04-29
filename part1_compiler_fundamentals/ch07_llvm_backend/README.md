# Chapter 7: LLVM Backend — From Custom IR to LLVM IR

## Learning Objectives

After this chapter you will:
- Understand what LLVM is and why AI compilers use it
- Translate your custom IR into LLVM IR using `llvmlite`
- Run LLVM's optimization passes on generated code
- JIT-compile and execute LLVM IR directly
- Compare your hand-written optimizations vs LLVM's `-O2`

## What is LLVM?

**LLVM** (Low Level Virtual Machine) is a compiler infrastructure that provides:
- A well-defined **Intermediate Representation** (LLVM IR)
- Powerful **optimization passes** (hundreds of them!)
- **Code generation** for many architectures (x86, ARM, RISC-V, ...)
- A **JIT compiler** that can compile and run code at runtime

```
    Your Compiler                        LLVM
  ┌──────────────┐                 ┌──────────────────┐
  │ Source        │                 │ LLVM IR          │
  │   ↓          │                 │   ↓              │
  │ Lexer        │                 │ Optimization     │
  │   ↓          │    llvmlite     │ Passes (-O2)     │
  │ Parser       │──────────────▶  │   ↓              │
  │   ↓          │                 │ Code Generation  │
  │ Custom IR    │                 │   ↓              │
  │   ↓          │                 │ Machine Code     │
  │ Custom Opts  │                 │ (or JIT Execute) │
  └──────────────┘                 └──────────────────┘
```

## Why LLVM Matters for AI Compilers

| AI Compiler | Uses LLVM? | How |
|-------------|-----------|-----|
| **TVM** | Yes | Generates LLVM IR for CPU kernels |
| **XLA** | Yes | CPU backend emits LLVM IR |
| **Triton** | Yes | Generates LLVM IR → PTX for GPU |
| **Halide** | Yes | Core backend is LLVM |
| **MLIR** | Part of LLVM | LLVM's multi-level IR framework |

## LLVM IR Crash Course

LLVM IR looks like typed assembly:

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
    %result = add i32 %a, %b
    ret i32 %result
}
```

Key features:
- **Typed**: every value has a type (`i32`, `float`, `double`, ...)
- **SSA form**: every register is assigned exactly once
- **Infinite registers**: `%0`, `%1`, `%result`, ...
- **Basic blocks**: labeled sections with one terminator (branch/return)

## Try It

```bash
python llvm_emitter.py          # translate custom IR → LLVM IR
python llvm_optimizer.py        # run LLVM optimization passes
python jit_runner.py            # JIT compile and execute!
python compare_optimizations.py # your optimizations vs LLVM's
```

**Note:** Requires `llvmlite` — install with `pip install llvmlite`

## Next Chapter

→ [Chapter 8: Computation Graphs](../../part2_ai_compiler_basics/ch08_computation_graphs/README.md) — 
  Now we enter the world of AI compilers!
