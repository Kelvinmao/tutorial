# Chapter 1: What is a Compiler?

## Learning Objectives

After this chapter you will understand:
- What a compiler does and why we need one
- The major phases of compilation
- How each phase transforms the program
- The difference between a compiler and an interpreter

## What is a Compiler?

A **compiler** is a program that translates source code written in one language
(the **source language**) into another language (the **target language**) — 
usually machine code or a lower-level intermediate form.

```
  Source Code    ──▶   [ COMPILER ]   ──▶   Target Code
  (high-level)                              (low-level)
```

### Compiler vs. Interpreter

| | Compiler | Interpreter |
|---|---|---|
| **When** | Translates *before* execution | Translates *during* execution |
| **Speed** | Faster at runtime (pre-compiled) | Slower (translates on the fly) |
| **Output** | Produces executable file | No executable produced |
| **Examples** | gcc, clang, rustc | CPython, Ruby MRI |

Many modern systems are **hybrid**: Python compiles to bytecode, then interprets it.
AI compilers like TVM compile a computation graph into optimized machine code.

## Phases of a Compiler

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     COMPILER PIPELINE                            │
  ├────────────┬────────────┬───────────┬──────────┬────────────────┤
  │            │            │           │          │                │
  │  Source    │  Lexical   │  Syntax   │ Semantic │  Intermediate  │
  │  Code     │  Analysis  │  Analysis │ Analysis │  Representation│
  │            │  (Lexer)   │  (Parser) │          │  (IR)          │
  │  "x = 2+3"│ ──────────▶│ ─────────▶│ ────────▶│ ──────────────▶│
  │            │  Tokens    │  AST      │ Typed    │  IR / SSA      │
  │            │            │           │  AST     │                │
  ├────────────┴────────────┴───────────┴──────────┴────────┬───────┤
  │                                                         │       │
  │  Optimization Passes                                    │ Code  │
  │  (constant folding, dead code elimination, ...)         │ Gen   │
  │  ─────────────────────────────────────────────────────▶ │ ─────▶│
  │                                                         │Target │
  └─────────────────────────────────────────────────────────┴───────┘
```

### Phase 1: Lexical Analysis (Lexer / Tokenizer)

Breaks the source text into **tokens** — the smallest meaningful units.

```
Input:  "x = 2 + 3"
Output: [ID("x"), EQUALS, INT(2), PLUS, INT(3)]
```

### Phase 2: Syntax Analysis (Parser)

Builds an **Abstract Syntax Tree (AST)** that captures the program's structure.

```
         Assign
        /      \
      "x"      Add
              /   \
             2     3
```

### Phase 3: Semantic Analysis

Checks that the program **makes sense**: types match, variables are declared, etc.

### Phase 4: Intermediate Representation (IR)

Converts the AST into a lower-level form that's easier to optimize.

```
t1 = 2
t2 = 3
t3 = t1 + t2
x  = t3
```

### Phase 5: Optimization

Applies transformations to make the code faster/smaller without changing behavior.

```
Before:  t1 = 2; t2 = 3; t3 = t1 + t2; x = t3
After:   x = 5                (constant folding!)
```

### Phase 6: Code Generation

Emits the final target code (assembly, machine code, C, LLVM IR, ...).

## Why This Matters for AI Compilers

AI compilers follow the **exact same pipeline**, but specialized for neural networks:

| Traditional Compiler | AI Compiler |
|---|---|
| Source code (C, Rust) | Computation graph (PyTorch, TF) |
| Tokens → AST | Graph nodes → DAG |
| IR (three-address code) | Tensor IR (loop nests) |
| Optimization passes | Operator fusion, tiling, vectorization |
| Machine code | Optimized CPU/GPU kernels |

Understanding the traditional pipeline first makes AI compiler concepts much easier.

## Try It

Run the demo to see all phases in action on a simple expression:

```bash
python demo_pipeline.py
```

## Next Chapter

→ [Chapter 2: Lexer](../ch02_lexer/README.md) — Build a tokenizer from scratch.
