# 🔧 AI Compiler Tutorial — From Zero to Building Your Own

A comprehensive, hands-on tutorial that takes you from **zero compiler knowledge** to building
a **working AI compiler** that can compile and run neural networks.

```
                           ╔══════════════════════════════════════╗
                           ║     AI Compiler Tutorial Roadmap     ║
                           ╚══════════════════════════════════════╝

  Part 1: Compiler Fundamentals          Part 2: AI Compiler Basics
  ┌──────────────────────────┐           ┌──────────────────────────┐
  │ Ch01 What is a Compiler  │           │ Ch08 Computation Graphs  │
  │ Ch02 Lexer               │──────────▶│ Ch09 Graph Optimizations │
  │ Ch03 Parser & AST        │           │ Ch10 Tensor IR           │
  │ Ch04 Semantic Analysis   │           │ Ch11 Loop Optimizations  │
  │ Ch05 Intermediate Rep.   │           └────────────┬─────────────┘
  │ Ch06 Optimization Passes │                        │
  │ Ch07 LLVM Backend        │                        ▼
  └──────────────────────────┘   Part 3: Advanced Topics
                                 ┌──────────────────────────┐
                                 │ Ch12 Memory Optimization  │
                                 │ Ch13 Hardware Mapping      │
                                 │ Ch14 Auto-Tuning           │
                                 │ Ch15 Quantization          │
                                 │ Ch16 Custom Kernels        │
                                 └────────────┬───────────────┘
                                              │
                                              ▼
                                 Part 4: Capstone
                                 ┌──────────────────────────┐
                                 │ Ch17 Mini AI Compiler 🎯  │
                                 │ Ch18 Real-World Frameworks│
                                 │ Ch19 What's Next          │
                                 └──────────────────────────┘
```

## Prerequisites

- **Python 3.10+** with basic proficiency
- **Basic ML knowledge** (what a neural network is, forward/backward pass)
- **gcc/g++** for C/C++ chapters
- No prior compiler knowledge needed!

## Quick Start

```bash
# Clone the repo
git clone <this-repo-url>
cd ai-compiler-tutorial

# One-command setup
bash setup.sh

# Start with Chapter 1
cd part1_compiler_fundamentals/ch01_what_is_a_compiler
python demo_pipeline.py
```

## Repository Structure

```
ai-compiler-tutorial/
├── utils/                              # Shared visualization & test helpers
│
├── part1_compiler_fundamentals/        # Build a compiler from scratch
│   ├── ch01_what_is_a_compiler/        # Compiler phases overview
│   ├── ch02_lexer/                     # Tokenization
│   ├── ch03_parser_ast/                # Parsing & AST construction
│   ├── ch04_semantic_analysis/         # Type checking & symbol tables
│   ├── ch05_intermediate_representation/ # IR, SSA, control flow graphs
│   ├── ch06_optimization_passes/       # Constant folding, DCE, CSE
│   └── ch07_llvm_backend/              # Emit LLVM IR, optimize, JIT
│
├── part2_ai_compiler_basics/           # AI-specific compiler concepts
│   ├── ch08_computation_graphs/        # DAGs, autodiff
│   ├── ch09_graph_optimizations/       # Operator fusion, layout transforms
│   ├── ch10_tensor_ir/                 # Tensor compute DSL, loop nests
│   └── ch11_loop_optimizations/        # Tiling, vectorization, parallelism
│
├── part3_advanced_topics/              # Production compiler techniques
│   ├── ch12_memory_optimization/       # Buffer sharing, liveness analysis
│   ├── ch13_hardware_mapping/          # Code generation for CPU/GPU
│   ├── ch14_auto_tuning/               # Search-based optimization
│   ├── ch15_quantization/              # INT8, mixed precision
│   └── ch16_custom_kernels/            # Hand-optimized C kernels
│
└── part4_capstone/                     # Putting it all together
    ├── ch17_mini_compiler/             # Full end-to-end AI compiler
    ├── ch18_real_world/                # TVM, XLA, Triton, ONNX Runtime
    └── ch19_whats_next/                # Papers, books, communities
```

## How Each Chapter Works

Every chapter contains:
- **README.md** — Theory explanation with diagrams
- **Runnable Python/C scripts** — Working implementations you can modify
- **`visualize_*.py`** — Generates visual output (graphs, heatmaps, diagrams)
- **`exercises.md`** — 3–5 graded exercises (easy → hard)

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Primary implementation language |
| C/C++ | Code generation target, kernel benchmarks |
| NumPy | Tensor operations, reference implementations |
| Matplotlib | Heatmaps, performance charts |
| Graphviz | AST, CFG, computation graph visualization |
| Rich | Color-coded terminal output |
| llvmlite | LLVM IR generation and JIT (Ch07) |
| gcc | Compile generated C code |

## License

MIT License — use freely for learning and teaching.
