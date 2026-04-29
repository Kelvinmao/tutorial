# Chapter 13 — Hardware Mapping & Code Generation

Move from abstract IR to **real executable code** — generate C for CPU and
simulate GPU thread/block mapping.

## What You'll Learn
- Translating tensor computations into C code
- Compiling and running generated kernels
- GPU execution model: threads, blocks, grids
- Mapping loop nests to GPU threads

## Files
| File | Description |
|------|-------------|
| `cpu_codegen.py` | Generate C code from loop nests |
| `gpu_mapping.py` | Simulate GPU thread/block mapping |
| `exercises.md` | Practice problems |

## Run
```bash
python cpu_codegen.py      # generates + compiles + runs C code
python gpu_mapping.py      # simulates GPU mapping for matrix multiply
```

## Key Idea

Hardware mapping is the step where an abstract tensor program becomes a
concrete execution plan. The same matrix multiplication can be represented as a
small loop nest, but the compiler still has to choose:

- which loop order to emit,
- which data layout to assume,
- how much work each CPU core or GPU thread owns,
- whether tiles fit in cache or shared memory,
- and which generated code is worth benchmarking.

`cpu_codegen.py` demonstrates the CPU side by emitting C loops and compiling
them. `gpu_mapping.py` does not launch a GPU kernel; it visualizes how output
elements would be assigned to GPU blocks and threads.

## Production Gap

Production AI compilers add target-specific lowering, vector intrinsics, memory
hierarchy modeling, runtime dispatch, and backend-specific code generation
such as LLVM, CUDA, ROCm, or vendor libraries. This chapter is a map of those
ideas, not a replacement for a real backend.
