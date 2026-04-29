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
