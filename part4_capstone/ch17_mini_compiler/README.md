# Chapter 17 — Mini AI Compiler (Capstone Project)

Bring everything together: build a complete mini compiler that takes
a neural network description and produces optimized executable code.

## Architecture
```
Model Description (Python DSL)
        ↓
   Graph Builder (computation graph)
        ↓
   Graph Optimizer (fusion, constant folding)
        ↓
   Tensor Optimizer (tiling, scheduling)
        ↓
   Code Generator (C code)
        ↓
   Compile & Run
```

## Files
| File | Description |
|------|-------------|
| `model_ir.py` | Model description DSL & graph IR |
| `optimizer.py` | Graph + tensor optimizations |
| `codegen.py` | Generate C code from optimized graph |
| `compiler.py` | Main driver: model → executable |
| `examples.py` | Example models (MLP, simple ConvNet) |
| `exercises.md` | Practice problems |

## Run
```bash
python compiler.py       # compile and run example model
python examples.py       # more model examples
```
