# Chapter 17 — Mini AI Compiler (Capstone Project)

Bring everything together: build a complete mini compiler that takes
a neural network description and produces optimized executable code.

## Architecture
```
Model Description (Python DSL)
        ↓
   Graph Builder (computation graph)
        ↓
   Graph Optimizer (fusion, dead-code elimination)
        ↓
   Code Generator (C code)
        ↓
   Compile & Run
```

## Files
| File | Description |
|------|-------------|
| `model_ir.py` | Model description DSL & graph IR |
| `optimizer.py` | Graph optimizations: fusion and dead-code elimination |
| `codegen.py` | Generate C code from optimized graph |
| `compiler.py` | Main driver: model → executable |
| `examples.py` | Example models (MLP, linear regression) |
| `exercises.md` | Practice problems |

## Run
```bash
python compiler.py       # compile and run example model
python examples.py       # more model examples
```

## What This Compiler Really Does

The capstone intentionally stays small enough to read in one sitting:

1. Build a neural-network graph with inputs, constants, matmul, add, ReLU, and
   softmax nodes.
2. Fuse common inference patterns such as `MatMul + Add` and `MatMul + ReLU`.
3. Emit C code with simple helper kernels.
4. Compile that C code with `gcc` and run the resulting executable.

This is not yet a production compiler: it does not schedule tiled kernels,
perform shape-polymorphic compilation, target GPUs, or import ONNX. Those are
good follow-up exercises once the end-to-end path is clear.
