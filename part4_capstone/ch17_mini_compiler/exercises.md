# Chapter 17 — Exercises

## Exercise 1: Add Conv2D Support
Extend `model_ir.py` and `codegen.py` to support 2D convolution.
Build a LeNet-like model and compile it.

## Exercise 2: Memory Optimization
Integrate the memory planner from Ch12 into the compiler.
Show buffer reuse in the generated C code.

## Exercise 3: Tiled MatMul Backend
Replace the naive matmul in `codegen.py` with the tiled version
from Ch16. Benchmark the improvement.

## Exercise 4: Graph Visualization
Add a `visualize()` method to `ModelGraph` that renders the graph
using graphviz. Show before/after optimization.

## Exercise 5: ONNX Import
Write a simple ONNX model loader that constructs a `ModelGraph`
from an ONNX protobuf file. Test with a small exported PyTorch model.

## Exercise 6: Multi-Batch Support
Extend the compiler to handle batch sizes > 1. The shapes should
propagate correctly through all ops.
