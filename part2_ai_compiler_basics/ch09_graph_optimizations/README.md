# Chapter 9: Graph-Level Optimizations

## Learning Objectives

After this chapter you will:
- Understand operator fusion and why it's critical for performance
- Implement graph-level constant folding
- Perform layout transformations (NCHW ↔ NHWC)
- Visualize optimized vs unoptimized graphs side-by-side

## Why Graph Optimization?

Before lowering to loops/kernels, AI compilers optimize the **graph structure**
itself. This can eliminate entire nodes and reduce memory traffic.

```
  Before fusion:                     After fusion:
  ┌────────┐                         ┌──────────────────┐
  │ Conv2D │ → write to memory       │ Conv2D+BN+ReLU   │
  └───┬────┘                         │ (fused kernel)    │
  ┌───▼────┐                         └──────────────────┘
  │BatchNorm│ → write to memory       Only 1 memory write!
  └───┬────┘
  ┌───▼────┐
  │  ReLU  │ → write to memory
  └────────┘
  3 memory writes!
```

## Key Graph Optimizations

### 1. Operator Fusion
Merge consecutive operations into a single fused kernel:
- **Conv + BN + ReLU** → single kernel (most common in CNNs)
- **MatMul + Add** → fused linear layer
- **Element-wise chains** → single fused element-wise kernel

### 2. Constant Folding
Evaluate nodes with constant inputs at compile time:
```
Const(shape) → Reshape → ... becomes Const(new_shape) → ...
```

### 3. Layout Transformation
Different hardware prefers different memory layouts:
- **NCHW** (channels first) — NVIDIA GPUs, PyTorch default
- **NHWC** (channels last) — CPUs, TensorFlow, Apple Silicon

## Try It

```bash
python operator_fusion.py           # see operator fusion in action
python constant_folding_graph.py    # graph-level constant folding
python layout_transform.py          # NCHW ↔ NHWC conversion
python visualize_fusion.py          # before/after comparison
```

## Next Chapter

→ [Chapter 10: Tensor IR](../ch10_tensor_ir/README.md)
