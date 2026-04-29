# Chapter 8: Computation Graphs

## Learning Objectives

After this chapter you will:
- Understand what a computation graph is and why AI frameworks use them
- Build a computation graph from tensor operations
- Implement reverse-mode automatic differentiation (backprop!)
- Visualize computation graphs with tensor shapes on edges

## From Traditional IR to Computation Graphs

In Part 1, we compiled **scalar** operations. AI compilers operate on **tensors**
(multi-dimensional arrays) organized as a **computation graph** (a DAG):

```
  Traditional Compiler              AI Compiler
  ──────────────────              ──────────────
  x = a + b                      Y = MatMul(X, W)
  (scalar arithmetic)            Z = Add(Y, bias)
                                 out = ReLU(Z)
                                 (tensor operations in a DAG)
```

## What is a Computation Graph?

A **computation graph** is a directed acyclic graph (DAG) where:
- **Nodes** = operations (MatMul, ReLU, Conv2D, ...)
- **Edges** = tensors flowing between operations

```
  Input X [2,784]    Weights W [784,256]
       \                /
        \              /
       ┌──────────────┐
       │  MatMul       │  → shape [2,256]
       └──────┬───────┘
              │
       ┌──────▼───────┐     Bias [256]
       │    Add        │────/
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │   ReLU        │  → shape [2,256]
       └──────────────┘
```

## Automatic Differentiation

The critical feature: given a computation graph for the **forward pass**,
we can automatically compute gradients for the **backward pass**:

```
Forward:  Z = X @ W + b  →  loss = MSE(Z, target)
Backward: ∂loss/∂W = X.T @ ∂loss/∂Z   (chain rule!)
```

## Try It

```bash
python comp_graph.py         # build and evaluate a computation graph
python autodiff.py           # see reverse-mode autodiff in action
python visualize_graph.py    # render computation graph with shapes
```

## Next Chapter

→ [Chapter 9: Graph Optimizations](../ch09_graph_optimizations/README.md)
