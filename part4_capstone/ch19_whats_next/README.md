# Chapter 19 — What's Next: Resources & Reading List

## Papers (Start Here)
1. **TVM** — "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (Chen et al., 2018)
2. **XLA** — "XLA: Optimizing Compiler for Machine Learning" (Google, 2017)
3. **Halide** — "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation" (Ragan-Kelley et al., 2013)
4. **Triton** — "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (Tillet et al., 2019)
5. **MLIR** — "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation" (Lattner et al., 2021)
6. **Ansor** — "Ansor: Generating High-Performance Tensor Programs for Deep Learning" (Zheng et al., 2020)
7. **FlexTensor** — "FlexTensor: An Automatic Schedule Exploration and Optimization Framework" (Zheng et al., 2020)

## Books
- *Engineering a Compiler* (Cooper & Torczon) — classic compiler textbook
- *Modern Compiler Implementation* (Appel) — another solid reference
- *The Dragon Book* (Aho et al.) — the definitive compilers reference

## Online Resources
- TVM tutorials: https://tvm.apache.org/docs/tutorial/
- MLIR documentation: https://mlir.llvm.org/
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- LLVM tutorial: https://llvm.org/docs/tutorial/

## Communities
- TVM Discuss forum
- MLIR Discord
- PyTorch compiler discussions (torch.compile / Inductor)
- r/MachineLearning, r/Compilers on Reddit

## Project Ideas
1. **Extend your mini compiler** with Conv2D, pooling, and batch norm
2. **Build an autotuner** that actually measures execution time (not just cost model)
3. **Write a Triton kernel** for a custom operation and integrate with PyTorch
4. **Implement a polyhedral scheduler** using ISL (Integer Set Library)
5. **Build a simple MLIR dialect** for your tensor operations
6. **Profile a real model** with TVM and compare against PyTorch eager mode

## Congratulations! 🎓
You've built an AI compiler from scratch — from lexing source code to generating
optimized machine code. You understand the key concepts that power TVM, XLA,
Triton, and other production AI compilers. Keep building!
