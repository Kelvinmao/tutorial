# Chapter 18 — Real-World AI Compiler Frameworks

Now that you've built your own mini compiler, explore how production
frameworks solve these same problems at scale.

## Frameworks Overview

### Apache TVM
- **What**: End-to-end deep learning compiler stack
- **Key ideas**: Relay IR, auto-scheduling (Ansor), VTA hardware backend
- **Install**: `pip install apache-tvm`
- **Docs**: https://tvm.apache.org/

### XLA (Accelerated Linear Algebra)
- **What**: Google's compiler for TensorFlow/JAX
- **Key ideas**: HLO IR, fusion, layout assignment, device-specific codegen
- **Used in**: TPU compilation, JAX `jit`

### Triton
- **What**: OpenAI's language for writing GPU kernels in Python
- **Key ideas**: Block-level programming, automatic tiling, MLIR backend
- **Install**: `pip install triton`

### MLIR (Multi-Level IR)
- **What**: Compiler infrastructure for building domain-specific compilers
- **Key ideas**: Dialects, progressive lowering, reusable passes
- **Part of**: LLVM project

### ONNX Runtime
- **What**: Cross-platform inference engine with graph optimizations
- **Key ideas**: ONNX format, execution providers, quantization tools
- **Install**: `pip install onnxruntime`

## Mapping to What You Learned

| Your Chapter | Production Equivalent |
|---|---|
| Ch02-03: Lexer/Parser | Framework model importers, ONNX parser |
| Ch05: IR | Relay (TVM), HLO (XLA), MLIR dialects |
| Ch06: Optimization passes | TVM relay passes, XLA passes |
| Ch08-09: Computation graphs | All frameworks use graph IR |
| Ch10-11: Tensor IR + loop opts | TVM TensorIR, Triton's block programs |
| Ch12: Memory | XLA buffer assignment, TVM memory planning |
| Ch14: Auto-tuning | TVM AutoTVM/Ansor, Triton's autotuner |
| Ch15: Quantization | ONNX Runtime quantization, TVM QNN |

## Next Steps
1. Pick ONE framework and work through its tutorial
2. Try compiling a real model (ResNet, BERT) with TVM or ONNX Runtime
3. Write a custom Triton kernel and benchmark against PyTorch
4. Read the TVM or XLA papers for deeper understanding
