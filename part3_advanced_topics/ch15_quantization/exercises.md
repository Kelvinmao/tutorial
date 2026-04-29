# Chapter 15 — Exercises

## Exercise 1: INT4 Quantization
Implement 4-bit quantization. How much additional error does this add
compared to INT8? Plot the error curve from 2-bit to 16-bit.

## Exercise 2: Per-Channel Quantization
Instead of one scale per tensor, use one scale per output channel.
Compare error with per-tensor quantization.

## Exercise 3: Quantization-Aware Training (Simulation)
Simulate QAT by adding fake quantization (quantize → dequantize) in
the forward pass during training. Use the straight-through estimator
for gradients. Compare final accuracy with post-training quantization.

## Exercise 4: Activation Quantization
Extend quantization to activations (not just weights). Run a small
MLP with quantized weights AND activations. Measure accuracy impact.

## Exercise 5: Dynamic Quantization
Implement dynamic quantization where the scale is computed per-batch
at inference time. Compare with static calibration.
