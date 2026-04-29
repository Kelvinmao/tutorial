# Chapter 15 — Quantization & Precision

Reduce model size and speed up inference by using lower-precision numbers.

## What You'll Learn
- INT8 quantization: scale + zero-point
- Calibration with representative data
- Mixed precision strategies
- Accuracy vs performance tradeoffs

## Files
| File | Description |
|------|-------------|
| `quantize.py` | INT8 symmetric & asymmetric quantization |
| `calibration.py` | Find optimal scale via calibration |
| `mixed_precision.py` | Layer-wise precision assignment |
| `visualize_quantization.py` | Distribution and error plots |
| `exercises.md` | Practice problems |

## Run
```bash
python quantize.py
python calibration.py
python mixed_precision.py
python visualize_quantization.py
```

## Key Idea

Quantization replaces high-precision floating-point values with lower-precision
integer values plus metadata such as scale and zero point. The compiler's job
is not just to change dtypes; it must preserve numerical meaning across
operators.

Common decisions include:

- **symmetric vs asymmetric** quantization,
- **per-tensor vs per-channel** scales,
- calibration data selection,
- accumulation type, usually int32 for int8 matmul,
- and where dequantize/requantize boundaries belong.

The scripts in this chapter show the mechanics on small tensors. Real
inference stacks also need operator coverage, accuracy validation, calibration
pipelines, and target-specific kernels that are actually faster than FP32.

## Common Failure Modes

Quantization can fail silently when calibration data is unrepresentative,
outliers dominate the scale, or mixed-precision choices introduce too much
rounding error in sensitive layers. Always compare model accuracy, not just
tensor-level reconstruction error.
