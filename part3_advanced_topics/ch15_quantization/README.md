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
