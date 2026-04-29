# Chapter 14 — Auto-Tuning & Search

Automatically find the best optimization parameters (tile sizes, unroll
factors, etc.) instead of hand-tuning.

## What You'll Learn
- Defining a search space of optimization knobs
- Cost models for estimating performance
- Evolutionary search for auto-tuning
- Visualizing the search process

## Files
| File | Description |
|------|-------------|
| `search_space.py` | Define tunable parameters |
| `cost_model.py` | Simple analytical cost model |
| `auto_tuner.py` | Evolutionary search |
| `visualize_search.py` | Plot search progress |
| `exercises.md` | Practice problems |

## Run
```bash
python auto_tuner.py          # run auto-tuning search
python visualize_search.py    # plot search history
```

## Key Idea

Many compiler decisions are too hardware-dependent to pick with a fixed rule.
Tile sizes, unroll factors, vector widths, and parallelization choices can
change performance dramatically across CPUs, GPUs, and tensor shapes. An
auto-tuner treats those choices as a search problem.

This chapter uses three pieces:

- a **search space** of legal configurations,
- a **cost model** that estimates which configurations should be fast,
- and an **evolutionary search** loop that keeps improving candidates.

The cost model here is analytical and intentionally simple. A production tuner
usually measures real kernels, caches historical results, rejects invalid
schedules early, and balances compile time against runtime speedup.

## What To Watch For

Auto-tuning can overfit one shape or machine. When you extend this chapter,
track both the best configuration and the assumptions behind it: tensor shape,
dtype, target CPU/GPU, compiler flags, and measured variance.
