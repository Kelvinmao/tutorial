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
