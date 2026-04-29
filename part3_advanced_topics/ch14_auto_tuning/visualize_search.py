#!/usr/bin/env python3
"""
Chapter 14 — Visualize auto-tuning search progress.

Usage:
    python auto_tuner.py          # first generate history
    python visualize_search.py    # then visualize
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    history_file = "search_history.json"
    if not os.path.exists(history_file):
        print("Run auto_tuner.py first to generate search_history.json")
        # Generate inline
        from auto_tuner import evolutionary_search
        _, _, history = evolutionary_search(128, 128, 128)
    else:
        with open(history_file) as f:
            history = json.load(f)

    gens = [h[0] for h in history]
    best = [h[1] for h in history]
    avg = [h[2] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, "b-", linewidth=2, label="Best cost")
    ax.plot(gens, avg, "r--", alpha=0.6, label="Avg cost")
    ax.fill_between(gens, best, avg, alpha=0.1, color="blue")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (lower = better)")
    ax.set_title("Auto-Tuning Search Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("search_progress.png", dpi=120)
    print("Saved → search_progress.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
