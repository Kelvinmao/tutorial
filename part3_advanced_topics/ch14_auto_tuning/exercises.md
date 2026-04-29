# Chapter 14 — Exercises

## Exercise 1: Simulated Annealing
Implement simulated annealing as an alternative search strategy.
Compare convergence speed with evolutionary search.

## Exercise 2: Random Search Baseline
Implement pure random search and plot its convergence alongside
evolutionary search. How many samples does random need to match?

## Exercise 3: Bayesian Optimization
Use a Gaussian Process surrogate model (from `sklearn`) to guide
the search. Compare sample efficiency with evolutionary search.

## Exercise 4: Multi-Objective
Extend the cost model to return both **latency** and **memory usage**.
Use Pareto dominance to find the set of non-dominated configs.

## Exercise 5: Transfer Learning
Save the best configs for different matrix sizes. When tuning a new
size, initialize the population with configs from similar sizes.
Measure how much this speeds up convergence.
