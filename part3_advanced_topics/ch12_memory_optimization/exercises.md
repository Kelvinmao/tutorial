# Chapter 12 — Exercises

## Exercise 1: First-Fit vs Best-Fit
Modify `greedy_buffer_sharing()` to use **first-fit** instead of best-fit.
Compare the number of physical buffers and total memory with both strategies.

## Exercise 2: Alignment Constraints
Real accelerators require buffers aligned to 64 or 256 bytes.
Add an `alignment` parameter to the memory planner that rounds up buffer sizes.

## Exercise 3: Memory Fragmentation
Add a metric that measures **fragmentation**: the ratio of wasted space
inside physical buffers (allocated but unused bytes) to total allocated bytes.
Print fragmentation statistics after planning.

## Exercise 4: Skip Connections
Add a ResNet-style skip connection to the op list (e.g., `Add(relu1_out, relu2_out)`).
Observe how it extends lifetimes and reduces sharing opportunities.
Visualize the difference.

## Exercise 5: Peak Memory Budget
Implement a `plan_with_budget(ops, max_bytes)` function that refuses to
allocate if peak memory would exceed `max_bytes`. When the budget is exceeded,
suggest which buffer to recompute instead of store.
