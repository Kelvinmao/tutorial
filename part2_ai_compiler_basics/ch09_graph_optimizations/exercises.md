# Chapter 9 — Exercises

## Exercise 9.1: Custom Fusion Pattern (Medium)
Add a fusion pattern for `Add + ReLU` (common in residual connections).
Test with a ResNet-like block that has a skip connection.

## Exercise 9.2: Fusion Profitability (Medium)
Not all fusions are beneficial. Add a profitability model that estimates:
- Memory saved (intermediate buffers eliminated)
- Compute overhead (fused kernel may be less optimized)
Only fuse when the estimated benefit exceeds a threshold.

## Exercise 9.3: Dead Node Elimination (Easy)
Implement dead node elimination for the graph: remove nodes whose outputs
are not used by any other node. This is the graph-level equivalent of DCE.

## Exercise 9.4: Subgraph Matching (Hard)
Implement a general pattern matcher that can find arbitrary subgraph
patterns in the computation graph. This is how production compilers
like TVM and XLA find fusion opportunities.

## Exercise 9.5: Layout Optimization (Medium)
Given a graph where different ops prefer different layouts, insert the
minimum number of layout conversions. This is a graph coloring problem!
