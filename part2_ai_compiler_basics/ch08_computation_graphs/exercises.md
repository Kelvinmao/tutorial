# Chapter 8 — Exercises

## Exercise 8.1: Add More Ops (Easy)
Add `sigmoid(x) = 1/(1+exp(-x))` and `softmax(x)` operations to the
computation graph, with correct backward functions. Verify with numerical
gradients.

## Exercise 8.2: Training Loop (Medium)
Use the computation graph + autodiff to actually *train* a 2-layer MLP on
a simple dataset (XOR problem). Implement SGD: `W -= lr * grad_W`.
Plot the loss curve over 100 iterations.

## Exercise 8.3: Gradient Accumulation (Easy)
What happens when a tensor is used by multiple downstream nodes?
Test with `y = x + x` — the gradient of x should be 2, not 1.
Does your autodiff handle this correctly?

## Exercise 8.4: Memory Usage (Medium)
Track the peak memory usage during forward and backward passes.
How much memory does the backward pass require compared to forward?
Why does backprop need to store intermediate activations?

## Exercise 8.5: Graph Visualization (Easy)
Extend the visualization to show:
- Gradient flow (add backward edges in red)
- Tensor values on hover (save as HTML with tooltips)
- Color nodes by their execution time
