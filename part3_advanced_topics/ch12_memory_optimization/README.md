# Chapter 12: Memory Optimization

## Learning Objectives

After this chapter you will:
- Understand why memory is the bottleneck in AI inference
- Implement liveness analysis on computation graphs
- Build a buffer sharing/reuse algorithm
- Visualize memory usage as a waterfall timeline

## The Memory Problem

Neural networks use huge amounts of memory for intermediate activations:
```
ResNet-50:  ~100 MB of intermediate activations (batch=1!)
GPT-2:     ~2 GB of intermediate activations
```

Without optimization, each operation allocates a new buffer. With **memory
planning**, buffers are reused once their data is no longer needed.

```
Without reuse:  [A][B][C][D][E]  = 5 buffers
With reuse:     [A][B][A][B][A]  = 2 buffers (3 reuses!)
```

## Try It

```bash
python memory_planner.py         # liveness analysis + buffer sharing
python inplace_mutation.py       # detect in-place operation opportunities
python visualize_memory.py       # memory waterfall diagram
```

## Next Chapter

→ [Chapter 13: Hardware Mapping](../ch13_hardware_mapping/README.md)
