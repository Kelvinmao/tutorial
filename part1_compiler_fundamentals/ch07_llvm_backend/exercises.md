# Chapter 7 — Exercises

## Exercise 7.1: Explore LLVM IR (Easy)
Modify the source expression and observe how the LLVM IR changes.
Try: `let x = 100 / 0` — what does LLVM do with division by zero?

## Exercise 7.2: LLVM Optimization Levels (Easy)
Run `python llvm_optimizer.py` and compare -O0 through -O3.
For the sample program, which level is "good enough"?
What additional optimization does -O3 do beyond -O2?

## Exercise 7.3: Floating Point Support (Medium)
Extend the LLVM emitter to support `double` (64-bit float) values.
When the MiniLang type is `float`, emit `fadd`/`fsub`/`fmul`/`fdiv`
instead of `add`/`sub`/`mul`/`sdiv`.

## Exercise 7.4: Function Calls (Hard)
Extend the LLVM emitter to support MiniLang function definitions and calls.
Generate proper LLVM `define` and `call` instructions. Test with:
```
def add(x: int, y: int) -> int:
    return x + y
let result = add(3, 4)
```

## Exercise 7.5: Emit Assembly (Medium)
Use `target_machine.emit_assembly()` to see the actual x86 assembly
generated from your LLVM IR. Compare the assembly before and after
optimization. How many instructions does LLVM eliminate?
