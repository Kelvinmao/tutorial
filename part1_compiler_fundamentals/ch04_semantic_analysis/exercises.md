# Chapter 4 — Exercises

## Exercise 4.1: Detect Duplicate Declarations (Easy)
Modify the type checker to detect when a variable is declared twice in the
same scope: `let x = 1; let x = 2` should be an error.

## Exercise 4.2: Function Arity Check (Medium)
Add a check that function calls pass the correct number of arguments:
```
def add(a: int, b: int) -> int: ...
add(1)        # Error: expected 2 args, got 1
add(1, 2, 3)  # Error: expected 2 args, got 3
```

## Exercise 4.3: Type Inference (Medium)
Currently `let z = x + y` requires looking up x and y's types to infer z's type.
Add support for inferring types through chains of operations:
```
let a = 1           # int
let b = 2.5         # float
let c = a + b       # float (promoted)
let d = c > 0       # bool
```
Verify the inferred types are correct.

## Exercise 4.4: Scope Visualization (Easy)
Extend `visualize_symbol_table.py` to also show which scope a variable
lookup resolves to. For example, if a function references a global variable,
draw a dashed arrow from the function scope to the global scope.

## Exercise 4.5: Type Coercion (Hard)
Implement automatic type coercion rules:
- `int` → `float` (widening, always safe)
- `float` → `int` (narrowing, emit warning)
- `bool` → `int` (true=1, false=0)
Insert explicit coercion nodes into the AST where needed.
