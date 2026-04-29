# Chapter 4: Semantic Analysis

## Learning Objectives

After this chapter you will:
- Understand why type checking is necessary beyond parsing
- Build a symbol table that tracks variable declarations and scopes
- Implement type inference and type checking for MiniLang
- Detect semantic errors: undeclared variables, type mismatches, etc.

## What is Semantic Analysis?

The parser checks that the program is **syntactically** correct (follows grammar
rules), but it doesn't check whether the program **makes sense**:

```python
let x: int = "hello"     # Syntax OK — but assigning string to int!
let y = x + z             # Syntax OK — but z is never declared!
```

**Semantic analysis** catches these logical errors by:
1. Building a **symbol table** of all variables and their types
2. **Type checking** every expression
3. Verifying **scope rules** (variables used within their scope)

## The Symbol Table

A symbol table maps variable names to their properties (type, scope level):

```
┌────────────────────────────────────────┐
│            Symbol Table                │
├──────────┬──────────┬─────────────────┤
│ Name     │ Type     │ Scope           │
├──────────┼──────────┼─────────────────┤
│ width    │ int      │ global (0)      │
│ height   │ int      │ global (0)      │
│ area     │ int      │ global (0)      │
│ square   │ fn→int   │ global (0)      │
│   x      │ int      │ square (1)      │
└──────────┴──────────┴─────────────────┘
```

Scopes are **nested**: a function body creates a new scope inside the global scope.

## Type Checking Rules

```
                ┌─────── int op int → int
                │
BinOp type ─────┼─────── float op float → float
                │
                ├─────── int op float → float  (promotion)
                │
                └─────── string + string → string (concat)

Comparison type ──────── any op any → bool
```

## Try It

```bash
python type_checker.py               # type-check a sample program
python visualize_symbol_table.py     # visualize scope tree
```

## Next Chapter

→ [Chapter 5: Intermediate Representation](../ch05_intermediate_representation/README.md)
