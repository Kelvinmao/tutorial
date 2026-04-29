# Chapter 1 — Exercises

## Exercise 1.1: Trace by Hand (Easy)
Take the expression `a = (10 - 2) * 3` and trace it through all 6 phases
**by hand** on paper. Write out:
1. The token list
2. The AST (draw it)
3. The three-address code IR
4. The optimized IR (what can be folded?)
5. The pseudo-assembly

Then run `python demo_pipeline.py "a = (10 - 2) * 3"` and compare.

## Exercise 1.2: Add a New Operator (Medium)
Modify `demo_pipeline.py` to support the `%` (modulo) operator.
You'll need to update: the tokenizer, the parser, the IR generator,
the optimizer, and the code generator.

Test with: `python demo_pipeline.py "x = 10 % 3"`
Expected result: `x = 1` after optimization.

## Exercise 1.3: Compiler vs. Interpreter (Easy)
Write a function `interpret(ast)` that **directly evaluates** an AST
(without generating IR or machine code). Compare:
- How many "steps" does the interpreter take for `2 + 3 * 4`?
- How many instructions does the compiler's optimized output have?

## Exercise 1.4: Error Handling (Hard)
Modify the pipeline to handle errors gracefully:
- What happens if you pass `"x = 2 +"` (incomplete expression)?
- Add error reporting that shows the position of the error.
- The error message should point to the problematic token.

## Exercise 1.5: Research (Easy)
Read about one real compiler (gcc, clang, or rustc) and identify which phases
from this chapter you can find in its documentation. Write a short paragraph
comparing the phases.
