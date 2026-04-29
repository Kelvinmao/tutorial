# Chapter 3 — Exercises

## Exercise 3.1: Operator Precedence (Easy)
Parse the expression `1 + 2 * 3 - 4 / 2` and draw the resulting AST by hand.
Verify by running `python visualize_ast.py "let x = 1 + 2 * 3 - 4 / 2"`.
Does `*` bind tighter than `+`? Why?

## Exercise 3.2: Add `for` Loops (Medium)
Add a `for` loop statement to the parser and AST:
```
for i in range(10):
    print(i)
```
You'll need: a `ForStmt` AST node, a `RANGE` keyword in the lexer,
and a `_parse_for()` method in the parser.

## Exercise 3.3: Error Messages (Medium)
Improve error messages to show the line of source code where the error occurred:
```
Parse error at L3:5: Expected RPAREN
  let y = foo(1, 2
                   ^
```
Hint: keep the original source string and use token line/col info.

## Exercise 3.4: Parse Tree vs AST (Easy)
What's the difference between a **parse tree** (concrete syntax tree) and an
**AST** (abstract syntax tree)? Modify the parser to optionally produce a
parse tree where every grammar rule becomes a node, including ones that
the AST omits (like parentheses).

## Exercise 3.5: Left vs Right Recursion (Hard)
The current parser uses left-to-right associativity for `+` and `*`.
Modify the parser to support right-associative operators (like `**` for power):
```
let x = 2 ** 3 ** 2   # should parse as 2 ** (3 ** 2) = 512, not (2**3)**2 = 64
```
Why can't you use the same iterative approach as for `+` and `*`?
