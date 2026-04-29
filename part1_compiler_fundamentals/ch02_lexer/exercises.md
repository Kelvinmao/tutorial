# Chapter 2 — Exercises

## Exercise 2.1: Extend the Lexer (Easy)
Add support for the `and`, `or`, `not` keywords. They should be recognized
as new token types `AND`, `OR`, `NOT`. Test with:
```
let x = true and false
let y = not x or true
```

## Exercise 2.2: Multi-line Strings (Medium)
Add support for triple-quoted strings (`"""..."""`). The string can span
multiple lines. Update the lexer and test with:
```
let msg = """hello
world"""
```

## Exercise 2.3: Error Recovery (Medium)
Currently the lexer stops at the first error. Modify it to **collect** errors
and continue scanning. At the end, report all errors with line/column info.
Test with a source that has multiple bad characters.

## Exercise 2.4: Benchmark (Easy)
Write a script that generates a large MiniLang program (10,000 lines) and
measures how long the lexer takes. How does it scale? Is it O(n)?

## Exercise 2.5: Regular Expression Lexer (Hard)
Rewrite the lexer using Python's `re` module with a single compiled regex
pattern (one pattern per token type, combined with `|`). Compare:
- Lines of code
- Performance (tokens/second)
- Error message quality
