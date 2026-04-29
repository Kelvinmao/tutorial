# Chapter 3: Parser & Abstract Syntax Tree

## Learning Objectives

After this chapter you will:
- Understand grammars and how they define language structure
- Know what an AST is and why it's different from a parse tree
- Build a recursive descent parser from scratch
- Visualize ASTs as graphviz trees

## What is Parsing?

The **parser** takes the flat token stream from the lexer and builds a
hierarchical **Abstract Syntax Tree (AST)** that captures the program's structure.

```
  [LET, ID("x"), COLON, TYPE_INT, EQUALS, INT(42)]
       │
       ▼
  ┌──────────┐
  │  PARSER  │
  └──────────┘
       │
       ▼
       LetDecl
      /   |    \
   "x"  "int"   Num(42)
```

## Grammar for MiniLang

We express the language syntax as a **context-free grammar** (CFG):

```
program     → statement*
statement   → let_decl | if_stmt | while_stmt | print_stmt
              | func_def | return_stmt | expr_stmt
let_decl    → "let" ID (":" type)? "=" expr NEWLINE
if_stmt     → "if" expr ":" NEWLINE INDENT statement+ DEDENT
              ("else" ":" NEWLINE INDENT statement+ DEDENT)?
while_stmt  → "while" expr ":" NEWLINE INDENT statement+ DEDENT
print_stmt  → "print" "(" expr ")" NEWLINE
func_def    → "def" ID "(" params? ")" ("->" type)? ":" NEWLINE
              INDENT statement+ DEDENT
return_stmt → "return" expr? NEWLINE
expr_stmt   → expr NEWLINE

expr        → comparison
comparison  → additive (("==" | "!=" | "<" | ">" | "<=" | ">=") additive)?
additive    → multiplicative (("+" | "-") multiplicative)*
multiplicative → unary (("*" | "/" | "%") unary)*
unary       → ("-")? primary
primary     → INT | FLOAT | BOOL | STRING | ID
              | ID "(" args? ")"         # function call
              | "(" expr ")"

type        → "int" | "float" | "bool"
params      → ID ":" type ("," ID ":" type)*
args        → expr ("," expr)*
```

## Recursive Descent Parsing

Each grammar rule becomes a function. The parser reads tokens left-to-right,
calling the appropriate function for each rule.

Example: parsing `2 + 3 * 4`
```
parse_additive()
  ├─ parse_multiplicative() → Num(2)
  ├─ see PLUS → consume it
  └─ parse_multiplicative()
       ├─ parse_primary() → Num(3)
       ├─ see STAR → consume it
       └─ parse_primary() → Num(4)
       └─ return BinOp(*, 3, 4)
  └─ return BinOp(+, 2, BinOp(*, 3, 4))    ← correct precedence!
```

## Try It

```bash
python parser.py                # parse sample program and print AST
python visualize_ast.py         # render AST as graphviz image
```

## Next Chapter

→ [Chapter 4: Semantic Analysis](../ch04_semantic_analysis/README.md)
