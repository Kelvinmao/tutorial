# Chapter 2: Lexer (Tokenizer)

## Learning Objectives

After this chapter you will:
- Understand what lexical analysis does and why it's the first phase
- Know the difference between tokens, lexemes, and patterns
- Build a hand-written lexer from scratch
- Visualize token streams with color-coded output

## What is a Lexer?

The **lexer** (also called **scanner** or **tokenizer**) reads raw source text
character by character and produces a stream of **tokens**.

```
  "x = 2 + 3.14"
       │
       ▼
  ┌─────────┐
  │  LEXER  │
  └─────────┘
       │
       ▼
  [ID("x"), EQ, INT(2), PLUS, FLOAT(3.14), EOF]
```

### Key Terminology

| Term | Meaning | Example |
|------|---------|---------|
| **Lexeme** | The raw character sequence | `"3.14"` |
| **Token type** | The category | `FLOAT` |
| **Token** | The pair (type, value) | `Token(FLOAT, 3.14)` |
| **Pattern** | Rule for recognizing a token type | `\d+\.\d+` |

### Why Separate Lexing from Parsing?

1. **Simplicity** — The parser doesn't need to worry about whitespace or comments
2. **Performance** — Character-by-character scanning is done once
3. **Modularity** — You can swap lexers without changing the parser

## Our Language: MiniLang

We'll build a lexer for **MiniLang**, a small language that grows across chapters:

```
# MiniLang — Chapter 2 subset
let x: int = 42
let y: float = 3.14
let z = x + y * 2
if z > 10:
    print(z)
```

Token types for MiniLang:
- Keywords: `let`, `if`, `else`, `while`, `print`, `def`, `return`
- Types: `int`, `float`, `bool`
- Literals: integers, floats, booleans (`true`/`false`), strings
- Operators: `+`, `-`, `*`, `/`, `=`, `==`, `!=`, `<`, `>`, `<=`, `>=`
- Delimiters: `(`, `)`, `:`, `,`
- Identifiers: variable/function names
- Comments: `# ...`

## Try It

```bash
python lexer.py                    # tokenize a sample program
python visualize_tokens.py         # color-coded visualization
```

## How the Lexer Works

The lexer uses a **state machine** approach — at each position, it looks at the
current character to decide which token to scan:

```
  ┌──── digit ──── scan_number() ──── INT or FLOAT
  │
  │──── alpha ──── scan_word() ──── KEYWORD or ID
  │
  │──── '"' ────── scan_string() ── STRING
  │
  │──── '+' ────── emit PLUS
  character ──┤
  │──── '=' ────── peek next:
  │                  '=' → emit EQEQ
  │                  else → emit EQUALS
  │
  │──── '#' ────── skip_comment()
  │
  └──── space ──── skip
```

## Next Chapter

→ [Chapter 3: Parser & AST](../ch03_parser_ast/README.md)
