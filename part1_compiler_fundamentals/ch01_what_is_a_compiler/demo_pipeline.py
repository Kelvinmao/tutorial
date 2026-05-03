#!/usr/bin/env python3
"""
Chapter 1 — Demo: Watch a tiny expression travel through every compiler phase.

Usage:
    python demo_pipeline.py              # default expression
    python demo_pipeline.py "y = 4 * x + 1"
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ─── Phase 0: Source Code ────────────────────────────────────────────────────

def show_source(source: str):
    console.print(Panel(source, title="[bold cyan]Phase 0 — Source Code[/]",
                        border_style="cyan"))

# ─── Phase 1: Lexer (Tokenizer) ────────────────────────────────────────────
#
# ALGORITHM: Lexical Analysis (Scanning / Tokenization)
#
# Historical context: Lexers date back to the 1950s–60s (Fortran, COBOL).
# The idea is that raw source text is just a stream of characters, which is
# hard for a parser to reason about. Lexing groups characters into meaningful
# "tokens" — the atoms of the language (keywords, numbers, operators).
#
# Problem solved: Convert a flat character stream into a sequence of
# classified tokens, discarding whitespace and comments.
#
#   Source: "y = 4 * x + 1"
#            │
#            ▼
#   ┌──────────────────────────────────────────────┐
#   │  y   =   4   *   x   +   1                  │
#   │  ↓   ↓   ↓   ↓   ↓   ↓   ↓                  │
#   │  ID  EQ  INT MUL ID  ADD INT                 │
#   └──────────────────────────────────────────────┘
#            │
#            ▼
#   Tokens: [(ID,"y"), (EQ,"="), (INT,"4"), (STAR,"*"),
#            (ID,"x"), (PLUS,"+"), (INT,"1"), (EOF)]
#
# How it works: Walk through the source character by character. At each
# position, decide what kind of token starts here (letter → identifier or
# keyword, digit → number, symbol → operator). Accumulate characters until
# the token ends, then emit a (type, value) record. This is essentially a
# finite state machine running over the input.
#
# This simplified version handles: identifiers, integers, floats, and
# single-character operators. Chapter 2 expands this into a full-featured
# lexer with keywords, strings, indentation tracking, and error reporting.
# ────────────────────────────────────────────────────────────────────────────

TOKEN_TYPES = {
    "ID": "green", "INT": "yellow", "FLOAT": "yellow",
    "PLUS": "red", "MINUS": "red", "STAR": "red", "SLASH": "red",
    "EQUALS": "magenta", "LPAREN": "blue", "RPAREN": "blue",
    "EOF": "dim",
}

def tokenize(source: str) -> list[dict]:
    """Minimal tokenizer for arithmetic expressions with assignment."""
    tokens = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
        elif ch.isalpha() or ch == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            tokens.append({"type": "ID", "value": source[start:i]})
        elif ch.isdigit():
            start = i
            while i < len(source) and (source[i].isdigit() or source[i] == '.'):
                i += 1
            val = source[start:i]
            tok_type = "FLOAT" if '.' in val else "INT"
            tokens.append({"type": tok_type, "value": val})
        elif ch == '+': tokens.append({"type": "PLUS", "value": "+"}); i += 1
        elif ch == '-': tokens.append({"type": "MINUS", "value": "-"}); i += 1
        elif ch == '*': tokens.append({"type": "STAR", "value": "*"}); i += 1
        elif ch == '/': tokens.append({"type": "SLASH", "value": "/"}); i += 1
        elif ch == '=': tokens.append({"type": "EQUALS", "value": "="}); i += 1
        elif ch == '(': tokens.append({"type": "LPAREN", "value": "("}); i += 1
        elif ch == ')': tokens.append({"type": "RPAREN", "value": ")"}); i += 1
        else:
            tokens.append({"type": "UNKNOWN", "value": ch}); i += 1
    tokens.append({"type": "EOF", "value": ""})
    return tokens


def show_tokens(tokens: list[dict]):
    table = Table(title="Tokens", box=box.ROUNDED)
    table.add_column("Index", style="dim")
    table.add_column("Type", style="bold")
    table.add_column("Value")
    for i, tok in enumerate(tokens):
        color = TOKEN_TYPES.get(tok["type"], "white")
        table.add_row(str(i), f"[{color}]{tok['type']}[/]",
                      f"[{color}]{tok['value']}[/]")
    console.print(Panel(table, title="[bold cyan]Phase 1 — Lexical Analysis[/]",
                        border_style="cyan"))

# ─── Phase 2: Parser (AST) ──────────────────────────────────────────────────
#
# ALGORITHM: Recursive Descent Parsing
#
# Historical context: Recursive descent was described in the early 1960s
# and became popular because it maps grammar rules directly to functions.
# Each grammar production (e.g., "expr → term (('+' | '-') term)*") becomes
# a function that calls other functions, making the code mirror the grammar.
#
# Problem solved: A flat token list has no structure — we need a tree
# (the Abstract Syntax Tree, AST) that captures how sub-expressions nest.
# For example, in "2 + 3 * 4", multiplication must bind tighter than
# addition — the AST must reflect: Add(2, Mul(3, 4)).
#
#   Tokens: ID(y) = INT(4) * ID(x) + INT(1)
#           │
#           ▼  Recursive Descent
#
#       ┌─── Assign ───┐
#       │              │
#     "y"          BinOp(+)
#                ┌─────┴─────┐
#            BinOp(*)      Num(1)
#           ┌────┴────┐
#         Num(4)    Var(x)
#
#   parse_additive()
#     └── parse_multiplicative()    ← called first (higher precedence)
#           └── parse_primary()     ← handles leaves: Num, Var, (expr)
#
# How it works:
# 1. Each precedence level gets its own function:
#    parse_additive() → handles +, - (lowest precedence)
#    parse_multiplicative() → handles *, / (higher precedence)
#    parse_primary() → handles literals, variables, parens (highest)
# 2. Higher-precedence functions are called *as operands* of lower ones.
#    So parse_additive() calls parse_multiplicative() for its left & right
#    children, which ensures * binds before +.
# 3. pos[0] (a mutable list used as a pointer) tracks the current token.
#    peek() looks at the current token, eat() consumes it.
# 4. Assignment is handled by lookahead: if the first two tokens are
#    ID EQUALS, parse it as an assignment; otherwise parse as expression.
#
# This approach is called "top-down" because we start from the start symbol
# and recursively descend into sub-rules. Chapter 3 expands this to handle
# if/while/function definitions and indentation-based blocks.
# ────────────────────────────────────────────────────────────────────────────

class ASTNode:
    pass

class Assign(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    def __repr__(self):
        return f"Assign({self.name}, {self.value})"

class BinOp(ASTNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
    def __repr__(self):
        return f"BinOp({self.op}, {self.left}, {self.right})"

class Num(ASTNode):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"Num({self.value})"

class Var(ASTNode):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"Var({self.name})"


def parse(tokens: list[dict]) -> ASTNode:
    """Recursive-descent parser for: <id> = <expr> | <expr>"""
    pos = [0]

    def peek():
        return tokens[pos[0]]

    def eat(expected_type=None):
        tok = tokens[pos[0]]
        if expected_type and tok["type"] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok['type']}")
        pos[0] += 1
        return tok

    def parse_expr():
        return parse_additive()

    def parse_additive():
        left = parse_multiplicative()
        while peek()["type"] in ("PLUS", "MINUS"):
            op = eat()["value"]
            right = parse_multiplicative()
            left = BinOp(op, left, right)
        return left

    def parse_multiplicative():
        left = parse_primary()
        while peek()["type"] in ("STAR", "SLASH"):
            op = eat()["value"]
            right = parse_primary()
            left = BinOp(op, left, right)
        return left

    def parse_primary():
        tok = peek()
        if tok["type"] in ("INT", "FLOAT"):
            eat()
            return Num(float(tok["value"]) if '.' in tok["value"] else int(tok["value"]))
        elif tok["type"] == "ID":
            eat()
            return Var(tok["value"])
        elif tok["type"] == "LPAREN":
            eat("LPAREN")
            node = parse_expr()
            eat("RPAREN")
            return node
        else:
            raise SyntaxError(f"Unexpected token: {tok}")

    # Check for assignment: ID = expr
    if (tokens[0]["type"] == "ID" and
        len(tokens) > 1 and tokens[1]["type"] == "EQUALS"):
        name = eat("ID")["value"]
        eat("EQUALS")
        value = parse_expr()
        return Assign(name, value)
    else:
        return parse_expr()


def ast_to_str(node: ASTNode, indent: int = 0) -> str:
    """Pretty-print AST as indented tree."""
    prefix = "  " * indent
    if isinstance(node, Assign):
        return (f"{prefix}Assign\n"
                f"{prefix}├─ name: {node.name}\n"
                f"{prefix}└─ value:\n"
                f"{ast_to_str(node.value, indent + 3)}")
    elif isinstance(node, BinOp):
        return (f"{prefix}BinOp '{node.op}'\n"
                f"{prefix}├─ left:\n{ast_to_str(node.left, indent + 3)}\n"
                f"{prefix}└─ right:\n{ast_to_str(node.right, indent + 3)}")
    elif isinstance(node, Num):
        return f"{prefix}Num({node.value})"
    elif isinstance(node, Var):
        return f"{prefix}Var({node.name})"
    return f"{prefix}???"


def show_ast(ast: ASTNode):
    console.print(Panel(ast_to_str(ast),
                        title="[bold cyan]Phase 2 — Syntax Analysis (AST)[/]",
                        border_style="cyan"))

# ─── Phase 3: Semantic Analysis ─────────────────────────────────────────────
#
# ALGORITHM: Type Checking via AST Traversal + Symbol Table
#
# Historical context: Semantic analysis became essential in the 1960s–70s
# as languages introduced type systems (Algol 68, Pascal, C). The parser
# checks *syntax* (structural validity) but cannot catch *meaning* errors
# like using an undeclared variable or adding a string to an integer.
#
# Problem solved: Verify that operations are semantically valid. Track
# which variables exist and what types they hold. Detect type mismatches.
#
#       ┌─── Assign ───┐
#       │              │
#     "y"          BinOp(+)       Symbol Table:
#                ┌─────┴────┐     ┌──────┬──────┐
#            BinOp(*)     Num(1)  │ Name │ Type │
#           ┌────┴────┐   :int   ├──────┼──────┤
#         Num(4)    Var(x)        │  x   │ int  │
#         :int      :int ←lookup  │  y   │ int  │
#                                 └──────┴──────┘
#         int * int = int
#              int + int = int
#                   y ← int   (recorded)
#
# How it works:
# 1. Walk the AST recursively. For each node, infer or check its type.
# 2. Maintain a symbol_table dict mapping variable names to their types.
# 3. For BinOp nodes: check both operands, then apply promotion rules
#    (e.g., int + float → float).
# 4. For Assign nodes: infer the type of the value, record it in the table.
# 5. For Var nodes: look up the variable in the table.
#
# This simplified version auto-declares unknown variables as "int".
# Chapter 4 builds a full scoped symbol table with error reporting.
# ────────────────────────────────────────────────────────────────────────────

def type_check(node: ASTNode, symbol_table: dict | None = None) -> str:
    """Minimal type checker — returns the inferred type."""
    if symbol_table is None:
        symbol_table = {}

    if isinstance(node, Num):
        return "float" if isinstance(node.value, float) else "int"
    elif isinstance(node, Var):
        if node.name not in symbol_table:
            symbol_table[node.name] = "int"  # assume int for free vars
        return symbol_table[node.name]
    elif isinstance(node, BinOp):
        lt = type_check(node.left, symbol_table)
        rt = type_check(node.right, symbol_table)
        # float promotion
        return "float" if "float" in (lt, rt) else "int"
    elif isinstance(node, Assign):
        vt = type_check(node.value, symbol_table)
        symbol_table[node.name] = vt
        return vt
    return "unknown"


def show_semantic(ast: ASTNode):
    sym = {}
    result_type = type_check(ast, sym)
    table = Table(title="Symbol Table", box=box.ROUNDED)
    table.add_column("Variable")
    table.add_column("Type")
    for name, typ in sym.items():
        table.add_row(name, typ)
    info = f"Expression type: [bold]{result_type}[/]\n"
    console.print(Panel.fit(
        info, title="[bold cyan]Phase 3 — Semantic Analysis[/]",
        border_style="cyan"))
    console.print(table)

# ─── Phase 4: IR (Three-Address Code) ──────────────────────────────────────
#
# ALGORITHM: AST → Three-Address Code (TAC) Lowering
#
# Historical context: Three-address code was formalized in the 1970s
# (Aho, Ullman — the "Dragon Book"). The key insight is that complex
# nested expressions can be flattened into a sequence of simple instructions,
# each with at most three operands: dst = src1 op src2.
#
# Problem solved: The AST is tree-shaped and tied to the source language.
# We need a flat, machine-like representation that is easy to optimize and
# translate to assembly. TAC is that intermediate representation (IR).
#
#       ┌─── Assign ───┐             Three-Address Code:
#       │              │              ┌─────────────────────┐
#     "y"          BinOp(+)     ──►   │ t0 = 4              │
#                ┌─────┴────┐         │ t1 = t0 * x         │
#            BinOp(*)     Num(1)      │ t2 = 1              │
#           ┌────┴────┐               │ t3 = t1 + t2        │
#         Num(4)    Var(x)            │ y  = t3             │
#                                     └─────────────────────┘
#   Tree (recursive)            ──►   Flat list (sequential)
#
# How it works:
# 1. Recursively walk the AST. For each sub-expression, generate IR
#    instructions and return the name of the register holding the result.
# 2. Each intermediate result gets a fresh temporary name (t0, t1, ...).
# 3. Leaf nodes (Num) emit a LOAD instruction: t0 = 42.
# 4. Binary ops emit: t2 = t0 + t1.
# 5. Assignments emit a COPY: x = t2.
#
# The result is a flat list of instructions that a backend can translate
# to assembly 1-to-1. Chapter 5 extends this with labels, jumps, branches,
# function calls, and builds a Control Flow Graph (CFG) from the IR.
# ────────────────────────────────────────────────────────────────────────────

class IRInstruction:
    def __init__(self, op, dst, src1, src2=None):
        self.op = op
        self.dst = dst
        self.src1 = src1
        self.src2 = src2

    def __repr__(self):
        if self.op == "LOAD":
            return f"{self.dst} = {self.src1}"
        elif self.op == "COPY":
            return f"{self.dst} = {self.src1}"
        else:
            return f"{self.dst} = {self.src1} {self.op} {self.src2}"


def generate_ir(node: ASTNode, instructions: list | None = None,
                counter: list | None = None) -> str:
    """Generate three-address code from AST. Returns the result register name."""
    if instructions is None:
        instructions = []
    if counter is None:
        counter = [0]

    def new_temp():
        name = f"t{counter[0]}"
        counter[0] += 1
        return name

    if isinstance(node, Num):
        t = new_temp()
        instructions.append(IRInstruction("LOAD", t, node.value))
        return t
    elif isinstance(node, Var):
        return node.name
    elif isinstance(node, BinOp):
        l = generate_ir(node.left, instructions, counter)
        r = generate_ir(node.right, instructions, counter)
        t = new_temp()
        instructions.append(IRInstruction(node.op, t, l, r))
        return t
    elif isinstance(node, Assign):
        val = generate_ir(node.value, instructions, counter)
        instructions.append(IRInstruction("COPY", node.name, val))
        return node.name
    return "?"


def show_ir(ast: ASTNode) -> list[IRInstruction]:
    instructions: list[IRInstruction] = []
    generate_ir(ast, instructions)
    ir_text = "\n".join(f"  {instr}" for instr in instructions)
    console.print(Panel(ir_text,
                        title="[bold cyan]Phase 4 — Intermediate Representation[/]",
                        border_style="cyan"))
    return instructions

# ─── Phase 5: Optimization ─────────────────────────────────────────────────
#
# ALGORITHMS: Constant Folding + Dead Code Elimination
#
# Historical context: Compiler optimizations emerged in the 1960s–70s
# (Frances Allen's work on control flow analysis won the Turing Award).
# The idea: if the compiler can compute something at compile time, don't
# make the CPU compute it at run time.
#
# Problem solved: Eliminate redundant computation. For example:
#   t0 = 2; t1 = 3; t2 = t0 * t1  →  t2 = 6  (constant folding)
#   Then t0 and t1 are unused       →  removed (dead code elimination)
#
#   Before optimization:          After optimization:
#   ┌───────────────────┐         ┌───────────────────┐
#   │ t0 = 4            │         │                   │
#   │ t1 = t0 * x       │  ──►   │ t1 = 4 * x        │  (t0 inlined)
#   │ t2 = 1            │         │                   │
#   │ t3 = t1 + t2      │  ──►   │ t3 = t1 + 1       │  (t2 inlined)
#   │ y  = t3           │         │ y  = t3           │
#   └───────────────────┘         └───────────────────┘
#   5 instructions                3 instructions (40% reduction)
#
# How it works (Constant Folding):
# 1. Maintain a dict "known" mapping register names to their constant values.
# 2. LOAD instructions record the constant: known["t0"] = 2.
# 3. For arithmetic instructions, look up both operands in "known".
#    If both are constants, compute the result immediately and replace
#    the instruction with a LOAD of the computed value.
# 4. Propagate the result: known["t2"] = 6.
#
# How it works (Dead Code Elimination):
# 1. Scan all instructions to find which registers are actually *read*
#    (appear as src1 or src2). Collect these into a "used" set.
# 2. Remove any instruction whose destination is a temporary (t-prefixed)
#    that is NOT in the used set — nobody reads it, so it's dead.
#
# Chapter 6 implements these as separate, reusable passes plus CSE.
# ────────────────────────────────────────────────────────────────────────────

def optimize_ir(instructions: list[IRInstruction]) -> list[IRInstruction]:
    """Simple constant-folding pass."""
    known = {}  # register -> known constant value
    optimized = []

    for instr in instructions:
        if instr.op == "LOAD":
            known[instr.dst] = instr.src1
            optimized.append(instr)
        elif instr.op == "COPY":
            if instr.src1 in known:
                known[instr.dst] = known[instr.src1]
            optimized.append(instr)
        else:
            v1 = known.get(instr.src1)
            v2 = known.get(instr.src2)
            if v1 is not None and v2 is not None:
                # Both operands are known constants — fold!
                ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
                       "*": lambda a, b: a * b, "/": lambda a, b: a / b}
                if instr.op in ops:
                    result = ops[instr.op](v1, v2)
                    known[instr.dst] = result
                    optimized.append(IRInstruction("LOAD", instr.dst, result))
                    continue
            optimized.append(instr)

    # Dead code elimination: remove unused temps
    used = set()
    for instr in optimized:
        if instr.src1 is not None and isinstance(instr.src1, str):
            used.add(instr.src1)
        if instr.src2 is not None and isinstance(instr.src2, str):
            used.add(instr.src2)

    final = []
    for instr in optimized:
        if instr.dst.startswith("t") and instr.dst not in used:
            continue  # dead temp
        final.append(instr)

    return final


def show_optimization(before: list[IRInstruction], after: list[IRInstruction]):
    before_text = "\n".join(f"  {i}" for i in before)
    after_text = "\n".join(f"  {i}" for i in after)
    text = (f"[dim]Before ({len(before)} instructions):[/]\n{before_text}\n\n"
            f"[bold green]After ({len(after)} instructions):[/]\n{after_text}")
    console.print(Panel(text,
                        title="[bold cyan]Phase 5 — Optimization[/]",
                        border_style="cyan"))

# ─── Phase 6: Code Generation ──────────────────────────────────────────────
#
# ALGORITHM: IR → Assembly Translation (Instruction Selection)
#
# Historical context: Code generation was one of the original compiler
# problems — early Fortran (1957) had to translate high-level math to
# IBM 704 machine instructions. Modern compilers use LLVM or GCC backends
# that handle instruction selection, register allocation, and scheduling.
#
# Problem solved: Translate the abstract IR instructions into concrete
# machine instructions that a CPU can execute.
#
#   IR Instruction           Assembly Output
#   ┌───────────────────┐    ┌──────────────────────────┐
#   │ t1 = 4 * x        │───►│ MOV  t1, x               │
#   │                   │    │ MUL  t1, #4              │
#   ├───────────────────┤    ├──────────────────────────┤
#   │ t3 = t1 + 1       │───►│ MOV  t3, t1              │
#   │                   │    │ ADD  t3, #1              │
#   ├───────────────────┤    ├──────────────────────────┤
#   │ y  = t3           │───►│ MOV  y,  t3              │
#   └───────────────────┘    └──────────────────────────┘
#
# How it works:
# 1. Walk the flat IR instruction list.
# 2. Map each IR opcode to the corresponding assembly mnemonic:
#    LOAD → MOV dst, #immediate
#    COPY → MOV dst, src
#    + → ADD, - → SUB, * → MUL, / → DIV
# 3. Emit the pseudo-assembly as text.
#
# This is a simplified 1-to-1 mapping. A real backend would also do:
# - Register allocation (map infinite virtual registers to finite HW regs)
# - Instruction scheduling (reorder to hide latency)
# - Peephole optimization (combine adjacent instructions)
#
# Chapter 7 replaces this with a real LLVM backend that produces native
# machine code and can JIT-execute it.
# ────────────────────────────────────────────────────────────────────────────

def codegen(instructions: list[IRInstruction]) -> str:
    """Generate pseudo-assembly from optimized IR."""
    lines = []
    for instr in instructions:
        if instr.op == "LOAD":
            lines.append(f"  MOV  {instr.dst}, #{instr.src1}")
        elif instr.op == "COPY":
            lines.append(f"  MOV  {instr.dst}, {instr.src1}")
        elif instr.op == "+":
            lines.append(f"  ADD  {instr.dst}, {instr.src1}, {instr.src2}")
        elif instr.op == "-":
            lines.append(f"  SUB  {instr.dst}, {instr.src1}, {instr.src2}")
        elif instr.op == "*":
            lines.append(f"  MUL  {instr.dst}, {instr.src1}, {instr.src2}")
        elif instr.op == "/":
            lines.append(f"  DIV  {instr.dst}, {instr.src1}, {instr.src2}")
    return "\n".join(lines)


def show_codegen(code: str):
    console.print(Panel(code,
                        title="[bold cyan]Phase 6 — Code Generation[/]",
                        border_style="cyan"))

# ─── Main Pipeline ─────────────────────────────────────────────────────────

def run_pipeline(source: str):
    console.print()
    console.rule("[bold magenta]Compiler Pipeline Demo[/]")
    console.print()

    # Phase 0
    show_source(source)
    console.print()

    # Phase 1
    tokens = tokenize(source)
    show_tokens(tokens)
    console.print()

    # Phase 2
    ast = parse(tokens)
    show_ast(ast)
    console.print()

    # Phase 3
    show_semantic(ast)
    console.print()

    # Phase 4
    ir = show_ir(ast)
    console.print()

    # Phase 5
    optimized = optimize_ir(ir)
    show_optimization(ir, optimized)
    console.print()

    # Phase 6
    code = codegen(optimized)
    show_codegen(code)
    console.print()

    console.rule("[bold green]Pipeline Complete![/]")


if __name__ == "__main__":
    expr = sys.argv[1] if len(sys.argv) > 1 else "result = 2 + 3 * 4"
    run_pipeline(expr)
