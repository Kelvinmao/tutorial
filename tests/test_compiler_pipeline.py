from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

for chapter_dir in [
    "part1_compiler_fundamentals/ch02_lexer",
    "part1_compiler_fundamentals/ch03_parser_ast",
    "part1_compiler_fundamentals/ch04_semantic_analysis",
    "part1_compiler_fundamentals/ch05_intermediate_representation",
    "part1_compiler_fundamentals/ch06_optimization_passes",
]:
    sys.path.insert(0, str(REPO_ROOT / chapter_dir))


from ast_nodes import BinOp, Comparison, IfStmt, IntLit, LetDecl, PrintStmt  # noqa: E402
from cfg_builder import build_cfg  # noqa: E402
from common_subexpr_elim import common_subexpr_elimination  # noqa: E402
from constant_folding import constant_folding  # noqa: E402
from dead_code_elimination import dead_code_elimination  # noqa: E402
from ir_builder import IRInstr, Op, build_ir  # noqa: E402
from lexer import TokenType, tokenize  # noqa: E402
from parser import parse_source  # noqa: E402
from type_checker import TypeChecker  # noqa: E402


def token_types(source: str) -> list[TokenType]:
    return [token.type for token in tokenize(source)]


def test_lexer_tracks_keywords_literals_indentation_and_dedentation() -> None:
    source = 'let answer: int = 42\nif answer >= 40:\n    print("ok")\n'

    tokens = tokenize(source)

    assert [token.type for token in tokens] == [
        TokenType.LET,
        TokenType.ID,
        TokenType.COLON,
        TokenType.TYPE_INT,
        TokenType.EQUALS,
        TokenType.INT,
        TokenType.NEWLINE,
        TokenType.IF,
        TokenType.ID,
        TokenType.GEQ,
        TokenType.INT,
        TokenType.COLON,
        TokenType.NEWLINE,
        TokenType.INDENT,
        TokenType.PRINT,
        TokenType.LPAREN,
        TokenType.STRING,
        TokenType.RPAREN,
        TokenType.NEWLINE,
        TokenType.DEDENT,
        TokenType.EOF,
    ]
    assert tokens[1].value == "answer"
    assert tokens[5].value == 42
    assert tokens[16].value == "ok"


def test_parser_preserves_expression_precedence_and_block_structure() -> None:
    program = parse_source(
        """\
let result = 1 + 2 * 3
if result > 5:
    print(result)
else:
    print(0)
"""
    )

    let_stmt = program.body[0]
    if_stmt = program.body[1]

    assert isinstance(let_stmt, LetDecl)
    assert let_stmt.name == "result"
    assert isinstance(let_stmt.value, BinOp)
    assert let_stmt.value.op == "+"
    assert isinstance(let_stmt.value.right, BinOp)
    assert let_stmt.value.right.op == "*"

    assert isinstance(if_stmt, IfStmt)
    assert isinstance(if_stmt.condition, Comparison)
    assert if_stmt.condition.op == ">"
    assert len(if_stmt.then_body) == 1
    assert len(if_stmt.else_body) == 1
    assert isinstance(if_stmt.then_body[0], PrintStmt)
    assert isinstance(if_stmt.else_body[0], PrintStmt)


def test_type_checker_accepts_valid_program_and_records_function_symbols() -> None:
    program = parse_source(
        """\
let width: int = 10
let height: float = width + 2.5

def scale(x: int) -> int:
    return x * 2

if height > 10:
    print(scale(width))
"""
    )

    symbols, errors = TypeChecker().check(program)

    assert errors == []
    assert symbols.lookup("width").sym_type == "int"
    assert symbols.lookup("height").sym_type == "float"
    assert symbols.lookup("scale").sym_type == "fn(int)→int"


def test_type_checker_reports_type_mismatch_and_undeclared_names() -> None:
    program = parse_source(
        """\
let x: int = "hello"
print(y)
"""
    )

    _, errors = TypeChecker().check(program)
    messages = [error.message for error in errors]

    assert any("Type mismatch in let 'x'" in message for message in messages)
    assert any("Undeclared variable 'y'" in message for message in messages)


def test_ir_builder_emits_three_address_code_for_arithmetic_and_print() -> None:
    instructions = build_ir(
        """\
let x = 2 + 3
print(x)
"""
    )

    assert [instr.op for instr in instructions] == [
        Op.LOAD_CONST,
        Op.LOAD_CONST,
        Op.ADD,
        Op.COPY,
        Op.PRINT,
    ]
    assert instructions[2].src1 == "t0"
    assert instructions[2].src2 == "t1"
    assert instructions[3].dst == "x"
    assert instructions[4].src1 == "x"


def test_cfg_builder_connects_conditional_branches_to_then_and_else_blocks() -> None:
    instructions = build_ir(
        """\
let x = 1
if x > 0:
    print(x)
else:
    print(0)
"""
    )

    cfg = build_cfg(instructions)
    branch_blocks = [
        block
        for block in cfg.blocks.values()
        if block.instructions and block.instructions[-1].op == Op.BRANCH
    ]

    assert len(branch_blocks) == 1
    assert set(branch_blocks[0].successors) == {"then0", "else1"}
    assert cfg.blocks["then0"].successors == ["endif2"]
    assert cfg.blocks["else1"].successors == ["endif2"]
    assert set(cfg.blocks["endif2"].predecessors) == {"then0", "else1"}


def test_constant_folding_replaces_constant_arithmetic_with_load_const() -> None:
    instructions = [
        IRInstr(Op.LOAD_CONST, "t0", "2"),
        IRInstr(Op.LOAD_CONST, "t1", "3"),
        IRInstr(Op.ADD, "t2", "t0", "t1"),
    ]

    optimized = constant_folding(instructions)

    assert optimized[-1].op == Op.LOAD_CONST
    assert optimized[-1].dst == "t2"
    assert optimized[-1].src1 == "5"


def test_dead_code_elimination_removes_unused_temporaries_but_keeps_side_effects() -> None:
    instructions = [
        IRInstr(Op.LOAD_CONST, "t0", "2"),
        IRInstr(Op.LOAD_CONST, "t1", "3"),
        IRInstr(Op.ADD, "t2", "t0", "t1"),
        IRInstr(Op.LOAD_CONST, "t_unused", "99"),
        IRInstr(Op.COPY, "result", "t2"),
        IRInstr(Op.PRINT, src1="result"),
    ]

    optimized = dead_code_elimination(instructions)

    assert all(instr.dst != "t_unused" for instr in optimized)
    assert optimized[-1].op == Op.PRINT
    assert optimized[-1].src1 == "result"


def test_common_subexpression_elimination_reuses_commutative_expressions() -> None:
    instructions = [
        IRInstr(Op.LOAD_CONST, "a", "5"),
        IRInstr(Op.LOAD_CONST, "b", "3"),
        IRInstr(Op.ADD, "t0", "a", "b"),
        IRInstr(Op.ADD, "t1", "b", "a"),
        IRInstr(Op.COPY, "result", "t1"),
    ]

    optimized = common_subexpr_elimination(instructions)

    assert optimized[3].op == Op.COPY
    assert optimized[3].dst == "t1"
    assert optimized[3].src1 == "t0"
