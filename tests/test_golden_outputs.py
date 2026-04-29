from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

for chapter_dir in [
    "part1_compiler_fundamentals/ch02_lexer",
    "part1_compiler_fundamentals/ch03_parser_ast",
    "part1_compiler_fundamentals/ch05_intermediate_representation",
]:
    sys.path.insert(0, str(REPO_ROOT / chapter_dir))


from ast_nodes import pretty_print  # noqa: E402
from ir_builder import build_ir  # noqa: E402
from lexer import tokenize  # noqa: E402
from parser import parse_source  # noqa: E402


def read_fixture(name: str) -> str:
    return (GOLDEN_DIR / name).read_text().strip()


def read_source(name: str) -> str:
    return (GOLDEN_DIR / name).read_text()


def test_control_flow_token_names_match_golden_output() -> None:
    source = read_source("control_flow.minilang")

    actual = "\n".join(token.type.name for token in tokenize(source))

    assert actual == read_fixture("control_flow.tokens")


def test_control_flow_ast_text_matches_golden_output() -> None:
    source = read_source("control_flow.minilang")

    actual = pretty_print(parse_source(source))

    assert actual == read_fixture("control_flow.ast")


def test_control_flow_ir_text_matches_golden_output() -> None:
    source = read_source("control_flow.minilang")

    actual = "\n".join(str(instruction).strip() for instruction in build_ir(source))

    assert actual == read_fixture("control_flow.ir")
