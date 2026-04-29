from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


SMOKE_SCRIPTS = [
    "part1_compiler_fundamentals/ch01_what_is_a_compiler/demo_pipeline.py",
    "part1_compiler_fundamentals/ch02_lexer/lexer.py",
    "part1_compiler_fundamentals/ch03_parser_ast/parser.py",
    "part1_compiler_fundamentals/ch04_semantic_analysis/type_checker.py",
    "part1_compiler_fundamentals/ch06_optimization_passes/constant_folding.py",
    "part2_ai_compiler_basics/ch08_computation_graphs/comp_graph.py",
    "part2_ai_compiler_basics/ch10_tensor_ir/tensor_expression.py",
    "part2_ai_compiler_basics/ch11_loop_optimizations/loop_tiling.py",
    "part3_advanced_topics/ch12_memory_optimization/memory_planner.py",
    "part3_advanced_topics/ch15_quantization/quantize.py",
]


@pytest.mark.parametrize("relative_path", SMOKE_SCRIPTS)
def test_representative_chapter_script_runs(relative_path: str) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(REPO_ROOT), env.get("PYTHONPATH", "")] if path
    )

    script = REPO_ROOT / relative_path
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"{relative_path} failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
