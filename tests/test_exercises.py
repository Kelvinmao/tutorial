"""Tests for machine-checkable exercises.

Run only exercises:  pytest -m exercise -v
Run one chapter:     pytest -m exercise -k ch08 -v

Unimplemented stubs raise ``NotImplementedError`` → the test is skipped
so students see a clear PASSED / SKIPPED / FAILED breakdown.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helper: import a module by file path without polluting sys.modules
# ---------------------------------------------------------------------------

_LOADED: dict[str, object] = {}


def _load(name: str, chapter_dir: str) -> object:
    """Load *name*.py from *chapter_dir*, adding the dir to sys.path first."""
    if name in _LOADED:
        return _LOADED[name]
    chapter_path = REPO / chapter_dir
    # Ensure transitive imports from sibling chapters work.
    for dep in (
        "part1_compiler_fundamentals/ch02_lexer",
        "part1_compiler_fundamentals/ch03_parser_ast",
        "part1_compiler_fundamentals/ch05_intermediate_representation",
        "part1_compiler_fundamentals/ch06_optimization_passes",
        "part2_ai_compiler_basics/ch08_computation_graphs",
        "part2_ai_compiler_basics/ch09_graph_optimizations",
        "part3_advanced_topics/ch12_memory_optimization",
        "part3_advanced_topics/ch15_quantization",
    ):
        p = str(REPO / dep)
        if p not in sys.path:
            sys.path.insert(0, p)

    file_path = chapter_path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED[name] = mod
    return mod


def _skip_if_stub(func, *args, **kwargs):
    """Call *func*; if it raises NotImplementedError, skip the test."""
    try:
        return func(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Chapter 2 — Lexer: and / or / not keywords
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.exercise
class TestCh02LogicKeywords:
    """Exercise 2.1: tokenize_with_logic_ops"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load(
            "ch02_exercises",
            "part1_compiler_fundamentals/ch02_lexer",
        )

    def test_and_or_not_recognised(self):
        tokens = _skip_if_stub(
            self.mod.tokenize_with_logic_ops,
            "true and false or not true",
        )
        type_names = [t[0] for t in tokens]
        assert "AND" in type_names, "Expected AND token"
        assert "OR" in type_names, "Expected OR token"
        assert "NOT" in type_names, "Expected NOT token"

    def test_original_keywords_still_work(self):
        tokens = _skip_if_stub(
            self.mod.tokenize_with_logic_ops,
            "let x = 1",
        )
        type_names = [t[0] for t in tokens]
        assert "LET" in type_names
        assert "ID" in type_names

    def test_and_is_not_an_identifier(self):
        tokens = _skip_if_stub(
            self.mod.tokenize_with_logic_ops,
            "and",
        )
        type_names = [t[0] for t in tokens]
        assert "AND" in type_names
        assert "ID" not in type_names or type_names.count("ID") == 0


# ═══════════════════════════════════════════════════════════════════════════
# Chapter 6 — Algebraic simplification
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.exercise
class TestCh06AlgebraicSimplification:
    """Exercise 6.1: algebraic_simplification"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load(
            "ch06_exercises",
            "part1_compiler_fundamentals/ch06_optimization_passes",
        )
        from ir_builder import IRInstr, Op
        self.IRInstr = IRInstr
        self.Op = Op

    def _make_ir(self, *specs):
        """Build IR from (op, dst, src1[, src2]) tuples."""
        return [self.IRInstr(*s) for s in specs]

    def test_add_zero_eliminated(self):
        ir = self._make_ir(
            (self.Op.LOAD_CONST, "t0", "5"),
            (self.Op.LOAD_CONST, "t1", "0"),
            (self.Op.ADD, "t2", "t0", "t1"),  # 5 + 0 → 5
        )
        result = _skip_if_stub(self.mod.algebraic_simplification, ir)
        # t2 should become a COPY of t0, not an ADD
        t2_instr = [i for i in result if i.dst == "t2"][0]
        assert t2_instr.op in (self.Op.COPY, self.Op.LOAD_CONST)

    def test_mul_one_eliminated(self):
        ir = self._make_ir(
            (self.Op.LOAD_CONST, "t0", "7"),
            (self.Op.LOAD_CONST, "t1", "1"),
            (self.Op.MUL, "t2", "t0", "t1"),  # 7 * 1 → 7
        )
        result = _skip_if_stub(self.mod.algebraic_simplification, ir)
        t2_instr = [i for i in result if i.dst == "t2"][0]
        assert t2_instr.op in (self.Op.COPY, self.Op.LOAD_CONST)

    def test_mul_zero_becomes_zero(self):
        ir = self._make_ir(
            (self.Op.LOAD_CONST, "t0", "42"),
            (self.Op.LOAD_CONST, "t1", "0"),
            (self.Op.MUL, "t2", "t0", "t1"),  # 42 * 0 → 0
        )
        result = _skip_if_stub(self.mod.algebraic_simplification, ir)
        t2_instr = [i for i in result if i.dst == "t2"][0]
        assert t2_instr.op == self.Op.LOAD_CONST
        assert t2_instr.src1 in ("0", "0.0", 0)

    def test_sub_self_becomes_zero(self):
        ir = self._make_ir(
            (self.Op.LOAD_CONST, "t0", "9"),
            (self.Op.SUB, "t1", "t0", "t0"),  # t0 - t0 → 0
        )
        result = _skip_if_stub(self.mod.algebraic_simplification, ir)
        t1_instr = [i for i in result if i.dst == "t1"][0]
        assert t1_instr.op == self.Op.LOAD_CONST
        assert t1_instr.src1 in ("0", "0.0", 0)


# ═══════════════════════════════════════════════════════════════════════════
# Chapter 8 — Sigmoid forward & backward
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.exercise
class TestCh08Sigmoid:
    """Exercise 8.1: sigmoid_forward / sigmoid_backward"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load(
            "ch08_exercises",
            "part2_ai_compiler_basics/ch08_computation_graphs",
        )

    def test_sigmoid_forward_basic(self):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        out = _skip_if_stub(self.mod.sigmoid_forward, x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, expected, atol=1e-7)

    def test_sigmoid_forward_at_zero(self):
        out = _skip_if_stub(self.mod.sigmoid_forward, np.array([0.0]))
        assert abs(float(out[0]) - 0.5) < 1e-10

    def test_sigmoid_forward_large_values(self):
        x = np.array([100.0, -100.0])
        out = _skip_if_stub(self.mod.sigmoid_forward, x)
        assert out[0] == pytest.approx(1.0, abs=1e-10)
        assert out[1] == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_backward(self):
        x = np.array([0.0, 1.0, -1.0])
        sig = 1.0 / (1.0 + np.exp(-x))
        grad_out = np.ones_like(x)
        grad = _skip_if_stub(self.mod.sigmoid_backward, sig, grad_out)
        expected = sig * (1 - sig)
        np.testing.assert_allclose(grad, expected, atol=1e-7)

    def test_sigmoid_backward_chain_rule(self):
        x = np.array([0.5, -0.5])
        sig = 1.0 / (1.0 + np.exp(-x))
        grad_out = np.array([2.0, 3.0])
        grad = _skip_if_stub(self.mod.sigmoid_backward, sig, grad_out)
        expected = sig * (1 - sig) * grad_out
        np.testing.assert_allclose(grad, expected, atol=1e-7)


# ═══════════════════════════════════════════════════════════════════════════
# Chapter 9 — Dead node elimination
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.exercise
class TestCh09DeadNodeElimination:
    """Exercise 9.3: dead_node_elimination"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load(
            "ch09_exercises",
            "part2_ai_compiler_basics/ch09_graph_optimizations",
        )

    @staticmethod
    def _node(name, op, inputs, output):
        return {"name": name, "op": op, "inputs": inputs, "output": output}

    def test_removes_dead_nodes(self):
        nodes = [
            self._node("n1", "MatMul", ["x", "w"], "h"),
            self._node("n2", "ReLU",   ["h"],       "r"),
            self._node("n3", "Add",    ["x", "w"],  "dead"),  # unused
            self._node("n4", "Softmax",["r"],       "out"),
        ]
        result = _skip_if_stub(
            self.mod.dead_node_elimination, nodes, {"out"},
        )
        names = [n["name"] for n in result]
        assert "n3" not in names, "'dead' node should be removed"
        assert set(names) == {"n1", "n2", "n4"}

    def test_keeps_all_when_all_needed(self):
        nodes = [
            self._node("n1", "Input", [], "a"),
            self._node("n2", "Neg",   ["a"], "b"),
        ]
        result = _skip_if_stub(
            self.mod.dead_node_elimination, nodes, {"b"},
        )
        assert len(result) == 2

    def test_preserves_order(self):
        nodes = [
            self._node("n1", "Input",  [], "a"),
            self._node("n2", "Input",  [], "b"),
            self._node("n3", "Add",    ["a", "b"], "c"),
            self._node("n4", "Neg",    ["a"],      "d"),  # dead
        ]
        result = _skip_if_stub(
            self.mod.dead_node_elimination, nodes, {"c"},
        )
        names = [n["name"] for n in result]
        assert names == ["n1", "n2", "n3"]

    def test_multiple_outputs(self):
        nodes = [
            self._node("n1", "Input",  [],     "a"),
            self._node("n2", "Neg",    ["a"],  "b"),
            self._node("n3", "Abs",    ["a"],  "c"),
            self._node("n4", "Add",    ["b"],  "d"),  # dead
        ]
        result = _skip_if_stub(
            self.mod.dead_node_elimination, nodes, {"b", "c"},
        )
        names = [n["name"] for n in result]
        assert "n4" not in names
        assert set(names) == {"n1", "n2", "n3"}


# ═══════════════════════════════════════════════════════════════════════════
# Chapter 12 — First-fit buffer sharing
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.exercise
class TestCh12FirstFit:
    """Exercise 12.1: first_fit_buffer_sharing"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load(
            "ch12_exercises",
            "part3_advanced_topics/ch12_memory_optimization",
        )
        from memory_planner import MemBuffer
        self.MemBuffer = MemBuffer

    def _make_buffers(self, specs):
        """specs: list of (name, size, birth, death)"""
        return {
            name: self.MemBuffer(name=name, size=size, birth=birth, death=death)
            for name, size, birth, death in specs
        }

    def test_basic_reuse(self):
        bufs = self._make_buffers([
            ("a", 100, 0, 1),
            ("b", 100, 2, 3),  # can reuse a's physical buffer
        ])
        mapping, reuses, physical = _skip_if_stub(
            self.mod.first_fit_buffer_sharing, bufs,
        )
        assert reuses >= 1, "b should reuse a's buffer"
        assert len(physical) == 1

    def test_no_reuse_when_overlapping(self):
        bufs = self._make_buffers([
            ("a", 100, 0, 3),
            ("b", 100, 1, 2),  # overlaps with a
        ])
        mapping, reuses, physical = _skip_if_stub(
            self.mod.first_fit_buffer_sharing, bufs,
        )
        assert reuses == 0
        assert len(physical) == 2

    def test_first_fit_picks_first_not_smallest(self):
        bufs = self._make_buffers([
            ("a", 200, 0, 0),  # big, freed after step 0
            ("b", 100, 0, 0),  # small, freed after step 0
            ("c", 50,  1, 1),  # should pick first fitting = a (200), not b (100)
        ])
        mapping, reuses, physical = _skip_if_stub(
            self.mod.first_fit_buffer_sharing, bufs,
        )
        assert reuses >= 1
        # First-fit should pick the first available buffer, not the
        # best-fit (smallest).  With sorted-by-birth order, 'a' appears
        # first so first-fit reuses 'a' for 'c'.
        assert mapping["c"] == mapping["a"]

    def test_returns_correct_structure(self):
        bufs = self._make_buffers([("x", 64, 0, 0)])
        mapping, reuses, physical = _skip_if_stub(
            self.mod.first_fit_buffer_sharing, bufs,
        )
        assert isinstance(mapping, dict)
        assert isinstance(reuses, int)
        assert isinstance(physical, list)
        assert "x" in mapping


# ═══════════════════════════════════════════════════════════════════════════
# Chapter 15 — INT4 quantization
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.exercise
class TestCh15Int4:
    """Exercise 15.1: quantize_int4 / dequantize_int4"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load(
            "ch15_exercises",
            "part3_advanced_topics/ch15_quantization",
        )

    def test_quantize_range(self):
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        x_q, scale = _skip_if_stub(self.mod.quantize_int4, x)
        assert x_q.dtype == np.int8
        assert np.all(x_q >= -7)
        assert np.all(x_q <= 7)

    def test_scale_correct(self):
        x = np.array([3.5, -3.5], dtype=np.float32)
        x_q, scale = _skip_if_stub(self.mod.quantize_int4, x)
        expected_scale = 3.5 / 7.0
        assert scale == pytest.approx(expected_scale, rel=1e-5)

    def test_round_trip(self):
        np.random.seed(42)
        x = np.random.randn(10).astype(np.float32) * 0.5
        x_q, scale = _skip_if_stub(self.mod.quantize_int4, x)
        x_deq = _skip_if_stub(self.mod.dequantize_int4, x_q, scale)
        assert x_deq.dtype == np.float32
        # 4-bit quantization error should be bounded
        max_err = np.max(np.abs(x - x_deq))
        assert max_err < scale * 1.5  # within ~1 quantization step

    def test_dequantize_dtype(self):
        x_q = np.array([1, -1, 7, -7], dtype=np.int8)
        x_deq = _skip_if_stub(self.mod.dequantize_int4, x_q, 0.1)
        assert x_deq.dtype == np.float32

    def test_zeros_roundtrip(self):
        x = np.zeros(5, dtype=np.float32)
        x_q, scale = _skip_if_stub(self.mod.quantize_int4, x)
        assert np.all(x_q == 0)
