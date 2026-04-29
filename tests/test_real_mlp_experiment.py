from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "part4_capstone/ch17_mini_compiler"))

from real_mlp_experiment import (  # noqa: E402
    build_deterministic_mlp,
    emit_native_mlp_c,
    run_experiment,
)


def test_native_mlp_codegen_embeds_deterministic_model_arrays() -> None:
    weights = build_deterministic_mlp()

    c_source = emit_native_mlp_c(weights, iterations=3)

    assert "#define INPUT_SIZE 16" in c_source
    assert "#define HIDDEN_SIZE 8" in c_source
    assert "#define OUTPUT_SIZE 4" in c_source
    assert "#define ITERATIONS 3" in c_source
    assert "static const float x[16]" in c_source
    assert "static const float w1[128]" in c_source
    assert "static const float b1[8]" in c_source
    assert "static const float w2[32]" in c_source
    assert "static const float b2[4]" in c_source
    assert "static void inference" in c_source


@pytest.mark.skipif(
    not any(shutil.which(candidate) for candidate in ("cc", "gcc", "clang")),
    reason="native MLP experiment requires a local C compiler",
)
def test_native_mlp_experiment_matches_numpy_reference() -> None:
    result = run_experiment(iterations=5)

    assert result.native_output.shape == (4,)
    assert result.numpy_output.shape == (4,)
    assert result.max_abs_diff < 1e-5
