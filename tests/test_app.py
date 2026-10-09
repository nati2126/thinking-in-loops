"""Phase 5 tests: the demo's helper functions (skipped if no trained checkpoint exists)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [ROOT / "app/model/ut.pt", ROOT / "checkpoints/ut_s0/best.pt", ROOT / "checkpoints/ut/best.pt"]
pytestmark = pytest.mark.skipif(not any(p.exists() for p in CANDIDATES), reason="no trained UT checkpoint")


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_app", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_halting_loop(demo):
    assert demo.halting_loop([0.1, 0.6, 0.9], 0.5) == 2
    assert demo.halting_loop([0.1, 0.2, 0.3], 0.5) == 3  # never halts -> last loop


def test_generate_and_solve(demo):
    from src.data.sudoku import from_string

    puzzle = demo.generate(7)
    assert from_string(puzzle).count(0) == 7
    state, _slider, html, fig, status = demo.solve_puzzle(puzzle, 12, 0.5)
    assert len(state["preds"]) == 12 and len(state["halt"]) == 12
    assert html.count("<td") == 16
    assert "Halted" in status or "Never crossed" in status
    html3, _ = demo.show_loop(state, 3, 0.5)
    assert "Loop 3" in html3


def test_bad_input_raises(demo):
    import gradio as gr

    with pytest.raises(gr.Error):
        demo.solve_puzzle("11" + "0" * 14, 8, 0.5)  # rule violation
    with pytest.raises(gr.Error):
        demo.solve_puzzle("123", 8, 0.5)  # wrong length
