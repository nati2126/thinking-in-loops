"""Gradio demo: watch a Universal Transformer solve 4x4 Sudoku one loop at a time.

Runs on CPU. Locally:  python app/app.py
On Hugging Face Spaces the README front-matter points ``app_file`` here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.config import DataConfig, ModelConfig  # noqa: E402
from src.data.sudoku import (  # noqa: E402
    NUM_CELLS,
    SIZE,
    from_string,
    has_conflicts,
    is_complete_valid,
    make_puzzle,
    matches_clues,
    solve,
    to_string,
)
from src.eval.metrics import halting_loop_index, predict_all_loops  # noqa: E402
from src.models.models import SudokuModel, build_model  # noqa: E402

MODEL_CANDIDATES = [
    ROOT / "app" / "model" / "ut.pt",  # exported by scripts/export_model.py (shipped with the Space)
    ROOT / "checkpoints" / "ut_s0" / "best.pt",
    ROOT / "checkpoints" / "ut" / "best.pt",
]
DEFAULT_MAX_LOOPS = ModelConfig().max_loops_train
MAX_LOOPS_LIMIT = 32
DEFAULT_THRESHOLD = 0.5

# Colours (light surface; the grid carries its own background so it reads in dark mode too).
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
CLUE_BG = "#e9e8e4"
CONF_LOW, CONF_HIGH = (232, 241, 252), (28, 92, 171)  # sequential blue ramp, light -> dark
WRONG = "#e34948"
SERIES = ["#2a78d6", "#eb6834"]

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def load_demo_model() -> tuple[SudokuModel, str]:
    """Load the first checkpoint found in MODEL_CANDIDATES onto the CPU."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model = build_model(ckpt["model_name"], ModelConfig(**ckpt["model_cfg"]))
            model.load_state_dict(ckpt["model_state"])
            return model.eval(), path.relative_to(ROOT).as_posix()
    raise FileNotFoundError(
        "No model checkpoint found. Train one (python scripts/train.py --model ut) "
        "and export it (python scripts/export_model.py)."
    )


MODEL, MODEL_SOURCE = load_demo_model()


# --------------------------------------------------------------------------- model + rendering helpers


def run_model(puzzle: list[int], max_loops: int) -> dict:
    """Run the UT for ``max_loops`` loops; returns per-loop predictions, confidences and halting probs."""
    lp = predict_all_loops(MODEL, torch.tensor([puzzle]), n_loops=max_loops, amp=False)
    return {
        "puzzle": puzzle,
        "preds": lp.preds[:, 0].tolist(),  # (L, 16)
        "conf": lp.confidence[:, 0].tolist(),  # (L, 16)
        "halt": lp.halt_prob[:, 0].tolist() if lp.halt_prob is not None else [0.0] * max_loops,  # (L,)
    }


def halting_loop(halt: list[float], threshold: float) -> int:
    """1-based loop at which the model halts (first p > threshold, else the last loop)."""
    idx = halting_loop_index(torch.tensor(halt).view(-1, 1), len(halt), threshold)
    return int(idx.item()) + 1


def _conf_color(c: float) -> tuple[str, str]:
    """Background and text colour for a confidence in [0.25, 1] (chance level is 0.25)."""
    t = max(0.0, min(1.0, (c - 1 / SIZE) / (1 - 1 / SIZE)))
    rgb = tuple(round(lo + (hi - lo) * t) for lo, hi in zip(CONF_LOW, CONF_HIGH))
    return f"rgb{rgb}", ("#ffffff" if t > 0.55 else INK)


def render_grid(puzzle: list[int], preds: list[int], conf: list[float], solution: list[int] | None) -> str:
    """HTML 4x4 grid: clues in grey, predictions shaded by confidence, mistakes outlined."""
    cells = []
    for i in range(NUM_CELLS):
        r, c = divmod(i, SIZE)
        borders = [
            f"border-top:{3 if r % 2 == 0 else 1}px solid {INK}",
            f"border-left:{3 if c % 2 == 0 else 1}px solid {INK}",
            f"border-right:{3 if c == SIZE - 1 else 0}px solid {INK}",
            f"border-bottom:{3 if r == SIZE - 1 else 0}px solid {INK}",
        ]
        if puzzle[i]:
            style = f"background:{CLUE_BG};color:{INK};font-weight:700"
            inner = str(puzzle[i])
        else:
            bg, fg = _conf_color(conf[i])
            wrong = solution is not None and preds[i] != solution[i]
            ring = f"box-shadow:inset 0 0 0 3px {WRONG};" if wrong else ""
            style = f"background:{bg};color:{fg};{ring}"
            mark = f'<span style="position:absolute;top:2px;left:5px;font-size:12px;color:{WRONG}">✗</span>' if wrong else ""
            inner = (f'{mark}{preds[i]}<span style="position:absolute;bottom:2px;right:4px;'
                     f'font-size:10px;font-weight:400;opacity:.85">{conf[i] * 100:.0f}%</span>')  # fmt: skip
        cells.append(
            f'<td style="position:relative;width:64px;height:64px;text-align:center;vertical-align:middle;'
            f'font-size:28px;font-family:ui-sans-serif,system-ui,sans-serif;{";".join(borders)};{style}">{inner}</td>'
        )
    rows = "".join("<tr>" + "".join(cells[r * SIZE : (r + 1) * SIZE]) + "</tr>" for r in range(SIZE))
    return (
        f'<div style="background:{SURFACE};padding:12px;border-radius:8px;display:inline-block">'
        f'<table style="border-collapse:collapse;margin:auto">{rows}</table></div>'
    )


def halting_figure(halt: list[float], conf: list[list[float]], puzzle: list[int], threshold: float, current: int):
    """Halting probability and mean blank-cell confidence per loop, with the halting point marked."""
    loops = list(range(1, len(halt) + 1))
    blanks = [i for i, v in enumerate(puzzle) if v == 0]
    mean_conf = [sum(c[i] for i in blanks) / max(len(blanks), 1) for c in conf]
    stop = halting_loop(halt, threshold)
    fig = Figure(figsize=(6.4, 3.0), facecolor=SURFACE)  # not registered with pyplot, so nothing leaks
    ax = fig.subplots()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(loops, halt, color=SERIES[0], marker="o", lw=2, ms=6, label="Halting probability")
    ax.plot(loops, mean_conf, color=SERIES[1], marker="s", lw=2, ms=5, label="Mean confidence (blank cells)")
    ax.axhline(threshold, color="#8a8984", ls="--", lw=1.2)
    ax.text(loops[-1], threshold, f"threshold {threshold:g}", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.axvline(stop, color=SERIES[0], lw=1, alpha=0.5)
    ax.annotate(f"halts at loop {stop}", (stop, halt[stop - 1]), xytext=(8, -18), textcoords="offset points",
                fontsize=9, color=INK)  # fmt: skip
    ax.plot([current], [halt[current - 1]], marker="o", ms=13, mfc="none", mec=INK, mew=1.5, ls="none", label="Loop shown")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(0.5, len(halt) + 0.5)
    ax.set_xticks(loops if len(loops) <= 16 else loops[::2])
    ax.set_xlabel("Loop", color=INK_2)
    ax.set_ylabel("Probability", color=INK_2)
    ax.tick_params(colors=INK_2)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- callbacks


def generate(n_empty: int) -> str:
    """Random easy puzzle with a unique solution."""
    puzzle, _ = make_puzzle(random.Random(), int(n_empty))
    return to_string(puzzle)


def _status(result: dict, threshold: float, info: dict) -> str:
    halt, preds = result["halt"], result["preds"]
    stop = halting_loop(halt, threshold)
    lines = [f"**Halted at loop {stop}** of {len(halt)} (halting prob {halt[stop - 1]:.3f} > {threshold:g})"
             if halt[stop - 1] > threshold else
             f"**Never crossed the threshold** - used all {len(halt)} loops (last halting prob {halt[-1]:.3f})"]  # fmt: skip
    for label, loop in (("at the halting loop", stop), (f"after all {len(halt)} loops", len(halt))):
        grid = preds[loop - 1]
        if info["solution"] is not None:
            ok = grid == info["solution"]
            lines.append(f"- Answer {label}: {'✅ correct' if ok else '❌ wrong'}")
        else:
            ok = is_complete_valid(grid) and matches_clues(result["puzzle"], grid)
            lines.append(f"- Answer {label}: {'✅ valid and consistent with the clues' if ok else '❌ breaks a rule'}")
    first_ok = next((t + 1 for t, g in enumerate(preds) if info["solution"] is not None and g == info["solution"]), None)
    if first_ok:
        lines.append(f"- First loop with the correct answer: {first_ok}")
    lines.append(f"- Puzzle: {info['n_empty']} blanks, solver backtracks: {info['backtracks']}, {info['uniqueness']}")
    lines.append(f"- Model: `{MODEL_SOURCE}` ({MODEL.num_parameters():,} parameters, one shared block)")
    return "\n".join(lines)


def solve_puzzle(puzzle_str: str, max_loops: int, threshold: float):
    """Parse, run the model and show the loop at which it halted."""
    try:
        puzzle = from_string(puzzle_str)
    except ValueError as err:
        raise gr.Error(f"Could not read the puzzle: {err}. Use 16 characters, 0 or . for empty cells.")
    if has_conflicts(puzzle):
        raise gr.Error("This puzzle breaks a Sudoku rule (a digit repeats in a row, column or box).")
    res = solve(puzzle, limit=2)
    if res.num_solutions == 0:
        raise gr.Error("This puzzle has no solution.")
    unique = res.num_solutions == 1
    info = {
        "solution": res.solution if unique else None,
        "n_empty": puzzle.count(0),
        "backtracks": res.backtracks,
        "uniqueness": "unique solution" if unique else "**not unique** (answers checked against the rules only)",
    }
    if not unique or not (DataConfig.min_empty <= info["n_empty"] <= DataConfig.max_empty):
        gr.Warning(f"The model was trained on unique puzzles with {DataConfig.min_empty}-{DataConfig.max_empty} blanks; "
                   "this one is outside that range.")  # fmt: skip
    result = run_model(puzzle, int(max_loops))
    result["info"] = info
    stop = halting_loop(result["halt"], threshold)
    html, fig = _view(result, stop, threshold)
    slider = gr.Slider(minimum=1, maximum=int(max_loops), value=stop, step=1, interactive=True)
    return result, slider, html, fig, _status(result, threshold, info)


def _view(result: dict, loop: int, threshold: float):
    t = max(1, min(int(loop), len(result["preds"]))) - 1
    html = render_grid(result["puzzle"], result["preds"][t], result["conf"][t], result["info"]["solution"])
    changed = sum(a != b for a, b in zip(result["preds"][t], result["preds"][t - 1])) if t > 0 else None
    caption = f"<p style='margin:6px 0 0;color:var(--body-text-color-subdued);font-size:13px'>Loop {t + 1}" + (
        f" · {changed} cell(s) changed since loop {t}" if changed is not None else "") + "</p>"  # fmt: skip
    html = f"<div style='text-align:center'>{html}{caption}</div>"
    return html, halting_figure(result["halt"], result["conf"], result["puzzle"], threshold, t + 1)


def show_loop(result: dict | None, loop: int, threshold: float):
    if not result:
        return gr.update(), gr.update()
    return _view(result, loop, threshold)


LEGEND = (
    "<div style='font-size:13px;color:var(--body-text-color);text-align:center'>"
    f"<span style='background:{CLUE_BG};padding:1px 6px;border-radius:3px;color:{INK}'>grey</span> given clue · "
    f"<span style='background:rgb{CONF_LOW};padding:1px 6px;border-radius:3px;color:{INK}'>light</span> → "
    f"<span style='background:rgb{CONF_HIGH};padding:1px 6px;border-radius:3px;color:#fff'>dark blue</span> "
    f"model confidence (shown in the corner) · <span style='color:{WRONG}'>✗ red outline</span> differs from the true solution"
    "</div>"
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Thinking in Loops - 4x4 Sudoku") as demo:
        gr.Markdown(
            "# Thinking in Loops\n"
            "A **Universal Transformer** solves 4x4 Sudoku by applying **one shared transformer block** again and "
            "again. After each loop it reads out a full answer, and a halting head estimates whether that answer "
            "is already correct. Generate a puzzle (or type one), press **Solve**, then scrub through the loops."
        )
        result_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                n_empty = gr.Slider(DataConfig.min_empty, DataConfig.max_empty, value=6, step=1, label="Empty cells")
                gen_btn = gr.Button("Generate easy puzzle")
                puzzle_box = gr.Textbox(
                    label="Puzzle (16 cells, row by row; 0 or . = empty)", value=generate(6), max_lines=1,
                    info="Example: 14.3 .3.. 3142 .23. (spaces are ignored)",
                )  # fmt: skip
                with gr.Accordion("Settings", open=False):
                    max_loops = gr.Slider(1, MAX_LOOPS_LIMIT, value=DEFAULT_MAX_LOOPS, step=1, label="Max loops",
                                          info=f"The model was trained with {DEFAULT_MAX_LOOPS} loops")  # fmt: skip
                    threshold = gr.Slider(0.05, 0.999, value=DEFAULT_THRESHOLD, step=0.001, label="Halting threshold",
                                          info="Stop at the first loop whose halting probability exceeds this")  # fmt: skip
                solve_btn = gr.Button("Solve", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2, min_width=340):
                loop_slider = gr.Slider(1, DEFAULT_MAX_LOOPS, value=1, step=1, label="Show loop", interactive=True)
                grid_html = gr.HTML()
                gr.HTML(LEGEND)
                halt_plot = gr.Plot(label="Halting probability per loop")

        gen_btn.click(generate, n_empty, puzzle_box).then(
            solve_puzzle, [puzzle_box, max_loops, threshold], [result_state, loop_slider, grid_html, halt_plot, status]
        )
        solve_btn.click(
            solve_puzzle, [puzzle_box, max_loops, threshold], [result_state, loop_slider, grid_html, halt_plot, status]
        )
        loop_slider.change(show_loop, [result_state, loop_slider, threshold], [grid_html, halt_plot])
        demo.load(
            solve_puzzle, [puzzle_box, max_loops, threshold], [result_state, loop_slider, grid_html, halt_plot, status]
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
