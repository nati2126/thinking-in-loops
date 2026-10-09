"""The four Phase-4 experiments. Each function writes a CSV + PNG to ``results/`` and returns its rows.

1. ``exp1_comparison``       - UT vs same-param and same-compute baselines.
2. ``exp2_loops_vs_difficulty`` - loops used before halting vs puzzle difficulty.
3. ``exp3_test_time_compute`` - accuracy when evaluating with 1..32 loops.
4. ``exp4_tbptt_ablation``    - truncated-backprop K = 1, 3, all: accuracy and memory.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ModelConfig
from ..data.dataset import SudokuSplit
from ..models.models import build_model
from ..training.step import forward_backward
from .metrics import LoopPredictions, grid_metrics, halting_loop_index, select_loop
from .plotting import MODEL_COLORS, MODEL_LABELS, MUTED, SERIES, TEXT_2, apply_style, plt, save


@dataclass
class RunResult:
    """A trained run evaluated on the test set with ``max_eval_loops`` loops."""

    name: str  # experiment-level name, e.g. "ut", "ut_k1"
    model: str  # architecture name
    seed: int
    params: int
    train_loops: int
    preds: LoopPredictions
    log: list[dict[str, float]]  # per-epoch training log rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def metrics_at(run: RunResult, test: SudokuSplit, loops: int, halting: bool, threshold: float = 0.5) -> dict[str, float]:
    """Test metrics when the model may use up to ``loops`` loops (forced to the last one if not halting)."""
    loops = min(loops, run.preds.n_loops)
    preds = run.preds.preds[:loops]
    if halting and run.preds.halt_prob is not None:
        idx = halting_loop_index(run.preds.halt_prob[:loops], loops, threshold)
    else:
        idx = torch.full((preds.shape[1],), loops - 1, dtype=torch.long)
    m = grid_metrics(select_loop(preds, idx), test.puzzles.cpu(), test.solutions.cpu())
    m["avg_loops"] = (idx.float() + 1).mean().item()
    return m


# --------------------------------------------------------------------------- experiment 1


def exp1_comparison(runs: list[RunResult], test: SudokuSplit, out_dir: Path) -> list[dict[str, Any]]:
    """Accuracy and parameter count of the UT and both baselines (mean ± std over seeds)."""
    variants = [
        ("same_param", "1-layer transformer (same params)", False),
        ("same_compute", "8-layer transformer (same compute)", False),
        ("ut", "UT, fixed 8 loops", False),
        ("ut", "UT, adaptive halting (<= 8 loops)", True),
    ]
    rows = []
    for name, label, halting in variants:
        group = [r for r in runs if r.name == name]
        if not group:
            continue
        ms = [metrics_at(r, test, r.train_loops, halting) for r in group]
        row: dict[str, Any] = {"model": label, "params": group[0].params, "seeds": len(group)}
        for key in ("cell_acc", "puzzle_acc", "validity", "avg_loops"):
            row[f"{key}_mean"], row[f"{key}_std"] = _mean_std([m[key] for m in ms])
        row["train_time_s_mean"] = _mean_std([sum(e["epoch_time_s"] for e in r.log) for r in group])[0]
        rows.append(row)
    write_csv(out_dir / "exp1_comparison.csv", rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [MODEL_COLORS["same_param"], MODEL_COLORS["same_compute"], MODEL_COLORS["ut"], MODEL_COLORS["ut"]]
    y = np.arange(len(rows))[::-1]
    errs = [100 * r["puzzle_acc_std"] for r in rows]
    vals = [100 * r["puzzle_acc_mean"] for r in rows]
    bars = ax.barh(y, vals, xerr=errs, color=colors[: len(rows)], height=0.6, edgecolor="white", linewidth=2,
                   error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})  # fmt: skip
    if len(rows) == 4:
        bars[3].set_hatch("///")
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.6, yi, f"{v:.2f}%", va="center", fontsize=9, color=TEXT_2)
    ax.set_yticks(y, [f"{r['model']}\n{r['params']:,} params" for r in rows])
    lo = max(0.0, math.floor(min(v - e for v, e in zip(vals, errs)) / 10) * 10 - 10)
    ax.set_xlim(lo, 100 + (100 - lo) * 0.12)
    ax.set_xlabel("Test full-puzzle accuracy (%)  -  mean ± std over seeds")
    ax.set_title("UT vs baselines on 4x4 Sudoku (4-8 blanks)")
    ax.grid(axis="y", visible=False)
    save(fig, out_dir / "exp1_comparison.png")
    return rows


def exp1b_novel_vs_seen(
    runs: list[RunResult], test: SudokuSplit, seen_mask: torch.Tensor, out_dir: Path
) -> list[dict[str, Any]]:
    """Extra analysis: accuracy on test puzzles that are / are not a symmetry image of a training puzzle.

    ``seen_mask[n]`` is True when augmentation could have produced test puzzle n exactly from the
    training set; the remaining ("novel") puzzles are the cleaner generalization test.
    """
    variants = [("same_param", "1-layer transformer (same params)", False),
                ("same_compute", "8-layer transformer (same compute)", False),
                ("ut", "UT, fixed 8 loops", False), ("ut", "UT, adaptive halting (<= 8 loops)", True)]  # fmt: skip
    rows = []
    for name, label, halting in variants:
        group = [r for r in runs if r.name == name]
        if not group:
            continue
        row: dict[str, Any] = {"model": label}
        for subset, mask in (("seen", seen_mask), ("novel", ~seen_mask)):
            sub = test.subset(mask.to(test.puzzles.device))
            accs = []
            for r in group:
                masked = dataclasses_replace_preds(r, mask)
                accs.append(metrics_at(masked, sub, r.train_loops, halting)["puzzle_acc"])
            row[f"{subset}_n"] = int(mask.sum())
            row[f"{subset}_puzzle_acc_mean"], row[f"{subset}_puzzle_acc_std"] = _mean_std(accs)
        rows.append(row)
    write_csv(out_dir / "exp1b_novel_vs_seen.csv", rows)
    return rows


def dataclasses_replace_preds(run: RunResult, mask: torch.Tensor) -> RunResult:
    """Copy of ``run`` whose predictions are restricted to the puzzles selected by ``mask``."""
    import dataclasses

    p = run.preds
    sub = LoopPredictions(p.preds[:, mask], p.confidence[:, mask], p.halt_prob[:, mask] if p.halt_prob is not None else None)
    return dataclasses.replace(run, preds=sub)


# --------------------------------------------------------------------------- experiment 2


def _first_stable_correct(run: RunResult, test: SudokuSplit, loops: int) -> torch.Tensor:
    """1-based first loop from which the prediction is correct at every later loop (<= loops); NaN if never."""
    correct = (run.preds.preds[:loops] == test.solutions.cpu().unsqueeze(0)).all(-1)  # (L, N)
    stays = correct.flip(0).int().cumprod(0).flip(0).bool()
    first = stays.float().argmax(0).float() + 1
    return torch.where(stays.any(0), first, torch.full_like(first, float("nan")))


def exp2_loops_vs_difficulty(runs: list[RunResult], test: SudokuSplit, out_dir: Path) -> list[dict[str, Any]]:
    """Average loops used by the halting head (threshold 0.5, cap = training loops) per difficulty bucket.

    Also reports the "oracle" loops: the first loop after which the answer is and stays correct,
    which shows how many loops the puzzle actually needed regardless of the halting head.
    """
    group = [r for r in runs if r.name == "ut"]
    if not group:
        return []
    loops = group[0].train_loops
    n_empty = test.n_empty.cpu()
    bt = test.backtracks.cpu().clamp(max=3)  # bucket 3 = "3+"
    per_seed = []
    for r in group:
        idx = halting_loop_index(r.preds.halt_prob[:loops], loops)
        chosen = select_loop(r.preds.preds[:loops], idx)
        solved = (chosen == test.solutions.cpu()).all(-1).float()
        per_seed.append((idx.float() + 1, _first_stable_correct(r, test, loops), solved))

    rows = []
    for kind, labels, values in (("n_empty", range(4, 9), n_empty), ("backtracks", range(0, 4), bt)):
        for v in labels:
            mask = values == v
            if mask.sum() == 0:
                continue
            halt = [h[mask].mean().item() for h, _, _ in per_seed]
            oracle = [np.nanmean(o[mask].numpy()) for _, o, _ in per_seed]
            acc = [s[mask].mean().item() for _, _, s in per_seed]
            row = {"difficulty": kind, "value": f"{v}+" if (kind == "backtracks" and v == 3) else str(v),
                   "n_puzzles": int(mask.sum())}  # fmt: skip
            row["halt_loops_mean"], row["halt_loops_std"] = _mean_std(halt)
            row["oracle_loops_mean"], row["oracle_loops_std"] = _mean_std(oracle)
            row["puzzle_acc_at_halt_mean"], _ = _mean_std(acc)
            rows.append(row)
    write_csv(out_dir / "exp2_loops_vs_difficulty.csv", rows)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ymax = 0.0
    for ax, kind, xlabel in zip(axes, ("n_empty", "backtracks"), ("Empty cells", "Solver backtracks")):
        sub = [r for r in rows if r["difficulty"] == kind]
        x = np.arange(len(sub))
        for key, label, color, marker in (
            ("halt_loops", "Loops used (halting head, p > 0.5)", SERIES[0], "o"),
            ("oracle_loops", "First loop with a stable correct answer", SERIES[1], "s"),
        ):
            m = np.array([r[f"{key}_mean"] for r in sub])
            s = np.array([r[f"{key}_std"] for r in sub])
            ax.plot(x, m, marker=marker, color=color, label=label)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
            ymax = max(ymax, float(np.nanmax(m + s)))
        ax.set_xticks(x, [f"{r['value']}\n(n={r['n_puzzles']})" for r in sub])
        ax.set_xlabel(xlabel)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.1)
    axes[0].set_ylabel("Loops (mean over test puzzles)")
    axes[0].set_title("Loops vs number of blanks")
    axes[1].set_title("Loops vs solver backtracks")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir / "exp2_loops_vs_difficulty.png")
    return rows


def exp2b_halting_threshold(
    runs: list[RunResult], test: SudokuSplit, out_dir: Path,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99, 0.999),
) -> list[dict[str, Any]]:  # fmt: skip
    """Extra analysis: accuracy vs loops used as the halting threshold varies (cap = training loops)."""
    group = [r for r in runs if r.name == "ut"]
    if not group:
        return []
    rows = []
    for thr in thresholds:
        ms = [metrics_at(r, test, r.train_loops, True, threshold=thr) for r in group]
        row: dict[str, Any] = {"threshold": thr}
        for key in ("puzzle_acc", "avg_loops"):
            row[f"{key}_mean"], row[f"{key}_std"] = _mean_std([m[key] for m in ms])
        rows.append(row)
    # Per-loop accuracy and calibration of the halting head (mean p vs fraction actually solved).
    calib = []
    for t in range(group[0].train_loops):
        acc = [(r.preds.preds[t] == test.solutions.cpu()).all(-1).float().mean().item() for r in group]
        p = [r.preds.halt_prob[t].mean().item() for r in group]
        calib.append({"loop": t + 1, "puzzle_acc_mean": _mean_std(acc)[0], "mean_halt_prob": _mean_std(p)[0]})
    write_csv(out_dir / "exp2b_halting_threshold.csv", rows)
    write_csv(out_dir / "exp2b_halting_calibration.csv", calib)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = [r["avg_loops_mean"] for r in rows]
    y = [100 * r["puzzle_acc_mean"] for r in rows]
    ax.plot(x, y, marker="o", color=SERIES[0], label="UT, halting head at threshold p")
    for i, (xi, yi, r) in enumerate(zip(x, y, rows)):
        below = i % 2 == 0  # alternate sides so neighbouring labels don't collide
        ax.annotate(f"p > {r['threshold']:g}", (xi, yi), textcoords="offset points",
                    xytext=(6, -14) if below else (-6, 8), ha="left" if below else "right",
                    fontsize=8, color=TEXT_2)  # fmt: skip
    ax.margins(x=0.08, y=0.12)
    fixed = 100 * _mean_std([metrics_at(r, test, r.train_loops, False)["puzzle_acc"] for r in group])[0]
    ax.axhline(fixed, color=MUTED, linestyle="--", linewidth=1.2, label=f"always run {group[0].train_loops} loops")
    ax.legend(loc="lower right")
    ax.set_xlabel("Average loops used before halting")
    ax.set_ylabel("Test full-puzzle accuracy (%)")
    ax.set_title("Halting threshold trades accuracy for loops")
    save(fig, out_dir / "exp2b_halting_threshold.png")
    return rows


def exp2c_checkpoint_choice(
    best: list[RunResult], last: list[RunResult], test: SudokuSplit, out_dir: Path
) -> list[dict[str, Any]]:
    """Extra analysis: halting quality of the selected (best-val) checkpoint vs the final-epoch checkpoint.

    The best checkpoint is the *earliest* epoch reaching the top val puzzle accuracy, which can be
    long before the halting head has finished improving.
    """
    rows = []
    for name in ("ut", "ut_k1", "ut_kall"):
        for label, runs in (("best (selected)", best), ("last epoch", last)):
            group = [r for r in runs if r.name == name]
            if not group:
                continue
            fixed = [metrics_at(r, test, r.train_loops, False) for r in group]
            halt = [metrics_at(r, test, r.train_loops, True) for r in group]
            row: dict[str, Any] = {"run": name, "checkpoint": label, "seeds": len(group)}
            row["fixed8_puzzle_acc_mean"], _ = _mean_std([m["puzzle_acc"] for m in fixed])
            row["halt_puzzle_acc_mean"], row["halt_puzzle_acc_std"] = _mean_std([m["puzzle_acc"] for m in halt])
            row["halt_avg_loops_mean"], _ = _mean_std([m["avg_loops"] for m in halt])
            rows.append(row)
    if rows:
        write_csv(out_dir / "exp2c_checkpoint_choice.csv", rows)
    return rows


# --------------------------------------------------------------------------- experiment 3


def exp3_test_time_compute(
    runs: list[RunResult], test: SudokuSplit, out_dir: Path, loop_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
) -> list[dict[str, Any]]:
    """Accuracy of the UT (trained with 8 loops) evaluated with different loop budgets."""
    group = [r for r in runs if r.name == "ut"]
    if not group:
        return []
    rows = []
    for loops in loop_grid:
        for halting in (False, True):
            ms = [metrics_at(r, test, loops, halting) for r in group]
            row: dict[str, Any] = {"max_loops": loops, "mode": "halting" if halting else "fixed"}
            for key in ("puzzle_acc", "cell_acc", "validity", "avg_loops"):
                row[f"{key}_mean"], row[f"{key}_std"] = _mean_std([m[key] for m in ms])
            rows.append(row)
    baselines = {}
    for name in ("same_param", "same_compute"):
        g = [r for r in runs if r.name == name]
        if g:
            baselines[name] = _mean_std([metrics_at(r, test, 1, False)["puzzle_acc"] for r in g])[0]
    write_csv(out_dir / "exp3_test_time_compute.csv", rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.log2(loop_grid)
    # Both UT series share the UT colour (colour = model); line style and marker separate them.
    all_lo = []
    for mode, label, style, marker in (("fixed", "UT, run exactly N loops", "-", "o"),
                                       ("halting", "UT, halting head p > 0.5 (at most N loops)", "--", "s")):  # fmt: skip
        sub = [r for r in rows if r["mode"] == mode]
        m = 100 * np.array([r["puzzle_acc_mean"] for r in sub])
        s = 100 * np.array([r["puzzle_acc_std"] for r in sub])
        ax.plot(x, m, marker=marker, linestyle=style, color=SERIES[0], label=label)
        ax.fill_between(x, m - s, m + s, color=SERIES[0], alpha=0.12, linewidth=0)
        all_lo.append(float((m - s).min()))
    for name, acc in baselines.items():
        ax.axhline(100 * acc, color=MODEL_COLORS[name], linestyle=":", linewidth=1.5, label=MODEL_LABELS[name])
    lo = math.floor(min(all_lo + [100 * a for a in baselines.values()]) / 5) * 5 - 3
    train_loops = group[0].train_loops
    ax.axvline(math.log2(train_loops), color=MUTED, linewidth=1, alpha=0.6)
    ax.text(math.log2(train_loops), lo + 0.5, f" trained with {train_loops} loops", fontsize=8, color=TEXT_2)
    ax.set_xticks(x, [str(n) for n in loop_grid])
    ax.set_ylim(lo, 101.5)
    ax.set_xlabel("Loop budget N at test time (log scale)")
    ax.set_ylabel("Test full-puzzle accuracy (%)")
    ax.set_title("Test-time compute: more loops than seen in training")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06))
    save(fig, out_dir / "exp3_test_time_compute.png")
    return rows


# --------------------------------------------------------------------------- experiment 4


def measure_step_memory(tbptt_k: int, model_cfg: ModelConfig, batch_size: int, device: torch.device) -> dict[str, float]:
    """Isolated CUDA memory measurement of one UT training step (MB).

    Returns ``total`` (peak of weights + grads + AdamW state + activations, relative to memory
    allocated before the model was built) and ``activations`` (the part above the resting state).
    """
    if device.type != "cuda":
        return {"total": float("nan"), "activations": float("nan")}
    import gc

    from ..config import TrainConfig
    from ..training.trainer import make_optimizer

    # Throwaway step so one-time CUDA/cuBLAS workspaces are allocated before the baseline is taken.
    warm = build_model("ut", model_cfg).to(device)
    forward_backward(warm, torch.zeros(2, 16, dtype=torch.long, device=device),
                     torch.ones(2, 16, dtype=torch.long, device=device), 1, 0, 0.5, amp=True)  # fmt: skip
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    process_base = torch.cuda.memory_allocated(device)
    torch.manual_seed(0)
    model = build_model("ut", model_cfg).to(device)
    opt = make_optimizer(model, TrainConfig())
    puzzles = torch.randint(0, 5, (batch_size, 16), device=device)
    solutions = torch.randint(1, 5, (batch_size, 16), device=device)
    for _ in range(2):  # the first step allocates AdamW state; measure the second
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        resting = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        forward_backward(model, puzzles, solutions, model_cfg.max_loops_train, tbptt_k, 0.5, amp=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        opt.step()
    del model, opt, puzzles, solutions
    gc.collect()
    torch.cuda.empty_cache()
    return {"total": (peak - process_base) / 2**20, "activations": (peak - resting) / 2**20}


def exp4_tbptt_ablation(
    runs: list[RunResult], test: SudokuSplit, out_dir: Path, model_cfg: ModelConfig, batch_size: int, device: torch.device
) -> list[dict[str, Any]]:
    """Truncated backprop K = 1, 3 and all loops: test accuracy, training memory and time."""
    variants = [("ut_k1", "K = 1", 1), ("ut", "K = 3 (default)", 3), ("ut_kall", f"K = all ({model_cfg.max_loops_train})", 0)]
    rows = []
    for name, label, k in variants:
        group = [r for r in runs if r.name == name]
        if not group:
            continue
        fixed = [metrics_at(r, test, r.train_loops, False) for r in group]
        halt = [metrics_at(r, test, r.train_loops, True) for r in group]
        row: dict[str, Any] = {"variant": label, "tbptt_k": k if k else model_cfg.max_loops_train, "seeds": len(group)}
        row["puzzle_acc_mean"], row["puzzle_acc_std"] = _mean_std([m["puzzle_acc"] for m in fixed])
        row["halt_puzzle_acc_mean"], row["halt_puzzle_acc_std"] = _mean_std([m["puzzle_acc"] for m in halt])
        row["halt_avg_loops_mean"], _ = _mean_std([m["avg_loops"] for m in halt])
        row["acc_32_loops_mean"], _ = _mean_std([metrics_at(r, test, 32, False)["puzzle_acc"] for r in group])
        mem = measure_step_memory(k, model_cfg, batch_size, device)
        row["step_peak_mem_mb"], row["step_activation_mem_mb"] = mem["total"], mem["activations"]
        row["train_time_s_mean"], _ = _mean_std([sum(e["epoch_time_s"] for e in r.log) for r in group])
        rows.append(row)
    if not rows:
        return rows
    write_csv(out_dir / "exp4_tbptt_ablation.csv", rows)

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.arange(len(rows))[::-1]
    labels = [r["variant"] for r in rows]
    acc = [100 * r["puzzle_acc_mean"] for r in rows]
    err = [100 * r["puzzle_acc_std"] for r in rows]
    ax1.barh(y, acc, xerr=err, color=SERIES[0], height=0.6, edgecolor="white", linewidth=2,
             error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})  # fmt: skip
    for yi, v, e in zip(y, acc, err):
        ax1.text(v + e + 0.4, yi, f"{v:.2f}%", va="center", fontsize=9, color=TEXT_2)
    ax1.set_yticks(y, labels)
    lo = max(0.0, math.floor(min(a - e for a, e in zip(acc, err)) / 10) * 10 - 10)
    ax1.set_xlim(lo, 100 + (100 - lo) * 0.15)
    ax1.set_xlabel("Test full-puzzle accuracy at 8 loops (%)")
    ax1.set_title("Accuracy")
    mem = [r["step_peak_mem_mb"] for r in rows]
    ax2.barh(y, mem, color=SERIES[1], height=0.6, edgecolor="white", linewidth=2)
    for yi, v in zip(y, mem):
        ax2.text(v, yi, f" {v:.1f} MB", va="center", fontsize=9, color=TEXT_2)
    ax2.set_yticks(y, labels)
    ax2.set_xlim(0, (max(mem) if not any(math.isnan(m) for m in mem) else 1) * 1.25)
    ax2.set_xlabel(f"Peak GPU memory per training step (MB, batch {batch_size})")
    ax2.set_title("Training memory")
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    save(fig, out_dir / "exp4_tbptt_ablation.png")
    return rows


# --------------------------------------------------------------------------- training curves


def plot_training_curves(runs: list[RunResult], out_dir: Path) -> None:
    """Validation puzzle accuracy per epoch for the three main models (first seed of each)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for name in ("same_param", "same_compute", "ut"):
        group = [r for r in runs if r.name == name]
        if not group:
            continue
        r = min(group, key=lambda g: g.seed)
        ep = [int(e["epoch"]) + 1 for e in r.log]
        acc = [100 * e["val_final_puzzle_acc"] for e in r.log]
        ax.plot(ep, acc, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val full-puzzle accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Validation accuracy during training (seed 0)")
    ax.legend(loc="lower right")
    save(fig, out_dir / "training_curves.png")
