"""Evaluation: per-loop predictions, halting selection and Sudoku metrics.

Metric definitions:
  * cell_acc     - accuracy over the *blank* cells only (clues are trivially copyable).
  * cell_acc_all - accuracy over all 16 cells.
  * puzzle_acc   - fraction of puzzles whose full 16-cell output equals the solution.
  * validity     - fraction of outputs that break no Sudoku rule (every row, column
                   and box is a permutation of 1-4), whether or not they match the clues.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.dataset import SudokuSplit
from ..data.sudoku import GROUPS, SIZE
from ..models.models import SudokuModel
from ..training.step import autocast_context

_GROUP_INDEX = torch.tensor(GROUPS)  # (12, 4)


def grid_validity(preds: torch.Tensor) -> torch.Tensor:
    """(N, 16) digits -> (N,) bool, True if no row/column/box contains a repeated digit."""
    groups = preds[:, _GROUP_INDEX.to(preds.device)]  # (N, 12, 4)
    target = torch.arange(1, SIZE + 1, device=preds.device)
    return (groups.sort(-1).values == target).all(-1).all(-1)


def grid_metrics(preds: torch.Tensor, puzzles: torch.Tensor, solutions: torch.Tensor) -> dict[str, float]:
    """Metrics for one prediction per puzzle. All tensors (N, 16) on the same device."""
    correct = preds == solutions
    blank = puzzles == 0
    return {
        "cell_acc": (correct & blank).sum().item() / max(blank.sum().item(), 1),
        "cell_acc_all": correct.float().mean().item(),
        "puzzle_acc": correct.all(-1).float().mean().item(),
        "validity": grid_validity(preds).float().mean().item(),
    }


@dataclass
class LoopPredictions:
    """Predictions after every loop, on CPU.

    Attributes:
        preds: (L, N, 16) predicted digits 1-4.
        confidence: (L, N, 16) softmax probability of the predicted digit.
        halt_prob: (L, N) sigmoid of the halting head, or None.
    """

    preds: torch.Tensor
    confidence: torch.Tensor
    halt_prob: torch.Tensor | None

    @property
    def n_loops(self) -> int:
        return self.preds.shape[0]


@torch.no_grad()
def predict_all_loops(
    model: SudokuModel, puzzles: torch.Tensor, n_loops: int | None = None, batch_size: int = 1024, amp: bool = True
) -> LoopPredictions:
    """Run the model on ``puzzles`` and collect predictions after every loop."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    preds, confs, halts = [], [], []
    for start in range(0, puzzles.shape[0], batch_size):
        batch = puzzles[start : start + batch_size].to(device)
        with autocast_context(device, amp):
            out = model(batch, n_loops=n_loops)
        probs = out.logits.float().softmax(-1)
        conf, cls = probs.max(-1)
        preds.append((cls + 1).cpu())
        confs.append(conf.cpu())
        if out.halt_logits is not None:
            halts.append(out.halt_logits.float().sigmoid().cpu())
    model.train(was_training)
    return LoopPredictions(torch.cat(preds, 1), torch.cat(confs, 1), torch.cat(halts, 1) if halts else None)


def halting_loop_index(halt_prob: torch.Tensor | None, n_loops: int, threshold: float = 0.5) -> torch.Tensor:
    """0-based loop at which each puzzle halts: first loop with p > threshold, else the last loop."""
    if halt_prob is None:
        raise ValueError("model has no halting head")
    above = halt_prob > threshold  # (L, N)
    first = above.float().argmax(0)  # first True (0 if none)
    return torch.where(above.any(0), first, torch.full_like(first, n_loops - 1))


def select_loop(preds: torch.Tensor, loop_idx: torch.Tensor) -> torch.Tensor:
    """Pick preds[loop_idx[n], n] for every puzzle n. preds: (L, N, 16) -> (N, 16)."""
    return preds.gather(0, loop_idx.view(1, -1, 1).expand(1, -1, preds.shape[-1])).squeeze(0)


def evaluate(
    model: SudokuModel,
    split: SudokuSplit,
    n_loops: int | None = None,
    batch_size: int = 1024,
    halt_threshold: float = 0.5,
    amp: bool = True,
) -> dict[str, float]:
    """Metrics after the final loop and, for halting models, at the halting loop.

    Keys: ``final_*`` (output after the last loop) and, if available,
    ``halt_*`` (output at the halting loop) plus ``avg_loops`` (1-based loops used).
    """
    lp = predict_all_loops(model, split.puzzles, n_loops, batch_size, amp)
    puzzles, solutions = split.puzzles.cpu(), split.solutions.cpu()
    out = {f"final_{k}": v for k, v in grid_metrics(lp.preds[-1], puzzles, solutions).items()}
    if lp.halt_prob is not None:
        idx = halting_loop_index(lp.halt_prob, lp.n_loops, halt_threshold)
        out.update({f"halt_{k}": v for k, v in grid_metrics(select_loop(lp.preds, idx), puzzles, solutions).items()})
        out["avg_loops"] = (idx.float() + 1).mean().item()
    return out
