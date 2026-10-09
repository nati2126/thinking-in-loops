"""Loss computation for one batch: deep supervision + halting + truncated backprop.

Truncated backprop through loops (TBPTT): the loops are split into segments of
at most K loops, aligned so the *last* segment has exactly K loops. The hidden
state is detached at segment boundaries and ``backward()`` is called once per
segment, so (a) each loop's loss sends gradients through at most K block
applications and (b) only one segment's activation graph is alive at a time,
which is what makes memory scale with K. Gradients are accumulated across
segments and the optimizer steps once per batch.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.models import SudokuModel


@dataclass
class StepStats:
    """Scalar statistics for one batch (losses already averaged over loops)."""

    loss: float
    ce: float
    halt_loss: float
    cell_acc: float  # all 16 cells, final loop
    puzzle_acc: float  # final loop


def loop_segments(n_loops: int, k: int) -> list[range]:
    """Split ``range(n_loops)`` into chunks of size ``k`` aligned to the end; ``k<=0`` means one chunk."""
    if k <= 0 or k >= n_loops:
        return [range(n_loops)]
    segments, end = [], n_loops
    while end > 0:
        start = max(0, end - k)
        segments.append(range(start, end))
        end = start
    return segments[::-1]


def autocast_context(device: torch.device, enabled: bool):
    """bf16 autocast on CUDA when enabled, otherwise a no-op context."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def forward_backward(
    model: SudokuModel,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    n_loops: int,
    tbptt_k: int,
    halt_loss_weight: float,
    amp: bool = True,
) -> StepStats:
    """Compute the deep-supervised loss for a batch and accumulate gradients (no optimizer step).

    Loss = mean over loops of [ CE(digits) + halt_loss_weight * BCE(halt, grid_correct) ].
    Non-recurrent models always use a single loop.
    """
    n_loops = n_loops if model.is_recurrent else 1
    targets = solutions - 1  # digits 1..4 -> classes 0..3
    device = puzzles.device
    total_ce = torch.zeros((), device=device)
    total_halt = torch.zeros((), device=device)
    h: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None

    for segment in loop_segments(n_loops, tbptt_k):
        with autocast_context(device, amp):
            e = model.embed(puzzles)  # recomputed per segment: its graph is freed by each backward()
            h = model.init_hidden(e) if h is None else h.detach()
            seg_loss = torch.zeros((), device=device)
            for _ in segment:
                h = model.step(h, e)
                logits, halt = model.readout(h)
                logits = logits.float()
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                seg_loss = seg_loss + ce / n_loops
                total_ce += ce.detach() / n_loops
                if halt is not None:
                    solved = (logits.argmax(-1) == targets).all(-1).float()
                    hl = F.binary_cross_entropy_with_logits(halt.float(), solved)
                    seg_loss = seg_loss + halt_loss_weight * hl / n_loops
                    total_halt += hl.detach() / n_loops
                last_logits = logits
        seg_loss.backward()

    assert last_logits is not None
    with torch.no_grad():
        correct = last_logits.argmax(-1) == targets
        cell_acc = correct.float().mean().item()
        puzzle_acc = correct.all(-1).float().mean().item()
    ce_v, halt_v = total_ce.item(), total_halt.item()
    return StepStats(ce_v + halt_loss_weight * halt_v, ce_v, halt_v, cell_acc, puzzle_acc)
