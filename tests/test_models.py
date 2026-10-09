"""Phase 2 tests: forward shapes, parameter parity, TBPTT segmenting and overfitting a batch."""

from __future__ import annotations

import pytest
import torch

from src.config import DataConfig, ModelConfig, set_seed
from src.data.dataset import build_datasets, load_split
from src.models.models import MODEL_NAMES, build_model
from src.training.step import forward_backward, loop_segments

CFG = ModelConfig()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shapes(name):
    model = build_model(name, CFG)
    puzzles = torch.randint(0, 5, (7, 16))
    out = model(puzzles, n_loops=5)
    loops = 5 if model.is_recurrent else 1
    assert out.logits.shape == (loops, 7, 16, 4)
    if model.has_halting:
        assert out.halt_logits is not None and out.halt_logits.shape == (loops, 7)
    else:
        assert out.halt_logits is None


def test_ut_loop_count_is_flexible():
    model = build_model("ut", CFG)
    puzzles = torch.randint(0, 5, (3, 16))
    assert model(puzzles).logits.shape[0] == CFG.max_loops_train
    assert model(puzzles, n_loops=32).logits.shape[0] == 32


def test_parameter_parity():
    ut, one, eight = (build_model(n, CFG).num_parameters() for n in MODEL_NAMES)
    assert ut - one == CFG.d_model + 1  # only the halting head differs
    assert eight > 6 * one


def test_attention_is_not_causal():
    """Changing the last cell must be able to change the first cell's output."""
    torch.manual_seed(0)
    model = build_model("same_param", CFG).eval()
    a = torch.zeros(1, 16, dtype=torch.long)
    b = a.clone()
    b[0, 15] = 3
    assert not torch.allclose(model(a).logits[0, 0, 0], model(b).logits[0, 0, 0])


@pytest.mark.parametrize(
    "n,k,expected",
    [(8, 3, [[0, 1], [2, 3, 4], [5, 6, 7]]), (8, 0, [list(range(8))]), (8, 8, [list(range(8))]), (4, 1, [[0], [1], [2], [3]])],
)
def test_loop_segments(n, k, expected):
    assert [list(r) for r in loop_segments(n, k)] == expected


def test_tbptt_cuts_gradient_to_early_loops():
    """With K=1 the embedding only receives gradient through a single block application per loop,
    so its gradient differs from full backprop."""
    set_seed(0)
    model = build_model("ut", CFG)
    puzzles = torch.randint(0, 5, (4, 16))
    sols = torch.randint(1, 5, (4, 16))
    grads = []
    for k in (1, 0):
        model.zero_grad()
        forward_backward(model, puzzles, sols, n_loops=4, tbptt_k=k, halt_loss_weight=0.5, amp=False)
        grads.append(model.embedding.token.weight.grad.clone())
    assert not torch.allclose(grads[0], grads[1])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_overfit_batch_of_32(name):
    set_seed(0)
    data_cfg = DataConfig()
    build_datasets(data_cfg, verbose=False)
    batch = load_split(data_cfg, "train").subset(slice(0, 32))
    model = build_model(name, CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        stats = forward_backward(model, batch.puzzles, batch.solutions, n_loops=4, tbptt_k=0, halt_loss_weight=0.5, amp=False)
        opt.step()
        if stats.puzzle_acc == 1.0:
            break
    assert stats.cell_acc > 0.99, f"{name}: cell acc {stats.cell_acc:.3f}"
    assert stats.puzzle_acc >= 0.97, f"{name}: puzzle acc {stats.puzzle_acc:.3f}"
