"""Phase 3 tests: LR schedule, an end-to-end tiny training run, checkpoints and resume."""

from __future__ import annotations

import csv

import pytest
import torch

from src.config import DataConfig, ModelConfig, TrainConfig
from src.models.models import build_model
from src.training.checkpoint import load_model
from src.training.step import forward_backward
from src.training.trainer import train, warmup_cosine


def test_warmup_cosine_shape():
    fn = warmup_cosine(total_steps=100, warmup_frac=0.1, min_ratio=0.05)
    assert fn(0) == pytest.approx(0.1)
    assert fn(9) == pytest.approx(1.0)
    assert fn(100) == pytest.approx(0.05)
    vals = [fn(s) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))  # monotone decay after warmup


@pytest.fixture
def tiny_cfgs(tmp_path):
    data = DataConfig(data_dir=str(tmp_path / "data"), n_train=256, n_val=64, n_test=64, seed=7)
    model = ModelConfig(d_model=32, n_heads=2, max_loops_train=4, same_compute_layers=2)
    train_cfg = TrainConfig(
        epochs=2, batch_size=64, eval_batch_size=64, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"),
    )  # fmt: skip
    return data, model, train_cfg


@pytest.mark.parametrize("name", ["ut", "same_param", "same_compute"])
def test_tiny_training_run_writes_logs_and_checkpoints(tiny_cfgs, name):
    data, model_cfg, train_cfg = tiny_cfgs
    train_cfg.model = name
    summary = train(train_cfg, model_cfg, data, verbose=False)
    with open(summary["log"]) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"]) * 1.5
    model, ckpt = load_model(summary["best_checkpoint"])
    assert ckpt["model_name"] == name and model.num_parameters() == summary["params"]


def test_resume_continues_from_last_epoch(tiny_cfgs):
    data, model_cfg, train_cfg = tiny_cfgs
    train(train_cfg, model_cfg, data, verbose=False)
    train_cfg.epochs, train_cfg.resume = 3, True
    summary = train(train_cfg, model_cfg, data, verbose=False)
    with open(summary["log"]) as f:
        epochs = [int(r["epoch"]) for r in csv.DictReader(f)]
    assert epochs == [0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="memory measurement needs CUDA")
def test_tbptt_memory_grows_with_k():
    device = torch.device("cuda")
    model = build_model("ut", ModelConfig()).to(device)
    puzzles = torch.randint(0, 5, (256, 16), device=device)
    sols = torch.randint(1, 5, (256, 16), device=device)
    peaks = []
    for k in (1, 0):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        forward_backward(model, puzzles, sols, n_loops=8, tbptt_k=k, halt_loss_weight=0.5, amp=True)
        peaks.append(torch.cuda.max_memory_allocated())
        model.zero_grad(set_to_none=True)
    assert peaks[0] < peaks[1]
