"""Training loop: AdamW + warmup/cosine, gradient clipping, CSV logging, best/last checkpoints, resume."""

from __future__ import annotations

import csv
import dataclasses
import gc
import math
import time
from pathlib import Path
from typing import Any

import torch

from ..config import DataConfig, ModelConfig, TrainConfig, resolve_device, set_seed
from ..data.augment import augment_batch
from ..data.dataset import load_split
from ..eval.metrics import evaluate
from ..models.models import SudokuModel, build_model
from .checkpoint import load_checkpoint, save_checkpoint
from .step import forward_backward

LOG_FIELDS = [
    "epoch", "step", "lr", "train_loss", "train_ce", "train_halt_loss", "train_cell_acc", "train_puzzle_acc",
    "val_final_cell_acc", "val_final_puzzle_acc", "val_final_validity",
    "val_halt_puzzle_acc", "val_avg_loops", "peak_mem_mb", "epoch_time_s",
]  # fmt: skip


def make_optimizer(model: SudokuModel, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices only (not norms, biases or embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embedding" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def warmup_cosine(total_steps: int, warmup_frac: float, min_ratio: float):
    """LR multiplier: linear warmup, then cosine decay from 1 to ``min_ratio``."""
    warmup = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def train(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    data_cfg: DataConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train ``train_cfg.model`` and return a summary dict (best val accuracy, paths, timing, memory)."""
    model_cfg = model_cfg or ModelConfig()
    data_cfg = data_cfg or DataConfig()
    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    run = train_cfg.resolved_run_name()
    ckpt_dir = Path(train_cfg.checkpoint_dir) / run
    log_path = Path(train_cfg.results_dir) / "runs" / run / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mem_base = 0
    if device.type == "cuda":  # measure memory relative to what was allocated before this run
        gc.collect()
        torch.cuda.empty_cache()
        mem_base = torch.cuda.memory_allocated(device)

    train_set = load_split(data_cfg, "train", device)
    val_set = load_split(data_cfg, "val", device)
    model = build_model(train_cfg.model, model_cfg).to(device)
    n_loops = model_cfg.max_loops_train if model.is_recurrent else 1
    if verbose:
        print(f"[{run}] model={train_cfg.model} params={model.num_parameters():,} device={device} "
              f"loops={n_loops} tbptt_k={train_cfg.tbptt_k or 'all'}")

    optimizer = make_optimizer(model, train_cfg)
    steps_per_epoch = math.ceil(len(train_set) / train_cfg.batch_size)
    total_steps = train_cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, train_cfg.warmup_frac, train_cfg.min_lr_ratio)
    )
    gen = torch.Generator(device=device).manual_seed(train_cfg.seed)
    start_epoch, best_acc, step = 0, -1.0, 0

    last_path = ckpt_dir / "last.pt"
    if train_cfg.resume and last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        gen.set_state(ckpt["generator"].cpu())
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        start_epoch, best_acc, step = ckpt["epoch"] + 1, ckpt["best_val_puzzle_acc"], ckpt["step"]
        if verbose:
            print(f"[{run}] resumed from epoch {ckpt['epoch']} (best val puzzle acc {best_acc:.4f})")
    elif log_path.exists():
        log_path.unlink()  # fresh run: start a fresh log

    train_time = 0.0
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        sums = dict(loss=0.0, ce=0.0, halt=0.0, cell=0.0, puzzle=0.0)
        perm = torch.randperm(len(train_set), generator=gen, device=device)
        for b in range(steps_per_epoch):
            idx = perm[b * train_cfg.batch_size : (b + 1) * train_cfg.batch_size]
            puzzles, solutions = train_set.puzzles[idx], train_set.solutions[idx]
            if train_cfg.augment:
                puzzles, solutions = augment_batch(puzzles, solutions, gen)
            optimizer.zero_grad(set_to_none=True)
            stats = forward_backward(
                model, puzzles, solutions, n_loops, train_cfg.tbptt_k, train_cfg.halt_loss_weight, train_cfg.amp
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            for key, val in zip(sums, (stats.loss, stats.ce, stats.halt_loss, stats.cell_acc, stats.puzzle_acc)):
                sums[key] += val / steps_per_epoch
        epoch_time = time.time() - t0
        train_time += epoch_time
        peak_mem = (torch.cuda.max_memory_allocated(device) - mem_base) / 2**20 if device.type == "cuda" else float("nan")

        val = evaluate(model, val_set, n_loops, train_cfg.eval_batch_size, train_cfg.halt_threshold, train_cfg.amp)
        row = {
            "epoch": epoch, "step": step, "lr": scheduler.get_last_lr()[0],
            "train_loss": sums["loss"], "train_ce": sums["ce"], "train_halt_loss": sums["halt"],
            "train_cell_acc": sums["cell"], "train_puzzle_acc": sums["puzzle"],
            "val_final_cell_acc": val["final_cell_acc"], "val_final_puzzle_acc": val["final_puzzle_acc"],
            "val_final_validity": val["final_validity"],
            "val_halt_puzzle_acc": val.get("halt_puzzle_acc", float("nan")),
            "val_avg_loops": val.get("avg_loops", float(n_loops)),
            "peak_mem_mb": peak_mem, "epoch_time_s": epoch_time,
        }  # fmt: skip
        _append_csv(log_path, row)

        extra = {"epoch": epoch, "step": step, "val_metrics": val, "train_cfg": dataclasses.asdict(train_cfg)}
        if val["final_puzzle_acc"] > best_acc:
            best_acc = val["final_puzzle_acc"]
            save_checkpoint(ckpt_dir / "best.pt", model, train_cfg.model, extra)
        save_checkpoint(
            last_path, model, train_cfg.model,
            {**extra, "best_val_puzzle_acc": best_acc, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "generator": gen.get_state(), "torch_rng": torch.get_rng_state()},
        )  # fmt: skip
        if verbose:
            halt = f" halt_acc={val['halt_puzzle_acc']:.4f} loops={val['avg_loops']:.2f}" if "avg_loops" in val else ""
            print(f"[{run}] ep {epoch:3d} loss={sums['loss']:.4f} train_puz={sums['puzzle']:.4f} "
                  f"val_cell={val['final_cell_acc']:.4f} val_puz={val['final_puzzle_acc']:.4f}{halt} "
                  f"mem={peak_mem:.0f}MB {epoch_time:.1f}s", flush=True)

    return {
        "run": run, "model": train_cfg.model, "params": model.num_parameters(), "best_val_puzzle_acc": best_acc,
        "best_checkpoint": str(ckpt_dir / "best.pt"), "log": str(log_path), "train_time_s": train_time,
    }  # fmt: skip


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
