"""Dataset generation, on-disk storage (.npz) and loading as tensors."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import DataConfig
from .sudoku import make_puzzle, solve, to_string

SPLITS = ("train", "val", "test")


@dataclass
class SudokuSplit:
    """One split held as tensors.

    Attributes:
        puzzles: (N, 16) long, 0 = empty.
        solutions: (N, 16) long, digits 1-4.
        n_empty: (N,) long, number of blanks (difficulty label 1).
        backtracks: (N,) long, solver backtracks before first solution (difficulty label 2).
    """

    puzzles: torch.Tensor
    solutions: torch.Tensor
    n_empty: torch.Tensor
    backtracks: torch.Tensor

    def __len__(self) -> int:
        return self.puzzles.shape[0]

    def to(self, device: torch.device | str) -> "SudokuSplit":
        return SudokuSplit(*(t.to(device) for t in (self.puzzles, self.solutions, self.n_empty, self.backtracks)))

    def subset(self, idx: torch.Tensor | slice) -> "SudokuSplit":
        return SudokuSplit(self.puzzles[idx], self.solutions[idx], self.n_empty[idx], self.backtracks[idx])


def split_path(cfg: DataConfig, split: str) -> Path:
    return Path(cfg.data_dir) / f"sudoku4_{split}.npz"


def generate_unique_puzzles(cfg: DataConfig, total: int) -> dict[str, np.ndarray]:
    """Generate ``total`` distinct puzzles with blank counts drawn uniformly from [min_empty, max_empty]."""
    rng = random.Random(cfg.seed)
    seen: set[str] = set()
    puzzles, solutions, n_empty, backtracks = [], [], [], []
    while len(seen) < total:
        k = rng.randint(cfg.min_empty, cfg.max_empty)
        puzzle, solution = make_puzzle(rng, k)
        key = to_string(puzzle)
        if key in seen:
            continue
        seen.add(key)
        result = solve(puzzle, limit=2)
        assert result.num_solutions == 1 and result.solution == solution
        puzzles.append(puzzle)
        solutions.append(solution)
        n_empty.append(k)
        backtracks.append(result.backtracks)
    return {
        "puzzles": np.asarray(puzzles, dtype=np.int8),
        "solutions": np.asarray(solutions, dtype=np.int8),
        "n_empty": np.asarray(n_empty, dtype=np.int16),
        "backtracks": np.asarray(backtracks, dtype=np.int16),
    }


def build_datasets(cfg: DataConfig, overwrite: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Generate train/val/test once and save each split to ``cfg.data_dir`` as .npz.

    Puzzles are unique as strings across the whole pool, then shuffled and split,
    so no puzzle string appears in more than one split.
    """
    paths = {s: split_path(cfg, s) for s in SPLITS}
    if not overwrite and all(p.exists() for p in paths.values()):
        if verbose:
            print(f"datasets already exist in {cfg.data_dir}; skipping generation")
        return paths
    sizes = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}
    t0 = time.time()
    pool = generate_unique_puzzles(cfg, sum(sizes.values()))
    order = np.random.default_rng(cfg.seed).permutation(len(pool["puzzles"]))
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    start = 0
    for split in SPLITS:
        idx = order[start : start + sizes[split]]
        start += sizes[split]
        np.savez_compressed(paths[split], **{k: v[idx] for k, v in pool.items()})
    if verbose:
        print(f"generated {start} puzzles in {time.time() - t0:.1f}s -> {cfg.data_dir}")
    return paths


def load_split(cfg: DataConfig, split: str, device: torch.device | str = "cpu") -> SudokuSplit:
    """Load a saved split as long tensors on ``device`` (generating datasets first if missing)."""
    path = split_path(cfg, split)
    if not path.exists():
        build_datasets(cfg)
    with np.load(path) as z:
        tensors = [torch.from_numpy(z[k].astype(np.int64)) for k in ("puzzles", "solutions", "n_empty", "backtracks")]
    return SudokuSplit(*tensors).to(device)
