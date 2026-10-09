"""Phase 1 tests: solver, generator, saved datasets and augmentation."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.config import DataConfig
from src.data.augment import augment_batch, random_transform
from src.data.dataset import SPLITS, build_datasets, load_split
from src.data.sudoku import (
    GROUPS,
    from_string,
    has_conflicts,
    is_complete_valid,
    make_puzzle,
    matches_clues,
    random_full_grid,
    solve,
    to_string,
)

SOLVED = from_string("1234 3412 2143 4321")


def test_groups_cover_each_cell_three_times():
    counts = np.zeros(16, dtype=int)
    for g in GROUPS:
        counts[list(g)] += 1
    assert len(GROUPS) == 12 and (counts == 3).all()


def test_rule_checks():
    assert is_complete_valid(SOLVED)
    broken = list(SOLVED)
    broken[0], broken[1] = broken[1], broken[0]
    assert not is_complete_valid(broken)
    assert has_conflicts([1, 1] + [0] * 14)
    assert not has_conflicts([0] * 16)


def test_solver_on_known_puzzle():
    puzzle = list(SOLVED)
    for i in (0, 5, 10, 15):
        puzzle[i] = 0
    res = solve(puzzle)
    assert res.solution == SOLVED and res.num_solutions == 1


def test_solver_detects_multiple_and_no_solutions():
    assert solve([0] * 16).num_solutions == 2  # capped at limit
    assert solve([1, 1] + [0] * 14).num_solutions == 0


def test_backtrack_count_nonnegative_and_deterministic():
    rng = random.Random(0)
    puzzle, _ = make_puzzle(rng, 8)
    a, b = solve(puzzle), solve(puzzle)
    assert a.backtracks == b.backtracks >= 0


@pytest.mark.parametrize("n_empty", [4, 6, 8])
def test_generator_unique_and_correct(n_empty):
    rng = random.Random(n_empty)
    for _ in range(50):
        puzzle, solution = make_puzzle(rng, n_empty)
        assert sum(v == 0 for v in puzzle) == n_empty
        assert is_complete_valid(solution)
        assert matches_clues(puzzle, solution)
        res = solve(puzzle, limit=2)
        assert res.num_solutions == 1
        assert res.solution == solution


def test_random_full_grid_valid():
    rng = random.Random(1)
    assert all(is_complete_valid(random_full_grid(rng)) for _ in range(100))


@pytest.fixture(scope="module")
def data_cfg() -> DataConfig:
    cfg = DataConfig()
    build_datasets(cfg, verbose=False)
    return cfg


def test_saved_splits_sizes_and_no_overlap(data_cfg):
    splits = {s: load_split(data_cfg, s) for s in SPLITS}
    assert [len(splits[s]) for s in SPLITS] == [data_cfg.n_train, data_cfg.n_val, data_cfg.n_test]
    keys = {s: {to_string(p) for p in splits[s].puzzles.tolist()} for s in SPLITS}
    for s in SPLITS:
        assert len(keys[s]) == len(splits[s]), f"duplicate puzzles inside {s}"
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])


@pytest.mark.parametrize("split", SPLITS)
def test_every_saved_puzzle_valid_and_unique(data_cfg, split):
    data = load_split(data_cfg, split)
    for p, s, ne, bt in zip(data.puzzles.tolist(), data.solutions.tolist(), data.n_empty.tolist(), data.backtracks.tolist()):
        assert data_cfg.min_empty <= ne <= data_cfg.max_empty and sum(v == 0 for v in p) == ne
        assert is_complete_valid(s) and matches_clues(p, s)
        res = solve(p, limit=2)
        assert res.num_solutions == 1 and res.solution == s and res.backtracks == bt


def test_augmentation_preserves_validity(data_cfg):
    data = load_split(data_cfg, "train").subset(slice(0, 2000))
    gen = torch.Generator().manual_seed(0)
    p, s = augment_batch(data.puzzles, data.solutions, gen)
    assert (p == 0).sum(1).tolist() == data.n_empty.tolist()  # blanks preserved
    changed = 0
    for pp, ss, p0 in zip(p.tolist(), s.tolist(), data.puzzles.tolist()):
        assert is_complete_valid(ss) and matches_clues(pp, ss)
        assert solve(pp).num_solutions == 1  # uniqueness is invariant under symmetries
        changed += pp != p0
    assert changed > 1900  # transforms actually do something


def test_transform_is_a_permutation_and_covers_all_symmetries():
    cell_index, digit_map = random_transform(4096, torch.Generator().manual_seed(1))
    assert (cell_index.sort(1).values == torch.arange(16)).all()
    assert (digit_map[:, 0] == 0).all() and (digit_map[:, 1:].sort(1).values == torch.arange(1, 5)).all()
    # 2 (row swaps) * 2 * 2 (col swaps) * 2 * 2 (transpose) = 32 distinct cell maps
    assert len({tuple(r) for r in cell_index.tolist()}) == 32


def test_all_symmetries_enumerates_768_valid_transforms():
    from src.data.augment import all_symmetries, apply_transform

    cell_maps, digit_maps = all_symmetries()
    assert cell_maps.shape == (32, 16) and digit_maps.shape == (24, 5)
    ci, di = cell_maps.repeat_interleave(24, 0), digit_maps.repeat(32, 1)
    grid = torch.tensor(SOLVED).unsqueeze(0).expand(768, -1)
    images = apply_transform(grid, ci, di)
    assert all(is_complete_valid(g) for g in images.tolist())
    # the random sampler only ever produces maps from this enumeration
    sampled, _ = random_transform(2048, torch.Generator().manual_seed(3))
    assert {tuple(r) for r in sampled.tolist()} <= {tuple(r) for r in cell_maps.tolist()}


def test_symmetry_overlap_detects_transformed_copy():
    from src.data.augment import augment_batch, symmetry_overlap

    rng = random.Random(5)
    a = torch.tensor([make_puzzle(rng, 6)[0] for _ in range(4)])
    b = torch.tensor([make_puzzle(rng, 6)[0] for _ in range(4)])
    moved, _ = augment_batch(a, a.clone(), torch.Generator().manual_seed(0))
    assert symmetry_overlap(a, moved).all()
    assert symmetry_overlap(a, a).all()
    assert symmetry_overlap(b, b).all() and len(symmetry_overlap(a, b)) == 4
