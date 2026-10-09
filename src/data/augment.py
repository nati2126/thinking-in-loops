"""Validity-preserving Sudoku augmentations, vectorised over a batch on any device.

Each sample gets an independent random transform composed of:
  * a digit permutation (relabel 1-4),
  * row swaps within each 2-row band,
  * column swaps within each 2-column stack,
  * an optional transpose.
The same transform is applied to a puzzle and its solution, so solutions stay valid.
"""

from __future__ import annotations

import torch

from .sudoku import BOX, SIZE


def _within_group_perm(batch: int, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    """(B, SIZE) index permutations that optionally swap the two lines inside each band/stack."""
    n_groups = SIZE // BOX
    swap = torch.rand(batch, n_groups, generator=generator, device=device) < 0.5  # (B, G)
    base = torch.arange(SIZE, device=device).view(1, n_groups, BOX).expand(batch, -1, -1)  # (B, G, BOX)
    flipped = base.flip(-1)
    return torch.where(swap.unsqueeze(-1), flipped, base).reshape(batch, SIZE)


def random_transform(
    batch: int, generator: torch.Generator | None = None, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one random symmetry per sample.

    Returns:
        cell_index: (B, 16) long; new_grid[:, i] = old_grid[:, cell_index[:, i]].
        digit_map: (B, SIZE + 1) long; digit_map[:, 0] == 0 so empties stay empty.
    """
    device = torch.device(device)
    row_perm = _within_group_perm(batch, generator, device)  # (B, 4)
    col_perm = _within_group_perm(batch, generator, device)
    src = row_perm.unsqueeze(2) * SIZE + col_perm.unsqueeze(1)  # (B, 4, 4): new[r, c] = old[rp[r], cp[c]]
    transpose = torch.rand(batch, generator=generator, device=device) < 0.5
    src = torch.where(transpose.view(-1, 1, 1), src.transpose(1, 2), src)
    cell_index = src.reshape(batch, SIZE * SIZE)

    perm = torch.argsort(torch.rand(batch, SIZE, generator=generator, device=device), dim=1) + 1
    digit_map = torch.cat([torch.zeros(batch, 1, dtype=torch.long, device=device), perm], dim=1)
    return cell_index, digit_map


def apply_transform(grids: torch.Tensor, cell_index: torch.Tensor, digit_map: torch.Tensor) -> torch.Tensor:
    """Apply a transform from :func:`random_transform` to (B, 16) integer grids."""
    moved = grids.long().gather(1, cell_index)
    return digit_map.gather(1, moved).to(grids.dtype)


def all_symmetries() -> tuple[torch.Tensor, torch.Tensor]:
    """Every transform the augmentation can produce: (32, 16) cell maps and (24, SIZE + 1) digit maps."""
    import itertools

    cell_maps = []
    for r_swaps in itertools.product((False, True), repeat=SIZE // BOX):
        for c_swaps in itertools.product((False, True), repeat=SIZE // BOX):
            for transpose in (False, True):
                rp = [b * BOX + (BOX - 1 - j if s else j) for b, s in enumerate(r_swaps) for j in range(BOX)]
                cp = [b * BOX + (BOX - 1 - j if s else j) for b, s in enumerate(c_swaps) for j in range(BOX)]
                src = [[rp[r] * SIZE + cp[c] for c in range(SIZE)] for r in range(SIZE)]
                if transpose:
                    src = [list(col) for col in zip(*src)]
                cell_maps.append([i for row in src for i in row])
    digit_maps = [[0, *p] for p in itertools.permutations(range(1, SIZE + 1))]
    return torch.tensor(cell_maps), torch.tensor(digit_maps)


def symmetry_overlap(reference: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """(N,) bool: True if some symmetry image of ``queries[n]`` is a puzzle in ``reference``.

    Used to find test puzzles that training-time augmentation could have produced exactly.
    """
    cell_maps, digit_maps = all_symmetries()
    ci = cell_maps.repeat_interleave(len(digit_maps), 0)  # (768, 16)
    di = digit_maps.repeat(len(cell_maps), 1)  # (768, 5)
    seen = {bytes(p) for p in reference.cpu().to(torch.uint8).numpy()}
    out = torch.zeros(len(queries), dtype=torch.bool)
    for n, q in enumerate(queries.cpu()):
        images = apply_transform(q.unsqueeze(0).expand(len(ci), -1), ci, di).to(torch.uint8).numpy()
        out[n] = any(bytes(x) in seen for x in images)
    return out


def augment_batch(
    puzzles: torch.Tensor, solutions: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one shared random transform per sample to both puzzles and solutions."""
    cell_index, digit_map = random_transform(puzzles.shape[0], generator, puzzles.device)
    return apply_transform(puzzles, cell_index, digit_map), apply_transform(solutions, cell_index, digit_map)
