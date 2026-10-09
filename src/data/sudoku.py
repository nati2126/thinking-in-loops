"""4x4 Sudoku primitives: rules, a backtracking solver and a puzzle generator.

Grids are flat sequences of 16 ints in row-major order. Digits are 1-4 and
0 marks an empty cell. Boxes are 2x2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

SIZE = 4  # digits 1..SIZE, SIZE x SIZE grid
BOX = 2  # box side length
NUM_CELLS = SIZE * SIZE
DIGITS = tuple(range(1, SIZE + 1))


def row_of(cell: int) -> int:
    return cell // SIZE


def col_of(cell: int) -> int:
    return cell % SIZE


def box_of(cell: int) -> int:
    return (row_of(cell) // BOX) * (SIZE // BOX) + col_of(cell) // BOX


# The 12 constraint groups (4 rows, 4 columns, 4 boxes), each a tuple of cell indices.
GROUPS: tuple[tuple[int, ...], ...] = (
    tuple(tuple(r * SIZE + c for c in range(SIZE)) for r in range(SIZE))
    + tuple(tuple(r * SIZE + c for r in range(SIZE)) for c in range(SIZE))
    + tuple(tuple(i for i in range(NUM_CELLS) if box_of(i) == b) for b in range(SIZE))
)

# PEERS[i] = cells sharing a row, column or box with cell i (excluding i).
PEERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(sorted({j for g in GROUPS if i in g for j in g} - {i})) for i in range(NUM_CELLS)
)


def has_conflicts(grid: Sequence[int]) -> bool:
    """True if two equal non-zero digits share a row, column or box."""
    for g in GROUPS:
        seen = [grid[i] for i in g if grid[i] != 0]
        if len(seen) != len(set(seen)):
            return True
    return False


def is_complete_valid(grid: Sequence[int]) -> bool:
    """True if ``grid`` is a fully filled grid that obeys every Sudoku rule."""
    if len(grid) != NUM_CELLS or any(v not in DIGITS for v in grid):
        return False
    return all(sorted(grid[i] for i in g) == list(DIGITS) for g in GROUPS)


def matches_clues(puzzle: Sequence[int], grid: Sequence[int]) -> bool:
    """True if ``grid`` agrees with every given (non-zero) cell of ``puzzle``."""
    return all(p == 0 or p == g for p, g in zip(puzzle, grid))


@dataclass
class SolveResult:
    """Outcome of :func:`solve`.

    Attributes:
        solution: First solution found (row-major), or None if unsolvable.
        num_solutions: Number of solutions found, capped at the search ``limit``.
        backtracks: Assignments undone (dead ends) before the first solution was
            found. Used as a difficulty signal.
    """

    solution: list[int] | None
    num_solutions: int
    backtracks: int


def solve(puzzle: Sequence[int], limit: int = 2) -> SolveResult:
    """Depth-first backtracking solver.

    Cells are filled in row-major order and digits tried in ascending order, so
    the backtrack count is deterministic for a given puzzle. The search stops
    after ``limit`` solutions; ``limit=2`` is enough to test uniqueness.
    """
    grid = list(puzzle)
    if has_conflicts(grid):
        return SolveResult(None, 0, 0)
    empties = [i for i, v in enumerate(grid) if v == 0]
    state = {"solutions": 0, "first": None, "backtracks": 0}

    def search(k: int) -> bool:
        """Returns True when the search should stop (limit reached)."""
        if k == len(empties):
            state["solutions"] += 1
            if state["first"] is None:
                state["first"] = list(grid)
            return state["solutions"] >= limit
        cell = empties[k]
        used = {grid[j] for j in PEERS[cell]}
        for d in DIGITS:
            if d in used:
                continue
            grid[cell] = d
            if search(k + 1):
                return True
            grid[cell] = 0
            if state["first"] is None:
                state["backtracks"] += 1
        return False

    search(0)
    return SolveResult(state["first"], state["solutions"], state["backtracks"])


def random_full_grid(rng: random.Random) -> list[int]:
    """Sample a uniformly-shuffled complete valid grid via randomized backtracking."""
    grid = [0] * NUM_CELLS

    def fill(cell: int) -> bool:
        if cell == NUM_CELLS:
            return True
        used = {grid[j] for j in PEERS[cell]}
        options = [d for d in DIGITS if d not in used]
        rng.shuffle(options)
        for d in options:
            grid[cell] = d
            if fill(cell + 1):
                return True
        grid[cell] = 0
        return False

    fill(0)
    return grid


def make_puzzle(rng: random.Random, n_empty: int, max_restarts: int = 100) -> tuple[list[int], list[int]]:
    """Generate a puzzle with exactly ``n_empty`` blanks and a unique solution.

    Starts from a random full grid and removes cells one at a time in random
    order, skipping any removal that would make the solution non-unique.
    If the target cannot be reached from a grid, a new grid is drawn.

    Returns:
        (puzzle, solution) as flat lists.
    """
    for _ in range(max_restarts):
        solution = random_full_grid(rng)
        puzzle = list(solution)
        order = list(range(NUM_CELLS))
        rng.shuffle(order)
        removed = 0
        for cell in order:
            if removed == n_empty:
                break
            puzzle[cell] = 0
            if solve(puzzle, limit=2).num_solutions == 1:
                removed += 1
            else:
                puzzle[cell] = solution[cell]
        if removed == n_empty:
            return puzzle, solution
    raise RuntimeError(f"could not generate a unique puzzle with {n_empty} empty cells")


def to_string(grid: Sequence[int]) -> str:
    """Compact 16-character representation, '0' for empty."""
    return "".join(str(int(v)) for v in grid)


def from_string(s: str) -> list[int]:
    """Parse a 16-character puzzle string. '.', '_' and '0' mean empty; whitespace is ignored."""
    chars = [ch for ch in s if not ch.isspace() and ch not in ",|"]
    if len(chars) != NUM_CELLS:
        raise ValueError(f"expected {NUM_CELLS} cells, got {len(chars)}")
    out = []
    for ch in chars:
        if ch in "._0":
            out.append(0)
        elif ch.isdigit() and 1 <= int(ch) <= SIZE:
            out.append(int(ch))
        else:
            raise ValueError(f"invalid cell character {ch!r}")
    return out


def pretty(grid: Sequence[int]) -> str:
    """Human-readable 4x4 rendering with box separators."""
    lines = []
    for r in range(SIZE):
        if r and r % BOX == 0:
            lines.append("------+------")
        cells = [str(grid[r * SIZE + c]) if grid[r * SIZE + c] else "." for c in range(SIZE)]
        lines.append(" " + " ".join(cells[:BOX]) + " | " + " ".join(cells[BOX:]))
    return "\n".join(lines)
