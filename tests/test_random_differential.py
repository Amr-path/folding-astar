"""Randomised differential test: Folding A* must agree with ground-truth A*
on every reachable instance and must produce a valid path on every claim of
reachability.

This is the test that catches regressions the hand-built edge cases miss.
"""

from __future__ import annotations

import random

import pytest

from folding_astar import astar, folding_astar, midline
from folding_astar.types import Cell, Grid


# ---------------------------------------------------------------------------
# Random symmetric grid generator
# ---------------------------------------------------------------------------

def _random_symmetric_grid(N: int, density: float, seed: int) -> Grid:
    rng = random.Random(seed)
    m = midline(N)
    grid: Grid = [[0] * N for _ in range(N)]
    for i in range(m):
        for j in range(N):
            if rng.random() < density:
                grid[i][j] = 1
                grid[N - 1 - i][j] = 1
    if N % 2 == 1:
        for j in range(N):
            if rng.random() < density:
                grid[m][j] = 1   # midline row, fixed under reflection
    return grid


def _random_free_cell(grid: Grid, rng: random.Random) -> Cell | None:
    free = [(i, j) for i in range(len(grid)) for j in range(len(grid[0]))
            if grid[i][j] == 0]
    if not free:
        return None
    return rng.choice(free)


def _path_is_valid(grid: Grid, path: list[Cell]) -> bool:
    if not path:
        return False
    for c in path:
        r, col = c
        if not (0 <= r < len(grid) and 0 <= col < len(grid[0])):
            return False
        if grid[r][col] != 0:
            return False
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


# ---------------------------------------------------------------------------
# The differential test, parametrised over a few configurations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_trials,sizes,densities,seed",
    [
        (200, [4, 5, 6, 7, 8, 10, 12], [0.0, 0.1, 0.2, 0.3], 42),
        (200, [16, 20, 25, 30], [0.0, 0.1, 0.2, 0.3], 137),
    ],
    ids=["small-grids", "medium-grids"],
)
def test_random_differential(num_trials: int, sizes: list[int],
                             densities: list[float], seed: int) -> None:
    rng = random.Random(seed)
    failures: list[str] = []
    pass_count = 0
    skip_count = 0

    for trial in range(num_trials):
        N = rng.choice(sizes)
        d = rng.choice(densities)
        grid = _random_symmetric_grid(N, d, rng.randint(0, 1 << 30))
        s = _random_free_cell(grid, rng)
        t = _random_free_cell(grid, rng)
        if s is None or t is None:
            skip_count += 1
            continue

        ref = astar(grid, s, t)
        path, info = folding_astar(grid, s, t)

        if ref is None and path is None:
            skip_count += 1
            continue
        if ref is None:
            failures.append(
                f"trial #{trial} N={N} d={d} s={s} t={t}: "
                f"A* found no path but Folding A* returned {path} (info={info!r})"
            )
            continue
        if path is None:
            failures.append(
                f"trial #{trial} N={N} d={d} s={s} t={t}: "
                f"Folding A* returned None but A* path length {len(ref) - 1} "
                f"(info={info!r})"
            )
            continue
        if not _path_is_valid(grid, path):
            failures.append(
                f"trial #{trial} N={N} d={d} s={s} t={t}: invalid path {path}"
            )
            continue
        if len(path) != len(ref):
            failures.append(
                f"trial #{trial} N={N} d={d} s={s} t={t}: length mismatch "
                f"A*={len(ref) - 1} vs Folding A*={len(path) - 1} "
                f"(info={info!r})"
            )
            continue
        pass_count += 1

    if failures:
        head = failures[:5]
        msg = (
            f"\n{len(failures)} differential failures out of {num_trials} trials.\n"
            f"first 5:\n  " + "\n  ".join(head)
        )
        pytest.fail(msg)

    # Sanity: at least some trials must have produced a real path comparison.
    assert pass_count > 0, (
        f"no reachable instances generated (trials={num_trials}, skipped={skip_count})"
    )
