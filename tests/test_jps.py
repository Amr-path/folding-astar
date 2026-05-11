"""Tests for the 4-connected JPS implementation.

JPS must agree with ground-truth A* on path length for every reachable
instance. The path returned must be a valid 4-connected obstacle-free
sequence."""

from __future__ import annotations

import random

import pytest

from folding_astar import astar, jps, verify_symmetry
from folding_astar.types import Cell, Grid


def _is_valid_path(grid: Grid, p: list[Cell] | None) -> bool:
    if not p:
        return False
    for r, c in p:
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return False
        if grid[r][c] != 0:
            return False
    for a, b in zip(p, p[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------

def test_jps_empty_grid_straight_path() -> None:
    grid: Grid = [[0] * 5 for _ in range(5)]
    p = jps(grid, (0, 0), (4, 4))
    assert p is not None
    assert len(p) == 9
    assert _is_valid_path(grid, p)


def test_jps_self_loop() -> None:
    grid: Grid = [[0] * 3 for _ in range(3)]
    p = jps(grid, (1, 1), (1, 1))
    assert p == [(1, 1)]


def test_jps_blocked_endpoint() -> None:
    grid: Grid = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    assert jps(grid, (0, 1), (2, 2)) is None
    assert jps(grid, (0, 0), (0, 1)) is None


def test_jps_unreachable() -> None:
    # Wall splitting the grid
    grid: Grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    assert jps(grid, (0, 0), (0, 2)) is None


# ---------------------------------------------------------------------------
# Forced-neighbour cases
# ---------------------------------------------------------------------------

def test_jps_forced_neighbour_corner() -> None:
    """An obstacle creates a forced perpendicular successor.

    Layout (S = start, G = goal, # = obstacle):
        S . # .
        . . . .
        . . . G
    Going east from S, after the obstacle the cell (0, 1) has the cell
    (0, 2) blocked but the perpendicular cell (1, 2) is reachable only
    via x; this is the canonical forced-neighbour case."""
    grid: Grid = [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p = jps(grid, (0, 0), (2, 3))
    ref = astar(grid, (0, 0), (2, 3))
    assert p is not None and ref is not None
    assert _is_valid_path(grid, p)
    assert len(p) == len(ref)


# ---------------------------------------------------------------------------
# Differential test against ground-truth A*
# ---------------------------------------------------------------------------

def _random_grid(N: int, density: float, seed: int) -> Grid:
    rng = random.Random(seed)
    grid: Grid = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if rng.random() < density:
                grid[i][j] = 1
    return grid


def _random_symmetric_grid(N: int, density: float, seed: int) -> Grid:
    rng = random.Random(seed)
    m = N // 2
    grid: Grid = [[0] * N for _ in range(N)]
    for i in range(m):
        for j in range(N):
            if rng.random() < density:
                grid[i][j] = 1
                grid[N - 1 - i][j] = 1
    return grid


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_jps_differential_against_astar_random(seed: int) -> None:
    """200 random pairs on random asymmetric grids: JPS must agree with A*
    on path length. JPS must produce valid paths."""
    rng = random.Random(seed)
    fails: list[str] = []
    matches = 0
    for _ in range(50):
        N = rng.choice([6, 8, 10, 12, 16, 20])
        d = rng.choice([0.0, 0.1, 0.2, 0.3])
        grid = _random_grid(N, d, rng.randint(0, 1 << 30))
        free = [(i, j) for i in range(N) for j in range(N) if grid[i][j] == 0]
        if len(free) < 2:
            continue
        s, t = rng.choice(free), rng.choice(free)
        if s == t:
            continue
        ref = astar(grid, s, t)
        p = jps(grid, s, t)
        if ref is None and p is None:
            continue
        if ref is None or p is None:
            fails.append(f"None mismatch at {s} -> {t}: ref={ref} jps={p}")
            continue
        if not _is_valid_path(grid, p):
            fails.append(f"invalid jps path {p} for {s} -> {t}")
            continue
        if len(ref) != len(p):
            fails.append(f"len mismatch {s} -> {t}: A*={len(ref)-1} JPS={len(p)-1}")
            continue
        matches += 1
    if fails:
        pytest.fail(f"{len(fails)} failures, first 3:\n  " + "\n  ".join(fails[:3]))
    assert matches > 0


@pytest.mark.parametrize("seed", [99, 137])
def test_jps_differential_against_astar_symmetric(seed: int) -> None:
    """Same differential test on horizontally-symmetric grids (the input
    family the rest of this paper cares about)."""
    rng = random.Random(seed)
    fails: list[str] = []
    matches = 0
    for _ in range(40):
        N = rng.choice([6, 8, 10, 12, 16, 20])
        d = rng.choice([0.0, 0.1, 0.2])
        grid = _random_symmetric_grid(N, d, rng.randint(0, 1 << 30))
        assert verify_symmetry(grid)
        free = [(i, j) for i in range(N) for j in range(N) if grid[i][j] == 0]
        if len(free) < 2:
            continue
        s, t = rng.choice(free), rng.choice(free)
        if s == t:
            continue
        ref = astar(grid, s, t)
        p = jps(grid, s, t)
        if ref is None and p is None:
            continue
        if ref is None or p is None:
            fails.append(f"None mismatch at {s} -> {t}: ref={ref} jps={p}")
            continue
        if not _is_valid_path(grid, p):
            fails.append(f"invalid jps path {p}")
            continue
        if len(ref) != len(p):
            fails.append(f"len mismatch {s} -> {t}: A*={len(ref)-1} JPS={len(p)-1}")
            continue
        matches += 1
    if fails:
        pytest.fail(f"{len(fails)} failures, first 3:\n  " + "\n  ".join(fails[:3]))
    assert matches > 0
