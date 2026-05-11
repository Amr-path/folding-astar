"""Tests for the BFS / Dijkstra baselines and the pluggable folding_search.

Each search algorithm must agree with ground-truth A* on path length for
every reachable instance. The folded variants must additionally produce
valid paths in the original grid (4-connected, obstacle-free)."""

from __future__ import annotations

import random

import pytest

from folding_astar import (
    astar, bfs, dijkstra,
    folding_astar, folding_bfs, folding_dijkstra,
    verify_symmetry,
)
from folding_astar.types import Cell, Grid


def _make_symmetric(top: list[list[int]], mid: list[int] | None = None) -> Grid:
    rows = [list(r) for r in top]
    if mid is not None:
        rows.append(list(mid))
    rows.extend(list(r) for r in reversed(top))
    return rows


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
# Standalone BFS and Dijkstra agree with A* on path length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alg", [bfs, dijkstra])
def test_uninformed_search_basic(alg: object) -> None:
    """Plain BFS/Dijkstra return optimal paths in trivial cases."""
    grid: Grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    p = alg(grid, (0, 0), (2, 2))  # type: ignore[operator]
    assert p is not None
    assert len(p) == 5  # length 4 (5 cells)
    p_self = alg(grid, (0, 0), (0, 0))  # type: ignore[operator]
    assert p_self == [(0, 0)]
    p_blocked = alg(grid, (1, 1), (0, 0))  # type: ignore[operator]
    assert p_blocked is None


def test_dijkstra_matches_bfs_matches_astar() -> None:
    """On unweighted 4-connected grids, all three return paths of equal length."""
    grid = _make_symmetric([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    rng = random.Random(7)
    free = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 0]
    for _ in range(20):
        s = rng.choice(free)
        t = rng.choice(free)
        a = astar(grid, s, t)
        b = bfs(grid, s, t)
        d = dijkstra(grid, s, t)
        if a is None:
            assert b is None and d is None
            continue
        assert b is not None and d is not None
        assert len(a) == len(b) == len(d), f"length disagreement at {s} -> {t}"


# ---------------------------------------------------------------------------
# Folded variants produce valid, optimal paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folded_alg", [folding_astar, folding_bfs, folding_dijkstra])
def test_folded_variants_match_astar_on_edge_cases(folded_alg: object) -> None:
    """Each folded variant must give the same path length as A* on the
    canonical edge cases."""
    cases = [
        (_make_symmetric([[0]*6, [0,1,0,1,0,0], [0,0,0,1,0,0]]), (0,2), (5,2)),
        (_make_symmetric([[0]*4, [0]*4]), (0,0), (3,1)),
        (_make_symmetric([[0]*5, [0,1,0,1,0], [0]*5], mid=[0]*5), (0,0), (6,0)),
        (_make_symmetric([[0]*6, [0,1,1,1,1,0], [0]*6]), (0,0), (0,5)),
        (_make_symmetric([[0]*6, [0]*6, [0]*6]), (1,0), (4,5)),
    ]
    for grid, s, t in cases:
        assert verify_symmetry(grid)
        ref = astar(grid, s, t)
        path, info = folded_alg(grid, s, t)  # type: ignore[operator]
        assert ref is not None and path is not None, f"None at {s} -> {t}"
        assert _is_valid_path(grid, path), f"invalid path at {s} -> {t}: {path}"
        assert len(path) == len(ref), (
            f"length mismatch at {s} -> {t}: ref={len(ref)-1} got={len(path)-1} info={info}"
        )


def test_folded_variants_fallback_on_asymmetric_grid() -> None:
    """Asymmetric input must trigger fallback to the underlying algorithm."""
    grid: Grid = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],     # asymmetric
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert not verify_symmetry(grid)
    for folded_alg in [folding_astar, folding_bfs, folding_dijkstra]:
        path, info = folded_alg(grid, (0, 0), (3, 3))  # type: ignore[operator]
        assert info == "fallback: not symmetric"
        ref = astar(grid, (0, 0), (3, 3))
        assert ref is not None and path is not None
        assert len(ref) == len(path)


# ---------------------------------------------------------------------------
# Randomised differential test for all variants
# ---------------------------------------------------------------------------

def _rand_sym(N: int, density: float, seed: int) -> Grid:
    rng = random.Random(seed)
    m = N // 2
    g: Grid = [[0] * N for _ in range(N)]
    for i in range(m):
        for j in range(N):
            if rng.random() < density:
                g[i][j] = 1
                g[N - 1 - i][j] = 1
    if N % 2 == 1:
        for j in range(N):
            if rng.random() < density:
                g[m][j] = 1
    return g


@pytest.mark.parametrize("folded_alg", [folding_astar, folding_bfs, folding_dijkstra])
def test_randomised_differential(folded_alg: object) -> None:
    """200 random symmetric grids, each folded variant must agree with A*."""
    rng = random.Random(13)
    fails: list[str] = []
    matches = 0
    for _ in range(200):
        N = rng.choice([5, 6, 8, 10, 12, 16, 20])
        d = rng.choice([0.0, 0.1, 0.2, 0.3])
        g = _rand_sym(N, d, rng.randint(0, 1 << 30))
        free = [(i, j) for i in range(N) for j in range(N) if g[i][j] == 0]
        if len(free) < 2:
            continue
        s, t = rng.choice(free), rng.choice(free)
        ref = astar(g, s, t)
        path, info = folded_alg(g, s, t)  # type: ignore[operator]
        if ref is None and path is None:
            continue
        if ref is None or path is None:
            fails.append(f"None mismatch at {s} -> {t}: ref={ref} got={path} ({info})")
            continue
        if not _is_valid_path(g, path):
            fails.append(f"invalid at {s} -> {t}: {path}")
            continue
        if len(ref) != len(path):
            fails.append(
                f"len mismatch {s} -> {t}: ref={len(ref)-1} got={len(path)-1} info={info}"
            )
            continue
        matches += 1
    if fails:
        pytest.fail(f"{len(fails)} failures of 200, first 3:\n  " + "\n  ".join(fails[:3]))
    assert matches > 0, "no reachable instances generated"
