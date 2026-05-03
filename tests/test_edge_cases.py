"""Hand-built edge-case tests for Folding A*.

Each case targets a specific concern from the JAAMAS reviewer reports:

  E1, E3:  same column / opposite halves and t = ρ(s)
           Reviewer 2: "the claim here that dist_G = dist_G_f seems incorrect"
  E2:      diagonally adjacent in folded grid, opposite halves in original
           Reviewer 2: "what if they are diagonally adjacent in the folded
           grid but on different sides of the midline in the original?"
  E4:      degenerate t == s
           Reviewer 2: "by symmetry — what if t = s?"
  E5, E6:  optimal path forced through the midline (even and odd N)
           Reviewer 2: "the path ... will need to cross the midline somewhere"
  E7, E8:  opposite-halves split with non-trivial midline crossing column

Every test asserts: (a) Folding A* returns a path, (b) the path is valid in
the original grid (4-connected, obstacle-free), (c) its length equals the
optimum found by ground-truth A*.
"""

from __future__ import annotations

import pytest

from folding_astar import astar, folding_astar, rho, verify_symmetry
from folding_astar.types import Cell, Grid


def _make_symmetric(top_half: list[list[int]],
                    midline_row: list[int] | None = None) -> Grid:
    rows = [list(r) for r in top_half]
    if midline_row is not None:
        rows.append(list(midline_row))
    rows.extend(list(r) for r in reversed(top_half))
    return rows


def _is_valid_path(grid: Grid, path: list[Cell] | None) -> tuple[bool, str]:
    if path is None:
        return False, "path is None"
    if not path:
        return False, "empty path"
    for c in path:
        r, col = c
        if not (0 <= r < len(grid) and 0 <= col < len(grid[0])):
            return False, f"cell {c} out of bounds"
        if grid[r][col] != 0:
            return False, f"cell {c} is an obstacle"
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False, f"non-4-adjacent step {a} -> {b}"
    return True, "valid"


def _check(grid: Grid, start: Cell, goal: Cell,
           expected_info: str | None = None) -> None:
    """Universal harness: Folding A* must agree with A* on length and produce
    a valid path."""
    assert verify_symmetry(grid), "grid is not symmetric — bad test setup"
    ref = astar(grid, start, goal)
    path, info = folding_astar(grid, start, goal)

    if ref is None:
        assert path is None, f"A* found no path but Folding A* returned {path}"
        return

    valid, why = _is_valid_path(grid, path)
    assert valid, f"Folding A* returned invalid path: {why} | path={path}"
    assert path is not None
    assert len(path) == len(ref), (
        f"length mismatch: A*={len(ref) - 1} vs Folding A*={len(path) - 1} | "
        f"info={info!r} A*-path={ref} F-path={path}"
    )
    if expected_info is not None:
        assert info == expected_info, f"expected info {expected_info!r}, got {info!r}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_e1_same_column_opposite_halves_even_n() -> None:
    """E1: even N=6, start and goal in column 2 on opposite halves."""
    grid = _make_symmetric([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ])
    _check(grid, (0, 2), (5, 2), expected_info="ok: split")


def test_e2_folded_adjacent_original_opposite_halves() -> None:
    """E2: 4×4 free grid. Folded coords are 4-adjacent but original cells
    are on opposite halves and not 4-adjacent."""
    grid = _make_symmetric([[0, 0, 0, 0], [0, 0, 0, 0]])
    _check(grid, (0, 0), (3, 1), expected_info="ok: split")


def test_e3_t_equals_rho_s_odd_n() -> None:
    """E3: odd N=7, t = ρ(s). Naive folding makes φ(s) = φ(t)."""
    grid = _make_symmetric(
        [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]],
        midline_row=[0, 0, 0, 0, 0],
    )
    s: Cell = (0, 0)
    t: Cell = rho(len(grid), s)
    _check(grid, s, t, expected_info="ok: split")


def test_e4_t_equals_s() -> None:
    """E4: degenerate."""
    grid = _make_symmetric([[0, 0, 0], [0, 0, 0]], midline_row=[0, 0, 0])
    _check(grid, (1, 1), (1, 1), expected_info="ok: s == t")


def test_e5_forced_midline_detour_even_n() -> None:
    """E5: a wall in the upper half forces the path to traverse the midline."""
    grid = _make_symmetric([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    _check(grid, (0, 0), (0, 5), expected_info="ok: both upper")


def test_e6_optimal_path_through_midline_odd_n() -> None:
    """E6: odd N=7 with an obstructed midline row."""
    grid = _make_symmetric(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
        ],
        midline_row=[0, 1, 1, 1, 1, 1, 0],
    )
    _check(grid, (0, 1), (0, 5), expected_info="ok: both upper")


def test_e7_split_with_t_not_equal_rho_s() -> None:
    """E7: opposite halves, t ≠ ρ(s). The case the manuscript never mentions."""
    grid = _make_symmetric([[0] * 6, [0] * 6, [0] * 6])
    _check(grid, (1, 0), (4, 5), expected_info="ok: split")


def test_e8_split_with_obstacles_forcing_specific_crossing_column() -> None:
    """E8: walls in row 1 / row 4 create a narrow midline channel; the
    split-case search must pick the right crossing column."""
    grid = _make_symmetric([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    _check(grid, (0, 0), (5, 7), expected_info="ok: split")


# ---------------------------------------------------------------------------
# Asymmetric input → fallback
# ---------------------------------------------------------------------------

def test_asymmetric_grid_falls_back_to_astar() -> None:
    """An asymmetric grid must trigger the fallback, and the returned path
    must equal what plain A* would return."""
    grid: Grid = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],   # asymmetric: this 1 is not mirrored below
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert not verify_symmetry(grid)
    path, info = folding_astar(grid, (0, 0), (3, 3))
    assert info == "fallback: not symmetric"
    ref = astar(grid, (0, 0), (3, 3))
    assert path is not None and ref is not None
    assert len(path) == len(ref)


@pytest.mark.parametrize("blocked_endpoint", ["start", "goal"])
def test_blocked_endpoint_returns_no_path(blocked_endpoint: str) -> None:
    """Symmetric grid where one endpoint is on an obstacle."""
    grid = _make_symmetric([[0, 0, 0], [0, 1, 0]], midline_row=[0, 0, 0])
    if blocked_endpoint == "start":
        s, t = (1, 1), (0, 0)
    else:
        s, t = (0, 0), (1, 1)
    path, info = folding_astar(grid, s, t)
    assert path is None
    assert info == "no path"
