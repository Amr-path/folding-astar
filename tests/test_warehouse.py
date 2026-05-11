"""Tests for the parametric warehouse generator."""

from __future__ import annotations

import pytest

from folding_astar import astar, folding_astar, verify_symmetry
from folding_astar.warehouse import (
    WarehouseSpec,
    build_warehouse,
    depot_cell,
    warehouse_endpoints,
)


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_aisles=0, rack_width=1, aisle_width=1, half_height=4, cross_aisle_period=2),
        dict(n_aisles=1, rack_width=0, aisle_width=1, half_height=4, cross_aisle_period=2),
        dict(n_aisles=1, rack_width=1, aisle_width=0, half_height=4, cross_aisle_period=2),
        dict(n_aisles=1, rack_width=1, aisle_width=1, half_height=0, cross_aisle_period=2),
        dict(n_aisles=1, rack_width=1, aisle_width=1, half_height=4, cross_aisle_period=-1),
        dict(n_aisles=1, rack_width=1, aisle_width=1, half_height=4, cross_aisle_period=2, border=-1),
        dict(n_aisles=1, rack_width=1, aisle_width=1, half_height=4, cross_aisle_period=2, parity="diagonal"),
    ],
)
def test_spec_rejects_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        WarehouseSpec(**kwargs)  # type: ignore[arg-type]


def test_spec_dimensions() -> None:
    spec = WarehouseSpec(
        n_aisles=3, rack_width=2, aisle_width=1, half_height=6, cross_aisle_period=3,
    )
    # interior cols: 4 racks * 2 + 3 aisles * 1 = 11; +2 borders = 13
    assert spec.total_cols == 13
    # even parity by default: total_rows = 2 * half_height = 12
    assert spec.total_rows == 12


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=5, cross_aisle_period=2),
        WarehouseSpec(n_aisles=3, rack_width=2, aisle_width=1, half_height=6, cross_aisle_period=3),
        WarehouseSpec(n_aisles=4, rack_width=2, aisle_width=2, half_height=8, cross_aisle_period=4),
        WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=5, cross_aisle_period=2, parity="odd"),
        WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=5, cross_aisle_period=0),
    ],
    ids=["small", "medium", "wide-aisle", "odd-parity", "no-cross-aisle"],
)
def test_build_warehouse_is_symmetric(spec: WarehouseSpec) -> None:
    grid = build_warehouse(spec)
    assert verify_symmetry(grid)
    assert len(grid) == spec.total_rows
    assert len(grid[0]) == spec.total_cols


def test_warehouse_has_obstacles_and_free_cells() -> None:
    """Sanity: a real warehouse is neither all-obstacles nor all-free."""
    spec = WarehouseSpec(n_aisles=3, rack_width=2, aisle_width=1, half_height=8, cross_aisle_period=3)
    grid = build_warehouse(spec)
    n_obs = sum(c for row in grid for c in row)
    n_free = sum(1 for row in grid for c in row if c == 0)
    assert n_obs > 0, "warehouse should contain rack obstacles"
    assert n_free > n_obs, "free cells should outnumber obstacles in this layout"


def test_aisles_are_actually_free() -> None:
    """Verify that the columns we declared as picking aisles are clear."""
    spec = WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=4, cross_aisle_period=0)
    grid = build_warehouse(spec)
    # Border is 1; so with rack_width=1, aisle_width=1 the columns from
    # j=1: rack, j=2: aisle, j=3: rack, j=4: aisle, j=5: rack
    aisle_cols = [2, 4]
    # Every row in the racked region (row 1 to row half_height-1) should have
    # the aisle columns free.
    for r in range(1, spec.half_height):
        for c in aisle_cols:
            assert grid[r][c] == 0, f"aisle column {c} not free at row {r}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_depot_is_free() -> None:
    spec = WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=5, cross_aisle_period=2)
    grid = build_warehouse(spec)
    d = depot_cell(spec)
    assert grid[d[0]][d[1]] == 0


def test_endpoints_are_free_and_distinct() -> None:
    spec = WarehouseSpec(n_aisles=3, rack_width=2, aisle_width=1, half_height=6, cross_aisle_period=3)
    grid = build_warehouse(spec)
    pairs = warehouse_endpoints(spec, n_pairs=50, seed=42)
    assert len(pairs) == 50
    for s, t in pairs:
        assert s != t
        assert grid[s[0]][s[1]] == 0
        assert grid[t[0]][t[1]] == 0


def test_endpoints_reproducible_seed() -> None:
    spec = WarehouseSpec(n_aisles=3, rack_width=2, aisle_width=1, half_height=6, cross_aisle_period=3)
    a = warehouse_endpoints(spec, n_pairs=20, seed=123)
    b = warehouse_endpoints(spec, n_pairs=20, seed=123)
    c = warehouse_endpoints(spec, n_pairs=20, seed=124)
    assert a == b
    assert a != c


def test_same_half_fraction() -> None:
    spec = WarehouseSpec(n_aisles=3, rack_width=2, aisle_width=1, half_height=8, cross_aisle_period=3)
    pairs = warehouse_endpoints(spec, n_pairs=200, seed=7, same_half_fraction=0.5)
    assert len(pairs) == 200
    # Roughly half should have both endpoints upper-half (rows < total/2).
    half = spec.total_rows // 2
    n_same = sum(1 for s, t in pairs if s[0] < half and t[0] < half)
    # Lenient bounds: 0.5 fraction means ~100 same-half pairs were *guaranteed*,
    # plus some incidental same-half pairs from the unconstrained half. So
    # n_same should be at least 100 and at most 200.
    assert 100 <= n_same <= 200


# ---------------------------------------------------------------------------
# End-to-end with the algorithm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=5, cross_aisle_period=2),
        WarehouseSpec(n_aisles=3, rack_width=2, aisle_width=1, half_height=6, cross_aisle_period=3),
        WarehouseSpec(n_aisles=2, rack_width=1, aisle_width=1, half_height=5, cross_aisle_period=2, parity="odd"),
    ],
    ids=["small", "medium", "odd-parity"],
)
def test_folding_astar_matches_astar_on_warehouse(spec: WarehouseSpec) -> None:
    """End-to-end: every (s, t) pair on a generated warehouse must give the
    same path length under A* and Folding A*."""
    grid = build_warehouse(spec)
    pairs = warehouse_endpoints(spec, n_pairs=30, seed=99)

    matched = 0
    skipped = 0
    for s, t in pairs:
        ref = astar(grid, s, t)
        path, info = folding_astar(grid, s, t)
        if ref is None and path is None:
            skipped += 1
            continue
        assert ref is not None and path is not None, (
            f"discrepancy at {s} -> {t}: A*={ref}, Folding A*={path} ({info})"
        )
        # Validate path
        for c in path:
            assert grid[c[0]][c[1]] == 0
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        # Optimality
        assert len(path) == len(ref), (
            f"length mismatch at {s} -> {t}: A*={len(ref)-1}, "
            f"Folding A*={len(path)-1} ({info})"
        )
        matched += 1
    assert matched > 0, "no reachable pairs in the generated warehouse"
