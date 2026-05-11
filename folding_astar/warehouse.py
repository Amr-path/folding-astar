"""Parametric warehouse layout generator.

Modern automated fulfilment-centre layouts are constructed from a small set
of repeating modules — parallel storage racks separated by picking aisles,
intersected at regular intervals by perpendicular cross-aisles — and the
cross-aisle pattern is routinely chosen to be symmetric across the
facility's main axis to balance traversal time. See Roodbergen and Vis
(EJOR, 2009) for a survey of the standard layout family. This module
generates instances of that family that are horizontally symmetric by
construction, suitable as Folding A* benchmarks.

Public API
----------
WarehouseSpec(...)
    Frozen dataclass describing a warehouse layout. Validates parameters
    on construction.

build_warehouse(spec) -> Grid
    Return the layout as a 2-D list of ints (0 = free, 1 = obstacle).
    The result is verified-symmetric: ``verify_symmetry(grid)`` is True.

warehouse_endpoints(spec, n_pairs, seed=...) -> list[(start, goal)]
    Return ``n_pairs`` random (start, goal) pairs drawn from the free
    cells. Reproducible given a seed. Excludes pairs where start == goal.

depot_cell(spec) -> Cell
    The conventional 'depot' / pickup-dropoff cell at the front-left
    corner of the warehouse (just outside the first picking aisle).

Layout family
-------------
The generator produces a single-block warehouse with optional cross-aisles:

  * ``border`` cells of free space form a perimeter.
  * Inside the perimeter, alternating *racks* (obstacles) and *aisles*
    (free cells) run vertically, with ``rack_width`` and ``aisle_width``
    controlling each. The total interior width must accommodate
    exactly ``n_aisles`` picking aisles separated by ``n_aisles + 1``
    racks.
  * Cross-aisles cut horizontally through the racks every
    ``cross_aisle_period`` rows. The first and last rows inside the
    perimeter are always cross-aisles (front and back of the
    warehouse).
  * Symmetry is enforced by mirroring the upper half across the
    horizontal midline. With even total height the fold sits between
    rows m-1 and m; with odd total height row m is fixed and we
    construct that row symmetrically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from folding_astar.folding import verify_symmetry
from folding_astar.types import Cell, Grid


__all__ = [
    "WarehouseSpec",
    "build_warehouse",
    "warehouse_endpoints",
    "depot_cell",
]


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WarehouseSpec:
    """Parameters of a warehouse layout. All values are cell counts."""

    n_aisles: int             # number of picking aisles (must be >= 1)
    rack_width: int           # cells across each rack (>= 1)
    aisle_width: int          # cells across each picking aisle (>= 1)
    half_height: int          # rows in the upper half (excluding midline if odd)
    cross_aisle_period: int   # cross-aisle every K rows; 0 for none beyond front/back
    border: int = 1           # free-cell border around the warehouse
    parity: str = "even"      # 'even' or 'odd' total height

    def __post_init__(self) -> None:
        if self.n_aisles < 1:
            raise ValueError("n_aisles must be >= 1")
        if self.rack_width < 1:
            raise ValueError("rack_width must be >= 1")
        if self.aisle_width < 1:
            raise ValueError("aisle_width must be >= 1")
        if self.half_height < 1:
            raise ValueError("half_height must be >= 1")
        if self.cross_aisle_period < 0:
            raise ValueError("cross_aisle_period must be >= 0")
        if self.border < 0:
            raise ValueError("border must be >= 0")
        if self.parity not in {"even", "odd"}:
            raise ValueError("parity must be 'even' or 'odd'")

    @property
    def total_cols(self) -> int:
        # Perimeter + (n_aisles + 1) racks + n_aisles aisles + perimeter.
        interior = (self.n_aisles + 1) * self.rack_width + self.n_aisles * self.aisle_width
        return interior + 2 * self.border

    @property
    def total_rows(self) -> int:
        # 2 * half_height + (1 if odd, 0 if even) + 2 * border on top/bottom.
        # The ``border`` is included in half_height by convention so that
        # the cross-aisle pattern is contiguous through the border.
        if self.parity == "odd":
            return 2 * self.half_height + 1
        return 2 * self.half_height


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build_upper_half(spec: WarehouseSpec) -> list[list[int]]:
    """Construct the upper half of the warehouse as a list of rows of length
    ``spec.total_cols``. The midline row (for odd parity) is built by the
    caller and concatenated separately."""
    cols = spec.total_cols
    rows = spec.half_height
    upper: list[list[int]] = [[0] * cols for _ in range(rows)]

    # Determine which interior columns belong to racks vs aisles.
    rack_cols: list[int] = []
    j = spec.border
    for i in range(spec.n_aisles + 1):
        for _ in range(spec.rack_width):
            rack_cols.append(j)
            j += 1
        # Skip the picking aisle (free cells)
        j += spec.aisle_width
    # All other interior cols are aisle cells.

    # Determine which rows are cross-aisles. Front of warehouse is row 0
    # (top) — entirely free. If cross_aisle_period > 0, every K rows after
    # row 0 is a cross-aisle, where K = cross_aisle_period.
    def is_cross_aisle(r: int) -> bool:
        if r < spec.border:
            return True   # the top border is free
        # Map r to a position 'within the racked region' so that period
        # measures distance between cross-aisles.
        if spec.cross_aisle_period == 0:
            return False
        r_in = r - spec.border
        return r_in % spec.cross_aisle_period == 0

    for r in range(rows):
        if is_cross_aisle(r):
            continue
        for c in rack_cols:
            upper[r][c] = 1

    return upper


def _midline_row(spec: WarehouseSpec) -> list[int]:
    """For odd-parity layouts the midline row is the reflection axis. We
    keep it as an additional cross-aisle (entirely free) so traversal can
    use it as a corridor."""
    return [0] * spec.total_cols


def build_warehouse(spec: WarehouseSpec) -> Grid:
    """Build the warehouse grid. Result is verified-symmetric."""
    upper = _build_upper_half(spec)
    if spec.parity == "odd":
        rows = upper + [_midline_row(spec)] + [list(r) for r in reversed(upper)]
    else:
        rows = upper + [list(r) for r in reversed(upper)]
    grid: Grid = rows
    if not verify_symmetry(grid):
        raise AssertionError(
            "warehouse generator produced an asymmetric grid — "
            "this is a bug, please report it"
        )
    return grid


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def depot_cell(spec: WarehouseSpec) -> Cell:
    """Conventional depot / pickup-dropoff: top-left free cell, just inside
    the perimeter border."""
    return (spec.border, spec.border)


def warehouse_endpoints(
    spec: WarehouseSpec,
    n_pairs: int,
    *,
    seed: int = 0,
    same_half_fraction: float | None = None,
) -> list[tuple[Cell, Cell]]:
    """Return ``n_pairs`` (start, goal) pairs drawn from the warehouse's
    free cells.

    If ``same_half_fraction`` is given (between 0 and 1), that fraction of
    pairs is constrained to have both endpoints in the upper half of the
    grid (Folding A*'s easy case); the remainder are unconstrained.
    Otherwise pairs are sampled uniformly from all free-cell pairs.
    """
    if n_pairs <= 0:
        return []
    grid = build_warehouse(spec)
    rows = len(grid)
    cols = len(grid[0])
    free_all = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
    if len(free_all) < 2:
        raise ValueError("warehouse has fewer than 2 free cells")
    half = rows // 2
    free_upper = [c for c in free_all if c[0] < half]

    rng = random.Random(seed)
    pairs: list[tuple[Cell, Cell]] = []
    if same_half_fraction is None:
        for _ in range(n_pairs):
            s = rng.choice(free_all)
            t = rng.choice(free_all)
            while t == s:
                t = rng.choice(free_all)
            pairs.append((s, t))
        return pairs

    n_same = int(round(n_pairs * same_half_fraction))
    n_split = n_pairs - n_same
    if n_same and len(free_upper) < 2:
        raise ValueError("not enough upper-half free cells for same-half pairs")
    for _ in range(n_same):
        s = rng.choice(free_upper)
        t = rng.choice(free_upper)
        while t == s:
            t = rng.choice(free_upper)
        pairs.append((s, t))
    for _ in range(n_split):
        s = rng.choice(free_all)
        t = rng.choice(free_all)
        while t == s:
            t = rng.choice(free_all)
        pairs.append((s, t))
    rng.shuffle(pairs)
    return pairs
