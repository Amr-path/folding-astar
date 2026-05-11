"""Folding A*: optimal grid pathfinding via horizontal-reflection symmetry.

This is the *corrected* version of the algorithm relative to the manuscript's
first submission. The fix is described in detail in docs/correctness.md.
The key idea: when start and goal are on opposite halves of the grid, the
naive 'fold-search-unfold' procedure cannot reconstruct the optimal path. We
canonicalise both endpoints into the upper half and find the optimal
midline-crossing column with one extra search.

Public API:
    folding_astar(grid, s, t)        -> (path | None, info_str)
    folding_bfs(grid, s, t)          -> (path | None, info_str)
    folding_dijkstra(grid, s, t)     -> (path | None, info_str)
    folding_search(grid, s, t, inner_search) -> (path | None, info_str)
        Generic version. The inner_search is the algorithm used to solve
        the constituent shortest-path subproblems on the folded grid.
    verify_symmetry(grid)
    fold_grid(grid)
    fold_vertex(N, cell)
    midline(N), rho(N, cell), in_upper_half(N, cell)
"""

from __future__ import annotations

from collections import deque
from typing import Callable

from folding_astar.astar import astar, is_free, neighbors4
from folding_astar.types import Cell, Grid


__all__ = [
    "folding_astar",
    "folding_bfs",
    "folding_dijkstra",
    "folding_search",
    "verify_symmetry",
    "fold_grid",
    "fold_vertex",
    "midline",
    "rho",
    "in_upper_half",
]


# Type of an inner search function: (grid, start, goal) -> path | None.
InnerSearch = Callable[[Grid, Cell, Cell], list[Cell] | None]


# ---------------------------------------------------------------------------
# Symmetry primitives
# ---------------------------------------------------------------------------

def midline(N: int) -> int:
    """Floor(N/2). For even N the fold occurs *between* rows m-1 and m.
    For odd N row m is the fixed axis (a fixed point of the reflection)."""
    return N // 2


def rho(N: int, cell: Cell) -> Cell:
    """Horizontal reflection: rho(i, j) = (N-1-i, j)."""
    return (N - 1 - cell[0], cell[1])


def in_upper_half(N: int, cell: Cell) -> bool:
    """A cell is upper-half if it lies in row < m (even N) or row <= m (odd N)."""
    m = midline(N)
    if N % 2 == 1:
        return cell[0] <= m
    return cell[0] < m


def fold_vertex(N: int, cell: Cell) -> Cell:
    """Quotient map Phi: pick the upper-half representative of cell's
    equivalence class under the reflection."""
    return cell if in_upper_half(N, cell) else rho(N, cell)


def verify_symmetry(grid: Grid) -> bool:
    """True iff the obstacle set is invariant under horizontal reflection."""
    N = len(grid)
    cols = len(grid[0])
    m = midline(N)
    for i in range(m):
        for j in range(cols):
            if grid[i][j] != grid[N - 1 - i][j]:
                return False
    return True


def fold_grid(grid: Grid) -> Grid:
    """Build the folded grid G_f.

    Vertices on the midline (odd N) inherit obstacle status from the original.
    Off-midline vertices: the folded cell (i, j) is free iff (i, j) is free in
    the original — which under symmetry is equivalent to (N-1-i, j) being
    free as well.
    """
    N = len(grid)
    m = midline(N)
    rows_f = m + 1 if N % 2 == 1 else m
    return [list(grid[i]) for i in range(rows_f)]


# ---------------------------------------------------------------------------
# The corrected end-to-end algorithm
# ---------------------------------------------------------------------------

def folding_search(
    grid: Grid, start: Cell, goal: Cell, inner_search: InnerSearch,
) -> tuple[list[Cell] | None, str]:
    """Generic Folding-Search dispatch. Same three-case logic as Folding A*,
    but the inner shortest-path subproblems on the folded grid are solved
    using `inner_search` (typically `astar`, `bfs`, or `dijkstra`).

    Returns (path, info) where path is a list of (row, col) cells in the
    *original* grid, or None if no path exists. info is a short string
    describing which case the algorithm took:
        'ok: s == t'        — degenerate (path is [s])
        'ok: both upper'    — both endpoints upper-half, naive fold suffices
        'ok: both lower'    — both endpoints lower-half, reflected variant
        'ok: split'         — endpoints on opposite halves, midline search
        'fallback: not symmetric' — input not horizontally symmetric;
                                     we ran inner_search on the original grid
        'no path'           — no path exists.
    """
    if not verify_symmetry(grid):
        return inner_search(grid, start, goal), "fallback: not symmetric"
    if not is_free(grid, start) or not is_free(grid, goal):
        return None, "no path"
    if start == goal:
        return [start], "ok: s == t"

    N = len(grid)
    s_up = in_upper_half(N, start)
    t_up = in_upper_half(N, goal)

    if s_up and t_up:
        return _solve_upper(grid, start, goal, inner_search), "ok: both upper"

    if not s_up and not t_up:
        s_im = rho(N, start)
        t_im = rho(N, goal)
        path_up = _solve_upper(grid, s_im, t_im, inner_search)
        if path_up is None:
            return None, "no path"
        return [rho(N, v) for v in path_up], "ok: both lower"

    return _solve_split(grid, start, goal, inner_search)


def folding_astar(grid: Grid, start: Cell, goal: Cell) -> tuple[list[Cell] | None, str]:
    """Folding A*: A* run on the quotient graph induced by horizontal-reflection
    symmetry. See `folding_search` for full documentation."""
    return folding_search(grid, start, goal, astar)


def folding_bfs(grid: Grid, start: Cell, goal: Cell) -> tuple[list[Cell] | None, str]:
    """Folding BFS: uninformed breadth-first search run via the same dispatch.
    Useful as a comparison baseline against standard BFS — this is the case
    in which the symmetry reduction translates most cleanly into wall-clock
    speedup, because BFS lacks the heuristic that lets A* implicitly avoid
    the symmetric half."""
    from folding_astar.search import bfs   # local to avoid circular import
    return folding_search(grid, start, goal, bfs)


def folding_dijkstra(grid: Grid, start: Cell, goal: Cell) -> tuple[list[Cell] | None, str]:
    """Folding Dijkstra. On unweighted 4-connected grids this returns the
    same shortest paths as Folding BFS but uses a binary-heap priority queue;
    kept as a separate baseline because the manuscript benchmarks against
    Dijkstra explicitly."""
    from folding_astar.search import dijkstra   # local to avoid circular import
    return folding_search(grid, start, goal, dijkstra)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solve_upper(
    grid: Grid, s: Cell, t: Cell, inner_search: InnerSearch,
) -> list[Cell] | None:
    """Both endpoints upper-half. The folded path on G_f is already valid
    in G because every cell visited is in the upper half."""
    folded = fold_grid(grid)
    return inner_search(folded, s, t)


def _solve_split(
    grid: Grid, s: Cell, t: Cell, inner_search: InnerSearch,
) -> tuple[list[Cell] | None, str]:
    """Start and goal on opposite halves. Strategy: reflect the lower endpoint
    into the upper half (call it t_image). The optimal path s -> t in G has
    the form (upper segment) -> (one fold-crossing edge) -> (mirror of upper
    segment from t_image to crossing point), and its total length is

        d_G_f(s, b) + cross_cost + d_G_f(b, t_image)

    where b ranges over the fold boundary (row m-1 in even N, or row m in
    odd N) and cross_cost is 1 for even N (the extra vertical edge) or 0
    for odd N (b = rho(b)). We minimise over b."""
    N = len(grid)
    folded = fold_grid(grid)
    cols = len(grid[0])

    # Make sure s is the upper one. Swap if not, then reverse at the end.
    swapped = False
    if not in_upper_half(N, s):
        s, t = t, s
        swapped = True

    t_image = rho(N, t)

    if N % 2 == 1:
        boundary_row = midline(N)
        cross_cost = 0
    else:
        boundary_row = midline(N) - 1
        cross_cost = 1

    boundary_cells = [(boundary_row, c) for c in range(cols)
                      if is_free(folded, (boundary_row, c))]
    if not boundary_cells:
        return None, "no path"

    d_s = _bfs_lengths(folded, s)
    d_t = _bfs_lengths(folded, t_image)

    best: tuple[int, Cell] | None = None
    for b in boundary_cells:
        if b not in d_s or b not in d_t:
            continue
        total = d_s[b] + cross_cost + d_t[b]
        if best is None or total < best[0]:
            best = (total, b)

    if best is None:
        return None, "no path"

    _, b = best

    upper_segment = inner_search(folded, s, b)
    if upper_segment is None:
        return None, "no path"
    folded_lower = inner_search(folded, b, t_image)
    if folded_lower is None:
        return None, "no path"

    if N % 2 == 1:
        # b is its own reflection. Drop duplicate b at the seam.
        lower_in_orig = [rho(N, v) for v in folded_lower[1:]]
    else:
        # Even N: the fold-crossing edge from b to rho(b) is added implicitly
        # because rho(folded_lower[0]) = rho(b) is the cell directly below b.
        lower_in_orig = [rho(N, v) for v in folded_lower]

    full = upper_segment + lower_in_orig
    if swapped:
        full = list(reversed(full))
    return full, "ok: split"


def _bfs_lengths(grid: Grid, source: Cell) -> dict[Cell, int]:
    """Single-source BFS distances on an unweighted grid. Used inside the
    split-case search to score candidate boundary cells without re-running
    A* once per cell."""
    if not is_free(grid, source):
        return {}
    dist: dict[Cell, int] = {source: 0}
    q: deque[Cell] = deque([source])
    while q:
        u = q.popleft()
        for v in neighbors4(grid, u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
