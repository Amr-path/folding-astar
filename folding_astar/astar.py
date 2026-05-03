"""Standard A* on a 4-connected grid. Used as ground truth for tests and as
the inner search routine inside Folding A*."""

from __future__ import annotations

import heapq

from folding_astar.types import Cell, Grid


__all__ = ["astar", "manhattan", "neighbors4", "is_free", "in_bounds"]


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(grid: Grid, cell: Cell) -> bool:
    r, c = cell
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def is_free(grid: Grid, cell: Cell) -> bool:
    return in_bounds(grid, cell) and grid[cell[0]][cell[1]] == 0


def neighbors4(grid: Grid, cell: Cell) -> list[Cell]:
    """4-connected free neighbours of `cell`."""
    r, c = cell
    out: list[Cell] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n = (r + dr, c + dc)
        if is_free(grid, n):
            out.append(n)
    return out


def astar(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Standard A* with Manhattan heuristic. Returns the optimal path
    (inclusive of start and goal) or None if no path exists. Treats blocked
    endpoints as 'no path'."""
    if not is_free(grid, start) or not is_free(grid, goal):
        return None
    if start == goal:
        return [start]

    open_q: list[tuple[int, int, Cell]] = []
    heapq.heappush(open_q, (manhattan(start, goal), 0, start))
    g: dict[Cell, int] = {start: 0}
    parent: dict[Cell, Cell | None] = {start: None}
    counter = 0

    while open_q:
        _, _, u = heapq.heappop(open_q)
        if u == goal:
            return _reconstruct(parent, u)
        for v in neighbors4(grid, u):
            ng = g[u] + 1
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                counter += 1
                heapq.heappush(open_q, (ng + manhattan(v, goal), counter, v))
    return None


def _reconstruct(parent: dict[Cell, Cell | None], end: Cell) -> list[Cell]:
    path: list[Cell] = []
    cur: Cell | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))
