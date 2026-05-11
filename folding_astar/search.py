"""Uninformed search algorithms: BFS and Dijkstra.

These are the comparison baselines against which Folding A* shows its
clearest wall-clock advantage. On a 4-connected unweighted grid, BFS and
Dijkstra return the same shortest path; the only difference is implementation
overhead (FIFO queue vs binary heap). Both are kept as separate functions so
the manuscript's per-baseline comparisons can be reproduced exactly.

Each function has the same signature as `astar(grid, start, goal)`:
    Grid, Cell, Cell -> list[Cell] | None
"""

from __future__ import annotations

import heapq
from collections import deque

from folding_astar.astar import is_free, neighbors4
from folding_astar.types import Cell, Grid


__all__ = ["bfs", "dijkstra", "InnerSearch"]


# A "search" function is anything with the same signature as astar / bfs / dijkstra.
# We use a Protocol-style alias rather than a typing.Protocol to keep the
# function registry simple.
from typing import Callable
InnerSearch = Callable[[Grid, Cell, Cell], list[Cell] | None]


def bfs(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Breadth-first search. Returns the shortest path or None.
    Treats blocked endpoints as 'no path'."""
    if not is_free(grid, start) or not is_free(grid, goal):
        return None
    if start == goal:
        return [start]
    parent: dict[Cell, Cell | None] = {start: None}
    q: deque[Cell] = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            return _reconstruct(parent, u)
        for v in neighbors4(grid, u):
            if v not in parent:
                parent[v] = u
                if v == goal:
                    return _reconstruct(parent, v)
                q.append(v)
    return None


def dijkstra(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Dijkstra's algorithm with unit edge weights. On 4-connected unweighted
    grids this returns the same shortest path as BFS but uses a binary heap
    instead of a FIFO queue (so it has more per-operation overhead). It is
    included separately because the manuscript benchmarks against Dijkstra as
    a distinct baseline."""
    if not is_free(grid, start) or not is_free(grid, goal):
        return None
    if start == goal:
        return [start]
    open_q: list[tuple[int, int, Cell]] = []
    heapq.heappush(open_q, (0, 0, start))
    g: dict[Cell, int] = {start: 0}
    parent: dict[Cell, Cell | None] = {start: None}
    counter = 0
    while open_q:
        d_u, _, u = heapq.heappop(open_q)
        if d_u > g[u]:
            continue   # stale entry
        if u == goal:
            return _reconstruct(parent, u)
        for v in neighbors4(grid, u):
            ng = g[u] + 1
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                counter += 1
                heapq.heappush(open_q, (ng, counter, v))
    return None


def _reconstruct(parent: dict[Cell, Cell | None], end: Cell) -> list[Cell]:
    path: list[Cell] = []
    cur: Cell | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))
