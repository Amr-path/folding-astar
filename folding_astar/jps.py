"""Jump Point Search (JPS) for 4-connected uniform-cost grids.

This is an independent implementation following Harabor and Grastien
(AAAI 2011), specialised to 4-connectivity. It is the comparison baseline
the manuscript needs to claim parity with the state of the art in
symmetry-aware grid pathfinding.

The algorithm:

    Plain A* expansion order, but successors of a node `x` reached from
    parent `p` in direction `d` are filtered to remove path-symmetric
    detours. From `x`, instead of expanding the four cardinal neighbours,
    we *jump* in each candidate direction until we either hit the goal,
    hit an obstacle, or find a node with a "forced neighbour" — a
    successor that cannot be reached more cheaply by going around `x`.

For 4-connectivity, the natural successor of `x` reached via direction
`d` is `x + d`. A perpendicular neighbour `x + e` (where `e ⊥ d`) is a
*forced* successor of `x` only if going from `p` through `e` is blocked
by an obstacle, so the only way to reach `x + e` is via `x` itself.

Public API:
    jps(grid, start, goal) -> path | None

Returns the same kind of result as `astar(...)`: a list of cells from
start to goal inclusive, or `None` if unreachable.
"""

from __future__ import annotations

import heapq

from folding_astar.astar import is_free, manhattan
from folding_astar.types import Cell, Grid


__all__ = ["jps"]


# Cardinal directions: (dr, dc).
_CARDINAL_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _has_forced_neighbour(grid: Grid, x: Cell, d: tuple[int, int]) -> bool:
    """Does `x` (reached moving in direction `d`) have a forced neighbour?

    For 4-connected JPS the rule, restated from Harabor 2011 §3 in the
    notation of this paper: when arriving at `x` from direction `d`, the
    perpendicular neighbour `x + e` is forced iff the cell behind it
    (`x - d + e`) is an obstacle. That is, the diagonal-corner cell that
    would otherwise be reachable from the parent without going through
    `x` is blocked, so `x + e` becomes a unique successor of `x`.
    """
    r, c = x
    dr, dc = d
    if dr == 0:
        # Moving horizontally; perpendicular axis is vertical.
        for er in (-1, 1):
            blocker = (r + er, c - dc)
            target = (r + er, c)
            if not is_free(grid, blocker) and is_free(grid, target):
                return True
    else:
        # Moving vertically; perpendicular axis is horizontal.
        for ec in (-1, 1):
            blocker = (r - dr, c + ec)
            target = (r, c + ec)
            if not is_free(grid, blocker) and is_free(grid, target):
                return True
    return False


def _jump(grid: Grid, x: Cell, d: tuple[int, int], goal: Cell) -> Cell | None:
    """Iterative jump from `x` in direction `d`. Returns the next jump
    point on this ray, or `None` if the very first step from `x` is
    blocked or out of bounds.

    A cell `nxt` becomes a jump point when any of these holds:

      1. `nxt == goal` — the goal lies on this ray.
      2. `nxt` has a forced neighbour — an obstacle adjacent to the
         path forces a perpendicular successor that would otherwise
         not be a successor of `nxt` under JPS pruning rules.
      3. The ray crosses the goal's row (for a vertical scan) or
         column (for a horizontal scan), and `nxt` is the cell at the
         crossing. This is the "turn-point" rule: from a cell in the
         goal's row, a perpendicular jump can reach the goal directly,
         so the crossing cell must enter the open list as a candidate
         expansion. Without this rule, 4-connected JPS on open grids
         fails to find paths whose start and goal are not collinear,
         because no jump ever lands inside the grid (every farthest-
         reachable cell is on a grid edge).
      4. The ray hits an obstacle or the grid boundary — `nxt` is then
         taken to be the previous cell ("farthest reachable").

    Rules 1–3 produce interior jump points; rule 4 is the fallback for
    open rays. The combination is what makes 4-connected JPS sound on
    arbitrary symmetric and asymmetric grids."""
    g_r, g_c = goal
    cur = x
    last_valid: Cell | None = None
    while True:
        nxt = (cur[0] + d[0], cur[1] + d[1])
        if not is_free(grid, nxt):
            return last_valid
        # Rule 1: goal on the ray.
        if nxt == goal:
            return nxt
        # Rule 3: ray crosses the goal's row or column. We check this
        # *before* the forced-neighbour rule because the turn-point is
        # the more specific successor on open grids.
        if d[0] != 0 and nxt[0] == g_r:
            return nxt
        if d[1] != 0 and nxt[1] == g_c:
            return nxt
        # Rule 2: forced neighbour from obstacle. We check both `d` and
        # `-d` because the canonical Harabor 2011 forced-neighbour rule
        # is direction-asymmetric: `nxt` is a jump point only if a
        # perpendicular successor is forced *given* the arrival
        # direction `d`. On 4-connected grids without diagonals this
        # asymmetry hides useful turn-corners that the scan passes
        # through but does not emit. Checking both arrival directions
        # costs one extra symmetric test per cell and preserves
        # optimality without losing JPS's corridor-compression benefit
        # on uncluttered rays.
        if _has_forced_neighbour(grid, nxt, d) or \
           _has_forced_neighbour(grid, nxt, (-d[0], -d[1])):
            return nxt
        last_valid = nxt
        cur = nxt


def _successors(grid: Grid, x: Cell, parent_dir: tuple[int, int] | None,
                goal: Cell) -> list[tuple[Cell, int]]:
    r"""Compute successor jump points of `x`.

    On 4-connected grids the canonical JPS pruning rules (Harabor 2011
    Sec. 3) are formulated for 8-connectivity and rely on diagonal
    moves to "turn"; on a pure 4-connected grid those diagonals do not
    exist, and the standard rules can stall on open grids where no
    forced neighbour or in-line goal is encountered. We therefore use
    the pragmatic "Cardinal JPS" variant (sometimes called CJPS): from
    every node we scan in all four cardinal directions, returning the
    farthest reachable cell on each ray as a jump point along with any
    earlier cell that has a forced neighbour, matches the goal's
    row/column, or is the goal itself. This preserves JPS's "skip
    empty corridors" property — the speedup over A* on grids with
    long open aisles — while remaining sound on open grids.

    Returns list of (jump_point, distance_from_x) pairs. The
    `parent_dir` argument is unused in this variant and kept only for
    future variants that re-enable Harabor 2011 pruning."""
    del parent_dir  # unused by Cardinal JPS
    out: list[tuple[Cell, int]] = []
    for d in _CARDINAL_DIRS:
        jp = _jump(grid, x, d, goal)
        if jp is None:
            continue
        # Distance is L1 because we only move along one axis between
        # consecutive jump points on a single ray.
        dist = abs(jp[0] - x[0]) + abs(jp[1] - x[1])
        out.append((jp, dist))
    return out


def jps(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Jump Point Search on a 4-connected grid. Returns the optimal path
    (inclusive of start and goal) or None if no path exists. Treats
    blocked endpoints as 'no path'."""
    if not is_free(grid, start) or not is_free(grid, goal):
        return None
    if start == goal:
        return [start]

    open_q: list[tuple[int, int, Cell, tuple[int, int] | None]] = []
    # Each entry: (f, tiebreaker, cell, parent_direction).
    # parent_direction is None for the start node.
    heapq.heappush(open_q, (manhattan(start, goal), 0, start, None))
    g: dict[Cell, int] = {start: 0}
    parent: dict[Cell, Cell | None] = {start: None}
    closed: set[Cell] = set()
    counter = 0

    while open_q:
        _, _, x, pd = heapq.heappop(open_q)
        if x in closed:
            continue
        closed.add(x)
        if x == goal:
            return _reconstruct(parent, x, grid)

        for jp, step_dist in _successors(grid, x, pd, goal):
            ng = g[x] + step_dist
            if jp not in g or ng < g[jp]:
                g[jp] = ng
                parent[jp] = x
                # Direction we arrived from
                direction = (
                    (1 if jp[0] > x[0] else -1 if jp[0] < x[0] else 0),
                    (1 if jp[1] > x[1] else -1 if jp[1] < x[1] else 0),
                )
                counter += 1
                heapq.heappush(
                    open_q, (ng + manhattan(jp, goal), counter, jp, direction)
                )
    return None


def _reconstruct(parent: dict[Cell, Cell | None], end: Cell, grid: Grid) -> list[Cell]:
    """Reconstruct the path from start to `end`. Between two consecutive
    jump points the path is a straight cardinal segment, so we emit every
    intermediate cell rather than just the jump points (otherwise the
    returned 'path' would not be 4-connected)."""
    jps_chain: list[Cell] = []
    cur: Cell | None = end
    while cur is not None:
        jps_chain.append(cur)
        cur = parent[cur]
    jps_chain.reverse()

    # Expand each (a, b) jump-point segment into its full cardinal walk.
    if not jps_chain:
        return []
    path: list[Cell] = [jps_chain[0]]
    for a, b in zip(jps_chain, jps_chain[1:]):
        dr = (1 if b[0] > a[0] else -1 if b[0] < a[0] else 0)
        dc = (1 if b[1] > a[1] else -1 if b[1] < a[1] else 0)
        cur_cell = a
        while cur_cell != b:
            cur_cell = (cur_cell[0] + dr, cur_cell[1] + dc)
            path.append(cur_cell)
    return path
