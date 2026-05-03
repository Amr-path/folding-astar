"""Canonical worked examples for the figures in the rewritten manuscript.

The original manuscript's Figure 1 was flagged by Reviewer 1 ("difficult to
follow") and Reviewer 2 ("Figure 1b is incorrect"). Figure 3 was flagged by
Reviewer 2 as well ("step 4 ... it is not clear how this path is to be
transformed into a path in the original graph. For example, it will need to
cross the midline somewhere"). These examples are constructed to make the
correct algorithm legible:

  EX_FIGURE_1 — both endpoints in upper half, optimal path stays in upper
                half. Easiest case to depict; shows the fold + unfold without
                the boundary-crossing wrinkle.
  EX_FIGURE_3 — start in upper, goal in lower. Optimal path crosses the
                midline. Shows the split-case search picking the optimal
                crossing column.

Each example exposes:
  - .grid                — the original symmetric grid
  - .start, .goal        — endpoints
  - .expected_length     — verified shortest-path length in the original grid
  - .info                — which case Folding A* reports
  - .render(path=None)   — ASCII rendering with optional path overlay

The expected lengths and infos are checked at import time against the
algorithm and standard A*. If anything ever drifts, import will fail
loudly rather than silently shipping a wrong example.
"""

from __future__ import annotations

from dataclasses import dataclass

from folding_astar.astar import astar
from folding_astar.folding import folding_astar, midline, verify_symmetry
from folding_astar.types import Cell, Grid


__all__ = ["WorkedExample", "EX_FIGURE_1", "EX_FIGURE_3", "ALL_EXAMPLES"]


@dataclass(frozen=True)
class WorkedExample:
    name: str
    description: str
    grid: Grid
    start: Cell
    goal: Cell
    expected_length: int
    expected_info: str

    def render(self, path: list[Cell] | None = None) -> str:
        """ASCII art with optional path overlay.
            .  free
            #  obstacle
            S  start
            G  goal
            -  midline (only drawn for odd N, on the fixed-point row)
            *  path cell (other than S, G)
        """
        N = len(self.grid)
        m = midline(N)
        cols = len(self.grid[0])
        path_set = {c for c in (path or []) if c not in (self.start, self.goal)}

        lines: list[str] = []
        # Header showing column indices for grids up to 99 columns.
        header_top = "    " + "".join(f"{j // 10 if j >= 10 else ' '}" for j in range(cols))
        header_bot = "    " + "".join(f"{j % 10}" for j in range(cols))
        lines.append(header_top)
        lines.append(header_bot)

        for i, row in enumerate(self.grid):
            chars: list[str] = []
            for j, cell in enumerate(row):
                if (i, j) == self.start:
                    chars.append("S")
                elif (i, j) == self.goal:
                    chars.append("G")
                elif (i, j) in path_set:
                    chars.append("*")
                elif cell == 1:
                    chars.append("#")
                else:
                    chars.append(".")
            lines.append(f" {i:2d} " + "".join(chars))
            # Draw fold marker between rows m-1 and m for even N, on row m for odd N.
            if N % 2 == 0 and i == m - 1:
                lines.append("    " + "-" * cols + "  <-- fold")
        if N % 2 == 1:
            # Replace the line for row m with one that flags it as the axis.
            axis_idx = 2 + m  # +2 for the two header rows
            lines[axis_idx] = lines[axis_idx] + "  <-- midline (axis)"

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EX_FIGURE_1: easy case, both endpoints upper-half.
# ---------------------------------------------------------------------------
# 6x6 grid. Obstacles form symmetric "speed bumps" in rows 1 and 4. Start
# and goal are both in the upper half; the optimal path is a clean
# left-to-right walk in row 0.
_FIG1_TOP = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
]
_FIG1_GRID: Grid = (
    [list(r) for r in _FIG1_TOP]
    + [list(r) for r in reversed(_FIG1_TOP)]
)
EX_FIGURE_1 = WorkedExample(
    name="figure_1",
    description=(
        "Both endpoints in the upper half. Optimal path stays in the upper "
        "half. Demonstrates the fold + unfold without midline crossing — the "
        "easiest case and the one we use to introduce the algorithm."
    ),
    grid=_FIG1_GRID,
    start=(0, 0),
    goal=(2, 5),
    expected_length=7,
    expected_info="ok: both upper",
)


# ---------------------------------------------------------------------------
# EX_FIGURE_3: split case, optimal path crosses the midline.
# ---------------------------------------------------------------------------
# 6x6 grid. Walls in rows 1 and 4 force the path to enter the midline gap.
# Start is in upper half, goal is in lower half — the *exact* case
# Reviewer 2 said Figure 3 step 4 had to handle and didn't.
_FIG3_TOP = [
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0],
]
_FIG3_GRID: Grid = (
    [list(r) for r in _FIG3_TOP]
    + [list(r) for r in reversed(_FIG3_TOP)]
)
EX_FIGURE_3 = WorkedExample(
    name="figure_3",
    description=(
        "Start in upper half, goal in lower half. Walls in rows 1 and 4 "
        "force the optimal path to traverse the midline. Demonstrates the "
        "corrected split-case search: Folding A* picks the optimal "
        "midline-crossing column and reflects the lower segment."
    ),
    grid=_FIG3_GRID,
    start=(0, 0),
    goal=(5, 5),
    expected_length=10,
    expected_info="ok: split",
)


ALL_EXAMPLES: tuple[WorkedExample, ...] = (EX_FIGURE_1, EX_FIGURE_3)


# ---------------------------------------------------------------------------
# Self-check at import time. If anything drifts (algorithm changes, expected
# values stale), import fails noisily.
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    for ex in ALL_EXAMPLES:
        assert verify_symmetry(ex.grid), f"{ex.name}: grid is not symmetric"
        ref = astar(ex.grid, ex.start, ex.goal)
        assert ref is not None, f"{ex.name}: A* found no path"
        assert len(ref) - 1 == ex.expected_length, (
            f"{ex.name}: expected length {ex.expected_length}, A* found {len(ref) - 1}"
        )
        path, info = folding_astar(ex.grid, ex.start, ex.goal)
        assert path is not None, f"{ex.name}: Folding A* returned no path"
        assert info == ex.expected_info, (
            f"{ex.name}: expected info {ex.expected_info!r}, got {info!r}"
        )
        assert len(path) - 1 == ex.expected_length, (
            f"{ex.name}: Folding A* length {len(path) - 1} != expected {ex.expected_length}"
        )
        for c in path:
            assert ex.grid[c[0]][c[1]] == 0, f"{ex.name}: path visits obstacle {c}"
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (
                f"{ex.name}: non-4-adjacent step {a} -> {b}"
            )


_selfcheck()
del _selfcheck
