"""folding-astar: optimal grid pathfinding via horizontal-reflection symmetry.

Public API:
    folding_astar(grid, start, goal) -> (path, info)
    astar(grid, start, goal)         -> path | None
    verify_symmetry(grid)            -> bool
    fold_grid(grid)                  -> folded grid
    fold_vertex(N, cell)             -> folded cell

See README.md for usage and docs/correctness.md for the relationship between
this implementation and the JAAMAS submission manuscript.
"""

from folding_astar.astar import astar
from folding_astar.folding import (
    fold_grid,
    fold_vertex,
    folding_astar,
    in_upper_half,
    midline,
    rho,
    verify_symmetry,
)
from folding_astar.types import Cell, Grid

__all__ = [
    "Cell",
    "Grid",
    "astar",
    "fold_grid",
    "fold_vertex",
    "folding_astar",
    "in_upper_half",
    "midline",
    "rho",
    "verify_symmetry",
]

__version__ = "0.1.0"
