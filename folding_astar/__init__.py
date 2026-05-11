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
    folding_bfs,
    folding_dijkstra,
    folding_search,
    in_upper_half,
    midline,
    rho,
    verify_symmetry,
)
from folding_astar.jps import jps
from folding_astar.movingai import (
    is_symmetric_band,
    largest_symmetric_window,
    parse_map,
    scan_for_symmetry,
    summarise_corpus,
)
from folding_astar.search import bfs, dijkstra
from folding_astar.stats import (
    WilcoxonResult,
    bootstrap_ci,
    cliffs_delta,
    wilcoxon_signed_rank,
)
from folding_astar.types import Cell, Grid
from folding_astar.warehouse import (
    WarehouseSpec,
    build_warehouse,
    depot_cell,
    warehouse_endpoints,
)

__all__ = [
    "Cell",
    "Grid",
    "WarehouseSpec",
    "WilcoxonResult",
    "astar",
    "bfs",
    "bootstrap_ci",
    "build_warehouse",
    "cliffs_delta",
    "depot_cell",
    "dijkstra",
    "fold_grid",
    "fold_vertex",
    "folding_astar",
    "folding_bfs",
    "folding_dijkstra",
    "folding_search",
    "in_upper_half",
    "is_symmetric_band",
    "jps",
    "largest_symmetric_window",
    "midline",
    "parse_map",
    "rho",
    "scan_for_symmetry",
    "summarise_corpus",
    "verify_symmetry",
    "warehouse_endpoints",
    "wilcoxon_signed_rank",
]

__version__ = "0.1.0"
