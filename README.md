# folding-astar

A correct, verified implementation of **Folding A\***: an optimal grid pathfinding
algorithm that exploits horizontal-reflection symmetry to halve the effective
search space.

## What this is

Folding A\* runs A\* on a quotient graph induced by horizontal-reflection symmetry,
then unfolds the result into a path on the original grid. On grids where
obstacles are mirrored across the horizontal midline, this halves the state
space without sacrificing optimality.

## Relationship to the manuscript

This repository implements the algorithm described in the manuscript
*"Folding A\*: An Efficient Grid Pathfinding Algorithm via Symmetry-Reduced
Search Space Exploration"* — but with two important corrections relative to
the manuscript's first submission:

1. **The algorithm has been corrected.** The version of `FoldingAStar` printed
   in the manuscript's Section 4.2 (Algorithms 1–4) is incorrect when start
   and goal are on opposite halves of the grid. Specifically, the closing
   line of the proof of Lemma 3.8 — `dist_G(u, v) = dist_{G_f}(Φ(u), Φ(v))` —
   is false in general. The implementation here uses the correct algorithm,
   which canonicalises both endpoints into the upper half and performs an
   extra search to find the optimal midline-crossing column when needed.
   See [`docs/correctness.md`](docs/correctness.md) for the full story.

2. **This repository supersedes [`cfa-star-experiments`](https://github.com/Amr-path/cfa-star-experiments).**
   That repository implemented an unrelated algorithm (a line-of-sight
   "portal" heuristic on standard A\*) that did not match the manuscript's
   description. This repository implements the algorithm the paper describes.

## Quick start

```python
from folding_astar import folding_astar, astar, verify_symmetry

# A horizontally-symmetric 6x6 grid (0 = free, 1 = obstacle).
grid = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
]
assert verify_symmetry(grid)

path, info = folding_astar(grid, start=(0, 0), goal=(5, 5))
print(path)        # path in the original grid
print(info)        # 'ok: split' or 'ok: both upper' etc.
```

If the input is not symmetric, `folding_astar` falls back to standard A\*
and reports `'fallback: not symmetric'`.

## Public API

```
folding_astar(grid, start, goal) -> (path | None, info: str)
astar(grid, start, goal)         -> path | None     # ground-truth reference
verify_symmetry(grid)            -> bool
fold_grid(grid)                  -> list[list[int]]
fold_vertex(N, cell)             -> cell
```

All grids use `(row, column)` coordinates. 4-connectivity. Free cells = 0,
obstacles = 1. Path lengths are counts of edges, not vertices.

## Correctness guarantees

This package is tested against ground-truth A\* in two ways:

- A hand-built **edge-case test suite** (`tests/test_edge_cases.py`) covering
  every special case identified in the JAAMAS reviewer reports: same column /
  opposite halves, folded-adjacent / original-opposite, `t = ρ(s)`, `t = s`,
  optimal path forced through the midline, even and odd N.
- A **randomised differential test** (`tests/test_random_differential.py`)
  that generates thousands of symmetric grids of varying sizes and obstacle
  densities and verifies that Folding A\* and standard A\* always agree on
  path length and that Folding A\*'s output is always a valid 4-connected
  obstacle-free path.

CI runs both on every push. See [`docs/algorithm.md`](docs/algorithm.md) for
the algorithm specification and proof sketches.

## Status

This is research-quality code. Phase 0 of the post-rejection rewrite is
complete: the algorithm is verified-correct on the edge cases the reviewers
raised and on randomised tests. Phases 1–6 (writing pass, real benchmarks
against JPS / RSR / GPPC, theoretical max-speedup analysis, figure rewrites,
applicability story, venue selection) are tracked separately.

## License

MIT. See [`LICENSE`](LICENSE).
