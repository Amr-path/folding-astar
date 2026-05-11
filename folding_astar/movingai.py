"""MovingAI benchmark corpus ingest.

The MovingAI Pathfinding Benchmarks (Sturtevant, 2012) are the de-facto
empirical baseline for grid pathfinding. Maps are stored in plain-text
``.map`` files with the following header:

    type octile
    height H
    width  W
    map
    <H rows of W characters each>

Cell character encoding (from the MovingAI format spec):

    .   passable terrain
    G   passable terrain (same as '.')
    @   out-of-bounds (obstacle)
    O   out-of-bounds (obstacle)
    T   tree (obstacle for 4-connected)
    S   swamp (passable, higher cost — we treat as obstacle in the
        4-connected variant for simplicity; this is the standard choice
        for unit-cost grid benchmarks)
    W   water (obstacle for 4-connected)

This module provides:

    parse_map(text) -> Grid
        Parse a MovingAI ``.map`` text blob into a Folding-A* compatible
        binary grid (0 = free, 1 = obstacle).

    largest_symmetric_window(grid) -> (h_start, h_end, area) | None
        Return the largest contiguous horizontal band whose obstacle
        pattern is invariant under reflection about its midline. The band
        spans rows ``[h_start, h_end)`` and is reported by its area
        (``(h_end - h_start) * width``).

    scan_for_symmetry(grid, min_height=20) -> dict
        Heuristic scan of an entire map for naturally-symmetric subgrids.
        Reports per-map summary statistics: largest symmetric band,
        symmetric area as a fraction of the map, and the median band
        height.

    summarise_corpus(directory) -> dict
        Scan every ``.map`` file in a directory and aggregate the
        per-map statistics into a corpus-level summary.

The point of this module is the §7.7 (Limitations of this evaluation)
question: *how much of a typical MovingAI map is naturally
horizontally symmetric?*  The answer informs whether folding-style
methods could ever be applied to such maps without re-ordering the rows
or imposing other transformations.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any, Iterable

from folding_astar.types import Grid


__all__ = [
    "parse_map",
    "is_symmetric_band",
    "largest_symmetric_window",
    "scan_for_symmetry",
    "summarise_corpus",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Characters treated as PASSABLE in the 4-connected variant. Everything
# else is an obstacle. This matches the standard MovingAI convention for
# unit-cost grid benchmarks.
_PASSABLE: frozenset[str] = frozenset({".", "G"})


def parse_map(text: str) -> Grid:
    """Parse a MovingAI ``.map`` text blob into a Folding-A* grid.

    Parameters
    ----------
    text:
        The full contents of a ``.map`` file as a single string.

    Returns
    -------
    Grid
        ``grid[i][j] == 0`` if the cell is passable, ``1`` otherwise.

    Raises
    ------
    ValueError
        If the header is malformed or the row count / row widths don't
        match the declared dimensions.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        raise ValueError("MovingAI map too short (need header + map keyword)")

    # Required header tokens. The order is fixed by the format spec.
    header = {}
    idx = 0
    while idx < len(lines) and lines[idx].strip().lower() != "map":
        line = lines[idx].strip()
        if line:
            parts = line.split(None, 1)
            if len(parts) == 2:
                header[parts[0].lower()] = parts[1]
        idx += 1
    if idx >= len(lines):
        raise ValueError("MovingAI map: missing 'map' keyword")

    try:
        height = int(header["height"])
        width = int(header["width"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"MovingAI map: missing/bad height or width ({e})") from e

    body = lines[idx + 1 : idx + 1 + height]
    if len(body) != height:
        raise ValueError(
            f"MovingAI map: declared height {height} but got {len(body)} rows"
        )

    grid: Grid = []
    for r, row in enumerate(body):
        if len(row) != width:
            raise ValueError(
                f"MovingAI map row {r}: declared width {width} but got "
                f"length {len(row)}"
            )
        grid.append([0 if ch in _PASSABLE else 1 for ch in row])

    return grid


# ---------------------------------------------------------------------------
# Symmetric-band detection
# ---------------------------------------------------------------------------


def is_symmetric_band(grid: Grid, h_start: int, h_end: int) -> bool:
    """True iff the row-band ``grid[h_start:h_end]`` is invariant under
    reflection about its own horizontal midline.

    The band is treated as a standalone grid for the purposes of the
    test; this matches the §3 definition of horizontal symmetry exactly.
    """
    if h_end <= h_start:
        return False
    H = h_end - h_start
    W = len(grid[h_start]) if h_start < len(grid) else 0
    m = H // 2  # number of strictly upper-half rows
    for i in range(m):
        row_top = grid[h_start + i]
        row_bot = grid[h_start + (H - 1 - i)]
        for j in range(W):
            if row_top[j] != row_bot[j]:
                return False
    return True


def largest_symmetric_window(grid: Grid) -> tuple[int, int, int] | None:
    """Return the largest contiguous horizontal band of the map whose
    obstacle pattern is invariant under reflection about its midline.

    Strategy: for each candidate centre row (or pair of rows, depending
    on parity), expand outward as far as the symmetry condition holds.
    This is O(H * H * W) in the worst case, but with cheap row equality
    checks it is acceptable for the largest published maps (1024×1024).

    Returns
    -------
    (h_start, h_end, area) or None
        ``area = (h_end - h_start) * W`` is the number of cells covered
        by the band. ``None`` if no band of height >= 2 is symmetric
        (extremely rare — a single duplicated row pair already qualifies).
    """
    H = len(grid)
    if H < 2:
        return None
    W = len(grid[0])

    best: tuple[int, int, int] | None = None

    # Odd-height bands: centre is a single row i. Expand by k on each side.
    for centre in range(H):
        k = 0
        while centre - k - 1 >= 0 and centre + k + 1 < H:
            if grid[centre - k - 1] != grid[centre + k + 1]:
                break
            k += 1
        h_start = centre - k
        h_end = centre + k + 1
        height = h_end - h_start
        if height >= 2:
            area = height * W
            if best is None or area > best[2]:
                best = (h_start, h_end, area)

    # Even-height bands: centre is between rows (i, i+1). Expand by k.
    for i in range(H - 1):
        if grid[i] != grid[i + 1]:
            continue  # not a valid even-band centre
        k = 0
        while i - k - 1 >= 0 and i + 1 + k + 1 < H:
            if grid[i - k - 1] != grid[i + 1 + k + 1]:
                break
            k += 1
        h_start = i - k
        h_end = i + 2 + k
        height = h_end - h_start
        if height >= 2:
            area = height * W
            if best is None or area > best[2]:
                best = (h_start, h_end, area)

    return best


def scan_for_symmetry(grid: Grid, min_height: int = 20) -> dict[str, Any]:
    """Compute symmetry statistics for a single map.

    Reports the largest symmetric band (by area), its height, its area
    as a fraction of the map area, and whether the map contains *any*
    band of height >= ``min_height``.

    The min_height threshold reflects practical utility: a 2-row band is
    technically symmetric but useless; we want bands tall enough that
    folding could meaningfully prune search.
    """
    H = len(grid)
    if H == 0:
        return {
            "map_height": 0, "map_width": 0, "map_area": 0,
            "best_band_height": 0, "best_band_area": 0,
            "best_band_frac": 0.0,
            "has_min_height_band": 0,
            "fully_symmetric": 0,
        }
    W = len(grid[0])
    area_total = H * W

    best = largest_symmetric_window(grid)
    if best is None:
        bh, ba = 0, 0
    else:
        h_start, h_end, ba = best
        bh = h_end - h_start

    fully_symmetric = 1 if is_symmetric_band(grid, 0, H) else 0

    return {
        "map_height": H,
        "map_width": W,
        "map_area": area_total,
        "best_band_height": bh,
        "best_band_area": ba,
        "best_band_frac": ba / area_total if area_total else 0.0,
        "has_min_height_band": 1 if bh >= min_height else 0,
        "fully_symmetric": fully_symmetric,
    }


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------


def _iter_map_files(directory: Path) -> Iterable[Path]:
    yield from sorted(directory.rglob("*.map"))


def summarise_corpus(
    directory: str | Path, min_height: int = 20
) -> dict[str, Any]:
    """Scan every ``.map`` file in ``directory`` and aggregate.

    Returns a dictionary with corpus-level totals and per-map records.
    Empty corpus is handled gracefully.
    """
    root = Path(directory)
    records: list[dict[str, Any]] = []
    for path in _iter_map_files(root):
        try:
            text = path.read_text()
            grid = parse_map(text)
        except (OSError, ValueError) as e:
            records.append({"path": str(path), "error": str(e)})
            continue
        stats = scan_for_symmetry(grid, min_height=min_height)
        stats["path"] = str(path)
        records.append(stats)

    parsed = [r for r in records if "error" not in r]
    n_parsed = len(parsed)
    n_failed = len(records) - n_parsed

    if n_parsed == 0:
        return {
            "n_maps_scanned": 0,
            "n_parse_failed": n_failed,
            "n_fully_symmetric": 0,
            "n_with_min_height_band": 0,
            "median_best_frac": 0.0,
            "mean_best_frac": 0.0,
            "records": records,
        }

    fracs = [r["best_band_frac"] for r in parsed]
    return {
        "n_maps_scanned": n_parsed,
        "n_parse_failed": n_failed,
        "n_fully_symmetric": sum(r["fully_symmetric"] for r in parsed),
        "n_with_min_height_band": sum(r["has_min_height_band"] for r in parsed),
        "median_best_frac": float(median(fracs)),
        "mean_best_frac": sum(fracs) / n_parsed,
        "records": records,
    }
