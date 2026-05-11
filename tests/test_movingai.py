"""Tests for the MovingAI ingest module.

The synthetic `.map` strings here use the canonical MovingAI format:

    type octile
    height H
    width  W
    map
    <H rows of W characters each>

We test:

  * Parsing of a well-formed map produces the expected (0/1) grid.
  * Malformed headers raise ValueError.
  * `is_symmetric_band` correctly identifies symmetric and asymmetric bands.
  * `largest_symmetric_window` finds the largest symmetric band including
    both odd-height and even-height bands.
  * `scan_for_symmetry` produces consistent statistics on hand-crafted maps.
  * Corpus summarisation handles empty directories and parse failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from folding_astar.movingai import (
    is_symmetric_band,
    largest_symmetric_window,
    parse_map,
    scan_for_symmetry,
    summarise_corpus,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_basic_4x4() -> None:
    text = (
        "type octile\n"
        "height 4\n"
        "width 4\n"
        "map\n"
        "....\n"
        ".@@.\n"
        ".@@.\n"
        "....\n"
    )
    g = parse_map(text)
    assert len(g) == 4
    assert all(len(row) == 4 for row in g)
    # Corner cells passable, centre block obstacles.
    assert g[0] == [0, 0, 0, 0]
    assert g[1] == [0, 1, 1, 0]
    assert g[2] == [0, 1, 1, 0]
    assert g[3] == [0, 0, 0, 0]


def test_parse_recognises_all_obstacle_chars() -> None:
    text = (
        "type octile\n"
        "height 1\n"
        "width 7\n"
        "map\n"
        ".G@OTSW\n"
    )
    g = parse_map(text)
    # '.' and 'G' passable; '@','O','T','S','W' obstacles
    assert g[0] == [0, 0, 1, 1, 1, 1, 1]


def test_parse_rejects_short_input() -> None:
    with pytest.raises(ValueError):
        parse_map("type octile\n")


def test_parse_rejects_missing_map_keyword() -> None:
    text = "type octile\nheight 1\nwidth 1\n.\n"  # no 'map' keyword
    with pytest.raises(ValueError):
        parse_map(text)


def test_parse_rejects_wrong_row_count() -> None:
    text = (
        "type octile\n"
        "height 3\n"
        "width 2\n"
        "map\n"
        "..\n"
        "..\n"  # only 2 rows, declared 3
    )
    with pytest.raises(ValueError):
        parse_map(text)


def test_parse_rejects_wrong_row_width() -> None:
    text = (
        "type octile\n"
        "height 2\n"
        "width 4\n"
        "map\n"
        "..\n"  # width 2, declared 4
        "....\n"
    )
    with pytest.raises(ValueError):
        parse_map(text)


# ---------------------------------------------------------------------------
# is_symmetric_band
# ---------------------------------------------------------------------------


def test_is_symmetric_band_obviously_symmetric() -> None:
    g = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
    ]
    assert is_symmetric_band(g, 0, 4) is True


def test_is_symmetric_band_obviously_asymmetric() -> None:
    g = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 0, 0, 0],  # differs from row 1 reflected
        [0, 0, 0, 0],
    ]
    assert is_symmetric_band(g, 0, 4) is False


def test_is_symmetric_band_odd_height() -> None:
    g = [
        [0, 0, 0],
        [1, 0, 1],  # midline; matches with itself
        [0, 0, 0],
    ]
    assert is_symmetric_band(g, 0, 3) is True


def test_is_symmetric_band_partial_window() -> None:
    g = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],  # part of an asymmetric overall map…
        [1, 1, 1, 1],  # but rows 1-2 alone are symmetric (identical pair)
        [0, 1, 0, 0],
    ]
    assert is_symmetric_band(g, 1, 3) is True
    assert is_symmetric_band(g, 0, 4) is False


def test_is_symmetric_band_degenerate() -> None:
    g = [[0, 0]]
    # A 0-height band is meaningless: contract says False.
    assert is_symmetric_band(g, 0, 0) is False


# ---------------------------------------------------------------------------
# largest_symmetric_window
# ---------------------------------------------------------------------------


def test_largest_symmetric_window_fully_symmetric_map() -> None:
    # 4x4 map that is entirely horizontally symmetric.
    g = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    res = largest_symmetric_window(g)
    assert res is not None
    h_start, h_end, area = res
    assert h_start == 0 and h_end == 4
    assert area == 16


def test_largest_symmetric_window_partial() -> None:
    # Rows 1..4 are a 4-row symmetric block; row 0 and row 5 break full
    # symmetry. Expect the algorithm to find the inner block.
    g = [
        [1, 0, 0, 0],     # asymmetric prefix
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 1],     # asymmetric suffix
    ]
    res = largest_symmetric_window(g)
    assert res is not None
    h_start, h_end, area = res
    assert (h_start, h_end) == (1, 5)
    assert area == 4 * 4


def test_largest_symmetric_window_no_symmetric_pair() -> None:
    # Adversarial case: every pair of rows differs and no duplicate rows
    # exist, so the only "symmetric" odd-band has height 1 (rejected).
    g = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
    res = largest_symmetric_window(g)
    # Either None (no qualifying band) or some short odd-band of height >=2.
    assert res is None or (res[1] - res[0]) >= 2


# ---------------------------------------------------------------------------
# scan_for_symmetry
# ---------------------------------------------------------------------------


def test_scan_for_symmetry_full_symmetry_recognised() -> None:
    g = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    stats = scan_for_symmetry(g, min_height=2)
    assert stats["fully_symmetric"] == 1
    assert stats["best_band_height"] == 3
    assert stats["has_min_height_band"] == 1
    assert stats["best_band_frac"] == 1.0


def test_scan_for_symmetry_no_useful_band() -> None:
    # Asymmetric and no tall band; expect a low score.
    g = [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    stats = scan_for_symmetry(g, min_height=4)
    assert stats["fully_symmetric"] == 0
    assert stats["has_min_height_band"] == 0


def test_scan_for_symmetry_empty_grid() -> None:
    stats = scan_for_symmetry([], min_height=2)
    assert stats["map_area"] == 0
    assert stats["fully_symmetric"] == 0


# ---------------------------------------------------------------------------
# summarise_corpus
# ---------------------------------------------------------------------------


def test_summarise_corpus_handles_empty_directory(tmp_path: Path) -> None:
    summary = summarise_corpus(tmp_path)
    assert summary["n_maps_scanned"] == 0
    assert summary["n_fully_symmetric"] == 0


def test_summarise_corpus_mixed_maps(tmp_path: Path) -> None:
    sym_text = (
        "type octile\n"
        "height 3\n"
        "width 3\n"
        "map\n"
        ".@.\n"
        "@@@\n"
        ".@.\n"
    )
    asym_text = (
        "type octile\n"
        "height 3\n"
        "width 3\n"
        "map\n"
        "...\n"
        "..@\n"
        "@..\n"
    )
    bad_text = "type octile\nheight 2\nwidth 2\nmap\n..\n"  # only 1 row body
    (tmp_path / "sym.map").write_text(sym_text)
    (tmp_path / "asym.map").write_text(asym_text)
    (tmp_path / "bad.map").write_text(bad_text)

    summary = summarise_corpus(tmp_path, min_height=2)
    assert summary["n_maps_scanned"] == 2
    assert summary["n_parse_failed"] == 1
    assert summary["n_fully_symmetric"] == 1
