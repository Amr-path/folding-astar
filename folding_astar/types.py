"""Shared type aliases. Keeps astar.py and folding.py from depending on each
other for type definitions (which would create a circular import)."""

from __future__ import annotations

Cell = tuple[int, int]   # (row, column)
Grid = list[list[int]]   # 0 = free, 1 = obstacle

__all__ = ["Cell", "Grid"]
