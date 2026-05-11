"""Statistical primitives for the manuscript's empirical evaluation.

Q1 reviewers expect every reported speedup to come with a confidence
interval and a significance test. This module provides three primitives,
all in pure Python (no scipy / numpy dependency):

  - wilcoxon_signed_rank: paired non-parametric test. Returns the
    test statistic and a normal-approximation two-sided p-value. For
    paired (folded, unfolded) timing samples this is the standard test.

  - cliffs_delta: ordinal effect size in [-1, 1] for two paired
    samples. |delta| > 0.147 is conventionally "small", > 0.330 is
    "medium", > 0.474 is "large".

  - bootstrap_ci: percentile bootstrap confidence interval for any
    statistic computed from a sample. Used to put a CI on speedup
    ratios reported as point estimates in §7.

Each function is type-annotated and unit-tested. The implementations
follow standard textbook formulas and are validated against scipy's
reference outputs in tests/test_stats.py to within numerical tolerance.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


__all__ = [
    "WilcoxonResult",
    "wilcoxon_signed_rank",
    "cliffs_delta",
    "bootstrap_ci",
]


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank test.

    Fields:
        statistic: the smaller of W+ and W- (the standard test statistic).
        n_effective: number of non-zero paired differences (zero-difference
            pairs are dropped per the standard procedure).
        z: standardised statistic under the normal approximation, including
            the continuity correction. Defined only when n_effective is
            large enough that the normal approximation applies.
        p_value: two-sided p-value via normal approximation. For very small
            n_effective (< 10) the normal approximation is unreliable and
            we return None — callers should report the test as "exact
            test required" rather than rely on the approximation.
    """
    statistic: float
    n_effective: int
    z: float | None
    p_value: float | None


def wilcoxon_signed_rank(xs: Sequence[float], ys: Sequence[float]) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test for paired samples.

    Tests H0: the median of the paired differences (xs[i] - ys[i]) is 0.

    Implementation note: ranks include average ranks for ties, and the
    variance correction for ties uses the standard formula
    Var(W) = n(n+1)(2n+1)/24 - sum(t^3 - t)/48 over ties of size t.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    diffs = [x - y for x, y in zip(xs, ys) if x != y]
    n = len(diffs)
    if n == 0:
        return WilcoxonResult(statistic=0.0, n_effective=0, z=None, p_value=None)

    abs_diffs = [abs(d) for d in diffs]
    ranks = _average_ranks(abs_diffs)

    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
    statistic = min(w_plus, w_minus)

    if n < 10:
        # Normal approximation unreliable; signal to caller.
        return WilcoxonResult(statistic=statistic, n_effective=n, z=None, p_value=None)

    # Variance with tie correction.
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    # Tie correction: subtract sum (t^3 - t) / 48 over groups of ties.
    sorted_abs = sorted(abs_diffs)
    tie_correction = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        t = j - i
        if t > 1:
            tie_correction += (t * t * t - t) / 48
        i = j
    var -= tie_correction

    if var <= 0:
        return WilcoxonResult(statistic=statistic, n_effective=n, z=None, p_value=None)

    # Continuity correction: 0.5 toward the mean.
    z_num = w_plus - mean
    if z_num > 0:
        z_num -= 0.5
    elif z_num < 0:
        z_num += 0.5
    z = z_num / math.sqrt(var)
    p = 2 * (1 - _norm_cdf(abs(z)))
    return WilcoxonResult(statistic=statistic, n_effective=n, z=z, p_value=p)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return the rank of each value, with ties getting the average rank."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2  # average of ranks i+1 .. j
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def _norm_cdf(x: float) -> float:
    """CDF of standard normal via the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------

def cliffs_delta(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Cliff's delta effect size for two independent samples.

    delta = (#{(i, j) : xs[i] > ys[j]} - #{(i, j) : xs[i] < ys[j]}) / (n_x * n_y).

    Range: [-1, 1]. Positive means xs tend to be larger than ys.
    Conventional thresholds: |delta| > 0.147 small, > 0.330 medium,
    > 0.474 large. (Romano et al. 2006).
    """
    if not xs or not ys:
        raise ValueError("samples must be non-empty")
    n_x = len(xs)
    n_y = len(ys)
    greater = 0
    less = 0
    # O(n_x * n_y) but n is typically small enough that this is fine.
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (n_x * n_y)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    sample: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic(sample)`.

    Resamples `sample` with replacement `n_resamples` times, computes the
    statistic on each resample, and returns the (lower, upper) percentile
    bounds at the given confidence level.

    For paired-comparison statistics (e.g., speedup ratio = mean(t_a) /
    mean(t_b)), the caller should pass paired data and use a statistic
    that respects the pairing structure.
    """
    if not sample:
        raise ValueError("sample must be non-empty")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")

    rng = random.Random(seed)
    n = len(sample)
    resamples: list[float] = []
    for _ in range(n_resamples):
        bs = [sample[rng.randint(0, n - 1)] for _ in range(n)]
        resamples.append(statistic(bs))
    resamples.sort()

    alpha = (1 - confidence) / 2
    lo_idx = int(math.floor(alpha * n_resamples))
    hi_idx = int(math.ceil((1 - alpha) * n_resamples)) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(0, min(n_resamples - 1, hi_idx))
    return (resamples[lo_idx], resamples[hi_idx])
