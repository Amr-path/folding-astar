# Correctness — relationship to the JAAMAS submission

This document explains why the algorithm in this repository differs from the
one printed in the manuscript *"Folding A\*: An Efficient Grid Pathfinding
Algorithm via Symmetry-Reduced Search Space Exploration"* (JAAMAS, 2026,
rejected). Two things are different:

1. The manuscript's Algorithms 1–4 are incorrect on inputs where start and
   goal lie on opposite halves of the grid. The implementation here uses a
   corrected procedure.
2. The previously cited code repository (`cfa-star-experiments`) implemented
   a different algorithm than the one the manuscript describes. The
   experimental numbers in Tables 2–7 were therefore not produced by the
   algorithm under review. This repository implements the algorithm the
   manuscript describes and supersedes the previous one for that purpose.

## Why the printed algorithm is incorrect

The proof of Lemma 3.8 in the manuscript shows only the forward direction:
every path of length *d* from *u* to *v* in *G* induces a path of length
≤ *d* from Φ(*u*) to Φ(*v*) in the folded graph *G*<sub>f</sub>. The proof
then closes with

> Critically, the shortest path distance is preserved:
> dist<sub>*G*</sub>(*u*, *v*) = dist<sub>*G*<sub>f</sub></sub>(Φ(*u*), Φ(*v*)).

This equality is false in general. Reviewer 2 of the original submission
flagged this exact line ("the claim here that dist_G = dist_G_f seems
incorrect"). A direct counterexample:

> Take the 6×6 grid that is free in column 2, with no obstacles affecting
> the relevant column. Let *u* = (0, 2), *v* = (5, 2). Then ρ(*v*) = (0, 2),
> so Φ(*u*) = Φ(*v*) = (0, 2), giving dist<sub>*G*<sub>f</sub></sub> = 0.
> But dist<sub>*G*</sub>(*u*, *v*) = 5.

This is the test labelled E1 in `tests/test_edge_cases.py`. The literal
printed algorithm returns a length-0 "path" for this input.

The deeper failure is in `UnfoldPath` (Algorithm 4 of the manuscript). Its
third branch (`add (N − 1 − i', j')`) is unreachable: by construction every
vertex in the folded grid has *i'* < *m* (even *N*) or *i'* ≤ *m* (odd *N*),
so the procedure always emits the upper-half image of the folded path. There
is no mechanism to reconstruct a path that genuinely uses the lower half of
*G*.

## What the correct algorithm does

The fix is to canonicalise both endpoints into the upper half before running
the folded search. There are three cases.

**Case A (both upper-half).** Run A\* on the folded grid; the path is
already valid in *G* because every visited vertex lies in the upper half.
This is what the manuscript described, and it is correct in this case.

**Case B (both lower-half).** Reflect both endpoints, solve as in case A,
reflect every vertex of the resulting path back. Symmetric.

**Case C (opposite halves — the case the manuscript missed).** Reflect the
lower endpoint into the upper half; call its image *t*<sub>im</sub>. The
optimal path *s* → *t* in *G* must cross the midline somewhere; by symmetry
the lower segment is the mirror of an upper-half path from the crossing
column to *t*<sub>im</sub>. We minimise

> total(*b*) = dist<sub>*G*<sub>f</sub></sub>(*s*, *b*) + cross_cost +
> dist<sub>*G*<sub>f</sub></sub>(*b*, *t*<sub>im</sub>)

over candidate boundary cells *b*, where cross_cost is 1 for even *N*
(the extra vertical edge between rows *m*−1 and *m*) and 0 for odd *N*
(*b* lies on the fixed-point row). One BFS from *s*, one BFS from
*t*<sub>im</sub>, plus an O(*N*) scan over boundary columns to pick the
minimiser. Two A\* calls then materialise the chosen segments and the
lower segment is reflected into place.

The implementation is in `folding_astar/folding.py`. Cases A, B, and C
report the strings `"ok: both upper"`, `"ok: both lower"`, and `"ok: split"`
respectively, plus `"ok: s == t"` for the degenerate case.

## Corrected lemma statement

The original Lemma 3.8 asserted distance equality. The correct statement
under the corrected algorithm is two-part:

**Lemma (folding inequality).** For any *u*, *v* ∈ *V*<sub>free</sub> in a
horizontally-symmetric grid,

> dist<sub>*G*<sub>f</sub></sub>(Φ(*u*), Φ(*v*)) ≤ dist<sub>*G*</sub>(*u*, *v*).

The reverse inequality holds *only* when *u* and *v* are on the same side
of the midline (or one of them lies on the midline in odd *N*). In general
it can fail by an arbitrarily large amount — *u* = (0, 2), *v* = (5, 2)
gives a slack of 5 in a 6×6 grid, and the slack scales with *N*.

**Theorem (correctness of the corrected algorithm).** For any
horizontally-symmetric grid *G* and endpoints *s*, *t* ∈ *V*<sub>free</sub>,
the corrected Folding A\* returns a path *π* in *G* with length
dist<sub>*G*</sub>(*s*, *t*) (or `None` if no path exists).

*Proof sketch.* Cases A and B reduce to standard A\* on a 4-connected
sub-grid; correctness follows from A\*'s admissibility under the Manhattan
heuristic. Case C decomposes the path at the optimal crossing column *b*\*.
By symmetry of *G*, the lower segment from *b* (or ρ(*b*)) to *t* in *G*
has the same length as the upper-segment A\* path from *b* to *t*<sub>im</sub>
in *G*<sub>f</sub>. The total cost we minimise is therefore exactly
dist<sub>*G*</sub>(*s*, *t*), and BFS gives us shortest distances to every
candidate *b*, so the minimiser is correct. □

## Empirical verification

The test suite covers:

- All four edge cases Reviewer 2 explicitly raised (same column / opposite
  halves; folded-adjacent / original-opposite; *t* = ρ(*s*); *t* = *s*).
- Optimal-path-through-midline cases for both even and odd *N*.
- Two-way split cases with non-trivial midline-column choice.
- Asymmetric-input fallback to standard A\*.
- 400 randomised differential trials across grid sizes 4×4 to 30×30 at
  obstacle densities 0%, 10%, 20%, 30%.

All pass. See `tests/test_edge_cases.py` and `tests/test_random_differential.py`.
The randomised test catches any regression that the hand-built suite would
miss, and is run on every push by CI.

## Implications for the manuscript rewrite

When the manuscript is rewritten for resubmission (target venues: SoCS,
ICAPS, JAIR), the following changes are non-negotiable:

1. **Lemma 3.8 must be restated as a one-directional inequality.** The
   equality form is false.
2. **Algorithm 1 (FoldingAStar) must be replaced** by the three-case
   procedure documented here. Algorithm 4 (UnfoldPath) becomes redundant
   in cases A and B and is replaced by the explicit reflection in case C.
3. **Theorem 5.2 (Optimality) must be rederived** under the corrected
   algorithm; the manuscript proof currently uses the false equality and
   cannot stand.
4. **All experimental numbers must be re-collected** with the corrected
   algorithm (and with a JPS / RSR baseline that is independently verified
   against the published reference implementation, per Reviewer 1's
   speedup-discrepancy concern).
5. **Figures 1 and 3 must be redrawn** from `EX_FIGURE_1` and `EX_FIGURE_3`
   in `folding_astar/examples.py`. The current Figure 3 in particular
   depicts a fold operation that does not produce its claimed result.
