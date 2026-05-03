# Algorithm specification

This is the specification of the Folding A\* algorithm as implemented in
`folding_astar/`. It supersedes the algorithm boxes (Algorithms 1–4) in
the JAAMAS submission manuscript. For the history of why this revision was
necessary, see [`correctness.md`](correctness.md).

## Notation

A grid graph *G* = (*V*, *E*) on an *N* × *N* discrete grid has vertex set

> *V* = { (*i*, *j*) : 0 ≤ *i*, *j* < *N* }

and edge set *E* given by 4-connectivity (each vertex is adjacent to its
horizontal and vertical neighbours when both lie inside the grid). An
obstacle set 𝒪 ⊆ *V* specifies impassable cells; the free set is
*V*<sub>free</sub> = *V* \ 𝒪.

We use **(row, column)** coordinates throughout. *i* is row, *j* is column.

The midline is *m* = ⌊*N*/2⌋. For even *N* the fold occurs *between*
rows *m* − 1 and *m*; for odd *N* row *m* is the fixed axis.

The reflection involution is

> ρ(*i*, *j*) = (*N* − 1 − *i*, *j*).

## Symmetry condition

*G* exhibits horizontal-reflection symmetry iff

> ∀ (*i*, *j*) ∈ *V*: (*i*, *j*) ∈ 𝒪 ⇔ ρ(*i*, *j*) ∈ 𝒪.

The symmetry check costs O(*N*²) (one pass over the upper half).

## Quotient map

The reflection induces an equivalence relation *u* ∼ *v* iff *u* = *v* or
*u* = ρ(*v*). The quotient map

> Φ(*u*) = *u* if *u* is in the upper half (*i* < *m* for even *N*,
> *i* ≤ *m* for odd *N*); else Φ(*u*) = ρ(*u*)

picks the upper-half representative of each class. The folded vertex set is

> *V*<sub>f</sub> = Φ(*V*<sub>free</sub>).

For even *N* this is exactly the upper half of *V*<sub>free</sub>; for odd
*N* it includes the midline row.

## Folding inequality

For any *u*, *v* ∈ *V*<sub>free</sub>,

> dist<sub>*G*<sub>f</sub></sub>(Φ(*u*), Φ(*v*)) ≤ dist<sub>*G*</sub>(*u*, *v*).

The reverse inequality holds only when *u* and *v* are on the same side of
the midline (or one of them lies on the midline in odd *N*). In general it
can fail: for *v* = ρ(*u*) the folded distance is 0 while the original
distance can be Θ(*N*).

This is the corrected statement of the manuscript's Lemma 3.8. It is
sufficient for the corrected algorithm but does not, by itself, suffice for
the manuscript's original "fold–search–unfold" outline.

## The algorithm

```
function FoldingAStar(G, s, t):
    if not VerifySymmetry(G):
        return AStar(G, s, t)             # fallback, info = "fallback: not symmetric"
    if s or t is on an obstacle:
        return None                        # info = "no path"
    if s == t:
        return [s]                         # info = "ok: s == t"
    s_up = inUpperHalf(N, s)
    t_up = inUpperHalf(N, t)
    if s_up and t_up:
        return SolveUpper(G, s, t)         # info = "ok: both upper"
    if not s_up and not t_up:
        path = SolveUpper(G, ρ(s), ρ(t))
        return [ρ(v) for v in path]        # info = "ok: both lower"
    return SolveSplit(G, s, t)             # info = "ok: split"


function SolveUpper(G, s, t):
    # Both endpoints are in the upper half (or on the midline for odd N).
    # The folded grid G_f is the upper-half sub-grid; the path A* finds in
    # G_f is a valid path in G with no further work.
    G_f = FoldGrid(G)
    return AStar(G_f, s, t)


function SolveSplit(G, s, t):
    # Exactly one of s, t is in the lower half. WLOG (after a swap if
    # needed) s is upper, t is lower. Let t_im = ρ(t).
    G_f = FoldGrid(G)
    cols = number of columns of G

    # Pick boundary row.
    if N is odd:
        boundary_row = m
        cross_cost = 0                    # b is its own reflection
    else:
        boundary_row = m - 1
        cross_cost = 1                    # extra vertical edge over the fold

    # BFS from s and from t_im to score every boundary cell.
    d_s   = BFS(G_f, s)
    d_t   = BFS(G_f, t_im)
    candidates = { (boundary_row, c) : c < cols, free in G_f, in d_s, in d_t }
    if candidates is empty: return None

    b* = argmin over b in candidates of  d_s[b] + cross_cost + d_t[b]

    upper = AStar(G_f, s, b*)
    lower = AStar(G_f, b*, t_im)
    if either is None: return None
    if N is odd:
        # b* is fixed by ρ. The lower segment starts the step *after* b*.
        return upper + [ρ(v) for v in lower[1:]]
    else:
        # The fold-crossing edge from b* to ρ(b*) is implicit: ρ(lower[0])
        # = ρ(b*) is the cell directly below b*.
        return upper + [ρ(v) for v in lower]
```

A swap-and-reverse wrapper turns the lower-and-upper case into the
upper-and-lower case before invoking `SolveSplit`.

## Correctness (sketch)

For both `SolveUpper` and the symmetric `SolveLower` reduction, the result
is a folded-grid A\* path; correctness follows from standard A\*
admissibility under the Manhattan heuristic on a 4-connected grid.

For `SolveSplit`, every path from *s* (upper) to *t* (lower) in *G* must
cross the midline at some boundary cell *b*. Decompose any optimal path
*π*\* into an upper segment *π*<sub>up</sub>: *s* → *b* and a lower
segment *π*<sub>low</sub>: *b'* → *t*, where *b'* is *b* itself for odd
*N* (axis row) or ρ(*b*) for even *N* (the row directly below the fold,
reached via one fold-crossing edge). By symmetry of obstacles,
*π*<sub>low</sub> reflected into the upper half is a path from *b* to
*t*<sub>im</sub> in *G*<sub>f</sub> of the same length, so

> |*π*\*| = dist<sub>*G*<sub>f</sub></sub>(*s*, *b*) + cross_cost(*N*) +
> dist<sub>*G*<sub>f</sub></sub>(*b*, *t*<sub>im</sub>).

Minimising over *b* picks the optimum; BFS yields the two distance maps
exactly. Materialising the chosen segments and reflecting the lower one
produces a valid 4-connected obstacle-free path in *G* with length
dist<sub>*G*</sub>(*s*, *t*). □

## Complexity

For an *N* × *N* grid:

- Symmetry verification: O(*N*²).
- `FoldGrid`: O(*N*²).
- `SolveUpper`: A\* on roughly *N*²/2 vertices, time O(*N*² log *N*),
  space O(*N*²).
- `SolveSplit`: two BFS calls (O(*N*²) each) plus an O(*N*) scan plus two
  A\* calls (O(*N*² log *N*) each). Total time O(*N*² log *N*); the
  constant is roughly twice that of `SolveUpper` because of the two A\*
  calls.

The asymptotic worst-case complexity matches standard A\*. The state-space
reduction is a constant factor of 2 in `SolveUpper` and `SolveLower`. In
`SolveSplit` the speedup is smaller because of the additional BFS and
second A\* call; we do not claim a 2× factor in the split case.

The original manuscript's Theorem 5.2 (Optimality) and the per-phase
timing breakdown in Section 7 must be rederived under this complexity
profile when the paper is rewritten — neither matches the corrected
algorithm.

## What is *not* claimed

- This algorithm does not exceed the asymptotic complexity of standard A\*.
  The "constant factor" speedup is real but it is a constant factor.
- This algorithm does not extend automatically to k-fold rotational or
  multi-axis symmetries. Reviewer 1 raised that as an interesting open
  question; it remains open.
- This algorithm does not handle *near*-symmetric grids with a theoretical
  optimality guarantee. The empirical robustness numbers in Section 7 of
  the manuscript do not survive the algorithmic correction in full and
  must be remeasured.
- This algorithm is not faster than JPS on grids without global symmetry.
  Reviewer 1's concern about the manuscript's anomalously-low JPS speedup
  numbers points at a separate issue (likely a buggy JPS implementation or
  too-small grids); we do not address that here.
