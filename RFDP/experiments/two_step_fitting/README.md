# Two-step renewal fitting for the indirect frontier

Status: formal Market v2 completed and fully accepted at2026-10-29 19:50 +08:00;
this fixed form is stopped after a negative accuracy result. Execution used
16workers x BLAS1 and frozen commit `08f44a63d3f50e1d2453024405d6b38aade837df`.
37 related CPU correctness checks passed. Model key `two_step_fitting` in the
shared `experiments.continuous_fitting` runner. [Full result](result.json):
mAP92.710585%, Rank-1 94.388361%; differences from the original are
-0.424317 and -0.771971 percentage points. Both query- and identity-paired
95% intervals are below zero for both metrics. All four stages exited0;
336788 actual solves and all saved scores/frontiers/bridges/rankings passed
readback, with maximum new-equation residual9.564e-10. Computation took48.14min
and full readback18.62min. Acceptance and accuracy are separate conclusions.
The initial v1 stopped before completing any query because concurrent legacy
logger imports raced to create a directory; the idempotent-creation fix and
[original failure](initialization-failure.json) preserve that distinction.
This is an internal fidelity modification, not a new input graph or a claim that
two-hop diffusion itself is novel.

## Decision after complete results

The continuous-source form lost accuracy; the target-preserving boundary-odds
form recovered most of that loss but still had no mAP point gain. Its completed
result and all-query bounded diagnosis are in `../boundary_odds/`. The latter
compared identical frontiers until the first ordered difference, at most five
batches per query: 590 ordered differences but only 116 changed sets. The median
first-stage direct curvature multiplier was 1.001305 and the median relative
score change was 0.000489. Thus that particular direct-coefficient correction
was mostly a small perturbation, not evidence for a stronger coefficient sweep.
Both fixed forms are stopped. These observations do not establish causality for
the full mAP difference.

We retain the original direct fit and source vector, and instead examine the
indirect nodes' missing fidelity curvature. Their original source beta is the
same regardless of how readily each node can reach the seeds in two steps.
The smoothing equation already uses their topology. The hypothesis is that
adding an explicit local absorption-time target supplies useful fidelity beyond
that smoothing; redundancy and harmful promotion remain possible.

## Local transition model and exact renewal interpretation

On a fixed original frontier U=A union B, W is symmetric and nonnegative,
h=W1>0 includes self loops, e is boundary mass to the current seed set L,
d=h+e, p=e/d, and S=H^(-1/2) W H^(-1/2). A={e>0}; B={e=0}.
Collapse L into an absorbing state. The transient transition matrix on this
local graph is P_UU=diag(1/d)W and its one-step absorption probability is p.
Edges outside the original local frontier are excluded in this interpretation,
as in the parent's stage construction; this is not a global-graph probability.

For i in B, the exact probability of first reaching L at step two is

    q_i = (P_UU p)_i = sum_j W_ij p_j / h_i.

The original two-hop frontier guarantees a positive path i->j->L, hence q_i>0.
No extra neighbors, paths, labels or camera information are introduced.

Define a trial that starts at i, takes two ordinary steps, and, if it fails to
hit L, instantly resets to i with zero extra step cost. Success can occur only
at step two since p_i=0. Its exact renewal equation is

    T_i = 2 + (1-q_i) T_i,       T_i = 2/q_i.

For a direct node, the analogous one-step reset trial has T_i=1/p_i, exactly
the parent's existing target. Trials are restarted at their own starting node;
they are only a derivation of local fidelity targets. We do not sample walks.
The period two is fixed by the existing indirect frontier, not an optimized
horizon. These reset times are NOT the unrestricted GMFPT of the original walk,
nor unbiased estimates or upper/lower bounds for it. The normalized variational
solution below is itself a fitted dissimilarity, not a literal expected time.

## Energy, closed form and proof

The original energy, omitting constants, is

    J0(psi) = .5 psi^T(I-S)psi
              + alpha/2 sum_A p_i(psi_i-1/p_i)^2 - beta sum_B psi_i.

Apply the same reciprocal-target fidelity construction to B with reciprocal
target q_i/2. Replace its linear term by

    beta/2 sum_B (q_i/2)(psi_i-2/q_i)^2.

After dropping constants this equals beta/4 sum_B q_i psi_i^2 - beta sum_B psi_i.
Hence the original linear source is preserved on BOTH A and B. Define

    D_B = beta/2 diag(1_B q),
    M2 = I-S+alpha diag(p)+D_B = M0+D_B,
    b0_i = alpha on A, beta on B,
    grad J2 = M2 psi-b0,         psi2 = M2^(-1)b0.

For any v, v^T(I-S)v=.5 sum_ij W_ij(v_i/sqrt(h_i)-v_j/sqrt(h_j))^2.
If every component has a boundary anchor, M0 is SPD; D_B>=0 therefore M2 is SPD.
Its off-diagonals are nonpositive, so it is a nonsingular M-matrix with positive
inverse on each connected block. b0>0 gives a unique strictly positive solution.
This is a derived data-dependent fidelity diagonal, not a numerical ridge used
to rescue a singular system. Unanchored components remain invalid.

The exact response identity on the SAME stage is

    psi2-psi0 = -M2^(-1) D_B psi0 <= 0 componentwise.

It proves scores decrease, not that their ordering improves. It is also not an
outer-trajectory comparison: different selections change later W/h/e/q.

In coordinates z=psi/sqrt(h), the certificate uses

    [L(W)+diag(a)] z = sqrt(h)*b0,
    a_i = alpha*h_i*e_i/(h_i+e_i) on A,
    a_i = beta/2 sum_j W_ij e_j/(h_j+e_j) on B.

This second expression avoids dividing and multiplying by h_i and is checked
directly from exact binary W/e with outward interval arithmetic. It is not
certified against the old linear equation. The positive-inverse residual bound
and actual returned first-five ordering are unchanged in strength. Exact
transposition symmetries of W/e preserve q too, so the existing sufficient tie
proof remains valid; unresolved ties are rejected, not tolerance-merged.

## Qualitative limits and falsifiable risks

- q is a graph absorption probability, not identity confidence. Several wrong
  paths or a wrongly admitted seed can yield large q and promote an impostor.
- q=0 gives the original indirect linear term after removing its divergent
  additive constant. The implementation uses the expanded energy and never
  divides by q; the model is defined even for generic test graphs beyond the
  production two-hop frontier. No probability floor is needed.
- If B is empty, the exact original equation is recovered. The source vector
  never changes. Common positive scaling of W/e preserves p,q,S and scores.
- The A/B switch at an exactly zero boundary remains: an arbitrarily weak
  new direct edge can still change the fitting rule. This experiment does not
  claim to solve that discontinuity.
- A two-step success model discounts longer excursions. It can help when such
  excursions trap a useful indirect node, but can miss useful long paths and
  reward a short spurious connection. The outer diffusion still has its original
  depth500 and step5; we do not truncate diffusion to two steps.
- q<=1 gives 0<=D_B<=beta/2 I and lambda_min(M2)>=lambda_min(M0). The largest
  eigenvalue can increase; condition number and total runtime need not improve.
- This is a single hypothesis following complete development results. Its
  correctness, provenance and low algebraic cost do not establish novelty or
  ReID benefit. Only the fixed full-population comparison addresses benefit.

## Cost and implementation checks

One sparse matrix-vector product Wp plus O(n) arithmetic adds the diagonal.
Sparse support and normalized smoothing remain unchanged. There is one SPD
linear solve per stage, no auxiliary original solve, nonlinear Newton loop or
matrix square. Extra storage is O(n); runtime includes normal sparse solves,
certification and the sequential outer trajectory.

Float64 outward arithmetic is attempted first; unresolved residual/order uses
24/40/80/120-digit correction with the SAME new matrix. Residual <=1e-8 and
strict ordered first-five are mandatory. The saved-score reader recomputes
q and the new equation, never repairs scores, and shares certificate arithmetic
with production. No ridge, bridge cutting, floors, tolerance relaxation or
numerical tail. Local8192 is a stop condition.

Tests independently construct an absorbing phase chain for 2/q, differentiate
the energy, verify the response identity and scale/all-direct limits, compare
against a 90-digit oracle, exercise sparse correction above the dense cap,
reject the wrong equation and corrupted scores, distinguish an indirect true
tie from a one-ULP perturbation, and read back the actual original rank interface
including indirect nodes, bridges and an incomplete last batch. These are
correctness checks, not additional retrieval/performance qualification.

## References and scope of attribution

1. Zheng, Fan, Han, *Hybrid Regularization of Diffusion Process for Visual
   Re-Ranking*, TIP30 (2021), 3705-3719,
   [DOI](https://doi.org/10.1109/TIP.2021.3064265), Eq8-13 and17-26:
   the parent energy, reciprocal direct target and GMFPT relationship. Restoring
   unnormalized coordinates simply recovers the published GMFPT family; we do
   not rename that as a new contribution.
2. Kim, Zaghen, Suleymanzade, Ryou, Hong, *Revisiting Random Walks for Learning
   on Graphs*, ICLR2025, [official paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cd51b67dcb19db4e9f0022f500076b00-Paper-Conference.pdf),
   section2.2: periodic restarts impose locality. Their learned RWNN,
   cover-time/universality theorems and experiments are not this ReID model.
   We borrow the local restart perspective, not their accuracy claims.
3. Janson and Peres, *Hitting Times for Random Walks with Restarts*,
   [SIAM journal source](https://doi.org/10.1137/100796352): restart policies
   change first-passage quantities. Our two-step renewal identity is derived
   above for this local finite chain; we do not use their optimal-policy or
   infinite-lattice asymptotic theorem as its proof.

The particular indirect quadratic fitting formula is our adaptation. Reading
the cited sections does not establish an exhaustive novelty search.
