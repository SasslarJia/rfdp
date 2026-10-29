# Continuous hybrid fitting: a quadratic candidate

**Status: selected for one fixed full-population empirical test.** The matrix
assembly, source-specific certified solver, original ranking adapter and complete
saved-score reader are implemented. Thirteen small correctness checks pass,
including the actual producer/reader file boundary; no real-query performance
qualification was added. There is still no complete ReID result for this model.
The prior nonlinear experiment remains a separate paused run.

## Why reconsider the previous change

The starting point is the mixed fitting and smoothness energy behind HyRDP, not
GPU utilization itself. The reviewed master's thesis, Chapter 4, retains a
quadratic energy even when introducing pairwise/tensor-product relationships;
its matrix iteration avoids explicitly constructing the enormous Kronecker
matrix. That does not make its n-by-n pairwise state cheap enough for Market by
itself. We borrow the principle of preserving tractable algebra, not the full
pairwise model or its empirical claims. Equations are rederived below rather
than copying ambiguous overall factors between the thesis's formulas.

Our nonlinear variant instead introduces a baseline solve to determine delta,
Newton iterations and nonlinear interval work at every I-HyRDP stage. The cost
review found that accelerating candidate generation alone cannot eliminate most
of this extra structure. We therefore retain the original quadratic smoothness
and examine the mixed fitting term.

## A concrete feature of the original stage equation

Fix one stage's symmetric nonnegative local W and boundary weights e. Let

    h = W 1 > 0, H = diag(h), p_i = e_i / (h_i + e_i),
    S = H^(-1/2) W H^(-1/2), M = I - S + alpha diag(p).

Here p is the one-step transition probability from a local node to the current
seed set in the stage graph. It is not an estimated probability of matching the
query identity. The original source is

    b0_i = alpha if e_i > 0, otherwise beta,
    M psi0 = b0.

For p_i > 0, the accurate fitting term, under a common half-energy convention,
is alpha p_i (psi_i - 1/p_i)^2 / 2. For p_i = 0, the original indirect fitting
uses beta |psi_i - xi| with xi above the solution. After dropping constants,
these produce the stated b0. For a fixed local vertex set and another anchor
remaining in the component, letting one e_i tend to zero changes M continuously
but switches b0_i from alpha to beta at zero. With alpha=5 and beta=25 this is a
finite jump caused by an arbitrarily weak direct connection. This is a precise
stage-equation observation, not proof that the original retrieval is defective;
the outer frontier can also change when an edge disappears.

## Proposed fitting energy and closed form

Replace the hard A/B fitting assignment by the following continuous mixture:

    Jc(psi) = 1/2 psi^T (I-S) psi
            + alpha/2 sum_i p_i (psi_i - 1)^2
            - beta sum_i (1-p_i) psi_i.

Equivalently, the last term is beta sum_i (1-p_i)(xi-psi_i), up to a constant.
It can be written as a weighted L1 rough-target term if xi is above the solution;
that interpretation is conditional, not an assumption that an arbitrary fixed
xi works for every graph. The affine energy above is the primary definition and
requires no xi parameter or extra constraint.

The direct target 1 represents one step to an absorbing seed with zero distance.
Using that target in this mixed, normalized energy is a modeling choice: this
is neither the original reciprocal-probability target nor an exact expected
hitting-time equation. It changes both the direct target and the indirect mixing
weight within one fitting component. It leaves W, smoothness and the quadratic
fitting coefficient alpha p unchanged.

Differentiation gives

    grad Jc = M psi - bc,
    bc = alpha p + beta (1-p),
    psi_c = M^(-1) bc.

Thus the **matrix is exactly the original M** at a fixed stage. The change is
O(n) right-hand-side arithmetic with one linear solve. There is no scale solve,
Newton loop, learned parameter, identity/camera input, added edge or ridge.
alpha=5 and beta=25 are retained as the candidate defaults; no new tunable
strength is introduced. A full I-HyRDP adapter would preserve seed expansion,
bridging, step=5 and depth=500.

## Existence, positivity and what the mechanism can change

For any vector v,

    v^T M v = 1/2 sum_ij W_ij (v_i/sqrt(h_i)-v_j/sqrt(h_j))^2
              + alpha sum_i p_i v_i^2.

If every connected component has an anchor with e_i>0, this expression is
positive for nonzero v. M is symmetric positive definite with nonpositive
off-diagonal entries, hence a nonsingular M-matrix. Its inverse is nonnegative;
bc is strictly positive, so the unique solution is strictly positive. An
unanchored component is rejected; no small diagonal correction is added.

At a fixed graph and surviving anchors, bc and the solution are continuous in
e. For alpha < beta, stronger seed-boundary probability reduces the source and
increases anchoring. Their combined effect may better distinguish strong and
weak direct connections. However, smoothness couples nodes and changes of e
also change M; this is not a proof of improved identity ranking or monotonic
ranking for every node.

The candidate could fail because weak direct links may be genuinely useful;
replacing the original reciprocal target could discard useful prior structure.
It also keeps degree normalization and the expensive original outer expansion.
There is no guarantee of a gain over HyRDP or complementarity with CSW.
Useful exact controls are:

- alpha=beta: original and continuous sources are identical on every stage.
- All nodes directly anchored with the same p: both sources are constant and
  their solutions differ only by a positive scalar, so rankings coincide.
- Common positive rescaling of W and e leaves S, p, M and bc unchanged.
- Equal local degrees and every node directly anchored: rankings coincide even
  when p varies across nodes. Since S 1=1, M 1=alpha p and b0=alpha 1,
  psi_c=(beta/alpha) psi0 + ((alpha-beta)/alpha) 1. This is a positive affine
  transformation, not a new ranking. The assertion is stage-local; it does not
  assert that all stages of a real query satisfy these conditions.

A nonuniform-boundary small-graph check demonstrates that ranking can change;
this candidate is not merely a renamed global score rescaling.

## Mechanism audit and remaining qualitative questions

For beta>alpha, the exact fixed-stage difference is

    bc-b0 = (beta-alpha) (1-p) 1_{e>0},
    psi_c-psi0 = M^(-1) [(beta-alpha) (1-p) 1_{e>0}] >= 0.

The direct source increment is largest for weakly connected A nodes and is then
propagated through M's inverse. This clarifies what is being changed: the
original direct-node advantage is weakened continuously. It is not an estimate
of which edges have the wrong identity. A weak true match can lose exactly the
advantage we want to remove from a weak false match. Degree heterogeneity and
the presence of B nodes can be important to where ranking changes arise; they
are not themselves evidence of a useful new mechanism.

With W fixed, differentiating with respect to one p_j gives

    d psi_c / d p_j = M^(-1) unit_j [alpha-beta-alpha psi_c,j].

For beta>=alpha this is componentwise nonpositive. Increasing one seed-boundary
probability lowers all affected dissimilarities, not necessarily that node's
rank relative to another. The algebra does not establish better top-five
selection or improved identity separation.

Continuity is local to anchored systems. As the last anchor becomes weak,
lambda_min(M) can approach zero and sensitivity can become arbitrarily large.
Eliminating the source jump is therefore not a uniform robustness bound.

The qualitative review covers the energy, solvability, positivity, invariances,
equivalence cases, source changes and stage cost. It is **not complete** in the
scientific sense needed to select a final method. Still unresolved are a
principled reason to prefer the unit target over the reciprocal target for
ReID, whether gains would exceed a simple degree/source effect, novelty relative
to prior source/fidelity formulations, and behavior under weak true/false links
and seed errors on real data. These mathematical properties do not establish retrieval gain or publication-level novelty. The fixed empirical decision below separates those questions from numerical readiness.

## Why select this hypothesis for a fixed empirical test

A direct local connection is a binary graph property, but a step from node i in
the full stage graph reaches the seed set with probability p_i=e_i/(h_i+e_i).
Assign the original direct/indirect costs alpha and beta to those two one-step
events. Their expectation is uniquely

    E[cost | i] = alpha p_i + beta (1-p_i).

This supplies a precise interpretation of the source replacement, rather than
claiming that continuity alone is beneficial. It is the affine interpolation
with endpoint costs beta and alpha, with no additional parameter. The one-step
event is known from the graph; it is not a true-identity or seed-purity event.

There is also an exact killed-walk representation of the resulting linear
system. Put z=H^(-1/2)psi, P=H^(-1)W, and

    K=diag(1/(1+alpha p)) P,
    g_i=b_i/[sqrt(h_i)(1+alpha p_i)].

Then z=g+Kz. K describes a walk that is killed at node i with probability
alpha p_i/(1+alpha p_i), and otherwise transitions using P. Finite connected
components with a positive anchor give rho(K)<1, so z is the expected accumulated
nonnegative reward g until killing. This is a reward-potential interpretation,
not an assertion that psi equals the GMFPT of the original full graph. Both the
old and new source share K; only the reward changes.

The qualitative decision is to test whether the probability-weighted source
improves the original hard direct/indirect assignment at the same stage-matrix
cost. It can fail on weak true matches and on contaminated seeds; the exact
ranking-equivalence cases above remain relevant. Novelty must still be assessed
against prior fidelity/source formulations before making a publication claim.
The test is justified by the explicit mechanism and bounded added algebra, not
by an assumed accuracy gain. Only one fixed configuration is scheduled; no
strength grid, partial-query AP selection or second component is introduced.

## Numerical implementation and cost boundary

`model.py` assembles the original matrix and either source. `certified.py`
implements a source-specific outward residual enclosure and actual first-five
certificate. Float64 candidates use CG or sparse LU; precision increases only
when the certificate is insufficient. The previous original-matrix refinement
utilities are reused, but their hard-source certificate and tie rule are not.
Simple exact W/e transposition symmetry is a sufficient tie proof; unresolved
symmetries are rejected rather than merged by tolerance. The inherited dense MP
fallback remains limited to 256 nodes. No dense-limit removal is used.

`full.py` uses the unchanged `FullIHyRDP.rank`, stores every actual returned score,
and evaluates identity labels only after all queries complete. `verify.py`
reconstructs the frontiers, incremental boundaries, bridges and entire gallery
ranking from saved points without solving or correcting them. It shares the
numerical certificate utilities with production; it is independent artifact
read-back, not an independently implemented interval proof. The synthetic
producer/reader check covers bridge events, exact and near ties and damaged
artifact rejection. It is not a performance qualification.

For example, let r=bc-M psi_hat, eta=max_i |r_i|/bc_i < 1, with validated outward
arithmetic. Inverse positivity implies

    (1-eta) psi_c <= psi_hat <= (1+eta) psi_c,
    psi_hat/(1+eta) <= psi_c <= psi_hat/(1-eta).

Those enclosures can certify the actual selected first five; unresolved cases
still need refinement and exact ties still need a symmetry proof for the new
coefficients. The production threshold remains 1e-8. Ordinary float64 residuals
in these unit checks are not a substitute for that certificate.

At identical W/e the matrix sparsity, eigenvalues, condition number and direct
factorization cost are identical to the original. A CG iteration has the same
matrix-vector work, but the changed right-hand side can affect its convergence.
Different rankings may change subsequent frontiers. Consequently we claim no
measured end-to-end speed or equal complete-query time. The design avoids adding
a nested nonlinear optimization to the already costly outer loop.

A GPU linear backend is compatible with this structure, but would still need
transfer/batching decisions and numerical checks. Matrices differ across stages
and queries; they cannot simply share one factorization. GPU use is an optional
implementation of a sound model, not the justification for the model.

## What Cheb-GR contributes to this assessment

Cheb-GR uses **Chebyshev's inequality**, not a Chebyshev polynomial approximation
to our inverse. Its per-row distance threshold replaces KNN sorting, then a
small number of GPU graph propagation steps is applied. It also starts from
query/gallery distance relationships, so the distinction is not “it computes
neighbours, while I-HyRDP computes every gallery node on every iteration.”
I-HyRDP constructs dynamic local frontiers and repeatedly solves them.

Our k20 graph is already prepared; replacing its construction cannot remove the
many subsequent solves. Transplanting Cheb-GR's threshold would change the input
graph, while replacing 100 expansion stages by three propagation steps would
change the method. Neither is an equivalent acceleration of the fixed model.
The useful design lesson is to keep operations sparse, regular and parallelizable.
A speed-only contribution is not selected here. Accuracy and cost must both be
reported honestly, even when ECN, KR or Cheb-GR are faster.

## Sources and attribution

1. Zheng et al., **Hybrid Regularization of Diffusion Process for Visual
   Re-Ranking**, IEEE TIP 30, 3705–3719, 2021.
   [DOI](https://doi.org/10.1109/TIP.2021.3064265).
   Original mixed energy and stage equation; reviewed with the local full text.
2. User-supplied *4-19-硕士论文.pdf*, Chapter 4, especially the pairwise quadratic
   formulation and matrix-form iteration (printed pp. 51–53), and the supplied
   chapter-analysis note. This is conceptual background, not a claim that the
   continuous source formula appears in that thesis.
3. Yang et al., **Graph Neural Networks Inspired by Classical Iterative
   Algorithms** (TWIRLS), ICML 2021.
   [Primary paper](https://proceedings.mlr.press/v139/yang21g.html).
   Energy-based propagation and robust iterative algorithms; not evidence that
   exact nested nonlinear optimization is appropriate for I-HyRDP.
4. Choi et al., **GREAD: Graph Neural Reaction-Diffusion Networks**, ICML 2023.
   [Primary paper](https://proceedings.mlr.press/v202/choi23a.html).
   Source/reaction terms offer an alternative to changing smoothness. Its neural
   architecture and source formula are different from this proposal; no GREAD
   accuracy result is transferred to ReID.
5. Yang et al., **Cheb-GR: Rethinking K-nearest Neighbor Search in Re-ranking for
   Person Re-identification**, CVPR 2025.
   [Primary paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.html).
   The threshold and computational structure were reviewed, including the GPU
   timing protocol; this proposal does not claim to implement Cheb-GR.

The particular continuous fitting formula is our proposed adaptation. These
sources motivate the question and algebraic constraints; none proves its ReID
benefit or establishes that this specific formula is novel.
