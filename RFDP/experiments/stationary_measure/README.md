# Degree-weighted hybrid fitting in one score coordinate

Execution policy (30 October 2026): the user has paused experimental runs and deferred Duke. Continue theoretical derivation, literature review and argument organization only. Do not launch further retrieval, tuning, diagnostic or performance runs until the user explicitly resumes experiments. Preserve both completed Market populations and the suspended nonlinear evidence.

Status: both fixed Market populations and their bounded mechanism reviews are complete. OSNet mAP/Rank-1 are 93.174274%/95.249406% (+0.039372/+0.089074 pp); TransReID gives 94.537030%/95.190024% (+0.006139/-0.089074 pp). All query- and identity-paired 95% intervals cross zero. See result.json and transfer-result.json. Both representations show a preference for higher-degree nodes, with beneficial and harmful early margin changes. Retain the candidate for its defined consistency mechanism; stable retrieval superiority and a dedicated Rank-1 advantage are not established. No full retrieval job is currently queued.

The experiment key is `stationary_measure`. This is one replacement of the fitting measure/coordinate convention, not a graph change, an additional indirect anchor, or post-hoc score division. The completed [coordinate and refinement audit](equivalence-and-refinement.md) distinguishes it from the original normalized equation and GMFPT, and proves fixed-frontier score consistency under a specified mass-preserving refinement. Seven exact rational checks pass. The completed [feature-transfer protocol](transfer-protocol.md) evaluates the same model on the existing Market TransReID representation; it does not test arbitrary duplicate-photo robustness. The next [conditional perturbation analysis](refinement-stability.md) relates approximate refinement to pairwise margins and still needs its own exact examples and justified practical perturbation protocol.

## Objective and implementation

At a fixed original I-HyRDP frontier, let W be the original symmetric nonnegative internal graph, h=W1>0, H=diag(h), e the original boundary weight, p=e/(h+e), A={e>0}, B={e=0}, and L=H-W. Self loops count in h but cancel in L. Keep alpha=5, beta=25 and the original direct target 1/p. Minimize in the **returned ranking coordinate x**:

    E(x) = (1/2) x^T L x
           + (alpha/2) sum_A h_i p_i (x_i-1/p_i)^2
           - beta sum_B h_i x_i.

The quadratic fitting term and the indirect linear term are integrated against the same degree measure as the reversible walk P=H^-1 W. Differentiating gives

    [L + alpha diag(h p)] x = H b0,
    b0_i = alpha on A, beta on B.

This deliberately changes the original objective. The original normalized-coordinate equation is M0 psi=b0, where M0=I-H^-1/2 W H^-1/2+alpha diag(p). With y=sqrt(H)x the candidate is

    M0 y = sqrt(H) b0,     x=H^-1/2 y.

The symmetric candidate solve therefore uses exactly the original sparse matrix at an identical frontier, with a changed right side and explicit conversion to x. Production certificates and actual sorting check x. Unpreconditioned sparse MP correction uses L+alpha diag(hp) and Hb0. No explicit inverse is formed.

## What follows and what does not

* If every connected component has a boundary anchor, v^T[L+alpha diag(hp)]v>0 for nonzero v. Its off-diagonal entries are nonpositive; its inverse is nonnegative. Positive Hb0 gives a unique positive solution.
* Dividing rows by h gives (I-P+alpha diag(p))x=b0. Define Q_ij=P_ij/(1+alpha p_i), c_i=b0_i/(1+alpha p_i). Then x=c+Qx is the expected accumulated positive cost of an auxiliary killed walk. Anchors reachable from each component make this expectation finite.
* This auxiliary walk is **not** the original GMFPT: the latter solves (L+diag(e))mu=h+e. In particular, even alpha=beta=1 gives killing probability p/(1+p), not p. Neither interpretation proves retrieval improvement.
* Scaling W and e by a common positive factor, even separately within disconnected components, leaves x unchanged. Constant h within each component recovers the original psi exactly. For an isolated direct self-loop node, x=1/p, preserving the original target.
* Naively returning original psi/sqrt(h) is a different rule. For W=diag(1,4), e=(1.125,4), original targets are (17/9,2); division changes them to (17/9,1) and reverses their order. The present objective preserves (17/9,2).
* Degree may encode useful density or confidence. Weighting the fitting term by h may favor incorrect dense regions and underweight correct sparse regions. The direct/indirect split and beta/alpha discontinuity remain. There is no identity-purity assumption, monotonic ranking improvement, or inferred complementarity with CSW.

## Provenance and novelty boundary

Zheng, Fan and Han, *Hybrid Regularization of Diffusion Process for Visual Re-Ranking*, IEEE TIP 30 (2021), 3705–3719, [DOI](https://doi.org/10.1109/TIP.2021.3064265), is the direct source. Its normalized smoothness and hybrid fitting motivate the coordinate audit; its Eqs. (20), (22)–(26) already discuss GMFPT and unnormalized/degree-weighted formulations. Degree weighting or an unnormalized Laplacian is therefore **not claimed as a new concept**. The particular hp curvature, original direct target, original alpha/beta values and returned-coordinate choice here are a project adaptation. Their value requires a full empirical result and further novelty comparison.

Andersen, Chung and Lang, *Local Graph Partitioning using PageRank Vectors* (FOCS 2006), [author paper](https://snap.stanford.edu/class/cs224w-readings/andersen06localgraph.pdf), Sections 2.1 and 2.4, distinguishes walk mass from degree-normalized sweep values. This supports checking the coordinate of a diffusion score; it does not authorize dividing HyRDP dissimilarities by degree or establish a ReID improvement. No formula in that paper is claimed as this candidate's objective.

## Cost and numerical contract

One sparse SPD solve per original solved batch. The preconditioned matrix equals the original M0; the change adds O(n) right-side and output transformations. There is no Newton iteration, extra W@p, learned parameter, or new neighborhood/horizon. At an identical frontier the matrix condition number is unchanged; RHS, stopping behavior, future frontiers and certification costs can change, so whole-query time is not guaranteed.

For actual stored x, independently recompute r=b0-H^-1(Lx+alpha diag(hp)x). Require its relative Euclidean residual <=1e-8 and eta=max_i |r_i|/b0_i<1. Inverse positivity yields componentwise solution bounds x/(1+eta) <= x* <= x/(1-eta). Strict interval separation certifies the actual ordered first five. Equal returned values are permitted only after exact exchange-symmetry proof and re-certification. Binary W/e and derived h,p are outward enclosed; float64 candidates escalate to 24/40/80/120-digit refinement only when necessary. No ridge, edge removal, floor, tolerance merging, relaxed threshold or numerical tail.

The saved-score verifier never re-solves or changes the points. It shares the row-certificate mathematical utilities, so it is operationally separate readback rather than a fully independent proof implementation. Necessary checks include an independent 90-digit solve, independent killed-walk resolvent, energy gradient, scaling/regular-degree limits, sparse correction above the dense threshold, exact ties versus one ULP, wrong-equation/damaged-score rejection and the unchanged actual rank/bridge/readback interface. See Test_RFDP/test_stationary_measure.py and test_continuous_fitting.py.
