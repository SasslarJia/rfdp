# Target-preserving boundary-mass fitting

Status: the fixed Market experiment completed and passed full read-back at
2026-10-29 18:02:32 +08:00, with 16 workers x BLAS1, execution commit
`5fa4044b9c11e9a717e70b7a0a1e44d845159530`. All four stages exited zero.
mAP is 93.132349%, Rank-1 95.190024%; differences from original I-HyRDP are
-0.002553 and +0.029691 percentage points. Both paired mAP confidence intervals
include zero. This fixed coefficient form has no mAP point gain and is stopped;
it is not classified as significantly harmful. [Complete result](result.json).
Computation took 47.54 minutes, full score/trajectory read-back 18.15 minutes.
The production engine and exact-score reader are shared
with `../continuous_fitting`; model key: `boundary_odds`.

## Why this change follows the result

Continuous fitting reduced mAP by 0.111473 pp and Rank-1 by 0.207838 pp.
The all-query, common-prefix audit in `../continuous_fitting/diagnosis.json`
recomputed the original equation for at most five batches, stopping at the first
ordered difference. It did not reconstruct complete original trajectories.
94.65% of first-batch direct-node probabilities were below 0.1; their mean
source changed from 5 to 24.6571. Of 3368 queries, 3123 diverged in those first
five batches, including 1692 with different selected sets. At those first
divergences, 1343 cross-camera positives were removed and 1198 added. Order-only
differences also occur and need not change the next frontier. The AP-down group
lost 109 cross-camera positives and added 62; the AP-up group lost 106 and added
138. These counts include same-identity query nodes as well as gallery nodes.
Restricting to evaluation-eligible gallery positives gives 1047 removed versus
1021 added overall, 87 versus 55 in the AP-down group, and 82 versus119 in the
AP-up group. These post-hoc associations support reconsidering the loss of the original
direct-node advantage; they are not a causal attribution of the entire mAP gap.

We stop the continuous-source form. The next hypothesis preserves its parent's
reciprocal target and hard direct/indirect division, and examines the **relative
weight of boundary fitting versus internal smoothing**. It does not change the
input graph, estimate identities, learn boundary edges or restore graph calibration.

## Boundary-mass derivation

For a fixed stage, W is the original symmetric nonnegative local matrix,
h=W1>0 includes original self loops, e is the original seed-boundary mass,
H=diag(h), S=H^(-1/2)WH^(-1/2), and p=e/(h+e). Let A={i:e_i>0}, B=A^c.
HyRDP's direct target is t_i=1/p_i. Its half-energy convention is

    J0(psi) = (1/2) psi^T(I-S)psi
              + (alpha/2) sum_A p_i (psi_i-t_i)^2
              - beta sum_B psi_i + constant.

Internal smoothing compares z_i=psi_i/sqrt(h_i). Give each boundary edge its
uncompressed weight, while retaining the original direct target. For a boundary
edge (i,l), use the target z_i*=t_i/sqrt(h_i) for node i. Summing the quadratic
penalties over the seed boundary yields

    (alpha/2) sum_i sum_{l in seeds} W_il (z_i-z_i*)^2
      = (alpha/2) sum_A (e_i/h_i) (psi_i-t_i)^2.

Thus replace p_i by r_i=e_i/h_i=p_i/(1-p_i) in the fitting curvature:

    Jo = (1/2) psi^T(I-S)psi
         + (alpha/2) sum_A r_i (psi_i-1/p_i)^2 - beta sum_B psi_i,
    Mo = I-S+alpha diag(e/h),
    bo_i = alpha (h_i+e_i)/h_i on A, beta on B,
    grad Jo = Mo psi-bo,           psi_o=Mo^(-1)bo.

This uses boundary/internal mass rather than boundary/total mass. The reference
values z_i* depend on i: this is a soft fidelity model retaining HyRDP's target,
**not** one common Dirichlet value for every seed, nor a derived exact hitting
time of the original walk. The modeling choice is the boundary penalty; the
displayed linear system then follows exactly. p and r are graph quantities,
not estimated identity probabilities. No new strength parameter is introduced.

## Solvability, interpretation and limitations

For any v, v^T Mo v = (1/2) sum_ij W_ij(v_i/sqrt(h_i)-v_j/sqrt(h_j))^2
+ alpha sum_i(e_i/h_i)v_i^2. If each component has a positive boundary anchor,
Mo is SPD with nonpositive off-diagonals. It is a nonsingular M-matrix;
bo>0 implies the unique solution is strictly positive. No numerical ridge is
used. An unanchored component is rejected.

The exact fixed-stage comparison with the original solution is

    Mo-M0 = alpha diag(p^2/(1-p)),
    bo-b0 = alpha p/(1-p) on A, zero on B,
    psi_o-psi_0 = Mo^(-1) D (t-psi_0), D=alpha diag(p^2/(1-p)).

The correction pulls toward the **existing** direct target. It is not uniformly
positive or negative because t-psi_0 need not have one sign. A node with strong
seed-boundary mass receives more fitting weight; if the seed or edge is wrong,
that can also increase its influence. This mechanism has no identity-purity
guarantee. It might still lose accuracy, or be redundant with the original fit.

For p near zero, the source changes by alpha*p+O(p^2), compared with the previous
candidate's nearly beta-alpha change. The curvature increment is O(p^2).
This is a coefficient expansion, not a uniform small solution-error theorem:
weak anchors can still make the inverse large. The original alpha/beta jump
at e=0 is intentionally retained. For p approaching one, curvature grows and
the node is increasingly fitted toward 1/p; the p=1 endpoint is excluded by h>0.
Strong anchoring can increase the largest eigenvalue, so a better condition
number or faster end-to-end runtime is not guaranteed. Mo-M0 is PSD, hence the
smallest eigenvalue cannot decrease on the same fixed stage.

Common positive scaling of W and e leaves S, r, p, Mo and bo unchanged.
When every node has the same degree and the same positive p, constant scores
equal to 1/p solve both equations, so there is no ranking change. Heterogeneous
systems can change rankings. The original outer seed updates and bridge rules
remain unchanged, but different selected nodes can change later systems.

## Cost and numerical certificate

Assembly adds O(n) diagonal/source arithmetic and preserves the sparse support.
Each stage solves one SPD linear equation: there is no baseline scale solve,
Newton loop, graph augmentation or matrix function. Runtime still includes the
original sequential stages, possible sparse fill-in and certificate work.

In z coordinates the checked equation is

    [L(W)+alpha diag(e)] z = sqrt(h)*bo.

The certificate encloses exact binary W/e, h and sqrt(h) with outward arithmetic
and evaluates this new equation at the actual returned MP scores. With the
positive inverse and |residual_i| <= eta*bo_i, eta<1, the true scores lie between
returned_score/(1+eta) and returned_score/(1-eta). Strict ordered first-five
separation and the normalized residual <=1e-8 remain required. True ties require
exact exchange symmetry. Precision increases only when needed; sparse residual
correction uses **Mo**, not the previous M0. Saved-score read-back never repairs
scores and shares the interval implementation with production.

## Sources and attribution

- Zheng, Fan and Han, *Hybrid Regularization of Diffusion Process for Visual
  Re-Ranking*, IEEE TIP 30 (2021), 3705–3719,
  [DOI 10.1109/TIP.2021.3064265](https://doi.org/10.1109/TIP.2021.3064265):
  parent mixed fitting and reciprocal target. The present coefficient is our
  modification, not a formula attributed to that paper.
- Zhu, Ghahramani and Lafferty, *Semi-Supervised Learning Using Gaussian Fields
  and Harmonic Functions*, ICML 2003, 912–919,
  [author PDF](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf), section 2:
  weighted quadratic graph energy and boundary partition explain why boundary
  edges contribute their summed mass to a diagonal. Our node-dependent soft
  targets and ReID objective differ from their harmonic label propagation.
- Calder and Slepcev, *Properly-weighted graph Laplacian for semi-supervised
  learning*, [2019 manuscript](https://arxiv.org/abs/1810.04351): boundary
  weighting can affect well-posedness at low label rates. Their asymptotic
  theorem and spatial weighting are not a proof or implementation of our odds
  coefficient. This reference is a limitation/background connection only.

The master's thesis Chapter 4 motivates preserving quadratic algebra, not this
particular coefficient. Novelty and retrieval gain remain to be established.

## Fixed empirical decision

Run Market OSNet 3368/19732 once, k20/gamma0.4, alpha5/beta25, step5/depth500,
16 workers with BLAS1. No labels enter graph, equation, or selection. This is a
development-set follow-up chosen after inspecting a previous complete result,
not an independent holdout claim. No strength sweep, second component or Duke.
Report complete mAP/Rank-1, query/identity paired bootstrap10000 seed260626,
all-score acceptance and total time. Positive point differences and stable
gains are separate claims. If this fixed form fails, diagnose the completed
result and reconsider the mechanism; do not silently tune it on partial AP.
