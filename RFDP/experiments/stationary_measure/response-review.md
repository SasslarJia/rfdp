# Degree response and the condition for an early-ranking benefit

This is a post-hoc mathematical and mechanism review after the fixed full-population experiment. It does not retroactively change that experiment's objective, evaluation or success criteria. The concrete question is whether using one walk-compatible score coordinate produces a useful change in the competition between correct and incorrect early candidates, at the cost of one original sparse system. A rise in a global metric alone does not answer this question.

## Exact fixed-frontier response

Keep the same W, h, e and b0 as the model README. Put P=H^-1 W, S=H^-1/2 W H^-1/2, M=I-S+alpha diag(p), and N=I-P+alpha diag(p). The original score satisfies M psi=b0; the candidate score satisfies N x=b0. Direct subtraction gives

    N(x-psi) = (P-S)psi,
    d := x-psi = N^-1 (P-S)psi.

N is similar to the SPD M and has a nonnegative inverse under the same anchored-component assumptions. No new solver or inverse is required in production. Each local forcing term is

    f_i = sum_j W_ij [1/h_i - 1/sqrt(h_i h_j)] psi_j.

Because psi_j>0, a higher-degree neighbor gives a positive term and a lower-degree neighbor gives a negative term. Both signs can occur, and N^-1 propagates their mixture. Consequently, the change is not universally degree-bias removal, nor a monotone promotion of correct matches. It vanishes when h is constant on each connected component. A nodewise sign conclusion requires a condition on the full propagated forcing, not just its own degree. Incorrect dense regions are a specific competing explanation.

For smaller-is-better scores, a particular correct node g outranks an incorrect node j precisely when

    x_j-x_g = (psi_j-psi_g) + (d_j-d_g) > 0.

Thus a sufficient first-place condition within a given eligible candidate set is that one correct g satisfies this inequality for every incorrect j. This is a conditional margin identity, not a theorem that graph degree encodes identity. Full-gallery Rank-1 additionally depends on eligibility and the later adaptive trajectory; a local frontier inequality cannot be presented as a proof of the final retrieval gain. Tests check the response identity, constant-degree limit, pair-gap relation and diagnostic margin semantics against independently solved small matrices.

## Bounded review protocol

Before running this review, freeze this document and diagnose.py in Git. Require the complete accepted 3368-query run and all input hashes. For every query, inspect at most the first five batches and stop at the first ordered selection divergence. Re-solve only the original equation at a still-common candidate frontier; candidate scores are read unchanged from its accepted trace. This is not a new retrieval experiment, parameter search, performance qualification or reconstruction of the full original trajectory.

Record the first frontier's log-degree spread and relative score change; verify the floating response closure; count first-divergence additions/removals, their degree percentiles and true gallery matches. Where the first frontier contains both an eligible positive and negative gallery node, report the minimum-negative minus minimum-positive margin for each method. These descriptive float margins are not production sorting certificates or final full-gallery Rank-1.

Cover all queries, with descriptive groups for AP up/down/unchanged and final Rank-1 gained/lost. The grouping uses completed results and is explicitly post-hoc. Report every group, including the three Rank-1 losses; do not treat selecting six gains as a proof of causality. The review can support a mechanism hypothesis or expose counterexamples; any new general superiority claim requires separately specified evidence. Keep mAP, Rank-1, uncertainty and cost together. Do not start a new formula merely because an interval crosses zero.

## Complete-run evidence

The original fixed run completed computation, paired analysis and full saved-score acceptance with all four exit codes zero. Compute time was 2890.6403 s (48.18 min) with 16 workers and one BLAS thread each; saved-score acceptance was 1055.4053 s (17.59 min), reported separately. The run has 336788 solved batches, no numerical tail and maximum recomputed relative residual 1.02484e-9. These are measured costs for this implementation, not an apples-to-apples speedup against literature timings.

Rank-1 gained six queries and lost three; mAP improved on 304 and decreased on 255. Both aggregate point estimates rise, with paired intervals still spanning zero. The current decision is to keep this candidate and examine its degree-response mechanism. Stable superiority, novelty, and a dedicated Rank-1 advantage are not established by this result.
