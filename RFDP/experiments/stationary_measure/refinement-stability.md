# From exact refinement to a conditional margin bound

The completed feature-transfer results motivate this **post-hoc theoretical question**, not a new accuracy success criterion: how far may a representation depart from the exact mass-preserving refinement in equivalence-and-refinement.md before its coarse score order can change? The following is an application of inverse-positive linear-system perturbation bounds. It is not a claim that node refinement or residual bounds are new concepts.

Let the coarse returned score solve `N x = b`, with `N=I-P+alpha diag(p)` and `b>0`. Let `R` copy each coarse score to its refined nodes. For another anchored, nonnegative graph let `N'=I-P'+alpha diag(p')`, `N' x'=b'>0`. Both systems use the same fixed alpha/beta rule. The refined graph need not obey exact refinement. Every refined component must retain an anchor; otherwise this statement does not apply.

Define row-aggregation error `E=P'R-RP`, boundary-probability error `delta_p=p'-Rp`, and source error `delta_b=b'-Rb`. Direct substitution gives the exact residual

    r = b' - N' R x
      = delta_b + E x - alpha delta_p * (R x),
    x' - R x = N'^(-1) r.

Here `*` is entrywise multiplication. This identity accounts for both propagation and fitting changes. In particular, direct/indirect membership changes can alter b from alpha to beta; dropping delta_b would conceal the original hard switch. Exact mass refinement has all three errors zero, recovering `x'=Rx`.

Since `N'^(-1)` is nonnegative, if a bound `epsilon<1` satisfies `|r_i| <= epsilon b'_i` in every row, then

    |x' - R x| <= epsilon x',
    (R x)/(1+epsilon) <= x' <= (R x)/(1-epsilon).

The first inequality follows by multiplying `|r| <= epsilon b'` by the nonnegative inverse; rearranging it gives the second. A directly checkable sufficient epsilon is `max_i |r_i|/b'_i`. A more conservative bound that avoids cancellation is the maximum of

    (|delta_b| + |E| x + alpha |delta_p| * (R x)) / b'.

For a coarse pair with lower score preferred, `x_g < x_j`, every refined copy of g remains ahead of every copy of j whenever

    (1-epsilon) x_j > (1+epsilon) x_g,

equivalently `epsilon < (x_j-x_g)/(x_j+x_g)`. This is a sufficient pairwise condition, not a necessary condition. It shows why a small relative score perturbation can reverse a near-tied pair while preserving a large-margin pair. No identity labels are required to compute the perturbation bound; labels would only identify which pairwise orders matter for an evaluation.

## What remains to connect to an experiment

This algebra assumes exact coarse scores. Any numerical evaluation must enclose the original solution, residual and epsilon; an ordinary floating difference is not a new strict certificate. Failure of the sufficient inequality means the bound is inconclusive, not that the ranking changes. Near-unanchored components may still produce loose bounds, and a change in direct/indirect membership may dominate the source error.

The relation between two actual feature graphs is not automatically a mass refinement. A justified grouping/perturbation construction must be specified without looking at retrieval outcomes before using this result as a robustness experiment. Rebuilding mutual kNN support, ordinary duplicate photographs and the entire evolving five-node selection trajectory remain outside the guarantee. No parameter choice, production change, new full retrieval run or robustness result is introduced here. Next, check this new implication on exact rational examples and counterexamples, then decide whether a meaningful real-input perturbation protocol follows; do not rerun the earlier seven exact-refinement checks merely to increase a test count.
