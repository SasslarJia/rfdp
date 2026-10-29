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

## Why source-support changes need a separate condition

The following is a paper calculation, not a new executed test. Take alpha=5,
beta=25, W=[[1,1],[1,1]], h=(2,2), and e=(2,eta). For eta>0 both nodes are
direct, b=(5,5), and

    N_eta = [[3, -1/2], [-1/2, 1/2 + 5 eta/(2+eta)]].

As eta tends to zero from above, the equations give x2=x1+10 and x1=4:
the solution tends to (4,14). At eta=0, the second source becomes beta=25.
The same limiting matrix now gives x2=x1+50, x1=12, hence x=(12,62).
The first node remains an anchor throughout. Thus existence and uniqueness do
not imply continuous scores across direct/indirect support changes. The score
jump is finite although the boundary-weight change tends to zero. This example
does not reverse the two nodes' order; it disproves unconditional score
continuity. The hard switch is inherited from the original hybrid rule.

For perturbations confined to fixed direct/indirect membership, delta_b=0.
For changes crossing zero it must be retained. This is also why an arbitrarily
small raw edge perturbation is not, by itself, a useful uniform ranking bound.

## From stage scores to an entire ranking

At a fixed vertex set, a sufficient induction requires the same initial seeds,
frontier membership, connected-component and bridge decisions, deterministic
tie rule, and the same **ordered** chosen nodes at every stage. The interval
condition must preserve the relevant internal top-five orders as well as the
selection boundary against all remaining nodes. The small-frontier branch must
retain its ID order, and the original-similarity tail order must also remain.
Under those conditions the seed-update sequence and final ranking follow
identically. This is a conditional implication, not a claim that those
conditions hold under general graph perturbations.

Refinement changes node counts, so a five-node batch and a depth-500 budget no
longer select the same number of coarse groups. Copying scores alone does not
establish this induction. Group-aware selection would be a different algorithm.
Furthermore, the first eligible gallery item is determined after query/junk
filtering and prefix construction; a best-positive/best-negative margin over a
whole first frontier need not determine final Rank-1.

## What remains to connect to an experiment

This algebra assumes exact coarse scores. Any numerical evaluation must enclose the original solution, residual and epsilon; an ordinary floating difference is not a new strict certificate. Failure of the sufficient inequality means the bound is inconclusive, not that the ranking changes. Near-unanchored components may still produce loose bounds, and a change in direct/indirect membership may dominate the source error.

The relation between two actual feature graphs is not automatically a mass refinement. A justified grouping/perturbation construction must be specified without looking at retrieval outcomes before using this result as a robustness experiment. Rebuilding mutual kNN support and ordinary duplicate photographs are not automatically covered. No parameter choice, production change, new full retrieval run or robustness result is introduced here. The user has paused experiments and disabled the recurring task; new checks and targeted validation in [research-summary.md](research-summary.md) remain proposals until experimental work is resumed. Do not rerun the earlier seven exact-refinement checks merely to increase a test count.
