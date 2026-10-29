# Coordinate audit and mass-preserving refinement

This review follows the completed OSNet Market experiment. The refinement question below is a new theoretical analysis, not its preregistered success criterion. It does not change the frozen model, original protocol, results or production solver.

## Three equations in their returned coordinates

Let H=diag(h), h=W1>0, E=diag(e), D=H+E, p=e/(h+e), L=H-W. Every component has an e>0 anchor. Set S=H^-1/2 W H^-1/2 and b0_i=alpha on direct nodes, beta otherwise.

| Form | Equation | Returned coordinate |
|---|---|---|
| Original HyRDP, Eq. (12)/(13) | [I-S+alpha diag(p)] psi=b0 | psi |
| GMFPT, Eq. (20)/(26) | (L+E) mu=(H+E)1 | mu |
| Current adaptation | [L+alpha diag(hp)] x=Hb0 | x |

Original HyRDP smoothness is psi^T(I-S)psi/2 and its direct target is 1/p. GMFPT smoothness is mu^T L mu/2, its direct fitting curvature is e and its indirect source is d=h on indirect nodes. Our direct curvature is alpha hp=alpha e h/(h+e); its indirect source is beta h. Thus it is a particular spatial weighting of the unnormalized hybrid family, not GMFPT itself, including when alpha=beta=1. The author's family-level comparison of GMFPT and HyRDP does not imply equality of these three returned scores.

With y=sqrt(H)x, the candidate has exactly the original matrix M0=I-S+alpha diag(p), but right side sqrt(H)b0. In y coordinates the direct target becomes sqrt(h)/p and the indirect source becomes beta sqrt(h). In the common returned x/psi coordinate, both the smoothness coordinate and fitting measure change coherently. Calling this merely multiplication of the old fitting terms by h while keeping the returned-coordinate smoothness unchanged would be incorrect. No production formula needs changing: the implementation already solves and certifies the stated x equation.

The original PDF was visually checked at journal pp. 3707–3709 (PDF pp. 3–5), Eqs. (8)–(16), (20), (22)–(26). The displayed edge sums in (8)/(22) and matrix halves have a factor-of-two ambiguity if sum(i,j) is read as all ordered edges. We use the explicit matrix energies, gradients and solutions in (12)/(26), consistent with the frozen implementation; this is not a reason to alter alpha. Local PDF SHA256: 272c979486fab2e9b13b44a2736feb2b2c0ab97238c16878c714874fa338ba26.

## A specific representational question

Can a relation region be split into equivalent smaller representatives without changing its total interaction mass, yet change dissimilarities elsewhere? This is a conditional consistency question, distinct from saying that dense regions have more reliable identities.

Replace node i by copies (i,a) with t_ia>0 and sum_a t_ia=1. Define the precise mass refinement

    W'_(ia,jb) = t_ia t_jb W_ij,    e'_(ia)=t_ia e_i.

All entries, including diagonal mass, follow this rule. Then h'_(ia)=t_ia h_i, p'_(ia)=p_i, and direct/indirect membership is unchanged. Let R lift a vector by (Rx)_(ia)=x_i. Put P=H^-1 W, N=I-P+alpha diag(p), and define primed matrices from the refined graph.

**Proposition.** The unique candidate solution obeys x'=Rx, so every coarse pairwise score difference is preserved.

**Proof.** For each coarse group j,

    sum_b P'_(ia,jb)=sum_b t_jb W_ij/h_i=P_ij.

Therefore P'R=RP, diag(p')R=Rdiag(p), N'R=RN and b0'=Rb0. N'x'=b0' has a unique solution: the symmetric system L'+alpha diag(h'p') is positive definite because every refined component retains a reachable anchor. Since N'Rx=RNx=Rb0, uniqueness gives x'=Rx. On the lifted subspace the quadratic energy and both degree-weighted fitting terms also equal their coarse values. This energy equality alone is not the uniqueness argument. QED.

GMFPT also satisfies this property: D'^-1 W' R=R D^-1W and its row right side is 1. Consequently this proposition distinguishes the current candidate from the original normalized-score HyRDP convention, not from all older diffusion methods.

## Exact counterexample for the original convention

Take alpha=5, beta=25 and

    W=[[2,1,1],[1,3,0],[1,0,3]],  e=(12,1/2,0).

Every coarse h is 4, so original and candidate scores agree: (732,792,9832)/91. Split only node 2 into four equal pieces with t=1/4. The original normalized solver then returns (1312,972,9756,9756,9756,9756)/91. The untouched nodes 0 and 1 reverse order. The candidate returns (732,792,9832,9832,9832,9832)/91 and preserves their interval. These are exact rational results without identity labels or numerical tolerance. They establish a possible representation-induced inversion, not an observed ReID error.

## Scope and value

The concrete contribution candidate is to retain HyRDP's direct target and hybrid source while making the returned dissimilarity consistent with a degree measure and the refinement above, using one original-size sparse SPD solve. The invariant is an established type of graph consistency, not a new mathematical concept. The hp coefficient and its use in this iterative retrieval framework remain a project adaptation; a first-in-literature claim is not established by this audit.

Ordinary duplicated photos, rebuilding a mutual-k20 graph, or sampling more images do not generally implement W'=t_i t_j W and e'=t_i e. Nor does this theorem preserve the complete iterative algorithm: selecting five nodes and stopping at 500 counts copies, and graph construction/bridges can change. Unequal-mass aliases may have equal mathematical scores without the exact exchange symmetry currently required by the production tie certificate. This proof therefore does not change that tie policy or assert a new production interface on arbitrary refined graphs.

For empirical early ranking, the exact response in response-review.md remains the relevant condition: d=N^-1(P-S)psi and the correct/incorrect margin changes by d_j-d_g. Refinement consistency does not determine the sign of that term on an ordinary ReID graph. The completed OSNet point gain and observed dense-node preference remain separate empirical observations. Dedicated top-1 superiority and realistic duplicate-image robustness are not established.

Seven dependency-free Fraction-based checks in Test_RFDP/test_stationary_refinement.py cover uniform and unequal refinement, operator intertwining/mass, the original inversion, GMFPT's shared property, alpha=beta=1 non-equivalence, and a counterexample to naive photo-duplication invariance. They are algebra checks, not a new query/performance qualification.

## Sources and reading scope

* Zheng, Fan, Han, IEEE TIP 30:3705–3719 (2021), DOI [10.1109/TIP.2021.3064265](https://doi.org/10.1109/TIP.2021.3064265). Original normalized/unnormalized energies and returned equations read from the local full PDF as specified above.
* Heitzig et al., *Node-weighted measures for complex networks with spatially embedded, sampled, or differently sized nodes*, EPJ B 85,38 (2012), [author manuscript](https://arxiv.org/pdf/1101.4757), §4.3, Eqs. (9)–(11), pp. 8–9, read in full for the splitting definition. Their node-weighted simple-graph construction motivates the consistency question; it is not identical to our weighted-edge factorization. The proof above is supplied directly.
* Buchholz, *Exact and Ordinary Lumpability in Finite Markov Chains*, Journal of Applied Probability 31(1):59–75 (1994), [publisher](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/exact-and-ordinary-lumpability-in-finite-markov-chains/2DC748F09D80BEEB03CCF18036E149D7), DOI 10.2307/3215235. Publisher abstract/metadata only; used to identify the established aggregation concept, not to assert an unread theorem or novelty priority. The matrix identity P'R=RP is proved here.
