# What this result supports

The fixed degree-weighted objective completed all 3368 Market queries and full saved-score acceptance. mAP is 93.174274% (+0.039372 pp), Rank-1 is 95.249406% (+0.089074 pp). All query- and identity-paired 95% intervals cross zero. Retain the model as a point-gain candidate; neither stable superiority nor a new Rank-1-oriented contribution follows from those two numbers alone. The compute time of 48.18 minutes with 16 workers restores practical feasibility relative to the suspended nonlinear experiment, but no equivalent-protocol speedup against the original parent or published fast re-rankers has been established.

## Mechanism evidence and counterexamples

The accepted scores obey the fixed-frontier response identity in response-review.md. A bounded, post-hoc comparison over all queries stops at the first ordered divergence or after five common batches. It does not replay either complete trajectory. The maximum floating closure error is 8.259e-13; this is an explanatory consistency check, separate from the full numerical acceptance.

There are 3127 ordered divergences within the first five batches, of which 1833 change the selected set. Of 2279 added node occurrences, the median within-frontier degree percentile is 92.21%; the corresponding 2279 removed occurrences have median 59.62%. Thus the evidence points to a pronounced preference for higher-degree nodes, rather than a generic elimination of density effects. Added true gallery matches number 1095 versus 1381 removed. These are first-divergence occurrences, not final recall: query nodes and deferred gallery matches can influence later propagation. In particular this comparison does not support a claim of universally purer seed expansion.

For the 3075 first frontiers containing both an eligible positive and negative gallery node, the local minimum-negative minus minimum-positive margin changes from negative to positive nine times and from positive to negative twice. All six final Rank-1 gains have a negative-to-positive local change and diverge in the first batch. Three of those gains change only the order within the first selected set, showing that changed early order can matter without changed seed membership.

All three final Rank-1 losses also diverge in the first batch. Two have a positive-to-negative local margin change. Query 640 has a negative local margin under both methods although its final original Rank-1 is correct. This is an explicit limit of the local diagnostic: the full original algorithm's seed handling, eligibility and later order cannot be replaced by a minimum score over the inspected frontier. The six gains and three losses are post-hoc groups; their association is not a causal proof or a statistically stable Rank-1 advantage. All-group statistics and both signs are retained in diagnosis.json.

## Research decision and next question

Keep the current fixed candidate. Its single-system cost and coordinate/measure interpretation are coherent, and both development point estimates improve. Do not claim novelty for degree weighting, claim identity confidence from degree, or rebrand the result as a preplanned Rank-1 theorem. Do not begin another coefficient sweep or add a compensating module.

The complete coordinate audit is now in equivalence-and-refinement.md. The current equation differs from the original normalized score and GMFPT, but belongs to an established degree-weighted hybrid family. In the same returned coordinate the smoothness and fitting measure change together. A concrete consistency property is proved: when a relation region is split under the specified mass-preserving rule, all coarse scores and margins are unchanged. An exact example reverses the original solver's order for two untouched nodes. GMFPT shares the property; ordinary duplicate photos and the whole five-at-a-time iterative trajectory are outside the proposition. Seven new exact rational checks pass.

This gives a more precise rationale than equating high degree with identity confidence. It leaves the empirical sign of the response term unresolved on general ReID graphs. The fixed feature-transfer validation below addresses that empirical question separately from the refinement theorem. The original OSNet protocol/results are not reclassified. A confidence interval spanning zero alone is not the decision rule.

## Completed fixed feature transfer

Market TransReID completed all 3368 queries and full acceptance at 23:39:12 on 29 October 2026, with all four exits zero. mAP is 94.537030% against 94.530890% for the same-feature original (+0.006139 pp); Rank-1 is 95.190024% against 95.279097% (-0.089074 pp). Query-paired 95% intervals are [-0.106263, 0.125978] pp for mAP and [-0.326603, 0.118765] pp for Rank-1; identity-paired intervals are [-0.111219, 0.128587] and [-0.298688, 0.118694] pp respectively. There are 211 AP gains, 234 decreases, 2923 unchanged queries, six Rank-1 gains and nine losses. Compute takes 43.85 minutes and score-readback acceptance 16.14 minutes on 16 workers; cache recovery is recorded separately.

The mAP point-change direction agrees across the two representations; the Rank-1 direction does not. These are two feature graphs on the same dataset, with uncertainty spanning zero in both. The evidence supports retaining a computationally practical, mathematically consistent candidate, but does not establish stable retrieval superiority, representation-independent first-match improvement, or duplicate-photo robustness. The mass-refinement proposition remains an exact statement under its prescribed transformation. Its practical relevance still requires a bridge from that transformation to observed representation changes.

The next bounded review is specified in transfer-diagnosis-protocol.md. It tests whether the higher-degree preference and early positive/negative margin associations observed on OSNet explain the changed signs on TransReID. All queries and both outcome directions are included; the fixed model will not be tuned in response. No additional full retrieval run is queued.

## Evidence

Full execution commit: 3ddf0520db18852401c636620147bd837754db0f. Full acceptance SHA256: 6415b36dea04eb2ab44de0a50877c37d33cc395cb750d484d62cb1df6f91aa3d. All four exit codes are zero; result.json records the full paired intervals and costs.

Diagnostic code/protocol frozen at 090a046266c5fe32d94f56c5db3e4d6ebcd98c12, two new response/margin correctness checks passed on S1. Diagnosis exit0, summary SHA256 48fa4ae2e4006b84da3a637e3e6137437390dbaa4b14297b6bf7f48fea263152. Raw query details remain at S1 `/root/autodl-tmp/rfdp-graph-routes-20261026/stationary-measure-diagnosis/queries.json`, SHA256 a5d2dab8b63835c6956513d079e4d9eb5f4940f7fdc1992324c7dc6ec12ba38f. No candidate query was rerun or selected by partial AP. The frozen formal source, all raw candidate traces and completed metadata remain intact.

TransReID execution remains frozen at 72d42e24ba15ea38094af4130a271419c1198a75. Acceptance SHA256 9766b716edcaa4fd5bc3f5839314ebe85af69739edec3487c0434e2e96870c97; complete statistics, input bindings and cost are in transfer-result.json. All original TransReID scores and traces remain on S1.
