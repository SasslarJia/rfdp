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

This gives a more precise rationale than equating high degree with identity confidence. It leaves the empirical sign of the response term unresolved on general ReID graphs. Preserve the fixed candidate and validate its transfer to the already accepted Market TransReID representation using transfer-protocol.md. Formula and parameters remain unchanged. This is empirical transfer on the same dataset, separate from the refinement theorem and from a dedicated Rank-1 superiority claim. The original OSNet protocol/results are not reclassified. A confidence interval spanning zero alone is not the decision rule.

## Evidence

Full execution commit: 3ddf0520db18852401c636620147bd837754db0f. Full acceptance SHA256: 6415b36dea04eb2ab44de0a50877c37d33cc395cb750d484d62cb1df6f91aa3d. All four exit codes are zero; result.json records the full paired intervals and costs.

Diagnostic code/protocol frozen at 090a046266c5fe32d94f56c5db3e4d6ebcd98c12, two new response/margin correctness checks passed on S1. Diagnosis exit0, summary SHA256 48fa4ae2e4006b84da3a637e3e6137437390dbaa4b14297b6bf7f48fea263152. Raw query details remain at S1 `/root/autodl-tmp/rfdp-graph-routes-20261026/stationary-measure-diagnosis/queries.json`, SHA256 a5d2dab8b63835c6956513d079e4d9eb5f4940f7fdc1992324c7dc6ec12ba38f. No candidate query was rerun or selected by partial AP. The frozen formal source, all raw candidate traces and completed metadata remain intact.
