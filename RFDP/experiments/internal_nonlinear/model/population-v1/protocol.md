# Internal nonlinear smoothing: formal population

User authorizes immediate formal experiment with 16 workers and BLAS1. No additional performance qualification or parameter search is scheduled. Use the existing internal objective at theta=.5 with each stage delta fixed from the original exact hybrid solution. The graph, original mixed fitting, seed updates, bridges, alpha=5, beta=25, step=5 and depth=500 are unchanged.

All 3368 Market queries and 19732 gallery items run. Actual MP scores, full traces and rankings are saved on S1. Every production solve requires the unchanged nonlinear residual bound <=1e-8 and strict actual front-five certificate. No numerical tail, truncation, ridge, threshold relaxation or label-based qualification selection. Existing fixed-query computed prefix/ranking hashes are checked for the nine overlapping queries; they do not select parameters.

After full computation: evaluate the whole population, paired query/identity 95% intervals with 10000 repetitions and seed260626 against original I-HyRDP, then read back every saved score/trace and independently rebuild all frontiers, bridges and complete rankings. Readback shares the locked mathematical enclosure utilities but does not call the producer solver or alter scores.

Execution: 16 spawned processes, BLAS1, 4GiB per-worker virtual-address cap. Account for shared read-only input mappings once and bound other mappings by the cap; retain RAM70%, GPU75%, disk free max25%/10GiB, and 40GiB total output reservation. Resource stops preserve evidence and do not manufacture rankings. Existing computed-query RSS peak2,538,135,552 bytes times1.5 fits within the chosen worker cap. This is an operational budget, not a guarantee on unseen-query allocations. No performance benchmark is launched.

Storage uses the server after explicitly authorized cleanup. Elements is fallback only. Baseline artifacts, raw features, weights and current dependencies remain protected. No second component or automatic Duke migration.
