# Fixed continuous-fitting experiment

One full Market1501 OSNet comparison with original I-HyRDP: 3368 queries and
19732 gallery images, identical frozen features and mutual k20/gamma0.4 graph.
The candidate changes only the fitting source to alpha*p+beta*(1-p), with
alpha=5, beta=25, p=e/(h+e). The original matrix, seed expansion, step=5,
depth=500, bridge rule and complete-gallery ordering contract remain.

This is a new scientific run. The paused nonlinear run's 626 query results are
not reusable candidate results. No partial candidate AP is evaluated, no new
parameter is tuned and no performance qualification is inserted before this
full run. The theoretical equivalence and failure cases are documented in README.

Execution: 16 workers, one BLAS thread each, 4 GiB address-space limit per worker.
Shared read-only feature/affinity mappings are counted once in the static
allocation budget. Existing paused processes remain included in current cgroup
usage. Before launch, enumerate all argv/cwd and process relations, verify the
Git commit and input SHA hashes, and require current+static budget <=70% RAM,
GPU <=75%, disk free >=max(25%,10 GiB). New output has a **32 GiB** independent
hard budget, fully reserved against current free space. This is an execution
budget, not an assertion that compression must fit; reaching it stops with
all evidence preserved. No automatic cleanup, truncation, retry or Elements
transfer is performed by the runner. The old run is not resumed by this script.

Every actual score vector must meet the exact-binary-input equation's outward
relative residual bound <=1e-8 and certify the ordered first five. Positive
M-matrix inverse bounds and source-specific intervals are used; unresolved
cases get precision refinement. True ties require exact symmetry and equal
actual returned scores, followed by residual re-certification. The 8192 local
size limit stops a run rather than restricting a frontier. No ridge, floors,
bridge removal, numerical completion tail or relaxed threshold.

Full scores/trace, prefixes and gallery permutations are saved on the server.
The chain is compute -> full paired analysis -> saved-score read-back acceptance.
Analysis uses the existing original-IHy baseline and query/identity bootstrap,
10000 repetitions, seed260626. Report mAP, Rank-1, both paired intervals and
execution time. Positive mAP point change and a stable gain are distinct;
acceptance and accuracy classification are also separate. Recompute every
stored stage's equation, selected nodes, bridge events and full ranking. The
reader shares certificate arithmetic, which is explicitly disclosed.

The Git checkout is immutable while executing. Configuration, input hashes and
commit are recorded in run-manifest.json. launch.py writes compute.exit,
analysis.exit, acceptance.exit and total.exit to its server control directory.
Failures retain partial raw evidence and require diagnosis before any new
execution decision. There is no automatic second component or Duke run.
