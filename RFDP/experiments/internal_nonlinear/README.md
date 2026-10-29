# I-HyRDP internal nonlinear smoothing

This directory on GitHub branch `test` is the source of truth for the current experiment. Edit and commit code here; deploy a recorded commit to the server. Do not create another dated local source/log directory for each change. Runtime traces, features, rankings, progress and failure evidence remain on the server and are excluded from Git. Keep only a compact current-state pointer outside the repository.

## Current implementation

`model/adaptive-precision-v1/` implements the same strongly convex nonlinear edge smoothing objective with adaptive interval precision. Production tries 24, 40, 80, then 120 decimal digits; the saved-score reader escalates independently. An inconclusive certificate increases precision. It never relaxes the residual threshold or accepts uncertified selections.

`model/population-adaptive-v2/` resumes completed query artifacts from the original population run, verifies their hashes, and computes only pending queries. The original scientific signature is retained; the arithmetic revision has a separate source lock. `model/population-v1/` and the original certificate/adapter are retained as the same model's reference implementation and migration dependency. They are not alternative research directions.

The model uses theta=0.5; each stage's delta is the original hybrid solution's weighted RMS edge difference, enclosed by interval arithmetic and held fixed during that solve. Original graph, fitting, seed and bridge rules remain unchanged. See [derivation](model/theory-explanation-20261028.md). Market settings are alpha=5, beta=25, step=5, depth=500, k=20, gamma=0.4, 3368 queries and 19732 gallery images. Production uses 16 workers and one BLAS thread each.

The current server deployment's adaptive lock is `ef6dba0cfb638fa24755e72ebefb760f63b69b56f4d2300591679e4f56581bb0`. Its nine source/config files and seven dependencies are copied byte-for-byte here. Existing repository core files also match all deployed reference hashes. The test harness below uses repository-relative paths.

## Correctness checks

Install NumPy, SciPy and mpmath 1.3.0. From the repository root:

```bash
python RFDP/experiments/internal_nonlinear/model/adaptive-precision-v1/check_adaptive.py
python RFDP/experiments/internal_nonlinear/model/population-adaptive-v2/check_resume.py
```

These check certified selections, exact and near ties, precision escalation, original outer diffusion trajectories and completed-query reuse. They are small correctness checks, not a retrieval performance qualification. Real saved stages at sizes 68, 2267 and 4316 also retained their original selections, with the adaptive scores accepted by the original 100-digit reader. No complete query was rerun for that check.

## Deployment

The source/config snapshots retain the original deployment paths so the current run is exactly traceable. For another machine, export with explicit paths:

```bash
python RFDP/experiments/internal_nonlinear/prepare_deployment.py \
  --runtime-root /path/to/new/runtime \
  --repo-root /path/to/rfdp \
  --cache-dir /path/to/feature-graph-cache \
  --mat-file /path/to/features.mat \
  --python /path/to/python
```

The exporter does not start a job or overwrite an existing model deployment. Supply the existing baseline summary/per-query data, run manifest, and fixed-query prefix references separately; these are experiment inputs, not source code. `population-adaptive-v2` requires a migration record and cannot silently start over or discard finished queries.

## Timing interpretation

The current algorithm performs roughly 100 sequential local solves per query, about 336,800 stage solves over Market. Each stage includes an original linear solution for delta, nonlinear Newton iterations and certification. This is structurally different from Cheb-GR's three global feature-propagation rounds. Adaptive precision reduces arithmetic overhead, not that stage count. A previously discussed 30–40% reduction was a conditional estimate, not a measured speedup. The current implementation has no efficiency claim against Cheb-GR.

The previous torch CUDA backend targets `ProjectedState.linear`; this nonlinear solver calls SciPy sparse Newton routines and does not use that backend. GPU candidate generation is not enabled by this revision.
