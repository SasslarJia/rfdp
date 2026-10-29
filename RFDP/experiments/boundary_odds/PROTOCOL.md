# Fixed boundary-mass experiment

One complete Market OSNet run, source/model key `boundary_odds`, 3368 queries and
19732 gallery nodes, original k20/gamma0.4 graph, alpha5/beta25, step5/depth500.
The energy and closed form are in README. Freeze this single form before its
outputs: no strength grid, partial AP selection, second component or new dataset.
This is iterative development on Market after a completed prior result.

Use the shared `experiments.continuous_fitting.launch` chain with a fresh output
`market-boundary-odds-fitting-v1`, never overwrite the previous run. Input SHA
hashes and the new Git commit are in the configuration and run manifest.
The previous execution checkout stays at 4cb6f46; deploy the new commit in a
separate worktree. No additional performance qualification is requested.

Execute 16 workers x BLAS1, each with **3.5 GiB address-space stop limit**. Reuse
the completed run's actual allocation evidence, not a new qualification run;
the old paused nonlinear resident memory remains in current cgroup consumption.
Count the read-only 2181754880-byte feature/affinity mappings once, and add
3 GiB for orchestration. Require current+shared+16*(worker_AS-shared)+3 GiB <=
70% of 90 GiB before dispatch, then retain runtime guards. This is a conservative
static allocation budget with a process stop limit, not a no-OOM guarantee.

Reserve a separate **16 GiB** output budget. The previous complete raw run used
11.11 GB. Compression/trajectory size can change, so reaching this hard limit
stops and preserves evidence rather than truncating it. Current free disk minus
the entire remaining reservation must exceed max(25%,10 GiB). GPU <=75%, RAM
<=70%, and both disk floors remain. No automatic deletion or Elements transfer.

Actual-score residual <=1e-8 and strict ordered first-five certificates use
L(W)+alpha diag(e), sqrt(h) and the new source. True ties need exact exchange
symmetry and re-certified equal returned scores. The original graph/seed/bridge
rules remain; local size8192 is a stop limit, not a frontier cap. No ridge,
floors, score repair during read-back, numerical tail or relaxed tolerance.

Run compute -> original-IHy paired analysis -> full saved-score acceptance,
including every frontier, bridge and complete-gallery rank. The artifact reader
shares certificate arithmetic with production. Paired query/identity bootstrap
uses10000 repetitions and seed260626. Report both full metrics, uncertainty,
acceptance status and all timing, regardless of direction. There is no automatic
retry or parameter search. Diagnose any failure before a new execution decision.

Correctness evidence before dispatch: 30 relevant CPU tests passed, including
the independent90-digit equation oracle, sparse refinement above the dense cap,
wrong-equation/damaged-score refusal, exact and one-ULP ties, and the actual
rank/bridge/full-readback fixture. One unrelated external-controls pipeline
smoke test was skipped because its optional dependencies are unavailable.
