# Fixed two-step fitting experiment

One Market OSNet development run:3368 queries/19732 gallery, original
k20/gamma.4, alpha5/beta25, step5/depth500,16workers x BLAS1.
Model key `two_step_fitting`; source b0 and direct fitting unchanged;
the indirect diagonal is beta*q/2, q_i=sum_j W_ij*p_j/h_i on e_i=0.
The trial length is two because the original indirect frontier is two-hop;
there is no horizon, strength, neighbor or precision-tolerance sweep.
No labels/cameras enter the solver, no second component, no automatic Duke.

Freeze code on GitHub test and deploy the exact commit in a separate S1 worktree.
The first dispatch used `market-two-step-fitting-v1`; initialization failed before
any completed query (see initialization-failure.json). After diagnosing and
fixing the logger directory race, the unchanged scientific definition uses
fresh run `market-two-step-fitting-v2` and `two-step-fitting-v2-config.json`.
The prior full results were used for mechanism selection, so this is continued
Market development, not an independent holdout claim. No new performance
qualification. Necessary new-equation and actual-interface checks precede launch.

Keep16workers/BLAS1 with3.5GiB AS perworker. Count current cgroup including
the old paused nonlinear run; reserve shared readonly inputs once and3GiB
orchestration using the existing static allocation calculation. RAM<=70%,
GPU<=75%, both disks retain max(25%,10GiB). Reserve a fresh16GiB output budget;
on budget failure stop with complete failure evidence, never truncate traces.
Clean only obsolete completed intermediates under the user's standing cleanup
authorization, after checking dependencies and retaining results/provenance.
Server first, Elements only if still necessary.

The launch chain is compute -> paired original-IHy analysis -> full saved-score
read-back. Each actual new-equation score residual<=1e-8 and ordered first-five
certificate is required. Exact symmetric ties only; original frontiers/bridges
and all gallery ranks reconstructed. Report all exits, metrics, query/identity
bootstrap10000 seed260626 intervals and timings. Acceptance and accuracy gains
are separate. No partial AP selection or automatic retry. If complete accuracy
does not improve, diagnose the mechanism before the next research decision.
