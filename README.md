# RFDP: visual re-ranking research

Code is maintained on the [`test` branch](https://github.com/SasslarJia/rfdp/tree/test).
The repository implements Zheng et al., *Hybrid Regularization of Diffusion
Process for Visual Re-Ranking*, IEEE TIP 2021, and explicitly separated research
variants. Frozen features and complete query/gallery evaluation are used; this
repository does not train a new backbone.

## Code map

| Directory | Role |
| --- | --- |
| `RFDP/MODEL/Paper_RFDP/` | Original HyRDP/GMFPT equations, dynamic I-HyRDP, precision and camera adapters |
| `RFDP/MODEL/{Aff,Asst,Iput,Enum}_RFDP/` | Shared graph and input utilities |
| `RFDP/EVAL/` | ReID evaluation |
| `RFDP/experiments/full_reid/` | Full-query input preparation, original methods and fixed comparison implementations |
| `RFDP/experiments/reliability/` | Accepted CSW + ordinary neighbourhood-support model and mechanism controls |
| `RFDP/experiments/stationary_measure/` | Current retained candidate: degree-weighted hybrid fitting, two accepted Market populations, theory and evidence |
| `RFDP/experiments/continuous_fitting/` | **Active shared quadratic runner, solver and verifier**; its original continuous-source model is retired |
| `RFDP/experiments/{boundary_odds,two_step_fitting}/` | Completed, retired fixed forms; concise results and status |
| `RFDP/experiments/internal_nonlinear/` | Frozen cost-paused model **and shared analysis utilities still imported by the current runner** |
| `RFDP/Test_RFDP/` | Correctness tests and small illustrative entry points |
| `docs/` | Reproduction instructions and historical source provenance |

Start with the [code and evidence index](docs/RESEARCH_INDEX.md),
[current research synthesis](RFDP/experiments/stationary_measure/research-summary.md),
[full ReID reproduction](docs/FULL_REID_REPRODUCTION.md) and
[accepted CSW implementation](RFDP/experiments/reliability/README.md).
`PaperFullIterative.py` is the dynamic full-experiment implementation; the
`Test_RFDP` local-subset examples are not substitutes for that protocol.

## Current research decision

**30 October 2026: experiments and the recurring task are paused by the user;
the user reports the server is powered off.** No automatic retrieval, tuning,
diagnostic, performance or Duke run is queued. Local theory and documentation
work does not authorize resuming experiments.

The retained candidate is [degree-weighted hybrid fitting](RFDP/experiments/stationary_measure/README.md).
Both Market populations and their bounded mechanism reviews are complete:

| Features | mAP / Rank-1 (%) | Change from original I-HyRDP (pp) |
| --- | ---: | ---: |
| OSNet | 93.174274 / 95.249406 | +0.039372 / +0.089074 |
| TransReID | 94.537030 / 95.190024 | +0.006139 / -0.089074 |

All computation, paired analysis and saved-score acceptance stages exited 0.
All query- and identity-paired 95% intervals cross zero. The current theoretical
focus is score consistency under a specified mass-preserving graph refinement,
and conditional pairwise stability under departures from it. Ordinary duplicate
images, rebuilt kNN graphs and full iterative ranking require separate evidence.
The candidate has not been run on Duke; original I-HyRDP and the five fixed
comparators already have the complete Market/Duke × OSNet/TransReID matrix.

Continuous-source, boundary-odds and two-step fitting fixed forms are retired.
The nonlinear form stopped at 626/3368 queries because of cost, without a full
accuracy result. Its saved artifacts and frozen revision remain references;
the old SIGSTOP process state must not be assumed to survive server shutdown.
Directory names do not determine whether code is unused: the current solver and
analysis still depend on modules under two historical experiment directories.

## Version and data policy

Git stores code, derivations and concise method status. Raw rankings, per-query
scores, features, weights and operational logs stay outside Git. Failed or
retired implementations can be recovered from history; they need not remain as
parallel working directories. The obsolete generic RFDP framework, unused
scratch entry points, IDE files, bytecode and a duplicate runner have been
removed from the active tree. Accepted CSW and comparison implementations remain.

Frozen results must be checked with their frozen source revision and original
manifests, recorded in the evidence index. The nonlinear execution used
`5e5ec194ef362c9bf179d73f8bd6ea0a10d71601`, not the latest cleaned working tree.
Its deployment exporter rejects incompatible repository inputs before writing.
Historical provenance JSON files are records, not manifests to silently update.

## Correctness checks

```bash
python -m pip install -r RFDP/experiments/reliability/requirements-cpu.txt
PYTHONPATH=RFDP:RFDP/experiments/reliability python -B -m unittest Test_RFDP.test_paper Test_RFDP.test_paper_safety Test_RFDP.test_paper_ranking_safety Test_RFDP.test_full_ihyrdp Test_RFDP.test_camera_ihyrdp Test_RFDP.test_reid_eval Test_RFDP.test_full_pipeline Test_RFDP.test_continuous_fitting
PYTHONPATH=RFDP:RFDP/experiments/reliability python -B -m unittest discover -s RFDP/experiments/reliability -p 'test_*.py'
```

The older utility tests `test_asst.py` and `test_xfrm.py` are outside this CPU
suite: they require optional Torch and an external IDSC dataset, respectively.
Tests are not benchmark results. Comparisons must use the same features,
query/gallery protocol and declared hardware, with preprocessing, ranking and
verification time distinguished. Retain original result summaries and report
both accuracy and computational cost, including unfavorable outcomes.
