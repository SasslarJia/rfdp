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
| `RFDP/experiments/continuous_fitting/` | New quadratic fitting hypothesis: derivation and small-graph assembly checks; no ReID result |
| `RFDP/experiments/internal_nonlinear/` | Frozen nonlinear experiment and cost review; population computation is paused |
| `RFDP/Test_RFDP/` | Correctness tests and small illustrative entry points |
| `docs/` | Reproduction instructions and historical source provenance |

Start with [full ReID reproduction](docs/FULL_REID_REPRODUCTION.md) and
[accepted CSW implementation](RFDP/experiments/reliability/README.md).
`PaperFullIterative.py` is the dynamic full-experiment implementation; the
`Test_RFDP` local-subset examples are not substitutes for that protocol.

## Current research decision

The nonlinear smoothing variant keeps a positive quadratic part but adds a
baseline solve for its scale, repeated Newton solves and nonlinear certification
at every expansion stage. Its full Market run is **paused for excessive cost**
with 626 completed queries. This is not an accuracy failure: no complete mAP
result is available. See its [cost review](RFDP/experiments/internal_nonlinear/README.md).

The next candidate changes the hybrid fitting source continuously with boundary
connection strength while preserving the original stage matrix. Its
[derivation and limitations](RFDP/experiments/continuous_fitting/README.md) explain
what is proved and what still needs retrieval evidence. No new experiment is
started by checking out this branch.

## Version and data policy

Git stores code, derivations and concise method status. Raw rankings, per-query
scores, features, weights and operational logs stay outside Git. Failed or
retired implementations can be recovered from history; they need not remain as
parallel working directories. The obsolete generic RFDP framework, unused
scratch entry points, IDE files, bytecode and a duplicate runner have been
removed from the active tree. Accepted CSW and comparison implementations remain.

Frozen results must be checked with their frozen source revision and original
manifests. The paused nonlinear execution uses
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
