# CSW + I-HyRDP with fixed neighbourhood support

This directory preserves the verified relation-support implementation used in
the current report. It extends camera weighting with **ordinary shared reciprocal
neighbour support**. It does not train a network, learn parameters online, or fuse
rankings with RRF. Core model/runner/verifiers are unchanged copies of the verified
experiment sources; provenance is in `docs/experiments/reliability-source-provenance.json`.

## Model and parameters

`reliability_model.py` provides `ReliabilityIHyRDP`, support construction and
hash-bound serialization. `rho=.5` is the camera prior and `lam=.25` is the
report's eta (relation-gating strength). The Python model constructor retains the
historical name `eta` for **rho**; the CLI uses unambiguous `--rho` / `--lam`.
`depth=500` counts cumulative seeds, not queries or a fixed candidate subset.

`run_candidate.py` covers every requested query and the complete gallery with
checkpoint hashes, exact provenance, complete rankings, traces and numerical
evidence. `high_precision_recovery.py` refines ill-conditioned solves without
changing the float64 equation or weakening the residual threshold.

## Run a full configuration

Use Linux for full CPU diffusion. GPU is used for feature extraction; this runner
is CPU/SciPy with workers and substantial RAM. Feature-cache creation requires
the existing full_reid dependencies. The optional development feature extractor
also requires a compatible PyTorch installation and the pinned model sources and
weights. No large data, rankings, weights or credentials are stored in Git.

From the repository root (replace DATA and OUT with real directories):

```bash
python -m pip install -r RFDP/experiments/reliability/requirements-cpu.txt
REPO="$PWD"
DATA=/path/to/market-resnet50.mat
OUT=/path/to/new-experiment
python RFDP/experiments/full_reid/build_feature_cache.py --mat "$DATA" --output "$OUT/cache"
python RFDP/experiments/reliability/prepare_support.py --repo "$REPO" --mat-path "$DATA" --cache-dir "$OUT/cache" --output "$OUT/support-ordinary.npz" --mode ordinary
python RFDP/experiments/reliability/run_candidate.py --repo "$REPO" --mat-path "$DATA" --cache-dir "$OUT/cache" --output-dir "$OUT/run" --protocol RFDP/experiments/reliability/PROTOCOL.md --method ordinary-r0p5-l0p25 --rho .5 --lam .25 --support-mode ordinary --support-file "$OUT/support-ordinary.npz" --depth 500 --workers 8 --expected-queries 3368 --expected-gallery 19732
python RFDP/experiments/reliability/verify_experiment.py --mat-path "$DATA" --output-dir "$OUT/run" --method ordinary-r0p5-l0p25 --protocol RFDP/experiments/reliability/PROTOCOL.md --support-file "$OUT/support-ordinary.npz" --depth 500 --expected-queries 3368 --expected-gallery 19732
python RFDP/experiments/reliability/verify_numerics.py --output-dir "$OUT/run" --method ordinary-r0p5-l0p25
```

Use a separate OUT for OSNet. Resume only the same frozen inputs and source;
source changes deliberately invalidate run/cache fingerprints. Existing archives
must be verified with their frozen sources/path mappings, not rewritten manifests.

## Uniform mechanism control

`uniform_control.py`, `run_uniform_candidate.py`, `verify_uniform_control.py`, and
`plan_uniform_suite.py` preserve the four-cell control and its strict input binding.
The control matches **global sparse off-diagonal mass**, keeps the diagonal, and
does not match each row or every local boundary. `plan_uniform_suite.py --help`
describes the required frozen selection, input hashes, caches and protocol.
The sibling `run_candidate_v2.py` is the frozen helper expected by this control;
it is byte-identical to the historical core runner, not a different method.

## Software checks (no real benchmark claim)

```bash
PYTHONPATH=RFDP:RFDP/experiments/reliability python -B -m unittest discover -s RFDP/experiments/reliability -p 'test_*.py'
python -B RFDP/experiments/reliability/verify_experiment.py --self-test
python -B RFDP/experiments/reliability/synthetic_integration.py --repo "$PWD" --output-dir /path/to/new-synthetic-check
python -B RFDP/experiments/reliability/check_uniform.py --repo "$PWD" --output-dir /path/to/new-uniform-check
```

Unit checks cover reciprocal support against independent set calculations,
camera/bridge consistency, positive gate limits, same-equation recovery and
tamper rejection. Synthetic checks exercise the actual ranking and verification
pipeline; they are not Market1501 performance evidence. Historical deployment
scripts with server-specific paths remain in the experiment archive, not here.
