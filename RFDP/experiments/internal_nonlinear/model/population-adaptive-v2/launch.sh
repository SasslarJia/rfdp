#!/bin/bash
set -u
cd /root/autodl-tmp/rfdp-graph-routes-20261026/internal-nonlinear-smoothing-v1-20261028/population-adaptive-v2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=/root/autodl-tmp/rfdp-graph-routes-20261026/market-internal-nonlinear-theta0p5-v1
PY=/root/miniconda3/bin/python
$PY -B -u full.py compute > compute.log 2>&1
rc=$?
echo "$rc" > compute.exit
if [ -d "$OUT" ]; then echo "$rc" > "$OUT/compute.exit"; fi
if [ "$rc" -eq 0 ]; then
  FEATURE_SHA=$($PY -c 'import json;print(json.load(open("'"$OUT"'/run-manifest.json"))["config"]["feature_sha256"])')
  $PY -B -u /root/autodl-tmp/rfdp-graph-routes-20261026/analyze_route.py --run "$OUT" --baseline-queries /root/autodl-tmp/rfdp-graph-routes-20261026/baselines/dynamic-depth-500.per-query.json --baseline-summary /root/autodl-tmp/rfdp-graph-routes-20261026/baselines/dynamic-depth-500.json --reference-name original-IHy --baseline-feature-sha "$FEATURE_SHA" > analysis.log 2>&1
  rc=$?
  echo "$rc" > analysis.exit
fi
if [ "$rc" -eq 0 ]; then
  $PY -B -u full.py acceptance > acceptance.log 2>&1
  rc=$?
  echo "$rc" > acceptance.exit
fi
echo "$rc" > total.exit
exit "$rc"
