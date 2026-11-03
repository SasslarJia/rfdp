"""Full official CA-Jaccard evaluation with the paper's fixed 2/4 expansion.

The pinned author reranking module is unmodified. Camera metadata is passed
to the method; identity labels are used only by the evaluators.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from EVAL.ReIDEval import evaluate_reid_rankings
from experiments.full_reid.common import sha, dump
from experiments.internal_nonlinear.runtime.analyze_route import paired_bootstrap

SOURCE_SHA = '75ea25170204baf67f61d48d5f43e6282bed548f4e0076f14d769dfa07f15d36'
POPULATION = {'market': (3368, 19732), 'duke': (2228, 17661)}
PARAMETERS = dict(k1=30, k2=6, ckrnns=True, k1_intra=5, k1_inter=20,
                  clqe=True, k2_intra=2, k2_inter=4)


def load_author(path):
    assert sha(path) == SOURCE_SHA, 'Frozen upstream source changed'
    spec = importlib.util.spec_from_file_location('pinned_caj_paper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_author(q, g, qcam, gcam, module):
    # Match the author's evaluator: squared Euclidean input; its core squares
    # the supplied distances internally. This convention also matches the
    # already accepted OSNet paper-parameter runs.
    def distance(x, y):
        return np.maximum(np.sum(x*x, axis=1)[:, None] +
                          np.sum(y*y, axis=1)[None, :] - 2*x@y.T, 0.)
    return module.re_ranking(distance(q, g), distance(q, q), distance(g, g),
                             np.r_[qcam, gcam], SimpleNamespace(**PARAMETERS), .3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=POPULATION, required=True)
    parser.add_argument('--frontend', choices=['osnet', 'transreid'], required=True)
    parser.add_argument('--mat', type=Path, required=True)
    parser.add_argument('--feature-sha256', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-queries', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    assert sha(args.mat) == args.feature_sha256
    module = load_author(args.source)
    if args.smoke_only:
        rng = np.random.default_rng(1404)
        x = rng.normal(size=(48, 16)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        cams = np.arange(48) % 3
        distances = run_author(x[:12], x[12:], cams[:12], cams[12:], module)
        assert distances.shape == (12, 36) and np.isfinite(distances).all()
        print(json.dumps(dict(status='pinned_author_api_checked', parameters=PARAMETERS,
                              upstream_sha256=SOURCE_SHA)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    data = loadmat(args.mat)
    q, g = [np.asarray(data[k], dtype=np.float32) for k in ['query_f', 'gallery_f']]
    qn, gn = len(q), len(g)
    assert (qn, gn) == POPULATION[args.dataset]
    for features in (q, g):
        assert np.isfinite(features).all()
        assert np.allclose(np.linalg.norm(features, axis=1), 1., atol=1e-5)
    qp, qc, gp, gc = [data[k].ravel() for k in
                      ['query_label', 'query_cam', 'gallery_label', 'gallery_cam']]
    config = dict(dataset=args.dataset, frontend=args.frontend, query_count=qn,
                  gallery_count=gn, feature_sha256=args.feature_sha256,
                  source_sha256=SOURCE_SHA, parameters=dict(**PARAMETERS, lambda_value=.3),
                  identity_labels_passed_to_algorithm=False, camera_used=True,
                  parameter_policy='paper section 4.2, k2 intra/inter 2/4; no input-specific search',
                  distance_convention='squared Euclidean input; author core squares internally')
    dump(args.output/'run-manifest.json', config)
    start = time.monotonic()
    distances = run_author(q, g, qc, gc, module)
    assert distances.shape == (qn, gn) and np.isfinite(distances).all()
    algorithm_seconds = time.monotonic() - start
    ranks = np.argsort(distances, axis=1, kind='stable').astype(np.int32) + qn
    np.save(args.output/'rankings.npy', ranks, allow_pickle=False)
    metric = evaluate_reid_rankings(ranks, range(qn), qp, qc, gp, gc, qn,
                                   max_rank=50, return_per_query=True)
    aps, r1 = [], []
    for i, row in enumerate(ranks):
        local = row.astype(int) - qn
        assert np.array_equal(np.sort(local), np.arange(gn))
        keep = (gp[local] != -1) & ~((gp[local] == qp[i]) & (gc[local] == qc[i]))
        hits = gp[local[keep]] == qp[i]
        n = np.sum((gp == qp[i]) & (gc != qc[i]) & (gp != -1))
        assert n == hits.sum() and n > 0
        ap = float(np.mean(np.arange(1, n+1) / (np.flatnonzero(hits)+1)))
        aps.append(ap)
        r1.append(int(hits[0]))
        assert abs(ap-metric['per_query'][i]['AP']) < 1e-12
        assert r1[-1] == metric['per_query'][i]['rank1']
    assert abs(np.mean(aps)-metric['mAP']) < 1e-12
    assert np.mean(r1) == metric['rank-1']
    dump(args.output/'metrics.json', metric)
    baseline = json.loads(args.baseline_queries.read_text())
    baseline = {r['query_id']: r for r in baseline.get('per_query', baseline.get('queries'))}
    assert set(baseline) == set(range(qn))
    for i, row in enumerate(metric['per_query']):
        for field in ['pid', 'camid', 'valid', 'positive_count']:
            assert row[field] == baseline[i][field]
    delta = np.array([[aps[i]-baseline[i]['AP'], r1[i]-baseline[i]['rank1']]
                      for i in range(qn)])
    rng = np.random.default_rng(260626)
    paired = {unit: {name: paired_bootstrap(delta[:, j], groups, rng)
                     for j, name in enumerate(['mAP', 'rank1'])}
              for unit, groups in [('query', np.arange(qn)), ('identity', qp)]}
    dump(args.output/'paired-ihyrdp.json', dict(reference='original-IHy', seed=260626,
         paired=paired, baseline_sha256=sha(args.baseline_queries)))
    repo = Path(__file__).resolve().parents[3]
    complete = dict(status='full_rankings_independently_verified', scope='full_standard_test',
        method='caj', **config, metrics={k: v for k, v in metric.items() if k != 'per_query'},
        rankings_sha256=sha(args.output/'rankings.npy'), runner_sha256=sha(__file__),
        execution_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        algorithm_seconds=algorithm_seconds, total_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    dump(args.output/'COMPLETE.json', complete)
    print(json.dumps({k: complete[k] for k in ['status', 'dataset', 'frontend', 'algorithm_seconds']}), flush=True)


if __name__ == '__main__':
    main()
