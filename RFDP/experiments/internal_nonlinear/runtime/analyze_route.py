"""Paired full-query analysis; never read partial runs as quality evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def paired_bootstrap(delta, groups, rng, repetitions=10000):
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    n = len(sizes)
    estimates = np.empty(repetitions)
    for lo in range(0, repetitions, 128):
        hi = min(lo+128, repetitions)
        chosen = rng.integers(0, n, size=(hi-lo, n))
        estimates[lo:hi] = sums[chosen].sum(axis=1)/sizes[chosen].sum(axis=1)
    return dict(point_pp=float(delta.mean()*100),
                percentile95_pp=(np.quantile(estimates, [.025,.975])*100).tolist(),
                repetitions=repetitions, groups=n,
                interpretation='descriptive paired resampling; not independent model retraining')


def load_queries(path):
    j = json.loads(path.read_text())
    values = j.get('per_query', j.get('queries'))
    if values is None:
        raise ValueError('Per-query evidence required')
    return {int(x['query_id']): x for x in values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline-queries', type=Path, required=True)
    p.add_argument('--baseline-summary', type=Path, required=True)
    p.add_argument('--reference-name', required=True)
    p.add_argument('--baseline-feature-sha', required=True)
    args = p.parse_args()
    complete = json.loads((args.run/'COMPUTE_COMPLETE.json').read_text())
    manifest = json.loads((args.run/'run-manifest.json').read_text())
    assert complete['status'] == 'computed_and_independently_evaluated'
    assert complete['signature'] == manifest['signature']
    assert manifest['config']['feature_sha256'] == args.baseline_feature_sha
    assert sha(args.run/'rankings.npy') == complete['rankings_sha256']
    metric = json.loads((args.run/'metrics.json').read_text())
    base_summary = json.loads(args.baseline_summary.read_text())
    base_summary = base_summary.get('metrics', base_summary)
    candidate = load_queries(args.run/'metrics.json')
    baseline = load_queries(args.baseline_queries)
    assert set(candidate) == set(baseline) == set(range(complete['query_count']))
    ids, pids, aps, r1s, baseline_ap, baseline_r1 = [], [], [], [], [], []
    for q in sorted(candidate):
        a, b = candidate[q], baseline[q]
        for field in ['pid', 'camid', 'valid', 'positive_count']:
            assert a[field] == b[field], (q, field)
        assert a['retrieved_positive_count'] == a['positive_count']
        assert b['retrieved_positive_count'] == b['positive_count']
        if not a['valid']:
            continue
        ids.append(q); pids.append(a['pid']); aps.append(a['AP']); r1s.append(a['rank1'])
        baseline_ap.append(b['AP']); baseline_r1.append(b['rank1'])
    for values, expected in [(aps, metric['mAP']), (r1s, metric['rank-1']),
                             (baseline_ap, base_summary['mAP']), (baseline_r1, base_summary['rank-1'])]:
        assert abs(np.mean(values)-expected) < 1e-12
    delta_ap = np.asarray(aps)-baseline_ap
    delta_r1 = np.asarray(r1s)-baseline_r1
    rng = np.random.default_rng(260626)
    result = dict(status='paired_full_query_verified', reference=args.reference_name,
                  candidate_mAP=metric['mAP'], candidate_rank1=metric['rank-1'],
                  baseline_mAP=base_summary['mAP'], baseline_rank1=base_summary['rank-1'],
                  query_count=len(ids), rank1_gained=int(np.sum(delta_r1>0)), rank1_lost=int(np.sum(delta_r1<0)),
                  AP_improved=int(np.sum(delta_ap>1e-12)), AP_decreased=int(np.sum(delta_ap < -1e-12)),
                  seed=260626, signature=complete['signature'],
                  baseline_feature_sha256=args.baseline_feature_sha,
                  baseline_per_query_sha256=sha(args.baseline_queries),
                  baseline_summary_sha256=sha(args.baseline_summary),
                  candidate_metrics_sha256=sha(args.run/'metrics.json'),
                  paired_query=dict(mAP=paired_bootstrap(delta_ap, ids, rng), rank1=paired_bootstrap(delta_r1, ids, rng)),
                  paired_identity=dict(mAP=paired_bootstrap(delta_ap, pids, rng), rank1=paired_bootstrap(delta_r1, pids, rng)))
    order = np.argsort(delta_ap, kind='stable')
    result['largest_changes_for_diagnostics'] = [dict(query_id=ids[i], pid=pids[i], AP_delta_pp=float(delta_ap[i]*100))
                                                for i in np.r_[order[:10], order[-10:]]]
    rows = [json.loads(x.read_text()) for x in (args.run/'queries').glob('*.json')]
    assert len(rows) == complete['query_count']
    result['solver_diagnostics'] = dict(
        max_residual=max(x['diagnostics']['max_relative_residual'] for x in rows),
        max_local_nodes=max(x['diagnostics']['max_unlabeled'] for x in rows),
        bridge_queries=sum(x['diagnostics']['bridge_events']>0 for x in rows),
        numerical_tail_total=sum(x['diagnostics']['numerical_tail_count'] for x in rows),
        summed_query_seconds=sum(x['seconds'] for x in rows),
        query_seconds_quantiles=np.quantile([x['seconds'] for x in rows], [0,.5,.9,.99,1]).tolist())
    out = args.run/('paired-'+args.reference_name+'.json')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','reference','candidate_mAP','candidate_rank1','paired_identity']}, indent=2))


if __name__ == '__main__':
    main()
