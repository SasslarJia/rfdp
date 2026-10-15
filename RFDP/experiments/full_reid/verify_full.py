"""Independent complete-permutation AP/CMC/INP and checkpoint verification."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import dump, load_features, sha, validate_ranking


def evaluate(rankings, query_pids, query_cams, gallery_pids, gallery_cams, max_rank=500):
    nq, ng = len(query_pids), len(gallery_pids)
    if rankings.shape != (nq, ng):
        raise ValueError('Not a full-query/full-gallery result')
    aps, inps, cmcs, rows = [], [], [], []
    for query in range(nq):
        validate_ranking(rankings[query], nq, ng)
        order = rankings[query] - nq
        valid = (gallery_pids[order] != -1) & ~((gallery_pids[order] == query_pids[query]) & (gallery_cams[order] == query_cams[query]))
        hits = np.flatnonzero(gallery_pids[order[valid]] == query_pids[query]) + 1
        if len(hits) == 0:
            rows.append(dict(query=query, valid=False))
            continue
        ap = float((np.arange(1, len(hits) + 1) / hits).mean())
        inp = float(len(hits) / hits[-1])
        aps.append(ap)
        inps.append(inp)
        cmcs.append((hits[0] <= np.arange(1, min(max_rank, ng) + 1)).astype(float))
        rows.append(dict(query=query, valid=True, AP=ap, INP=inp,
                         first_match=int(hits[0]), valid_positives=len(hits)))
    if not aps:
        raise ValueError('No valid queries')
    return dict(mAP=float(np.mean(aps)), mINP=float(np.mean(inps)), CMC=np.mean(cmcs, axis=0).tolist(),
                valid_queries=len(aps), query_count=nq, gallery_count=ng), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['original', 'ihyrdp', 'csw', 'k-reciprocal', 'ecn-rankdist'])
    args = parser.parse_args()
    data, nq, ng = load_features(args.mat_path)
    out = args.output_dir
    (out / 'COMPLETE.json').unlink(missing_ok=True)
    manifest = json.loads((out / 'run-manifest.json').read_text())
    if manifest['config']['feature_sha256'] != sha(args.mat_path):
        raise ValueError('Evaluation feature mismatch')
    if args.methods != manifest['config']['methods']:
        raise ValueError('Verification must cover every configured method in order')
    result, checks = {}, {}
    for method in args.methods:
        path = out / f'{method}.rankings.npy'
        rankings = np.load(path, mmap_mode='r')
        if method in ('ihyrdp', 'csw'):
            for query in range(nq):
                record = json.loads((out / method / f'{query:04d}.json').read_text())
                rp = out / method / f'{query:04d}.rankings.npy'
                pp = out / method / f'{query:04d}.prefix.npy'
                prefix = np.load(pp)
                trace_nodes = [query] + [n for step in record['trace'] for n in step['selected']]
                if (record['signature'] != manifest['signature'] or record['query'] != query
                        or record['mode'] != method or record['rankings_sha256'] != sha(rp)
                        or record['prefix_sha256'] != sha(pp)
                        or len(prefix) != manifest['config']['depth']
                        or len(np.unique(prefix)) != len(prefix)
                        or np.any(prefix < 0) or np.any(prefix >= nq + ng)
                        or not np.array_equal(prefix, trace_nodes)
                        or not np.array_equal(rankings[query], np.load(rp))
                        or record['diagnostics']['status'] != 'complete_diffusion'
                        or record['diagnostics']['numerical_tail_count'] != 0
                        or record['diagnostics']['max_relative_residual'] > 1e-8):
                    raise ValueError('Invalid diffusion checkpoint: ' + str(rp))
                gallery_prefix = prefix[prefix >= nq]
                if not np.array_equal(rankings[query, :len(gallery_prefix)], gallery_prefix):
                    raise ValueError('Ranking does not preserve diffusion prefix')
        metrics, rows = evaluate(rankings, *[data[k].ravel() for k in
                                            ['query_label', 'query_cam', 'gallery_label', 'gallery_cam']])
        original = json.loads((out / f'{method}.json').read_text())
        original = original.get('metrics', original)
        for field in ['mAP', 'mINP']:
            if not np.isclose(metrics[field], original[field], rtol=0, atol=1e-12):
                raise ValueError('Independent metric mismatch: ' + method + '/' + field)
        if not np.allclose(metrics['CMC'][:len(original['cmc'])], original['cmc'], rtol=0, atol=1e-12):
            raise ValueError('Independent CMC mismatch')
        result[method] = metrics
        checks[method] = dict(rankings_sha256=sha(path), permutation_checks=nq)
        dump(out / f'{method}.independent-per-query.json', rows)
    dump(out / 'verified-results.json', dict(signature=manifest['signature'], results=result, checks=checks))
    dump(out / 'COMPLETE.json', dict(status='verified', signature=manifest['signature'], queries=nq, gallery=ng, methods=args.methods))
    print(json.dumps({k: dict(mAP=v['mAP'], rank1=v['CMC'][0]) for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
