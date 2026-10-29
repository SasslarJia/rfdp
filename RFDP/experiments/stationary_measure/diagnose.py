"""Post-hoc degree-response review of an accepted run, not a new model trial.

All queries, at most five shared batches, stop at the first ordered divergence.
Float summaries explain already certified scores; they are not certificates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gzip
import json
import multiprocessing
from pathlib import Path
import numpy as np
from scipy import sparse
from ..continuous_fitting import diagnose as shared
from ..continuous_fitting.certified import solve, unpack
from ..continuous_fitting.full import dump, sha, now


def degree_response(w, e, original, candidate, alpha=5.):
    """N(x-psi)=(P-S)psi; report floating closure, not a new solve."""
    h = np.asarray(w.sum(axis=1)).ravel()
    p = e/(h+e)
    walk = sparse.diags(1/h) @ w
    symmetric = sparse.diags(1/np.sqrt(h)) @ w @ sparse.diags(1/np.sqrt(h))
    operator = sparse.diags(1+alpha*p)-walk
    forcing = np.asarray((walk-symmetric) @ original).ravel()
    closure = operator @ (candidate-original)-forcing
    scale = np.linalg.norm(operator @ candidate)+np.linalg.norm(operator @ original)
    return dict(log_degree_std=float(np.std(np.log(h))),
                relative_score_change=float(np.linalg.norm(candidate-original)/np.linalg.norm(original)),
                relative_response_closure=float(np.linalg.norm(closure)/scale)), h


def local_margin(scores, positive, negative):
    """Minimum valid-negative score minus minimum positive score on this frontier."""
    if not np.any(positive) or not np.any(negative):
        return None
    return float(np.min(scores[negative])-np.min(scores[positive]))


def compare(query):
    cfg, graph = shared.CFG, shared.GRAPH
    labels, cams = shared.LABELS, shared.CAMS
    boundary = np.asarray(graph[[query]].sum(axis=0)).ravel()
    record = dict(query=query, ap_difference=shared.CANDIDATE[query]['AP']-shared.BASELINE[query]['AP'],
                  rank1_difference=shared.CANDIDATE[query]['rank1']-shared.BASELINE[query]['rank1'],
                  first_divergence_batch=None)
    with gzip.open(Path(cfg['output'])/'queries'/f'{query:04d}.trace.jsonl.gz', 'rt') as stream:
        for batch in range(5):
            row = json.loads(next(stream))
            assert row['iteration'] == batch
            u = np.asarray(row['unlabeled'])
            e = boundary.copy()
            for i, j, value in row['bridges']:
                e[i] += value
            original = candidate = h = None
            if row['returned_scores'] is not None:
                w, eu = graph[u][:, u], e[u]
                values, _, event = solve(w, eu, alpha=cfg['alpha'], beta=cfg['beta'], source='original')
                selected = u[event['returned_prefix_local']].tolist()
                original = np.asarray([float(v) for v in values])
                candidate = np.asarray([float(v) for v in unpack(row['returned_scores'])])
                response, h = degree_response(w, eu, original, candidate, cfg['alpha'])
                record['max_response_closure'] = max(record.get('max_response_closure', 0.), response['relative_response_closure'])
                if batch == 0:
                    record['first_stage'] = response
                    positive = (u >= cfg['queries']) & (labels[u] == labels[query]) & (cams[u] != cams[query])
                    negative = (u >= cfg['queries']) & (labels[u] != labels[query]) & (labels[u] != -1)
                    old_margin = local_margin(original, positive, negative)
                    if old_margin is not None:
                        record['first_stage_local_gallery_margin'] = dict(
                            original=old_margin, candidate=local_margin(candidate, positive, negative),
                            positive_count=int(positive.sum()), negative_count=int(negative.sum()))
            else:
                selected = u.tolist()
            if selected != row['selected']:
                old, new = set(selected), set(row['selected'])
                local = {int(v): i for i, v in enumerate(u)}
                def describe(nodes):
                    items = []
                    for node in sorted(nodes):
                        k = local[node]
                        items.append(dict(node=node, direct=bool(e[node]>0),
                            gallery_positive=bool(node>=cfg['queries'] and labels[node]==labels[query] and cams[node]!=cams[query]),
                            degree_percentile=float(np.mean(h<=h[k])),
                            relative_score_shift=float((candidate[k]-original[k])/original[k])))
                    return items
                record.update(first_divergence_batch=batch, set_changed=old!=new,
                              original_selected=selected, candidate_selected=row['selected'],
                              removed=describe(old-new), added=describe(new-old))
                break
            for i in selected:
                start, stop = graph.indptr[i:i+2]
                boundary[graph.indices[start:stop]] += graph.data[start:stop]
    return record


def quantiles(values):
    return np.quantile(values, [0,.1,.5,.9,.99,1]).tolist() if len(values) else None


def summarize(rows):
    changed = [r for r in rows if r['first_divergence_batch'] is not None]
    margins = [r['first_stage_local_gallery_margin'] for r in rows if 'first_stage_local_gallery_margin' in r]
    return dict(queries=len(rows), diverged_within_first_five=len(changed),
        ordered_divergence_histogram={str(i):sum(r['first_divergence_batch']==i for r in changed) for i in range(5)},
        changed_sets=sum(r['set_changed'] for r in changed),
        swaps={key:dict(count=sum(len(r[key]) for r in changed),
            gallery_positive=sum(v['gallery_positive'] for r in changed for v in r[key]),
            degree_percentile_quantiles=quantiles([v['degree_percentile'] for r in changed for v in r[key]])) for key in ('removed','added')},
        first_log_degree_std_quantiles=quantiles([r['first_stage']['log_degree_std'] for r in rows if 'first_stage' in r]),
        first_score_relative_change_quantiles=quantiles([r['first_stage']['relative_score_change'] for r in rows if 'first_stage' in r]),
        first_local_gallery_margin=dict(eligible_queries=len(margins),
            positive_to_negative=sum(v['original']>0 and v['candidate']<0 for v in margins),
            negative_to_positive=sum(v['original']<0 and v['candidate']>0 for v in margins),
            note='Restricted to the first frontier, not the final full-gallery Rank-1; float descriptive signs.'))


def main(config, output):
    cfg = json.loads(Path(config).read_text()); run = Path(cfg['output']); output = Path(output)
    assert cfg['source'] == 'stationary_measure'
    accepted = json.loads((run/'FULL_ACCEPTANCE.json').read_text())
    assert accepted['queries'] == cfg['queries'] == 3368
    assert accepted['status'] == 'accepted_full_population'
    assert len(list((run/'queries').glob('*.json'))) == cfg['queries']
    assert not output.exists()
    cg = Path('/sys/fs/cgroup')
    assert int((cg/'memory.current').read_text())+17*2**30 <= .7*int((cg/'memory.max').read_text())
    for path, expected in cfg['input_sha256'].items():
        assert sha(path) == expected, path
    output.mkdir()
    try:
        with multiprocessing.get_context('spawn').Pool(16, initializer=shared.initialize, initargs=(cfg,)) as pool:
            rows = list(pool.imap_unordered(compare, range(cfg['queries']), chunksize=4))
        rows.sort(key=lambda r:r['query'])
        dump(output/'queries.json', rows)
        groups = dict(all=rows, AP_up=[r for r in rows if r['ap_difference']>1e-12],
            AP_down=[r for r in rows if r['ap_difference'] < -1e-12],
            AP_same=[r for r in rows if abs(r['ap_difference'])<=1e-12],
            rank1_gained=[r for r in rows if r['rank1_difference']>0],
            rank1_lost=[r for r in rows if r['rank1_difference']<0])
        dump(output/'summary.json', dict(time=now(), model=cfg['source'],
            scope='All queries; at most five common batches, stop at first ordered divergence.',
            post_hoc=True, causal_claim=False, original_full_trajectory_reconstructed=False,
            float_diagnostics_not_certificates=True,
            max_response_closure=max(r.get('max_response_closure', 0.) for r in rows),
            groups={name:summarize(values) for name, values in groups.items()},
            accepted_run_sha256=sha(run/'FULL_ACCEPTANCE.json'), query_details_sha256=sha(output/'queries.json')))
        print((output/'summary.json').read_text(), flush=True)
    except Exception as exc:
        dump(output/'FAILED.json', dict(time=now(), error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.config, args.output)
