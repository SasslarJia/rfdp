"""Bounded first-frontier analysis of accepted scores, including GMFPT.

One saved first frontier per query; no reranking trajectory or parameter search.
Sparse right-hand-side solves evaluate the decomposition without forming F.
Floating diagnostics are not ordering certificates or new full-gallery metrics.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import gzip
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.io import loadmat

from ..continuous_fitting.certified import unpack
from ..continuous_fitting.model import assemble_system
from ..full_reid.common import REPO, dump, sha


def components(w, e, alpha=5., beta=25.):
    w = sparse.csr_matrix(w, dtype=float)
    e = np.asarray(e, dtype=float)
    h = np.asarray(w.sum(axis=1)).ravel()
    direct = e > 0
    indirect = ~direct
    system = assemble_system(w, e, alpha, beta, 'stationary_measure')
    p = system.boundary_probability
    root = np.sqrt(h)
    rhs = np.column_stack((alpha*h*direct, h*indirect, alpha*h*p,
                           alpha*root*direct, root*indirect, alpha*root*p))
    solution = splu(system.matrix.tocsc()).solve(rhs)
    u, v, mass = solution[:, 0], solution[:, 1], solution[:, 2]
    old_u, old_v, old_mass = (root*solution[:, k] for k in (3, 4, 5))
    gm_matrix = sparse.diags(h+e)-w
    gm_rhs = np.column_stack(((h+e)*direct, h*indirect, e))
    gm_solution = splu(gm_matrix.tocsc()).solve(gm_rhs)
    residuals = []
    for a, x, b in ((system.matrix, solution, rhs), (gm_matrix, gm_solution, gm_rhs)):
        residuals.extend(np.linalg.norm(a@x-b, axis=0)/np.maximum(np.linalg.norm(b, axis=0), 1.))
    return dict(h=h, p=p, u=u, v=v, mass=mass, score=u+beta*v,
                original_u=old_u, original_v=old_v, original_mass=old_mass,
                original=old_u+beta*old_v, gmfpt_u=gm_solution[:, 0],
                gmfpt_v=gm_solution[:, 1], gmfpt_mass=gm_solution[:, 2],
                gmfpt=gm_solution[:, 0]+gm_solution[:, 1],
                max_linear_residual=float(max(residuals)))


def margin(score, positive, negative):
    if not np.any(positive) or not np.any(negative):
        return None
    return float(np.min(score[negative])-np.min(score[positive]))


def quantiles(values):
    return np.quantile(values, [0, .1, .5, .9, 1]).tolist() if values else None


def summarize(rows):
    solved = [r for r in rows if r['solved']]
    eligible = [r for r in solved if r['margins'] is not None]
    comparisons = {}
    for other in ('original', 'gmfpt'):
        comparisons[other] = dict(eligible=len(eligible),
            negative_to_positive=sum(r['margins'][other]<0<r['margins']['current'] for r in eligible),
            positive_to_negative=sum(r['margins']['current']<0<r['margins'][other] for r in eligible),
            positive_both=sum(r['margins']['current']>0 and r['margins'][other]>0 for r in eligible),
            nonpositive_both=sum(r['margins']['current']<=0 and r['margins'][other]<=0 for r in eligible))
    return dict(queries=len(rows), solved=len(solved), no_solve=len(rows)-len(solved),
        max_decomposition_error=max((r['decomposition_error'] for r in solved), default=0.),
        max_linear_residual=max((r['max_linear_residual'] for r in solved), default=0.),
        max_mass_error=max((r['mass_error'] for r in solved), default=0.),
        max_gmfpt_mass_error=max((r['gmfpt_mass_error'] for r in solved), default=0.),
        max_range_violation=max((r['range_violation'] for r in solved), default=0.),
        original_mass_deviation_quantiles=quantiles([r['original_mass_deviation'] for r in solved]),
        original_boundary_out_of_range_queries=sum(r['original_range_violation']>1e-10 for r in solved),
        degree_log_std_quantiles=quantiles([r['degree_log_std'] for r in solved]),
        local_margin_comparisons=comparisons,
        indirect_gap_dominates=sum(abs(r['pair']['indirect_gap'])>abs(r['pair']['boundary_gap']) for r in eligible))


def main(config, output, max_seconds):
    start = time.monotonic()
    cfg = json.loads(Path(config).read_text())
    run, output = Path(cfg['output']), Path(output)
    assert cfg['source'] == 'stationary_measure'
    acceptance = json.loads((run/'FULL_ACCEPTANCE.json').read_text())
    assert acceptance['status'] == 'accepted_full_population'
    assert acceptance['queries'] == cfg['queries']
    assert not output.exists()
    cap = int(Path('/sys/fs/cgroup/memory.max').read_text())
    assert int(Path('/sys/fs/cgroup/memory.current').read_text())+4*2**30 < .7*cap
    resource.setrlimit(resource.RLIMIT_AS, (4*2**30, 4*2**30))
    cache = Path(cfg['cache'])
    assert sha(cache/'graph.npz') == cfg['graph_sha256']
    assert sha(cfg['mat']) == cfg['feature_sha256']
    graph = sparse.load_npz(cache/'graph.npz').astype(float).tocsr()
    graph.eliminate_zeros(); graph.sort_indices()
    data = loadmat(cfg['mat'], variable_names=['query_label','query_cam','gallery_label','gallery_cam'])
    labels = np.r_[data['query_label'].ravel(), data['gallery_label'].ravel()]
    cams = np.r_[data['query_cam'].ravel(), data['gallery_cam'].ravel()]
    metric = json.loads((run/'metrics.json').read_text())
    baseline = json.loads(Path(cfg['baseline_queries']).read_text())
    candidate = {r['query_id']:r for r in metric['per_query']}
    parent = {r['query_id']:r for r in baseline.get('per_query', baseline.get('queries'))}
    output.mkdir()
    rows = []
    try:
        for q in range(cfg['queries']):
            assert time.monotonic()-start < max_seconds, 'Bounded analysis time limit'
            if q % 64 == 0:
                assert int(Path('/sys/fs/cgroup/memory.current').read_text()) < .7*cap
            with gzip.open(run/'queries'/f'{q:04d}.trace.jsonl.gz', 'rt') as stream:
                row = json.loads(next(stream))
            assert row['iteration'] == 0 and row['labeled'] == [q]
            record = dict(query=q, solved=row['returned_scores'] is not None,
                          AP_delta=candidate[q]['AP']-parent[q]['AP'],
                          rank1_delta=candidate[q]['rank1']-parent[q]['rank1'])
            if record['solved']:
                ids = np.asarray(row['unlabeled'])
                boundary = np.asarray(graph[[q]].sum(axis=0)).ravel()
                for i, j, weight in row['bridges']:
                    boundary[i] += weight
                e = boundary[ids]
                c = components(graph[ids][:, ids], e, cfg['alpha'], cfg['beta'])
                saved = np.array([float(v) for v in unpack(row['returned_scores'])])
                error = float(np.max(np.abs(c['score']-saved))/max(1., np.max(np.abs(saved))))
                assert error < 1e-8 and c['max_linear_residual'] < 1e-8
                t = 1/c['p'][e>0]
                low, high = float(t.min()), float(t.max())
                def violation(v):
                    return float(max(0., low-v.min(), v.max()-high)/max(1., high))
                positive = (ids>=cfg['queries']) & (labels[ids]==labels[q]) & (cams[ids]!=cams[q])
                negative = (ids>=cfg['queries']) & (labels[ids]!=labels[q]) & (labels[ids]!=-1)
                m = {name:margin(c[key], positive, negative) for name,key in
                     [('current','score'),('original','original'),('gmfpt','gmfpt')]}
                record.update(size=len(ids), direct=int(np.sum(e>0)), indirect=int(np.sum(e==0)),
                    degree_log_std=float(np.std(np.log(c['h']))), decomposition_error=error,
                    max_linear_residual=c['max_linear_residual'],
                    mass_error=float(np.max(np.abs(c['mass']-1))),
                    gmfpt_mass_error=float(np.max(np.abs(c['gmfpt_mass']-1))),
                    original_mass_deviation=float(np.max(np.abs(c['original_mass']-1))),
                    range_violation=violation(c['u']), original_range_violation=violation(c['original_u']),
                    target_min=low, target_max=high, margins=m if m['current'] is not None else None)
                if m['current'] is not None:
                    pos = np.flatnonzero(positive); neg = np.flatnonzero(negative)
                    i = pos[np.argmin(c['score'][pos])]; j = neg[np.argmin(c['score'][neg])]
                    record['pair'] = dict(positive=int(ids[i]), negative=int(ids[j]),
                        boundary_gap=float(c['u'][j]-c['u'][i]),
                        indirect_gap=float(cfg['beta']*(c['v'][j]-c['v'][i])),
                        current_gap=float(c['score'][j]-c['score'][i]),
                        original_gap=float(c['original'][j]-c['original'][i]),
                        gmfpt_gap=float(c['gmfpt'][j]-c['gmfpt'][i]))
            rows.append(record)
        dump(output/'queries.json', rows)
        groups = dict(all=rows, rank1_gained=[r for r in rows if r['rank1_delta']>0],
                      rank1_lost=[r for r in rows if r['rank1_delta']<0])
        dump(output/'summary.json', dict(status='complete_first_frontier_analysis',
            scope='All queries, saved first frontier only; no full GMFPT retrieval.',
            floating_descriptive_analysis=True, post_hoc=True, parameters=dict(alpha=cfg['alpha'],beta=cfg['beta']),
            seconds=time.monotonic()-start, groups={k:summarize(v) for k,v in groups.items()},
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            accepted_run_sha256=sha(run/'FULL_ACCEPTANCE.json'), query_details_sha256=sha(output/'queries.json')))
        print(json.dumps(json.loads((output/'summary.json').read_text())['groups']['all']))
    except BaseException as error:
        dump(output/'FAILED.json', dict(error=repr(error), completed=len(rows), seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    main(args.config, args.output, args.max_seconds)
