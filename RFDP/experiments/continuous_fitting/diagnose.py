"""Post-hoc, bounded common-prefix comparison after complete acceptance.

Recompute the original equation only while its ordered selections agree with
the saved candidate. Stop on the first difference or after five batches. This
does not reconstruct a full original trajectory and never tests a new model.
Identity labels are used only for the completed-run diagnosis.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, gzip, json, multiprocessing, resource
from pathlib import Path
import numpy as np
from scipy import sparse, io
from .certified import solve, unpack
from .full import sha, dump, now


def load_query_reference(path):
    """Read either preserved baseline schema without changing its source file."""
    data = json.loads(Path(path).read_text())
    rows = data['queries'] if 'queries' in data else data['per_query']
    assert isinstance(rows, list) and rows
    assert all(row['query_id'] == i for i, row in enumerate(rows)), 'Reference query order'
    return rows


def initialize(config):
    global CFG, GRAPH, LABELS, CAMS, CANDIDATE, BASELINE
    resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
    CFG = config
    GRAPH = sparse.load_npz(Path(config['cache'])/'graph.npz').astype(float).tocsr()
    data = io.loadmat(config['mat'], variable_names=['query_label', 'gallery_label', 'query_cam', 'gallery_cam'])
    LABELS = np.concatenate([data[k].ravel() for k in ('query_label', 'gallery_label')])
    CAMS = np.concatenate([data[k].ravel() for k in ('query_cam', 'gallery_cam')])
    CANDIDATE = json.loads((Path(config['output'])/'metrics.json').read_text())['per_query']
    BASELINE = load_query_reference(config['baseline_queries'])
    assert len(BASELINE) == len(CANDIDATE) == config['queries']


def compare(query):
    path = Path(CFG['output'])/'queries'/f'{query:04d}.trace.jsonl.gz'
    boundary = np.asarray(GRAPH[[query]].sum(axis=0)).ravel()
    record = dict(query=query, pid=int(LABELS[query]),
                  ap_difference=CANDIDATE[query]['AP']-BASELINE[query]['AP'])
    with gzip.open(path, 'rt') as stream:
        for batch in range(5):
            row = json.loads(next(stream)); u = np.asarray(row['unlabeled'])
            assert row['iteration'] == batch
            e = boundary.copy()
            for i, j, v in row['bridges']: e[i] += v
            q = None
            if row['returned_scores'] is not None:
                w = GRAPH[u][:, u]; eu = e[u]
                scores, residual, event = solve(w, eu, source='original')
                selected = u[event['returned_prefix_local']].tolist()
                h = np.asarray(w.sum(axis=1)).ravel(); p = eu/(h+eu)
                q = np.asarray(w @ p).ravel()/(h+eu) if CFG['source']=='two_step_fitting' else None
                if batch == 0:
                    direct = p[p > 0]
                    original_values=np.array([float(v) for v in scores])
                    candidate_values=np.array([float(v) for v in unpack(row['returned_scores'])])
                    record.update(first_direct_count=len(direct), first_p=direct.tolist(),
                                  first_score_relative_change=float(np.linalg.norm(candidate_values-original_values)/np.linalg.norm(original_values)))
                    if q is not None:
                        record.update(first_indirect_q=q[eu==0].tolist(),
                                      first_score_increase_count=int(np.sum(candidate_values > original_values + 1e-8*np.maximum(1,abs(original_values)))))
            else:
                selected = u.tolist()
            if selected != row['selected']:
                old, new = set(selected), set(row['selected'])
                def describe(ids):
                    return [dict(node=int(i), same_identity=bool(LABELS[i]==LABELS[query]),
                                 cross_camera_positive=bool(LABELS[i]==LABELS[query] and CAMS[i]!=CAMS[query]),
                                 direct=bool(e[i]>0)) for i in sorted(ids)]
                record.update(first_divergence_batch=batch, original_selected=selected,
                              candidate_selected=row['selected'], removed=describe(old-new), added=describe(new-old),
                              set_changed=old!=new)
                if q is not None:
                    local={int(node):k for k,node in enumerate(u)}
                    for key in ('removed','added'):
                        for node in record[key]:
                            k=local[node['node']]
                            node['two_step_probability']=float(q[k])
                            node['added_fidelity_curvature']=float(CFG['beta']*q[k]/2) if eu[k]==0 else 0.
                break
            for i in selected:
                a,b=GRAPH.indptr[i:i+2];boundary[GRAPH.indices[a:b]]+=GRAPH.data[a:b]
        else:
            record['first_divergence_batch'] = None
    return record


def main(config, output):
    cfg=json.loads(Path(config).read_text()); run=Path(cfg['output']);output=Path(output)
    assert cfg['source'] in ('continuous','boundary_odds','two_step_fitting')
    assert (run/'FULL_ACCEPTANCE.json').exists() and (run/'compute.exit').read_text().strip()=='0'
    assert len(list((run/'queries').glob('*.json')))==3368
    assert not output.exists()
    cg=Path('/sys/fs/cgroup')
    assert int((cg/'memory.current').read_text())+17*2**30 <= .7*int((cg/'memory.max').read_text()),'Diagnosis memory reservation'
    output.mkdir()
    for path,expected in cfg['input_sha256'].items():assert sha(path)==expected,path
    with multiprocessing.get_context('spawn').Pool(16,initializer=initialize,initargs=(cfg,)) as pool:
        rows=list(pool.imap_unordered(compare,range(3368),chunksize=4))
    rows.sort(key=lambda r:r['query'])
    probabilities=np.concatenate([r.pop('first_p') for r in rows if 'first_p' in r])
    alpha,beta=cfg['alpha'],cfg['beta']
    if cfg['source']=='continuous':
        increments=(beta-alpha)*(1-probabilities)
        curvature_ratio=np.ones_like(probabilities)
    elif cfg['source']=='boundary_odds':
        increments=alpha*probabilities/(1-probabilities)
        curvature_ratio=1/(1-probabilities)
    elif cfg['source']=='two_step_fitting':
        increments=np.zeros_like(probabilities)
        curvature_ratio=np.ones_like(probabilities)
    else:raise ValueError('Diagnostic coefficient summary is not defined for this model')
    def group(rs):
        changed=[r for r in rs if r['first_divergence_batch'] is not None]
        return dict(queries=len(rs), diverged_within_first_five=len(changed),
                    ordered_divergence_histogram={str(i):sum(r['first_divergence_batch']==i for r in changed) for i in range(5)},
                    changed_sets=sum(r['set_changed'] for r in changed),
                    removed_true=sum(v['same_identity'] for r in changed for v in r['removed']),
                    added_true=sum(v['same_identity'] for r in changed for v in r['added']),
                    removed_cross_camera=sum(v['cross_camera_positive'] for r in changed for v in r['removed']),
                    added_cross_camera=sum(v['cross_camera_positive'] for r in changed for v in r['added']),
                    removed_gallery_positive=sum(v['cross_camera_positive'] and v['node']>=3368 for r in changed for v in r['removed']),
                    added_gallery_positive=sum(v['cross_camera_positive'] and v['node']>=3368 for r in changed for v in r['added']))
    extra={}
    if cfg['source']=='two_step_fitting':
        indirect=np.concatenate([r.pop('first_indirect_q') for r in rows if 'first_indirect_q' in r])
        extra=dict(indirect_q_quantiles=np.quantile(indirect,[0,.1,.5,.9,.99,1]).tolist(),
                   indirect_curvature_quantiles=np.quantile(beta*indirect/2,[0,.1,.5,.9,.99,1]).tolist(),
                   first_stage_score_increase_count=sum(r.get('first_score_increase_count',0) for r in rows),
                   score_comparison_note='Float diagnostic only; increases use 1e-8 relative guard and do not replace production certificates.',
                   indirect_swap_counts={key:dict(total=sum(not v['direct'] for r in rows for v in r.get(key,[])),
                        gallery_positive=sum(not v['direct'] and v['cross_camera_positive'] and v['node']>=3368 for r in rows for v in r.get(key,[]))) for key in ('removed','added')})
    dump(output/'queries.json',rows)
    dump(output/'summary.json',dict(time=now(),scope='all queries, at most first five batches, stop at first divergence',
          causal_claim=False, original_full_trajectory_reconstructed=False,
          counting_unit='Selected node occurrences at first ordered divergence; original cross_camera fields include query and gallery nodes.',
          accepted_run_sha256=sha(run/'FULL_ACCEPTANCE.json'),
          p_quantiles=np.quantile(probabilities,[0,.1,.5,.9,.99,1]).tolist(),
          fraction_direct_p_below_0p1=float(np.mean(probabilities<.1)),
          model=cfg['source'],
          mean_direct_source_increment=float(np.mean(increments)),
          direct_curvature_ratio_quantiles=np.quantile(curvature_ratio,[0,.1,.5,.9,.99,1]).tolist(),
          first_score_relative_change_quantiles=np.quantile([r['first_score_relative_change'] for r in rows if 'first_score_relative_change' in r],[0,.1,.5,.9,.99,1]).tolist(),
          groups={name:group(rs) for name,rs in [('all',rows),('AP_down',[r for r in rows if r['ap_difference'] < -1e-12]),
             ('AP_up',[r for r in rows if r['ap_difference'] > 1e-12]),('AP_same',[r for r in rows if abs(r['ap_difference']) <= 1e-12])]},
          query_details_sha256=sha(output/'queries.json'),**extra))
    print((output/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.config,a.output)
