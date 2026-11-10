"""Describe TH perturbations on saved frontiers, without solving or re-ranking.

The floating-point diagnostics are descriptive, not score certificates or
counterfactual retrieval results. Initial/middle/final states are fixed per
retained query; the middle and final states are conditioned on the TH run.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from experiments.continuous_fitting.certified import unpack


def frame(graph, row, theta, alpha, beta):
    u=np.asarray(row['unlabeled'],dtype=int)
    targets=np.asarray(row['labeled'],dtype=int)
    w=graph[u][:,u].tocsr()
    e=np.asarray(graph[u][:,targets].sum(axis=1)).ravel()
    where={int(v):i for i,v in enumerate(u)}
    for outside,inside,value in row['bridges']:
        assert inside in targets
        e[where[outside]]+=value
    h=np.asarray(w.sum(axis=1)).ravel();root=np.sqrt(h)
    m=e/len(targets);s=np.sqrt(np.maximum(0.,(1.-m)*(1.+m)))
    assert np.all((m>=0)&(m<=1))
    values=np.asarray([float(x) for x in unpack(row['returned_scores'])])
    z=values/root
    base=(h*z-w@z)/root
    geometric=((s*(w@s)+m*m.sum())*z-s*(w@(s*z))-m*(m@z))/root
    forcing=theta*(geometric-base)
    rhs=np.where(e>0,alpha,beta)
    i=np.repeat(np.arange(len(u)),np.diff(w.indptr));j=w.indices
    keep=i<j;i,j=i[keep],j[keep]
    coupling=w.data[keep]/(root[i]*root[j])
    factor=(1.-theta)+theta*s[i]*s[j]
    r=m/root
    rank_one=theta*max(0.,float(r.sum()**2-r@r))/2.
    modified=float(coupling@factor)+rank_one
    return dict(target_count=len(targets),frontier=len(u),direct=int((e>0).sum()),
        m_max=float(m.max()),m_mean_direct=float(m[e>0].mean()) if np.any(e>0) else 0.,
        rank_one_coupling_share=rank_one/modified if modified else 0.,
        existing_edge_attenuation=float(coupling@(1.-factor)/coupling.sum()) if coupling.sum() else 0.,
        original_rhs_relative_shift=float(np.max(np.abs(forcing)/rhs)))


def analyze(root):
    root=Path(root);run=root/'runs/th_target'
    cfg=json.loads((root/'configs/th_target.json').read_text())
    accepted=json.loads((run/'FULL_ACCEPTANCE.json').read_text())
    assert accepted['status']=='accepted_full_population' and accepted['queries']==cfg['queries']
    graph=sparse.load_npz(Path(cfg['cache'])/'graph.npz').astype(np.float64).tocsr()
    details=[];hashes={}
    for q in cfg['retained_queries']:
        path=run/'queries'/f'{q:04d}.trace.jsonl.gz'
        hashes[str(q)]=hashlib.sha256(path.read_bytes()).hexdigest()
        metadata=json.loads((run/'queries'/f'{q:04d}.json').read_text())
        assert hashes[str(q)]==metadata['trace_sha256']
        with gzip.open(path,'rt') as stream:
            rows=[json.loads(line) for line in stream]
        for phase,index in [('initial',0),('middle',len(rows)//2),('final',len(rows)-1)]:
            row=rows[index]
            if row['returned_scores'] is None:continue
            details.append(dict(query=q,phase=phase,iteration=row['iteration'],
                **frame(graph,row,cfg['theta'],cfg['alpha'],cfg['beta'])))
    aggregates={}
    for phase in ('initial','middle','final'):
        rows=[x for x in details if x['phase']==phase]
        aggregates[phase]={key:dict(median=float(np.median([x[key] for x in rows])),
            minimum=float(min(x[key] for x in rows)),maximum=float(max(x[key] for x in rows)))
            for key in ('target_count','frontier','m_max','m_mean_direct',
                        'rank_one_coupling_share','existing_edge_attenuation','original_rhs_relative_shift')}
    result=dict(scope='Saved TH states only; no solve or ranking; descriptive float64 quantities',
        limitation='Fixed retained queries, not independent samples; later states depend on TH trajectory; no causal metric attribution',
        frames=len(details),trace_sha256=hashes,aggregates=aggregates,details=details)
    path=run/'th-mechanism.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),aggregates=aggregates)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root)
