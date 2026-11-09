"""Describe the completed channel run's saved graph states; never solve/rank.

Uses the fixed retained queries and their initial/final frontiers. Identity
labels describe edge types after the experiment; they are never fed to the
method. Repeated edge/node exposures are not independent causal observations.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat


def frame(graph,cameras,pids,indices):
    u=np.asarray(indices,dtype=int);w=graph[u][:,u].tocsr()
    h=np.asarray(w.sum(axis=1)).ravel()
    rows=np.repeat(np.arange(len(u)),np.diff(w.indptr));cols=w.indices
    off=rows!=cols;same=cameras[u][rows]==cameras[u][cols]
    ds=np.bincount(rows,weights=np.where(off & same,w.data,0.),minlength=len(u))
    dx=np.bincount(rows,weights=np.where(off & ~same,w.data,0.),minlength=len(u))
    both=(ds>0)&(dx>0);count=(ds>0).astype(int)+(dx>0).astype(int);d=ds+dx
    fs=np.divide(d,count*ds,out=np.zeros(len(u)),where=ds>0)
    fx=np.divide(d,count*dx,out=np.zeros(len(u)),where=dx>0)
    factor=(np.where(same,fs[rows],fx[rows])+np.where(same,fs[cols],fx[cols]))/2
    keep=rows<cols;i,j=rows[keep],cols[keep]
    original=w.data[keep];modified=original*factor[keep]
    cross=cameras[u][i]!=cameras[u][j]
    known=(pids[u][i]>=0)&(pids[u][j]>=0)
    wrong=known&(pids[u][i]!=pids[u][j])
    result=dict(nodes=len(u),both_channels=int(both.sum()),
        cross_dominant_both=int((both&(dx>ds)).sum()),intra_dominant_both=int((both&(ds>dx)).sum()),
        edges=len(i),amplified_over_2=int((factor[keep]>2).sum()),
        amplified_over_5=int((factor[keep]>5).sum()),max_edge_factor=float(factor[keep].max(initial=0)))
    groups={'all':np.ones(len(i),dtype=bool),'cross':cross,'same_camera':~cross,
            'known':known,'different_identity':wrong,'different_identity_cross':wrong&cross,
            'different_identity_same_camera':wrong&~cross}
    for coordinate,denom in [('weight',np.ones(len(i))),('coupling',np.sqrt(h[i]*h[j]))]:
        for name,mask in groups.items():
            result[f'{coordinate}_original_{name}']=float((original[mask]/denom[mask]).sum())
            result[f'{coordinate}_modified_{name}']=float((modified[mask]/denom[mask]).sum())
    return result


def analyze(root):
    root=Path(root);cfg=json.loads((root/'configs/camera_channel.json').read_text())
    run=root/'runs/camera_channel'
    accepted=json.loads((run/'FULL_ACCEPTANCE.json').read_text())
    assert accepted['status']=='accepted_full_population' and accepted['queries']==cfg['queries']
    graph=sparse.load_npz(Path(cfg['cache'])/'graph.npz').astype(np.float64).tocsr()
    cameras=np.load(cfg['camera_ids_file'],allow_pickle=False)
    labels=loadmat(cfg['mat'],variable_names=['query_label','gallery_label'])
    pids=np.r_[labels['query_label'].ravel(),labels['gallery_label'].ravel()]
    assert graph.shape==(len(pids),len(pids)) and len(cameras)==len(pids)
    details=[];hashes={}
    for q in cfg['retained_queries']:
        path=run/'queries'/f'{q:04d}.trace.jsonl.gz'
        data=path.read_bytes();hashes[str(q)]=hashlib.sha256(data).hexdigest()
        record=json.loads((run/'queries'/f'{q:04d}.json').read_text())
        assert hashes[str(q)]==record['trace_sha256']
        with gzip.open(path,'rt') as f:
            first=f.readline();last=first
            for line in f:last=line
        for phase,line in [('initial',first),('final',last)]:
            row=json.loads(line)
            details.append(dict(query=q,phase=phase,iteration=row['iteration'],
                                **frame(graph,cameras,pids,row['unlabeled'])))
    aggregates={}
    for phase in ('initial','final'):
        items=[r for r in details if r['phase']==phase]
        keys=set(items[0])-{'query','phase','iteration','max_edge_factor'}
        total={k:sum(r[k] for r in items) for k in sorted(keys)}
        total['max_edge_factor']=max(r['max_edge_factor'] for r in items)
        total['cross_dominant_fraction_among_both']=total['cross_dominant_both']/max(1,total['both_channels'])
        for coordinate in ('weight','coupling'):
            for version in ('original','modified'):
                prefix=f'{coordinate}_{version}_'
                total[prefix+'cross_fraction']=total[prefix+'cross']/total[prefix+'all']
                total[prefix+'different_identity_fraction_known']=total[prefix+'different_identity']/total[prefix+'known']
        aggregates[phase]=total
    result=dict(scope='posthoc descriptions on fixed 32 retained queries, initial/final states; no new solve',
        caveat='Repeated exposures; trajectory-conditioned final states; not causal proof or a population sample of independent edges',
        identity_labels_used_only_for_analysis=True,frames=len(details),trace_sha256=hashes,
        aggregates=aggregates,details=details)
    output=run/'channel-mechanism.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),frames=len(details),aggregates=aggregates)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root)
