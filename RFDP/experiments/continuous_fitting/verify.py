"""Rebuild full trajectories from saved points, without running a solver.

Frontier, boundary, bridge and full-ranking reconstruction derives from the
previous verified reader. The new source uses its own equation certificate.
Producer and reader share certificate utilities; this is independent artifact
read-back, not two independent implementations of interval arithmetic.
"""
import json,gzip,hashlib
import numpy as np
from scipy.sparse.csgraph import connected_components
from .certified import unpack,certify_saved as certify_scores

def bridge_edges(graph,full,labels,active,sums):
    inside=np.flatnonzero(np.isin(labels,list(active)));choices=[]
    for comp in range(int(labels.max())+1):
        if comp in active:continue
        outside=np.flatnonzero(labels==comp);best=None
        for start in range(0,len(outside),128):
            ids=outside[start:start+128]
            block=np.asarray(full[np.ix_(ids,inside)],dtype=np.float64)/sums[ids,None]
            i,j=np.unravel_index(np.argmax(block),block.shape)
            candidate=(float(block[i,j]),int(ids[i]),int(inside[j]))
            if best is None or (-candidate[0],candidate[1:])<(-best[0],best[1:]):best=candidate
        if best is not None and best[0]>0:choices.append(best)
    choices.sort(key=lambda t:(-t[0],t[1],t[2]))
    return [[i,j,float(full[i,j])] for _,i,j in choices[:5]]

def verify_query(graph,full,similarity,query,prefix,rank,rows,source,qn,depth,check=lambda:None):
    n=graph.shape[0];prefix=list(map(int,prefix));rank=list(map(int,rank))
    assert len(prefix)==depth and prefix[0]==query and len(set(prefix))==depth
    assert all(0<=i<n for i in prefix)
    nc,labels=connected_components(graph,directed=False);sizes=np.bincount(labels)
    sums=np.asarray(full.sum(axis=1,dtype=np.float64))
    labeled=[query];mask=np.zeros(n,dtype=bool);mask[query]=True
    boundary=np.asarray(graph[[query]].sum(axis=0)).ravel();active={int(labels[query])}
    calls=batches=no_solve=bridges=edges=0;maximum=0.;reported=0.;max_size=0
    for t in rows:
        check();assert len(labeled)<depth and t['iteration']==batches
        assert t['labeled']==labeled
        expected_bridges=[]
        if len(labeled)==int(sizes[list(active)].sum()) and len(labeled)<n:
            expected_bridges=bridge_edges(graph,full,labels,active,sums)
            assert expected_bridges
        assert t['bridges']==expected_bridges
        ew=boundary.copy()
        for i,j,v in expected_bridges:assert not mask[i] and mask[j];ew[i]+=v
        a=(ew>0)&~mask
        # Reachability from direct rows, independent of producer's sparse matvec.
        b=np.zeros(n,dtype=bool)
        for node in np.flatnonzero(a):b[graph.indices[graph.indptr[node]:graph.indptr[node+1]]]=True
        b&=~mask&~a;u=np.flatnonzero(a|b)
        assert 0<len(u)<=8192
        max_size=max(max_size,len(u))
        assert t['direct']==np.flatnonzero(a).tolist() and t['indirect']==np.flatnonzero(b).tolist()
        assert t['unlabeled']==u.tolist()
        residual=float(t['residual']);assert np.isfinite(residual) and 0<=residual<=1e-8
        reported=max(reported,residual)
        if len(u)<=5:
            assert t['returned_scores'] is None and t['event'] is None and residual==0
            selected=u[:depth-len(labeled)].tolist();no_solve+=1
        else:
            values=unpack(t['returned_scores']);ev=t['event']
            digest=hashlib.sha256(json.dumps(t['returned_scores'],separators=(',',':')).encode()).hexdigest()
            assert ev['returned_scores_sha256']==digest and ev['size']==len(u) and ev['source']==source
            c=certify_scores(graph[u][:,u],ew[u],values,source=source)
            selected=u[c['prefix']][:depth-len(labeled)].tolist()
            assert c['prefix']==ev['returned_prefix_local']
            maximum=max(maximum,c['relative']);calls+=1
        assert selected==t['selected']==prefix[len(labeled):len(labeled)+len(selected)]
        mask[selected]=True;labeled.extend(selected)
        for node in selected:
            begin,end=graph.indptr[node:node+2]
            boundary[graph.indices[begin:end]]+=graph.data[begin:end]
        active.update(map(int,labels[selected]));batches+=1;bridges+=bool(expected_bridges);edges+=len(expected_bridges)
    assert labeled==prefix and batches==calls+no_solve
    used=set(prefix);expected=[i for i in prefix if i>=qn]
    expected+=sorted((i for i in range(qn,n) if i not in used),key=lambda i:(-float(similarity[i]),i))
    assert rank==expected and sorted(rank)==list(range(qn,n))
    return dict(solver_calls=calls,iterations=batches,no_solve_batches=no_solve,
                bridge_events=bridges,bridge_edges=edges,max_recomputed_residual=maximum,
                max_reported_residual=reported,max_unlabeled=max_size)

def read_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
