"""Read-back verification from actual saved scores and original W/e.
No candidate solve or runner callback is used. The exact-input baseline/delta
enclosure and tree bound are shared locked mathematical utilities, not an
algorithmically independent proof of those utilities.
"""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
import mpmath as mp
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'adaptive-precision-v1'))
import adaptive_certificate as sc
from certificate import iv,bounds

def unpack(values):
    out=[]
    for t in values:
        if len(t)!=4 or any(type(x)!=int for x in t):raise ValueError('MP encoding')
        sign,man,exp,bc=t
        if sign!=0 or man<=0 or man.bit_length()!=bc or not man&1 or abs(exp)>100000:
            raise ValueError('Nonfinite/nonpositive/noncanonical MP score')
        out.append(mp.make_mpf(tuple(t)))
    return out

def certify_scores(w,e,values,theta):
    if theta not in (0.,.5):raise ValueError('Fixed theta only')
    p=sc.problem(w,e); old=mp.iv.dps
    # This validates saved points; it never corrects or replaces them.
    try:
        for dps in (24,40,100,160,240):
            mp.iv.dps=dps
            with mp.workdps(dps):
                data=sc.interval_data(p);roots=data['roots']
                z=[iv(x)/r for x,r in zip(values,roots)]
                if len(z)!=p['n']:raise ValueError('Score count')
                if theta:
                    try:
                        baseline,q,_=sc.refined_baseline(p,dps)
                        delta,_=sc.baseline_box(p,baseline)
                    except ArithmeticError:continue
                else:delta=iv(0)
                # Row-based flux differs from producer's undirected-edge scatter.
                g=[];th=iv(theta);one_minus=1-th;delta2=delta**2;zero=bounds(delta)[1]==0
                for i,row in enumerate(p['rows']):
                    total=data['d'][i]*z[i]-data['c'][i]
                    for j,wgt in row.items():
                        if i==j:continue
                        t=z[i]-z[j]
                        if theta==0 or zero:f=t
                        else:f=one_minus*t+th*delta*t/mp.iv.sqrt(delta2+t**2)
                        total+=iv(wgt)*f
                    g.append(total)
                relative=bounds(mp.iv.sqrt(sum(((v/r)**2 for v,r in zip(g,roots)),iv(0))/
                                          data['bnorm']))[1]
                if theta==0:
                    # Original inverse-positive system gives a tighter neutral enclosure.
                    eta=max(bounds(abs(v/r)/iv(b))[1] for v,r,b in zip(g,roots,p['b']))
                    if eta>=1:continue
                    boxes=[(bounds(iv(x)/(1+iv(eta)))[0],bounds(iv(x)/(1-iv(eta)))[1]) for x in values]
                else:
                    mus=sc.curvature_bounds(p,theta)
                    radii=[bounds(mp.iv.sqrt(sum((g[i]**2 for i in ids),iv(0)))/iv(mu))[1]
                           for ids,mu in zip(p['components'],mus)]
                    boxes=[bounds(iv(x)+r*mp.iv.mpf([-radii[p['labels'][i]],radii[p['labels'][i]]]))
                           for i,(x,r) in enumerate(zip(values,roots))]
                if relative>mp.mpf('1e-8') or any(lo<=0 for lo,hi in boxes):continue
                order=sorted(range(len(values)),key=lambda i:(values[i],i));valid=True;ties=0
                for k,i in enumerate(order[:5]):
                    for j in order[k+1:]:
                        if boxes[i][1]<boxes[j][0]:continue
                        if values[i]==values[j] and i<j and sc.exact_twins(p,i,j):
                            ties+=1;continue
                        valid=False;break
                    if not valid:break
                if valid:return dict(prefix=order[:5],relative=float(relative),dps=dps,tie_pairs=ties)
        raise ValueError('Saved actual scores not certified; no correction or relaxed tolerance')
    finally:mp.iv.dps=old

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

def verify_query(graph,full,similarity,query,prefix,rank,rows,theta,qn,depth,check=lambda:None):
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
            assert ev['returned_scores_sha256']==digest and ev['size']==len(u) and ev['theta']==theta
            c=certify_scores(graph[u][:,u],ew[u],values,theta)
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
