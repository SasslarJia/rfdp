"""Sparse exact-input nonlinear certification. No dense MP solve at any size.
Candidate refinement is numerical; outward enclosures alone decide acceptance.
Shared delta is fixed from the exact original stage solution.
"""
from pathlib import Path
import sys, math
from collections import deque
from fractions import Fraction as F
import numpy as np
import mpmath as mp
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'sparse-certificate-v1'))
from certificate import rat,iv,scalar,bounds
from sparse_solver import assemble,laplacian

def problem(w,e,alpha=5.,beta=25.):
    w=sparse.csr_matrix(w,dtype=float).copy();e=np.asarray(e,dtype=float)
    w.sort_indices()
    if w.shape!=(len(e),len(e)) or not 0<len(e)<=8192 or not w.has_canonical_format:
        raise ValueError('Invalid shape/canonical input; 8192 is a stop, not truncation')
    if not np.isfinite(w.data).all() or not np.isfinite(e).all() or (w.data<0).any() or (e<0).any():
        raise ValueError('Invalid binary weights')
    if not np.isfinite([alpha,beta]).all() or min(alpha,beta)<=0 or (w-w.T).nnz:
        raise ValueError('Invalid fitting or symmetry')
    w.eliminate_zeros();n=len(e)
    rows=[{int(w.indices[k]):rat(w.data[k]) for k in range(w.indptr[i],w.indptr[i+1])} for i in range(n)]
    h=[sum(row.values(),F(0)) for row in rows]
    if min(h)<=0:raise ValueError('Nonpositive h')
    ee=list(map(rat,e));d=[rat(alpha)*x*y/(x+y) for x,y in zip(h,ee)]
    b=[rat(alpha if y>0 else beta) for y in ee]
    nc,labels=connected_components(w,directed=False)
    components=[[] for _ in range(nc)]
    for i,k in enumerate(labels):components[k].append(i)
    if any(not any(d[i]>0 for i in ids) for ids in components):raise ValueError('Unanchored')
    edges=[(i,j,v) for i,row in enumerate(rows) for j,v in row.items() if i<j]
    return dict(n=n,w=w,e=e,rows=rows,h=h,d=d,b=b,edges=edges,
                labels=labels,components=components,alpha=alpha,beta=beta)

def prepare_mp(p):
    roots=[mp.sqrt(scalar(h)) for h in p['h']]
    return dict(d=[scalar(x) for x in p['d']],c=[r*scalar(b) for r,b in zip(roots,p['b'])],
                roots=roots,edges=[(i,j,scalar(w)) for i,j,w in p['edges']])

def gradient(q,z,theta=0.,delta=0):
    g=[d*x-c for d,x,c in zip(q['d'],z,q['c'])]
    for i,j,w in q['edges']:
        t=z[i]-z[j]
        f=w*t if theta==0 or delta==0 else w*((1-theta)*t+theta*delta*t/mp.sqrt(delta*delta+t*t))
        g[i]+=f;g[j]-=f
    return g

def refined_baseline(p,dps):
    q=prepare_mp(p);float_model=assemble(p['w'],p['e'],p['alpha'],p['beta'])
    lu=splu(float_model['a'].tocsc());z=[scalar(v) for v in float_model['z0']]
    history=[];target=mp.power(10,-min(60,dps//2))
    for it in range(13):
        g=gradient(q,z);rel=max(abs(v)/c for v,c in zip(g,q['c']));history.append(mp.nstr(rel,12))
        if rel<target and min(z)>0:return z,q,history
        if it==12:break
        scale=max(abs(v) for v in g)
        step=lu.solve(np.array([float(-v/scale) for v in g]))
        if not np.isfinite(step).all():break
        z=[v+scale*scalar(t) for v,t in zip(z,step)]
    raise ArithmeticError('Sparse MP original refinement failed; no dense fallback '+str(history))

def baseline_box(p,z):
    roots=[mp.iv.sqrt(iv(h)) for h in p['h']]
    ci=[r*iv(b) for r,b in zip(roots,p['b'])]
    zz=[iv(v) for v in z]
    g=[iv(d)*v-c for d,v,c in zip(p['d'],zz,ci)]
    for i,j,w in p['edges']:
        f=iv(w)*(zz[i]-zz[j]);g[i]+=f;g[j]-=f
    eta=max(bounds(abs(v)/c)[1] for v,c in zip(g,ci))
    if eta>=1 or min(z)<=0:raise ArithmeticError('Original M-matrix enclosure failed')
    boxes=[mp.iv.mpf([bounds(v/(1+iv(eta)))[0],bounds(v/(1-iv(eta)))[1]]) for v in zz]
    # Zero edge differences are permitted separately on each component.
    constant=all(all(p['d'][i]>0 for i in ids) and
                 len({p['h'][i]*p['b'][i]**2/p['d'][i]**2 for i in ids})==1
                 for ids in p['components'])
    if constant or not p['edges']:delta=iv(0)
    else:
        mass=sum((w for _,_,w in p['edges']),F(0))
        delta=mp.iv.sqrt(sum((iv(w)*(boxes[i]-boxes[j])**2 for i,j,w in p['edges']),iv(0))/iv(mass))
        if bounds(delta)[0]<=0:raise ArithmeticError('Exact delta zero/nonzero unresolved')
    return delta,eta

def curvature_bounds(p,theta):
    # Outward tree path sums avoid huge exact-Fraction denominators along long paths.
    # C is an upper bound on trace of a path-based inverse-energy majorant.
    # 1/C is a LOWER eigenvalue bound; all rounding remains outward.
    result=[]
    for ids in p['components']:
        root=max(ids,key=lambda i:p['d'][i]);dist={root:1/iv(p['d'][root])};todo=deque([root])
        while todo:
            i=todo.popleft()
            for j,w in p['rows'][i].items():
                if j!=i and j not in dist:
                    dist[j]=dist[i]+1/((1-iv(theta))*iv(w));todo.append(j)
        if len(dist)!=len(ids):raise ArithmeticError('Tree incomplete')
        C=sum(dist.values(),iv(0));mu=bounds(1/C)[0]
        if mu<=0:raise ArithmeticError('No positive curvature enclosure')
        result.append(mu)
    return result

def exact_twins(p,i,j):
    if i==j:return True
    if any(p[k][i]!=p[k][j] for k in ['h','d','b']):return False
    ri,rj=p['rows'][i],p['rows'][j]
    if ri.get(i,F(0))!=rj.get(j,F(0)):return False
    # Transposition preserves W, h, e-derived d, c and the shared scalar delta.
    # Uniqueness then fixes z_i=z_j, and equal h fixes psi_i=psi_j.
    return {k:v for k,v in ri.items() if k not in (i,j)}=={k:v for k,v in rj.items() if k not in (i,j)}

def first_five(p,points,boxes,allow_ties):
    left=set(range(p['n']));prefix=[];ties=[]
    for _ in range(min(5,p['n'])):
        i=min(left,key=lambda j:(points[j],j))
        candidates=[j for j in left if j<i and boxes[j][0]<=boxes[i][1] and
                    boxes[i][0]<=boxes[j][1] and allow_ties and exact_twins(p,i,j)]
        if candidates:i=min(candidates)
        for j in left:
            if i==j or boxes[i][1]<boxes[j][0]:continue
            if allow_ties and i<j and exact_twins(p,i,j):
                ties.append([i,j]);continue
            return None,ties
        prefix.append(i);left.remove(i)
    return prefix,ties

def check_candidate(p,z,delta,theta,dps,allow_ties=True):
    if len(z)!=p['n'] or any(not mp.isfinite(x) for x in z):raise ValueError('Invalid candidate')
    zz=[iv(x) for x in z];roots=[mp.iv.sqrt(iv(h)) for h in p['h']]
    g=[iv(d)*x-r*iv(b) for d,x,r,b in zip(p['d'],zz,roots,p['b'])]
    zero=bounds(delta)[1]==0
    for i,j,w in p['edges']:
        t=zz[i]-zz[j]
        f=iv(w)*t if theta==0 or zero else iv(w)*((1-iv(theta))*t+iv(theta)*delta*t/mp.iv.sqrt(delta**2+t**2))
        g[i]+=f;g[j]-=f
    relative=bounds(mp.iv.sqrt(sum(((x/r)**2 for x,r in zip(g,roots)),iv(0))/
                                      sum((iv(b)**2 for b in p['b']),iv(0))))[1]
    mus=curvature_bounds(p,theta);radii=[]
    for ids,mu in zip(p['components'],mus):
        radii.append(bounds(mp.iv.sqrt(sum((g[i]**2 for i in ids),iv(0)))/iv(mu))[1])
    boxes=[bounds(r*(x+mp.iv.mpf([-radii[p['labels'][i]],radii[p['labels'][i]]])))
           for i,(r,x) in enumerate(zip(roots,zz))]
    points=[mp.sqrt(scalar(h))*scalar(x) for h,x in zip(p['h'],z)]
    prefix,ties=first_five(p,points,boxes,allow_ties)
    positive=all(lo>0 for lo,hi in boxes)
    return dict(accepted=bool(relative<=mp.mpf('1e-8') and positive and prefix is not None),
                relative=relative,mu=mus,radius=radii,intervals=boxes,points=points,prefix=prefix,
                exact_transposition_ties=ties,positive=positive,dps=dps)

def refine_nonlinear(p,q,z,delta,theta,dps):
    # Sparse float Hessian is only a correction preconditioner.
    # Residual and accumulated iterate use arbitrary precision; final IV check is independent.
    mid=sum(bounds(delta))/2;z=list(z);history=[]
    base=assemble(p['w'],p['e'],p['alpha'],p['beta'])
    target=mp.power(10,-min(60,dps//2))
    for it in range(13):
        g=gradient(q,z,scalar(theta),mid);rel=max(abs(v)/c for v,c in zip(g,q['c']))
        history.append(mp.nstr(rel,12))
        if rel<target:return z,history
        if it==12:break
        t=np.array([float(z[i]-z[j]) for i,j,_ in p['edges']])
        k=np.ones(len(t)) if theta==0 or mid==0 else (1-theta)+theta/np.hypot(1,t/float(mid))**3
        H=laplacian(base,base['weight']*k)+sparse.diags(base['d'])
        scale=max(abs(v) for v in g);lu=splu(H.tocsc())
        step=lu.solve(np.array([float(-v/scale) for v in g]))
        if not np.isfinite(step).all():break
        trial=[v+scale*scalar(t) for v,t in zip(z,step)]
        newg=gradient(q,trial,scalar(theta),mid)
        if max(abs(v)/c for v,c in zip(newg,q['c']))>=rel:break
        z=trial
    raise ArithmeticError('Sparse nonlinear MP correction failed; no tolerance fallback '+str(history))

def certify(w,e,z,theta=.5,alpha=5.,beta=25.,refine=True,allow_ties=True,precisions=(80,120)):
    if not 0<=theta<1:raise ValueError('Theta domain')
    p=problem(w,e,alpha,beta);old=mp.iv.dps;attempts=[]
    try:
        for dps in precisions:
            mp.iv.dps=dps
            with mp.workdps(dps):
                z0,q,bh=refined_baseline(p,dps);delta,eta=baseline_box(p,z0)
                candidate=[scalar(x) for x in z]
                cert=check_candidate(p,candidate,delta,theta,dps,allow_ties)
                nh=[];initial=cert['accepted']
                if not initial and refine:
                    candidate,nh=refine_nonlinear(p,q,candidate,delta,theta,dps)
                    cert=check_candidate(p,candidate,delta,theta,dps,allow_ties)
                attempts.append(dict(dps=dps,baseline_refinement=bh,nonlinear_refinement=nh,
                                     initial_candidate_certified=initial,accepted=cert['accepted']))
                cert.update(problem=p,delta_box=delta,baseline_eta=eta,z=candidate,attempts=attempts,
                            dense_MP_used=False,AP_evaluated=False)
                if cert['accepted'] or not refine:return cert
        return cert
    finally:mp.iv.dps=old

def summary(c):
    return dict(accepted=c['accepted'],dps=c['dps'],relative=mp.nstr(c['relative'],24),
                max_z_radius=mp.nstr(max(c['radius']),24),min_mu=mp.nstr(min(c['mu']),24),
                delta_interval=[mp.nstr(v,30) for v in bounds(c['delta_box'])],
                baseline_eta=mp.nstr(c['baseline_eta'],24),prefix=c['prefix'],
                exact_transposition_ties=c['exact_transposition_ties'],attempts=c['attempts'],
                dense_MP_used=False)

