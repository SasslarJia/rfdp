"""Independent outward MP certificate for small fixtures, max 64 nodes.
Exact binary W/e -> exact sums/rational d. Delta is enclosed from the exact
original hybrid solution; the rounded solver's delta is never trusted.
Production-scale baseline enclosures and vectorized certificates are pending.
"""
from fractions import Fraction as F
from collections import deque
import numpy as np
import mpmath as mp
from mpmath.libmp import to_rational
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def rat(x):
    if isinstance(x,F):return x
    if isinstance(x,mp.mpf):
        a,b=to_rational(x._mpf_);return F(a,b)
    return F(float(x))
def iv(x):
    q=rat(x);return mp.iv.mpf(q.numerator)/mp.iv.mpf(q.denominator)
def scalar(q):
    q=rat(q);return mp.mpf(q.numerator)/q.denominator
def bounds(v):return mp.make_mpf(v._mpi_[0]),mp.make_mpf(v._mpi_[1])
def absolute(v):return max(abs(t) for t in bounds(v))

def exact_problem(w,e,alpha=5.,beta=25.):
    w=sparse.csr_matrix(w,dtype=float);n=len(e)
    if n>64:raise ValueError('Small-fixture certificate limit; no dense allocation above 64')
    if w.shape!=(n,n) or not n or not w.has_canonical_format or (w-w.T).nnz:raise ValueError('Invalid symmetric input')
    if not np.isfinite(w.data).all() or (w.data<0).any() or not np.isfinite(e).all() or (np.asarray(e)<0).any():raise ValueError('Invalid input values')
    rows=[[(int(w.indices[k]),rat(w.data[k])) for k in range(w.indptr[i],w.indptr[i+1])] for i in range(n)]
    h=[sum((v for _,v in row),F(0)) for row in rows]
    if any(x<=0 for x in h):raise ValueError('Nonpositive degree')
    ee=[rat(v) for v in e];aa=rat(alpha)
    d=[aa*x*y/(x+y) for x,y in zip(h,ee)]
    b=[rat(alpha if x>0 else beta) for x in ee]
    if min(b)<=0:raise ValueError('Nonpositive source')
    nc,labels=connected_components(w,directed=False)
    if any(not any(d[i]>0 for i in range(n) if labels[i]==k) for k in range(nc)):raise ValueError('Unanchored')
    edges=[(i,j,v) for i,row in enumerate(rows) for j,v in row if j>i and v>0]
    return dict(n=n,rows=rows,h=h,d=d,b=b,edges=edges,labels=labels,nc=nc)

def spectral_lower(p,theta):
    # For each anchored spanning tree: |x_i|² <= (1/d_root +
    # sum_path 1/((1-theta)w))*[sum_edges (1-theta)w(dx)²+sum d*x²].
    # Sum over nodes; max over connected components gives 1/mu.
    m=F(1)-rat(theta);totals=[]
    for k in range(p['nc']):
        ids=[i for i in range(p['n']) if p['labels'][i]==k]
        root=max(ids,key=lambda i:p['d'][i]);dist={root:1/p['d'][root]};queue=deque([root])
        while queue:
            i=queue.popleft()
            for j,w in p['rows'][i]:
                if j!=i and w>0 and j not in dist:
                    dist[j]=dist[i]+1/(m*w);queue.append(j)
        if len(dist)!=len(ids):raise ValueError('Spanning tree incomplete')
        totals.append(sum(dist.values(),F(0)))
    return 1/max(totals)

def baseline_enclosure(p):
    n=p['n'];hroot=[mp.sqrt(scalar(v)) for v in p['h']]
    a=mp.matrix(n)
    for i,row in enumerate(p['rows']):
        a[i,i]=scalar(sum((w for j,w in row if i!=j),F(0))+p['d'][i])
        for j,w in row:
            if i!=j:a[i,j]=-scalar(w)
    c=[r*scalar(b) for r,b in zip(hroot,p['b'])]
    z=list(mp.lu_solve(a,mp.matrix(c)))
    if min(z)<=0:raise ArithmeticError('Invalid MP baseline candidate')
    ci=[mp.iv.sqrt(iv(h))*iv(b) for h,b in zip(p['h'],p['b'])]
    zi=[iv(v) for v in z];eta=mp.mpf(0)
    for i,row in enumerate(p['rows']):
        g=sum((iv(w)*(zi[i]-zi[j]) for j,w in row if i!=j),iv(0))+iv(p['d'][i])*zi[i]-ci[i]
        eta=max(eta,bounds(abs(g)/ci[i])[1])
    if eta>=1:raise ArithmeticError('Baseline enclosure not certified')
    enclosure=[(bounds(v/(1+iv(eta)))[0],bounds(v/(1-iv(eta)))[1]) for v in zi]
    # Exact constant solution iff all d>0 and (c_i/d_i)^2 is constant.
    constant=all(v>0 for v in p['d']) and len({h*b*b/(d*d) for h,b,d in zip(p['h'],p['b'],p['d'])})==1
    if constant or not p['edges']:delta=iv(0)
    else:
        boxes=[mp.iv.mpf([iv(lo).a,iv(hi).b]) for lo,hi in enclosure]
        mass=sum((w for _,_,w in p['edges']),F(0))
        delta=mp.iv.sqrt(sum((iv(w)*(boxes[i]-boxes[j])**2 for i,j,w in p['edges']),iv(0))/iv(mass))
        if bounds(delta)[0]<=0:raise ArithmeticError('Delta zero/nonzero not resolved')
    return z,delta,eta

def certify(w,e,z,theta=.5,alpha=5.,beta=25.,dps=80,tolerance=1e-8):
    if not 0<=theta<1:raise ValueError('Theta domain')
    old=mp.iv.dps;mp.iv.dps=dps
    try:
        with mp.workdps(dps):
            p=exact_problem(w,e,alpha,beta);_,delta,baseline_eta=baseline_enclosure(p)
            zz=[iv(x) for x in z];di=[iv(x) for x in p['d']]
            roots=[mp.iv.sqrt(iv(h)) for h in p['h']]
            cc=[r*iv(b) for r,b in zip(roots,p['b'])]
            g=[d*v-c for d,v,c in zip(di,zz,cc)]
            for i,j,w in p['edges']:
                t=zz[i]-zz[j]
                if theta==0 or bounds(delta)[1]==0:flux=iv(w)*t
                else:flux=iv(w)*((1-iv(theta))*t+iv(theta)*delta*t/mp.iv.sqrt(delta**2+t**2))
                g[i]+=flux;g[j]-=flux
            norm=mp.iv.sqrt(sum((x**2 for x in g),iv(0)))
            relative=bounds(mp.iv.sqrt(sum(((x/r)**2 for x,r in zip(g,roots)),iv(0))/sum((iv(b)**2 for b in p['b']),iv(0))))[1]
            mu=spectral_lower(p,theta);radius=bounds(norm/iv(mu))[1]
            score_boxes=[r*(v+mp.iv.mpf([-iv(radius).b,iv(radius).b])) for r,v in zip(roots,zz)]
            intervals=[bounds(v) for v in score_boxes]
            point_scores=[mp.sqrt(scalar(h))*scalar(x) for h,x in zip(p['h'],z)]
            order=sorted(range(len(z)),key=lambda i:(point_scores[i],i))
            ordered=all(intervals[order[i]][1]<min(intervals[j][0] for j in order[i+1:]) for i in range(min(5,len(z)-1)))
            positive=all(lo>0 for lo,hi in intervals)
            accepted=relative<=mp.mpf(tolerance) and positive and ordered
            return dict(accepted=bool(accepted),ordered_first_five=bool(ordered),positive=bool(positive),
                nonlinear_relative_residual_upper=mp.nstr(relative,24),score_coordinate_radius_z=mp.nstr(radius,24),
                spectral_lower_exact=[mu.numerator,mu.denominator],delta_interval=[mp.nstr(v,30) for v in bounds(delta)],
                baseline_relative_component_residual_upper=mp.nstr(baseline_eta,24),intervals=intervals,
                point_scores=point_scores,problem=p,delta_box=delta,
                scope='small-fixture exact-binary-input nonlinear objective; no real retrieval qualification')
    finally:mp.iv.dps=old

def independent_nonlinear_oracle(cert,initial,theta=.5,dps=80):
    # Separate MP Newton assembly; used only to cross-check a valid enclosure.
    with mp.workdps(dps):
        p=cert['problem'];n=p['n'];delta=sum(bounds(cert['delta_box']))/2
        z=mp.matrix([scalar(v) for v in initial]);root=[mp.sqrt(scalar(v)) for v in p['h']]
        for it in range(20):
            g=mp.matrix([scalar(d)*z[i]-root[i]*scalar(p['b'][i]) for i,d in enumerate(p['d'])])
            H=mp.diag([scalar(d) for d in p['d']])
            for i,j,ww in p['edges']:
                w=scalar(ww);t=z[i]-z[j]
                rr=mp.sqrt(1+(t/delta)**2) if delta else mp.mpf(1)
                a=1-theta+theta/rr;k=1-theta+theta/rr**3
                g[i]+=w*a*t;g[j]-=w*a*t
                H[i,i]+=w*k;H[j,j]+=w*k;H[i,j]-=w*k;H[j,i]-=w*k
            if max(abs(x) for x in g)<mp.mpf('1e-60'):break
            z-=mp.lu_solve(H,g)
        else:raise ArithmeticError('MP oracle failed')
        return [z[i]*root[i] for i in range(n)]

