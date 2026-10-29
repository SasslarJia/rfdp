"""Sparse mathematical/engineering qualification implementation.
Not installed in the retrieval runner. W/e and original hybrid fitting are fixed.
"""
import math
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

def assemble(w,e,alpha=5.,beta=25.):
    w=sparse.csr_matrix(w,dtype=float).copy();w.sort_indices()
    e=np.array(e,dtype=float,copy=True)
    if not w.has_canonical_format:raise ValueError('Duplicate coordinates')
    w.eliminate_zeros()
    if w.shape!=(len(e),len(e)) or not 0<len(e)<=8192:raise ValueError('Domain size; no truncation')
    if not np.isfinite(w.data).all() or not np.isfinite(e).all() or (w.data<0).any() or (e<0).any():raise ValueError('Invalid weights')
    if not np.isfinite([alpha,beta]).all() or min(alpha,beta)<=0:raise ValueError('Invalid fitting')
    if (w-w.T).nnz:raise ValueError('Exact symmetry required')
    n,labels=connected_components(w,directed=False)
    if any(not np.any(e[labels==i]>0) for i in range(n)):raise ValueError('Unanchored component')
    h=np.array([math.fsum(w.data[w.indptr[i]:w.indptr[i+1]]) for i in range(len(e))])
    if (h<=0).any():raise ValueError('Nonpositive original degree')
    edges=sparse.triu(w,k=1,format='coo');ii,jj,weight=edges.row,edges.col,edges.data
    d=alpha*h*e/(h+e);c=np.sqrt(h)*np.where(e>0,alpha,beta)
    p=dict(w=w,e=e,h=h,ii=ii,jj=jj,weight=weight,d=d,c=c,alpha=alpha,beta=beta)
    p['a']=laplacian(p,weight)+sparse.diags(d)
    z0=spsolve(p['a'].tocsc(),c)
    if not np.isfinite(z0).all() or (z0<=0).any():raise ArithmeticError('Initial original solve failed')
    t=z0[ii]-z0[jj]
    delta=math.sqrt(float(np.dot(weight,t*t)/weight.sum())) if len(weight) else 0.
    p.update(z0=z0,delta=delta)
    return p

def laplacian(p,weight):
    i,j=p['ii'],p['jj'];n=len(p['e'])
    diag=np.bincount(i,weights=weight,minlength=n)+np.bincount(j,weights=weight,minlength=n)
    return sparse.diags(diag)+sparse.coo_matrix((np.r_[-weight,-weight],(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()

def terms(p,z,theta=.5):
    if not 0<=theta<1:raise ValueError('0 <= theta < 1')
    i,j,w,d,c,delta=[p[k] for k in ['ii','jj','weight','d','c','delta']]
    t=z[i]-z[j];n=len(z)
    if theta==0 or delta==0:
        conductance=np.ones(len(w));curvature=conductance
    else:
        ratio=np.hypot(1.,t/delta)
        conductance=(1-theta)+theta/ratio
        curvature=(1-theta)+theta/ratio**3
    flux=w*conductance*t
    g=np.bincount(i,weights=flux,minlength=n)-np.bincount(j,weights=flux,minlength=n)+d*z-c
    H=laplacian(p,w*curvature)+sparse.diags(d)
    return g,H,conductance

def energy_change(p,z,step,theta):
    ld=np.longdouble;i,j=p['ii'],p['jj']
    z=z.astype(ld);step=step.astype(ld);t=z[i]-z[j];dt=step[i]-step[j]
    if theta==0 or p['delta']==0:change=t*dt+ld('.5')*dt*dt
    else:
        delta=ld(p['delta']);r=np.hypot(ld(1),t/delta);rn=np.hypot(ld(1),(t+dt)/delta)
        change=(1-theta)*(t*dt+ld('.5')*dt*dt)+theta*(2*t*dt+dt*dt)/(r+rn)
    return np.sum(p['weight'].astype(ld)*change,dtype=ld)+np.sum(p['d'].astype(ld)*(z*step+ld('.5')*step*step),dtype=ld)-np.sum(p['c'].astype(ld)*step,dtype=ld)

def solve(w,e,theta=.5,alpha=5.,beta=25.,rtol=1e-11,maxiter=100):
    p=assemble(w,e,alpha,beta);z=p['z0'].copy();history=[]
    for iteration in range(maxiter+1):
        g,H,a=terms(p,z,theta);relative=float(np.linalg.norm(g)/np.linalg.norm(p['c']))
        history.append(dict(iteration=iteration,relative_gradient=relative,hessian_nnz=H.nnz))
        if relative<=rtol:
            return dict(z=z,scores=np.sqrt(p['h'])*z,conductance=a,model=p,history=history)
        if iteration==maxiter:raise ArithmeticError('Nonlinear convergence not achieved')
        step=spsolve(H.tocsc(),-g);slope=float(np.dot(g,step))
        if not np.isfinite(step).all() or not slope<0:raise ArithmeticError('Newton direction')
        rate=1.
        for trial in range(60):
            change=energy_change(p,z,rate*step,theta)
            if change<=1e-4*rate*slope:
                z+=rate*step;break
            rate*=.5
        else:raise ArithmeticError('Armijo failed; no objective fallback')

