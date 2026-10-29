"""Small-graph mathematical prototype, NOT a qualified retrieval runner.
Only the inner smoothness energy changes. Original local W, h, e, hybrid
fitting coefficients and output coordinates are held fixed within each solve.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def assemble(w, e, alpha=5., beta=25.):
    w=np.asarray(w,dtype=float);e=np.asarray(e,dtype=float)
    if w.ndim!=2 or w.shape!=(len(e),len(e)) or not len(e) or len(e)>128:
        raise ValueError("Prototype domain is 1..128 nodes; no truncation")
    if not np.isfinite(w).all() or not np.isfinite(e).all() or (w<0).any() or (e<0).any() or not np.array_equal(w,w.T):
        raise ValueError("Finite nonnegative exactly symmetric W and nonnegative e required")
    n,labels=connected_components(sparse.csr_matrix(w),directed=False)
    if any(not np.any(e[labels==k]>0) for k in range(n)):
        raise ValueError("Original hybrid problem has an unanchored component")
    h=w.sum(1)
    if np.any(h<=0):raise ValueError("Nonpositive original local degree")
    ii,jj=np.where(np.triu(w,1)>0);weight=w[ii,jj]
    incidence=np.zeros((len(ii),len(e)))
    incidence[np.arange(len(ii)),ii]=1.;incidence[np.arange(len(ii)),jj]=-1.
    lap=np.einsum('ki,k,kj->ij',incidence,weight,incidence,optimize=False)
    d=alpha*h*e/(h+e);c=np.sqrt(h)*np.where(e>0,alpha,beta)
    a=lap+np.diag(d);z0=np.linalg.solve(a,c)
    differences=incidence@z0
    delta=float(np.sqrt(np.dot(weight,differences**2)/weight.sum())) if len(weight) else 0.
    return dict(w=w.copy(),e=e.copy(),h=h,incidence=incidence,weight=weight,d=d,c=c,a=a,z0=z0,delta=delta)

def terms(p,z,theta=.5):
    if not 0<=theta<1:raise ValueError("Fixed proposal requires 0 <= theta < 1")
    b,w,d,c,delta=[p[k] for k in ['incidence','weight','d','c','delta']]
    t=b@z
    if theta==0 or delta==0:
        energy=.5*np.dot(w,t*t)+.5*np.dot(d,z*z)-c@z
        return energy,p['a']@z-c,p['a'],np.ones(len(w))
    r=np.hypot(1.,t/delta)
    # stable Charbonnier value: delta^2*(r-1) = t^2/(r+1).
    phi=(1-theta)*.5*t*t+theta*t*t/(r+1)
    conductance=(1-theta)+theta/r
    curvature=(1-theta)+theta/r**3
    energy=np.dot(w,phi)+.5*np.dot(d,z*z)-c@z
    gradient=b.T@(w*conductance*t)+d*z-c
    hessian=np.einsum('ki,k,kj->ij',b,w*curvature,b,optimize=False)+np.diag(d)
    return energy,gradient,hessian,conductance

def energy_change(p,z,step,theta):
    # Evaluate E(z+step)-E(z) directly; avoid subtracting two large energies.
    ld=np.longdouble
    b=p['incidence'].astype(ld);w=p['weight'].astype(ld)
    z=z.astype(ld);step=step.astype(ld);t=b@z;dt=b@step
    if theta==0 or p['delta']==0:
        difference=t*dt+ld('.5')*dt*dt
    else:
        delta=ld(p['delta']);r=np.hypot(ld(1),t/delta);rn=np.hypot(ld(1),(t+dt)/delta)
        difference=(1-theta)*(t*dt+ld('.5')*dt*dt)+theta*(2*t*dt+dt*dt)/(r+rn)
    return np.dot(w,difference)+np.dot(p['d'].astype(ld),z*step+ld('.5')*step*step)-np.dot(p['c'].astype(ld),step)

def solve(w,e,theta=.5,alpha=5.,beta=25.,rtol=1e-11,maxiter=100):
    p=assemble(w,e,alpha,beta);z=p['z0'].copy();history=[]
    for iteration in range(maxiter+1):
        f,g,h,conductance=terms(p,z,theta)
        relative=float(np.linalg.norm(g)/np.linalg.norm(p['c']))
        history.append(dict(iteration=iteration,energy=float(f),relative_gradient=relative))
        if relative<=rtol:
            # Numerical diagnostic, not an outward interval certificate.
            mu=(1-theta)*np.linalg.eigvalsh(p['a'])[0]
            return dict(z=z,scores=np.sqrt(p['h'])*z,baseline_scores=np.sqrt(p['h'])*p['z0'],
                delta=p['delta'],conductance=conductance,history=history,
                diagnostic_error_bound=float(np.linalg.norm(g)/mu),model=p)
        if iteration==maxiter:raise RuntimeError("Nonlinear KKT tolerance not reached")
        step=np.linalg.solve(h,-g);slope=g@step
        if not slope<0:raise RuntimeError("Newton direction is not descent")
        rate=1.
        for k in range(60):
            candidate=z+rate*step
            change=energy_change(p,z,rate*step,theta)
            if change<=1e-4*rate*slope:
                z=candidate;break
            rate*=.5
        else:raise RuntimeError("Armijo failure; no fallback to a different objective")

