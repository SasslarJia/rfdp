import json,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from prototype import assemble,terms,solve
rng=np.random.default_rng(260628)
rows=[]
for case in range(24):
    n=6+case%13
    w=np.eye(n)
    for i in range(n-1):w[i,i+1]=w[i+1,i]=rng.uniform(.1,1.)
    for i in range(n):
        for j in range(i+2,n):
            if rng.random()<.2:w[i,j]=w[j,i]=rng.uniform(.05,1.)
    e=np.zeros(n);e[rng.choice(n,size=1+case%3,replace=False)]=rng.uniform(.2,2.,1+case%3)
    original_w=w.copy();original_e=e.copy()
    p=assemble(w,e);a=solve(w,e,0.);r=solve(w,e,.5)
    h=p['h'];s=w/np.sqrt(h[:,None]*h[None,:])
    original=np.linalg.solve(np.eye(n)-s+np.diag(5*e/(h+e)),np.where(e>0,5.,25.))
    neutral=np.max(np.abs(a['scores']-original))/np.max(np.abs(original))
    z=r['z'];perm=rng.permutation(n);rp=solve(w[perm][:,perm],e[perm],.5)
    permutation=np.max(np.abs(rp['scores']-r['scores'][perm]))/np.max(np.abs(r['scores']))
    independent=minimize(lambda v:terms(p,v)[0],p['z0'],jac=lambda v:terms(p,v)[1],
        hess=lambda v:terms(p,v)[2],method='trust-exact',options={'gtol':1e-8,'maxiter':200})
    independent_error=np.max(np.abs(independent.x-z))/max(1.,np.max(np.abs(z)))
    k=rng.normal(size=n);k/=np.linalg.norm(k);point=.9*p['z0'];eps=1e-4
    f,g,H,_=terms(p,point)
    finite_grad=(terms(p,point+eps*k)[0]-terms(p,point-eps*k)[0])/(2*eps)
    grad_error=abs(finite_grad-g@k)/max(1.,abs(g@k))
    hess_error=np.linalg.norm((terms(p,point+eps*k)[1]-terms(p,point-eps*k)[1])/(2*eps)-H@k)/max(1.,np.linalg.norm(H@k))
    minimum=np.linalg.eigvalsh(H-.5*p['a'])[0]
    assert neutral<1e-10 and permutation<1e-9 and independent_error<1e-7
    assert grad_error<1e-5 and hess_error<1e-6 and minimum>-1e-11
    assert np.array_equal(w,original_w) and np.array_equal(e,original_e) and np.min(z)>0
    assert r['conductance'].min()>=.5 and r['conductance'].max()<=1
    energies=[v['energy'] for v in r['history']]
    assert all(y<=x+1e-10*max(1.,abs(x)) for x,y in zip(energies,energies[1:]))
    rows.append(dict(case=case,n=n,neutral_relative_error=float(neutral),permutation_relative_error=float(permutation),
        independent_solution_relative_error=float(independent_error),gradient_difference_error=float(grad_error),
        hessian_difference_error=float(hess_error),hessian_lower_bound_eigenvalue=float(minimum),
        final_relative_gradient=r['history'][-1]['relative_gradient'],iterations=len(r['history'])-1,
        scores_changed_relative=float(np.max(np.abs(r['scores']-original))/np.max(np.abs(original)))))
# Query-dependent conductance on exactly the same input W.
w=np.array([[1.,1.,0.,0.],[1.,1.,.2,0.],[0.,.2,1.,1.],[0.,0.,1.,1.]])
r1=solve(w,np.array([1.,0.,0.,0.]));r2=solve(w,np.array([0.,0.,0.,1.]))
dependence=float(np.max(np.abs(r1['conductance']-r2['conductance'])))
assert dependence>1e-3
# Pure Charbonnier (theta=1) is not a valid automatic replacement here:
# z1=0,z2=t, one unit edge, d2=0,c2=25 -> sqrt(1+t^2)-1-25*t -> -infinity.
counter=[float(np.hypot(1.,t)-1-25*t) for t in [10.,100.,1000.]]
assert counter[2]<counter[1]<counter[0]
rejected=[]
for tag,w,e in [('unanchored',np.eye(2),np.zeros(2)),('negative',np.array([[1.,-.1],[-.1,1.]]),np.ones(2))]:
    try:solve(w,e)
    except ValueError:rejected.append(tag)
    else:raise AssertionError(tag)
out=dict(time=datetime.datetime.now().astimezone().isoformat(),status='small_graph_mathematical_prototype_checks_passed_not_retrieval_qualified',
    random_graphs=len(rows),theta=.5,delta_rule='weighted RMS of original hybrid solution edge differences per outer-stage; no labels',
    rows=rows,query_conditioned_conductance_difference=dependence,pure_robust_unbounded_counterexample=counter,rejected=rejected,
    labels_loaded=False,ReID_AP_evaluated=False,server_job_started=False,
    remaining=['sparse implementation and high-precision nonlinear residual/rank certification','original solver neutral interface qualification',
    'fixed real queries without AP selection','resource and throughput pilot before a population protocol'])
here=Path(__file__).resolve().parent
out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [here/'prototype.py',Path(__file__)]}
(here/'synthetic-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows']}))

