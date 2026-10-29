"""Bind the internal nonlinear objective at the existing solver interface.
The outer FullIHyRDP.rank, graph, local degrees, bridge and seed rules are untouched.
Theta=0 delegates to the already-installed original accurate-v3 chain.
"""
from pathlib import Path
import sys,time,hashlib,json
import numpy as np
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parent))
import adaptive_certificate as sc
from sparse_solver import solve as candidate_solve
from certificate import iv,bounds,scalar
SOLVER_ID='internal-charbonnier-hybrid-certified-adaptive-v1'
RESIDUAL_TOL=1e-8
MAX_LOCAL=8192

class InternalFailure(np.linalg.LinAlgError):
    pass

def packed(x):
    if not isinstance(x,mp.mpf):x=scalar(x)
    return [int(k) for k in x._mpf_]

def score_hash(scores):
    return hashlib.sha256(json.dumps([packed(x) for x in scores],separators=(',',':')).encode()).hexdigest()

def input_hash(w,e,alpha,beta):
    h=hashlib.sha256()
    for v,d in [(w.shape,'<i8'),(w.indptr,'<i8'),(w.indices,'<i8'),(w.data,'<f8'),
                (e,'<f8'),([alpha,beta],'<f8')]:
        h.update(np.asarray(v,dtype=d).tobytes())
    return h.hexdigest()

def equal_representatives(c):
    # Only exact symmetry-proven groups can share a numerical representative.
    p=c['problem'];values=list(c['points']);parent=list(range(p['n']))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in c['exact_transposition_ties']:
        if not sc.exact_twins(p,i,j):raise InternalFailure('Invalid exact tie proof')
        parent[root(j)]=root(i)
    groups={}
    for i in range(p['n']):groups.setdefault(root(i),[]).append(i)
    out=[]
    for g in groups.values():
        if len(g)<2:continue
        lo=max(c['intervals'][i][0] for i in g);hi=min(c['intervals'][i][1] for i in g)
        if lo>hi:raise InternalFailure('Exact tie intervals disjoint')
        common=(lo+hi)/2
        for i in g:values[i]=common
        out.append(g)
    return values,out

def certify_returned(c,values,theta):
    """Independent residual/order check in the actual RETURNED psi coordinates.
    z=psi/sqrt(exact h) is interval-valued, not the rounded mp candidate quotient.
    """
    p=c['problem'];delta=c['delta_box'];old=mp.iv.dps;mp.iv.dps=c['dps']
    try:
        with mp.workdps(c['dps']):
            data=sc.interval_data(p);roots=data['roots'];psi=[iv(x) for x in values]
            z=[x/r for x,r in zip(psi,roots)]
            g=[d*x-cv for d,x,cv in zip(data['d'],z,data['c'])]
            zero=bounds(delta)[1]==0
            th=iv(theta);one_minus=1-th;delta2=delta**2
            for i,j,w in data['edges']:
                t=z[i]-z[j]
                flux=w*t if theta==0 or zero else w*(one_minus*t+th*delta*t/mp.iv.sqrt(delta2+t**2))
                g[i]+=flux;g[j]-=flux
            rel=bounds(mp.iv.sqrt(sum(((x/r)**2 for x,r in zip(g,roots)),iv(0))/
                                          data['bnorm']))[1]
            radii=[]
            for ids,mu in zip(p['components'],c['mu']):
                radii.append(bounds(mp.iv.sqrt(sum((g[i]**2 for i in ids),iv(0)))/iv(mu))[1])
            boxes=[bounds(x+r*mp.iv.mpf([-radii[p['labels'][i]],radii[p['labels'][i]]]))
                   for i,(x,r) in enumerate(zip(psi,roots))]
            if any(lo<=0 for lo,hi in boxes) or rel>mp.mpf('1e-8'):
                raise InternalFailure('Returned-score residual or positivity not certified')
            # This is the exact expression used by the unmodified outer rank.
            order=np.lexsort((np.arange(p['n']),np.asarray(values,dtype=object))).tolist()
            ties=[]
            for k in range(min(5,p['n'])):
                i=order[k]
                for j in order[k+1:]:
                    if boxes[i][1]<boxes[j][0]:continue
                    if values[i]==values[j] and i<j and sc.exact_twins(p,i,j):
                        ties.append([i,j]);continue
                    raise InternalFailure('Actual returned np.lexsort prefix not certified')
            prefix=order[:min(5,p['n'])]
            proof=[]
            for k,i in enumerate(prefix):
                rest=order[k+1:]
                strict=[j for j in rest if not(values[i]==values[j] and sc.exact_twins(p,i,j))]
                proof.append(dict(local_node=i,score_interval=[packed(x) for x in boxes[i]],
                    strict_remaining_lower=None if not strict else packed(min(boxes[j][0] for j in strict))))
            # Upward binary64 conversion only for legacy diagnostic interface.
            rf=float(rel)
            if mp.mpf(rf)<rel:rf=float(np.nextafter(rf,np.inf))
            return dict(relative_float_upper=rf,relative_exact_upper=packed(rel),
                        prefix=prefix,rank_proof=proof,exact_ties=ties,
                        radii_upper=[packed(x) for x in radii],intervals=boxes)
    finally:mp.iv.dps=old

def solve(w,e,alpha=5.,beta=25.,theta=.5):
    if theta!=.5:raise ValueError('This candidate interface fixes theta=.5')
    if len(e)>MAX_LOCAL:raise InternalFailure('Local cap; no truncation')
    started=time.monotonic();before=input_hash(w,e,alpha,beta)
    candidate=candidate_solve(w,e,theta=theta,alpha=alpha,beta=beta)
    candidate_seconds=time.monotonic()-started;last=None
    precision_attempts=[]
    for dps in (24,40,80,120):
        try:c=sc.certify(w,e,candidate['z'],theta=theta,alpha=alpha,beta=beta,precisions=(dps,),float_model=candidate['model'])
        except ArithmeticError as exc:
            precision_attempts.append(dict(dps=dps,status='raise_precision',reason=str(exc)));last=str(exc);continue
        precision_attempts.append(dict(dps=dps,status='certified' if c['accepted'] else 'raise_precision'))
        if not c['accepted']:last='Nonlinear certificate inconclusive';continue
        with mp.workdps(dps):
            values,groups=equal_representatives(c)
            try:returned=certify_returned(c,values,theta)
            except InternalFailure as exc:
                precision_attempts[-1].update(status='raise_precision',reason=str(exc));last=str(exc);continue
            if input_hash(w,e,alpha,beta)!=before:raise InternalFailure('Input changed')
            report=dict(solver_id=SOLVER_ID,kind='internal-nonlinear-returned-score-certified',
                        target='exact-binary-W-e-stage-shared-exact-baseline-RMS-delta',
                        theta=theta,size=len(e),input_sha256=before,returned_scores_sha256=score_hash(values),
                        candidate_seconds=candidate_seconds,total_seconds=time.monotonic()-started,
                        Newton_iterations=len(candidate['history'])-1,
                        certificate=sc.summary(c),returned_relative_residual_upper=returned['relative_float_upper'],
                        returned_relative_exact_upper=returned['relative_exact_upper'],
                        returned_prefix_local=returned['prefix'],rank_proof=returned['rank_proof'],
                        exact_tie_groups=groups,exact_tie_certificate='exact-input-transposition-unique-nonlinear-minimizer',
                        dps=dps,precision_attempts=precision_attempts,output_type='mpf-object',h_e_fitting_unchanged=True,
                        delta_bounds=[packed(x) for x in bounds(c['delta_box'])],
                        mu_lower=[packed(x) for x in c['mu']],z_radii_upper=returned['radii_upper'])
            return np.asarray(values,dtype=object),returned['relative_float_upper'],report
    raise InternalFailure('Returned-score certification failed at bounded precisions: '+str(last))

def install(module,events,theta=.5):
    if theta not in (0.,.5):raise ValueError('Only neutral theta0 and declared theta.5')
    original=module.solve_hybrid_sparse
    def adapted(w,e,alpha=5.,beta=25.,backend='auto',diagnostics=None):
        if len(e)>MAX_LOCAL:raise InternalFailure('Local cap; no truncation')
        if theta==0:
            # Return the original exact objects and residual without recasting.
            scores,residual=original(w,e,alpha,beta,backend=backend,diagnostics=diagnostics)
            events.append(dict(kind='internal-neutral-original-delegation',size=len(e),theta=0.,
                               returned_prefix_local=np.lexsort((np.arange(len(e)),scores))[:5].tolist(),
                               returned_scores_sha256=score_hash(scores),returned_relative_residual_upper=residual))
            return scores,residual
        if backend not in ('auto','direct','cg'):raise ValueError('Invalid solver backend')
        try:scores,residual,report=solve(w,e,alpha,beta,theta)
        except (ArithmeticError,ValueError,RuntimeError) as exc:
            raise InternalFailure(str(exc)) from exc
        events.append(report)
        if diagnostics is not None:
            # Existing counters count outer solver calls; inner counts stay in report.
            diagnostics.update(cg_calls=0,cg_iterations=0,direct_calls=1,uncertified_cg=0)
        return scores,residual
    module.solve_hybrid_sparse=adapted
    return original

