"""Correctness checks for the new precision schedule; no retrieval qualification."""
import sys,json,importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));import adaptive_adapter as new
sys.path.insert(0,str(P.parent/'runner-interface-v1'));import internal_adapter as old
sys.path.insert(0,str(P.parent/'population-v1'));from verify_artifact import certify_scores
rng=np.random.default_rng(260629)
def chain(n):
 v=rng.uniform(.2,2.,n-1);return sparse.diags([v,np.ones(n),v],[-1,0,1],format='csr')
fixtures=[]
for n in [9,17]:
 w=chain(n);e=np.zeros(n);e[[0,n//2]]=1.;fixtures.append(('chain'+str(n),w,e))
w=sparse.block_diag([chain(6),chain(7)],format='csr');e=np.zeros(13);e[[0,6]]=1.;fixtures.append(('components_shared_delta',w,e))
w=sparse.csr_matrix(np.ones((7,7)));e=np.ones(7);fixtures.append(('exact_ties',w,e.copy()));e[0]=np.nextafter(1.,2.);fixtures.append(('one_ulp',w,e))
rows=[]
for name,w,e in fixtures:
 x,rr,report=new.solve(w,e);y,_,original=old.solve(w,e)
 assert report['returned_prefix_local']==original['returned_prefix_local']
 check=certify_scores(w,e,list(x),.5)
 assert check['prefix']==report['returned_prefix_local'] and check['relative']<=1e-8
 if name=='one_ulp':assert check['prefix'][0]==0 and all(0 not in g for g in report['exact_tie_groups'])
 rows.append(dict(case=name,dps=report['dps'],attempts=report['precision_attempts'],same_prefix=True,independent_saved_score_residual=check['relative']))
# Force a low-precision inconclusive proof to verify escalation, never silent acceptance.
real=new.sc.certify
calls=[]
def inconclusive(*a,**kw):
 calls.append(kw['precisions'][0]);c=real(*a,**kw)
 if kw['precisions']==(24,):c['accepted']=False
 return c
new.sc.certify=inconclusive
try:
 _,_,r=new.solve(fixtures[0][1],fixtures[0][2]);assert calls==[24,40] and r['dps']==40
finally:new.sc.certify=real
# Perturbed saved scores must still be rejected by the independent reader.
values,_,_=new.solve(fixtures[0][1],fixtures[0][2]);values[0]*=2
try:certify_scores(fixtures[0][1],fixtures[0][2],list(values),.5)
except ValueError:rejection=True
else:raise AssertionError('Corrupt scores accepted')
# Original outer Algorithm 1: check whole trajectories including bridge/no-solve batches.
source=P.parents[3]/'MODEL/Paper_RFDP/PaperFullIterative.py'
def load(name,adapter):
 spec=importlib.util.spec_from_file_location(name,source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ev=[];adapter.install(m,ev,.5);return m,ev
outer=[]
for name,w,seed,depth in [('connected',chain(23),3,21),('bridge',sparse.block_diag([chain(4),chain(13),chain(9)],format='csr'),1,24)]:
 w.eliminate_zeros();x=rng.uniform(.1,1.,(w.shape[0],5));full=x@x.T;sim=full[:,seed];records=[]
 for tag,adapter in [('old',old),('adaptive',new)]:
  m,ev=load('adaptive_check_'+name+tag,adapter);ts=[]
  prefix,diag=m.FullIHyRDP(w,full,alpha=5,beta=25,step=5).rank([seed],nq=depth,input_similarity=sim,numerical_tail=False,trace=lambda t:ts.append({k:t[k] for k in ['labeled','unlabeled','selected','bridges']}))
  remaining=np.array([i for i in range(len(sim)) if i not in set(prefix)])
  rank=np.r_[prefix,remaining[np.lexsort((remaining,-sim[remaining]))]]
  records.append((prefix.tolist(),rank.tolist(),ts))
 assert records[0]==records[1]
 outer.append(dict(case=name,prefix_rank_and_selection_trace_identical=True,batches=len(records[0][2])))
result=dict(status='correctness_passed',cases=rows,forced_escalation=calls,corrupt_saved_scores_rejected=rejection,outer=outer,model_parameters_unchanged=True,performance_qualification=False)
(P/'correctness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
