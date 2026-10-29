"""Targeted tests that exercise paths not reached by the initial fixtures."""
import json,time,datetime,sys,resource
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy import sparse
from scalable_certificate import certify,summary
from sparse_solver import solve
from certificate import certify as old_cert,independent_nonlinear_oracle
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]))
from accurate_hybrid import Block
rng=np.random.default_rng(26062844);n=13
w=np.eye(n)
for i in range(n-1):w[i,i+1]=w[i+1,i]=rng.uniform(.3,2.)
e=np.zeros(n);e[[0,7,12]]=[1.,1.4,.7]
a=solve(w,e);c=certify(w,e,a['z']*1.001)
assert c['accepted'] and not c['attempts'][0]['initial_candidate_certified']
assert len(c['attempts'][0]['nonlinear_refinement'])>=2
old=old_cert(w,e,a['z']);oracle=independent_nonlinear_oracle(old,a['z'])
assert all(lo<x<hi for (lo,hi),x in zip(c['intervals'],oracle))
# Nearly equal nodes are resolved at high precision, never merged by a tolerance.
w=np.ones((7,7));e=np.ones(7);e[0]=np.nextafter(1.,2.)
a=solve(w,e);near=certify(w,e,a['z'])
assert near['accepted'] and near['prefix'][0]==0
assert all(0 not in pair for pair in near['exact_transposition_ties'])
assert len(near['attempts'][0]['nonlinear_refinement'])>=2
# Large neutral algebra is independently accepted by the unmodified original adapter.
n=1600;v=rng.uniform(.3,2.,n-1)
w=sparse.diags([v,np.ones(n),v],[-1,0,1],format='csr')
e=np.zeros(n);e[::40]=1.
a=solve(w,e,theta=0.)
neutral=certify(w,e,a['z'],theta=0.)
with mp.workdps(neutral['dps']):
    scores=[mp.sqrt(mp.mpf(h.numerator)/h.denominator)*z for h,z in zip(neutral['problem']['h'],neutral['z'])]
    original=Block(w,e,5.,25.).certify(scores,neutral['dps'])
assert original['accepted'] and neutral['accepted']
out=dict(time=datetime.datetime.now().astimezone().isoformat(),status='targeted_sparse_refinement_and_exact_tie_paths_passed',
         perturbed_start=summary(c),near_tie=summary(near),large_neutral=dict(n=n,certificate=summary(neutral),
         original_adapter_relative=mp.nstr(original['relative'],24)),AP_evaluated=False,real_query_qualified=False)
(P/'targeted-results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

