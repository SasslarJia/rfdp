import json,hashlib,datetime,time,resource,sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from prototype import solve as dense_solve
from sparse_solver import solve
from certificate import certify,independent_nonlinear_oracle
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from accurate_hybrid import Block as OriginalBlock
rng=np.random.default_rng(26062818);rows=[];started=time.monotonic()
for case in range(12):
    n=6+case;w=np.eye(n)
    for i in range(n-1):w[i,i+1]=w[i+1,i]=rng.uniform(.2,2.)
    for i in range(n):
        for j in range(i+2,n):
            if rng.random()<.15:w[i,j]=w[j,i]=rng.uniform(.1,1.)
    e=np.zeros(n);e[rng.choice(n,2,replace=False)]=rng.uniform(.5,3.,2)
    csr=sparse.csr_matrix(w);before=(csr.data.copy(),e.copy())
    a=solve(csr,e);dense=dense_solve(w,e)
    neutral=solve(csr,e,theta=0.)
    original_certificate=OriginalBlock(csr,e,5.,25.).certify(neutral['scores'],80)
    assert original_certificate['accepted']
    agreement=np.max(np.abs(a['scores']-dense['scores']))/np.max(np.abs(dense['scores']))
    cert=certify(csr,e,a['z']);assert cert['accepted'],{k:v for k,v in cert.items() if k not in ['problem','delta_box','point_scores','intervals']}
    oracle=independent_nonlinear_oracle(cert,a['z'])
    assert all(lo<x<hi for (lo,hi),x in zip(cert['intervals'],oracle))
    scaled=solve(csr*8.,e*8.);invariance=np.max(np.abs(scaled['scores']-a['scores']))/np.max(np.abs(a['scores']))
    assert agreement<1e-8 and invariance<1e-9
    assert np.array_equal(csr.data,before[0]) and np.array_equal(e,before[1])
    bad=certify(csr,e,a['z']*1.1)
    assert not bad['accepted']
    rows.append(dict(case=case,n=n,sparse_dense_relative=agreement,uniform_weight_scale_relative=invariance,
        neutral_original_adapter_relative=float(original_certificate['relative']),
        nonlinear_residual_upper=cert['nonlinear_relative_residual_upper'],z_error_radius=cert['score_coordinate_radius_z'],
        all_MP_oracle_scores_strictly_inside_intervals=True,first_five_strictly_certified=True,
        distorted_candidate_rejected=True,iterations=len(a['history'])-1))
# Symmetric real ties must remain inconclusive without a tie proof.
w=np.ones((6,6));e=np.ones(6);a=solve(w,e);tie=certify(w,e,a['z'])
assert not tie['accepted'] and not tie['ordered_first_five']
assert float(tie['delta_interval'][1])==0
# Refuse unanchored inputs rather than add a ridge.
try:solve(sparse.eye(6),np.zeros(6))
except ValueError:unanchored=True
else:raise AssertionError('Unanchored silently accepted')
# Larger sparse engineering fixture, NOT a population or server budget trial.
n=1500;weights=rng.uniform(.5,2.,n-1)
w=sparse.diags([weights,np.ones(n),weights],[-1,0,1],format='csr')
e=np.zeros(n);e[::30]=1.
t=time.monotonic();large=solve(w,e);large_seconds=time.monotonic()-t
assert large['history'][-1]['relative_gradient']<=1e-11
# No dense arrays of shape n*n or m*n retained by the implementation.
assert all(not(isinstance(v,np.ndarray) and v.ndim==2) for v in large['model'].values())
out=dict(time=datetime.datetime.now().astimezone().isoformat(),status='sparse_and_small_fixture_interval_checks_passed_not_retrieval_qualified',
    fixtures=rows,symmetric_tie_rejected_without_proof=True,exact_zero_delta_recognized=True,unanchored_rejected=unanchored,
    large_sparse_fixture=dict(n=n,nnz=w.nnz,seconds=large_seconds,iterations=len(large['history'])-1,
        max_hessian_nnz=max(x['hessian_nnz'] for x in large['history']),relative_gradient=large['history'][-1]['relative_gradient']),
    local_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    peak_platform='macOS ru_maxrss bytes; includes dense/MP small-fixture checks; not a server worker budget',
    total_seconds=time.monotonic()-started,labels_loaded=False,AP_evaluated=False,server_job_started=False,
    certificate_domain_max_nodes=64,certificate_scope='exact binary W/e, exact rational h/d and outward 80-decimal nonlinear gradient; delta enclosed from exact original hybrid solution',
    remaining=['Production-scale baseline/delta enclosure and efficient certificate; current dense MP baseline limited to 64',
    'Full-precision solve/refinement and exact tie handling if certificate inconclusive',
    'Independent real original-runner neutral interface/rank qualification and full-query resource pilot',
    'Fixed internal-component retrieval protocol; theta/delta remain a mathematical draft'])
p=Path(__file__).resolve().parent
out['source_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.glob('*.py')}
(p/'qualification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='fixtures'}))

