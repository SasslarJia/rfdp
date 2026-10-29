import sys,json,time,datetime,resource,traceback,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy import sparse
from scalable_certificate import certify,summary,exact_twins,problem
from sparse_solver import solve
from certificate import certify as old_cert,independent_nonlinear_oracle,bounds
P=Path(__file__).resolve().parent
rng=np.random.default_rng(26062839);results=[];started=time.monotonic()
try:
    # New fixtures compare the new sparse certificate with a separate dense MP oracle.
    for case in range(8):
        n=7+case;w=np.eye(n)
        for i in range(n-1):w[i,i+1]=w[i+1,i]=rng.uniform(.1,2.)
        for i in range(n):
            for j in range(i+2,n):
                if rng.random()<.2:w[i,j]=w[j,i]=rng.uniform(.1,1.)
        e=np.zeros(n);e[[0,n-1]]=rng.uniform(.3,2.,2)
        a=solve(w,e);c=certify(w,e,a['z'])
        assert c['accepted']
        old=old_cert(w,e,a['z']);oracle=independent_nonlinear_oracle(old,a['z'])
        assert all(lo<x<hi for (lo,hi),x in zip(c['intervals'],oracle))
        # Interval delta must enclose the high-precision dense baseline delta midpoint.
        lo,hi=bounds(c['delta_box']);olo,ohi=bounds(old['delta_box'])
        assert lo<=ohi and olo<=hi
        bad=certify(w,e,a['z']*1.1,refine=False)
        assert not bad['accepted']
        results.append(dict(case=case,n=n,certificate=summary(c),oracle_enclosed=True,distorted_rejected=True))
    print('small completed',flush=True)
    # Exact transposition proof, including the quadratic zero-scale branch.
    w=np.ones((7,7));e=np.ones(7);a=solve(w,e)
    t=certify(w,e,a['z']);assert t['accepted'] and t['prefix']==[0,1,2,3,4]
    assert not certify(w,e,a['z'],allow_ties=False)['accepted']
    # A one-ULP asymmetric row breaks exact equality; no approximate tie proof.
    w[0,2]=w[2,0]=np.nextafter(1.,2.)
    assert not exact_twins(problem(w,e),0,1)
    # Two disconnected components with different constant solutions still have exact delta=0.
    w=sparse.block_diag([np.ones((3,3)),2*np.ones((4,4))],format='csr')
    e=np.r_[np.ones(3),np.ones(4)*3];a=solve(w,e);c=certify(w,e,a['z'])
    assert bounds(c['delta_box'])==(mp.mpf(0),mp.mpf(0)) and c['accepted']
    # A weak real bridge forces arbitrary-precision correction without deleting it.
    n=12;ww=np.ones(n-1);ww[5]=2.**-24
    w=sparse.diags([ww,np.ones(n),ww],[-1,0,1],format='csr')
    e=np.zeros(n);e[[0,11]]=[1.,1.3];a=solve(w,e);c=certify(w,e,a['z'])
    assert c['accepted'];weak=summary(c)
    print('edge cases completed',flush=True)
    large=[]
    for n,degree in [(1500,2),(6500,12)]:
        # Sparse graph at production local-node scale; no ReID data or labels.
        rr=[];cc=[];vv=[]
        for offset in range(1,degree//2+1):
            ids=np.arange(n-offset);v=rng.uniform(.5,2.,len(ids))
            rr.extend([ids,ids+offset]);cc.extend([ids+offset,ids]);vv.extend([v,v])
        rr.append(np.arange(n));cc.append(np.arange(n));vv.append(np.ones(n))
        w=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n,n)).tocsr()
        e=np.zeros(n);e[::30]=rng.uniform(.5,2.,len(e[::30]))
        tm=time.monotonic();a=solve(w,e);solve_seconds=time.monotonic()-tm
        tm=time.monotonic();c=certify(w,e,a['z']);cert_seconds=time.monotonic()-tm
        assert c['accepted'],summary(c)
        large.append(dict(n=n,nnz=w.nnz,solve_seconds=solve_seconds,certificate_seconds=cert_seconds,
                          certificate=summary(c),Newton_iterations=len(a['history'])-1))
        print(json.dumps(large[-1]),flush=True)
    for w,e in [(sparse.eye(5),np.zeros(5))]:
        try:certify(w,e,np.ones(5))
        except ValueError:pass
        else:raise AssertionError('Unanchored accepted')
    out=dict(time=datetime.datetime.now().astimezone().isoformat(),status='scalable_synthetic_certificates_passed_not_real_query_qualified',
             small=results,exact_ties=summary(t),one_ulp_twin_rejected=True,
             disconnected_constant_delta_zero=True,weak_bridge=weak,large=large,
             seconds=time.monotonic()-started,local_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             platform='macOS ru_maxrss bytes; aggregate local process, not server worker admission',
             no_dense_MP_path=True,labels_loaded=False,AP_evaluated=False,server_job_started=False)
    out['source_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in P.glob('*.py')}
    (P/'qualification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['small','large']}))
except Exception as exc:
    f=P/('FAILED-attempt-'+datetime.datetime.now().strftime('%H%M%S')+'.json')
    f.write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),completed=results),indent=2)+'\n')
    raise

