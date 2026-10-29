"""Resume integrity and adaptive read-back tests, without real query reruns."""
import sys,json,tempfile
from pathlib import Path
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import full
from verify_artifact import certify_scores
from adaptive_adapter import solve
rng=np.random.default_rng(26062910);checks=[]
for name,e in [('ordinary',np.r_[1.,np.zeros(8)]),('near',np.r_[np.nextafter(1.,2.),np.ones(8)])]:
 w=sparse.csr_matrix(np.ones((9,9))) if name=='near' else sparse.diags([rng.uniform(.2,2.,8),np.ones(9)],[-1,0],format='csr');w=(w+w.T)*.5
 values,_,ev=solve(w,e);v=certify_scores(w,e,list(values),.5)
 assert v['prefix']==ev['returned_prefix_local'] and v['relative']<=1e-8
 checks.append(dict(case=name,producer_dps=ev['dps'],reader_dps=v['dps'],same_prefix=True))
with tempfile.TemporaryDirectory() as td:
 full.OUT=Path(td);q=full.OUT/'queries';q.mkdir();(full.OUT/'run-manifest.json').write_text(json.dumps({'signature':'test'}))
 row={'query':2,'theta':.5,'signature':'test','diagnostics':{'status':'complete_diffusion','numerical_tail_count':0}}
 for ext,key in [('trace.jsonl.gz','trace_sha256'),('prefix.npy','prefix_sha256'),('rankings.npy','rankings_sha256')]:
  p=q/('0002.'+ext);p.write_bytes(b'test-'+ext.encode());row[key]=full.sha(p)
 (q/'0002.json').write_text(json.dumps(row));assert full.completed_queries()==[2]
 (q/'0003.trace.jsonl.gz').write_bytes(b'incomplete');assert full.completed_queries()==[2]
 (q/'0002.rankings.npy').write_bytes(b'corrupted')
 try:full.completed_queries()
 except AssertionError:pass
 else:raise AssertionError('Completed corrupt query was reusable')
r=dict(status='passed',readback=checks,resume_skips_complete_only=True,corrupt_completed_artifact_rejected=True,unfinished_trace_not_complete=True)
(P/'resume-readback-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
