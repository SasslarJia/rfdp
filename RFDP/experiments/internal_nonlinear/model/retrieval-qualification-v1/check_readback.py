"""Exercise the NEW artifact reader with already-produced synthetic trajectories.
No old retrieval or solver qualification is rerun.
"""
import json,gzip,tempfile,copy,datetime,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from verify_artifact import verify_query,read_rows
from accept_fixed import require_complete
P=Path(__file__).resolve().parent
old=P.parent/'runner-interface-v1/synthetic-traces.json'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='ce1334619d3a81ef10f1f7a75624e0a6fa78b99b17b0ce4ddda5d56386a329b3'
raw=json.loads(old.read_text());rng=np.random.default_rng(26062819)
def graph(n):
    w=np.eye(n)
    for i in range(n-1):w[i,i+1]=w[i+1,i]=rng.uniform(.2,1.7)
    for i in range(n-3):w[i,i+3]=w[i+3,i]=rng.uniform(.1,.8)
    return w
fixtures=[sparse.csr_matrix(graph(37)),sparse.block_diag([graph(4),graph(13),graph(18)],format='csr'),
          sparse.csr_matrix(np.ones((19,19)))]
near=np.ones((19,19));near[0,7]=near[7,0]=np.nextafter(1.,2.);fixtures.append(sparse.csr_matrix(near))
results=[];example=None
with tempfile.TemporaryDirectory() as temp:
 for w,item in zip(fixtures,raw):
    w.eliminate_zeros();w.sort_indices();x=rng.uniform(.1,1.,(w.shape[0],9));full=x@x.T
    q=item['seed'];sim=full[:,q]
    for arm,theta in [(item['arms'][1],0.),(item['arms'][2],.5)]:
        rows=copy.deepcopy(arm['trace']);ev=iter(arm['events'])
        for t in rows:t['event']=None if t['returned_scores'] is None else next(ev)
        assert next(ev,None) is None
        p=Path(temp)/'scores.jsonl.gz'
        with gzip.open(p,'wt') as f:
            for t in rows:f.write(json.dumps(t,separators=(',',':'))+'\n')
        result=verify_query(w,full,sim,q,arm['prefix'],arm['full_ranking'],read_rows(p),theta,0,item['nq'])
        results.append(dict(fixture=item['fixture'],theta=theta,**result))
        if example is None and theta==.5:example=(w,full,sim,q,arm,rows,item['nq'])
 w,full,sim,q,arm,rows,depth=example
 rejected=[]
 for name in ['selected_order','actual_score','missing_batch','missing_scores','wrong_bridge']:
    bad=copy.deepcopy(rows)
    if name=='selected_order':bad[0]['selected'][0],bad[0]['selected'][1]=bad[0]['selected'][1],bad[0]['selected'][0]
    elif name=='actual_score':
        bad[0]['returned_scores'][0][2]+=1
        # Rehash so a numeric corruption is not rejected merely by a stale digest.
        bad[0]['event']['returned_scores_sha256']=hashlib.sha256(json.dumps(bad[0]['returned_scores'],separators=(',',':')).encode()).hexdigest()
    elif name=='missing_batch':bad.pop()
    elif name=='missing_scores':bad[0]['returned_scores']=None
    else:bad[0]['bridges']=[[1,q,.5]]
    try:verify_query(w,full,sim,q,arm['prefix'],arm['full_ranking'],bad,.5,0,depth)
    except (AssertionError,ValueError,TypeError):rejected.append(name)
    else:raise AssertionError('Tampering accepted: '+name)
 try:require_complete(Path(temp)/'absent')
 except (AssertionError,FileNotFoundError):rejected.append('unfinished_input')
 else:raise AssertionError('Unfinished input accepted')
 result=dict(status='artifact_readback_and_corruption_rejection_passed_not_real_qualification',
             time=datetime.datetime.now().astimezone().isoformat(),rows=results,rejected=rejected,
             old_retrieval_rerun=False,real_data_loaded=False,AP_evaluated=False)
 (P/'readback-check.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
