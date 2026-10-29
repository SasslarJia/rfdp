"""Check three saved stage inputs against old selections and old 100+ digit reader."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,gzip,importlib.util,resource
from pathlib import Path
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'adaptive-precision-v1'))
import adaptive_adapter as a
spec=importlib.util.spec_from_file_location('original_readback',P.parent/'population-v1/verify_artifact.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
refs=json.loads((P/'references.json').read_text());graph=sparse.load_npz(Path(refs['cache'])/'graph.npz').astype(np.float64).tocsr();graph.eliminate_zeros();graph.sort_indices()
run=Path('/root/autodl-tmp/rfdp-graph-routes-20261026/market-internal-nonlinear-theta0p5-v1');query=0
boundary=np.asarray(graph[[query]].sum(axis=0)).ravel();checks=[]
with gzip.open(run/'queries/0000.trace.jsonl.gz','rt') as f:
 for row in map(json.loads,f):
  if row['iteration'] in [0,49,99] and row['returned_scores'] is not None:
   u=np.array(row['unlabeled']);e=boundary[u].copy()
   for i,j,v in row['bridges']:e[np.flatnonzero(u==i)[0]]+=v
   w=graph[u][:,u];assert a.input_hash(w,e,5,25)==row['event']['input_sha256']
   values,res,event=a.solve(w,e);assert event['returned_prefix_local']==row['event']['returned_prefix_local']
   cert=reader.certify_scores(w,e,list(values),.5);assert cert['prefix']==event['returned_prefix_local']
   checks.append(dict(query=query,iteration=row['iteration'],size=len(u),precision=event['dps'],precision_attempts=event['precision_attempts'],old_selected_equal=True,original_high_precision_reader_dps=cert['dps'],reader_residual=cert['relative'],candidate_seconds=event['candidate_seconds'],solver_seconds=event['total_seconds']))
  for node in row['selected']:
   begin,end=graph.indptr[node:node+2];boundary[graph.indices[begin:end]]+=graph.data[begin:end]
assert len(checks)==3
result=dict(status='correctness_passed',checks=checks,new_queries_run=0,AP_read=False,full_query_qualification=False)
(P/'saved-stages-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
