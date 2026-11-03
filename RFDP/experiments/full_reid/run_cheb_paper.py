"""Full official-population Cheb-GR with the paper's declared dataset settings.

OSNet uses the paper's CNN (AGW) reference for that dataset. TransReID uses
its own published setting. The two pinned upstream function bodies are unedited.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat
import torch

from EVAL.ReIDEval import evaluate_reid_rankings
from experiments.full_reid.common import sha, dump
from experiments.internal_nonlinear.runtime.analyze_route import paired_bootstrap

PAPER_LAMBDA={('market','osnet'):6.,('duke','osnet'):5.3,
              ('market','transreid'):5.3,('duke','transreid'):4.3}
POPULATION={'market':(3368,19732),'duke':(2228,17661)}
SOURCE_SHA='b927d20c894f06e580f8c99eadc399b24534eaecadf5496b46486d0bd175f29f'


def author_functions(path):
    assert sha(path)==SOURCE_SHA,'Frozen upstream source changed'
    names=['fast_mergesetfeat4','fast_run_gcr'];tree=ast.parse(path.read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes)==sorted(names)
    namespace=dict(np=np,torch=torch,time=time,defaultdict=defaultdict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
    return namespace


def run_author(q,g,threshold,functions):
    cfg=SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=True,BETA1=.2,BETA2=.2,
        LAMBDA1=threshold,LAMBDA2=threshold,SCALE=0.,GAL_ROUND=3,MODE='nonsym',ENABLE_GCR=True),
        COMMON=SimpleNamespace(VERBOSE=True))
    with torch.inference_mode():
        return functions['fast_run_gcr'](cfg,[torch.from_numpy(q),np.zeros(len(q),int),None,
                                            torch.from_numpy(g),np.zeros(len(g),int),None])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=POPULATION,required=True)
    p.add_argument('--frontend',choices=['osnet','transreid'],required=True)
    p.add_argument('--mat',type=Path,required=True);p.add_argument('--feature-sha256',required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-queries',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args();assert sha(args.mat)==args.feature_sha256
    threshold=PAPER_LAMBDA[(args.dataset,args.frontend)];functions=author_functions(args.source)
    torch.set_num_threads(4);torch.manual_seed(20261020)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    total=torch.cuda.get_device_properties(0).total_memory;assert 22*2**30<.75*total
    torch.cuda.set_per_process_memory_fraction(22*2**30/total,0)
    torch.cuda.reset_peak_memory_stats()
    if args.smoke_only:
        rng=np.random.default_rng(1404);x=rng.normal(size=(48,16)).astype(np.float32)
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        q,g=run_author(x[:12],x[12:],threshold,functions)
        assert q.shape==(12,16) and g.shape==(36,16) and torch.isfinite(q).all() and torch.isfinite(g).all()
        print(json.dumps(dict(status='upstream_call_and_resource_check_only',lambda_threshold=threshold,
              upstream_sha256=SOURCE_SHA,peak_cuda_bytes=torch.cuda.max_memory_allocated())));return
    args.output.mkdir(parents=True,exist_ok=False)
    data=loadmat(args.mat);q,g=[np.asarray(data[k],dtype=np.float32) for k in ['query_f','gallery_f']]
    qn,gn=len(q),len(g);assert (qn,gn)==POPULATION[args.dataset]
    assert np.isfinite(q).all() and np.isfinite(g).all()
    assert np.allclose(np.linalg.norm(q,axis=1),1.,atol=1e-5) and np.allclose(np.linalg.norm(g,axis=1),1.,atol=1e-5)
    config=dict(dataset=args.dataset,frontend=args.frontend,query_count=qn,gallery_count=gn,
        feature_sha256=args.feature_sha256,source_sha256=SOURCE_SHA,lambda_threshold=threshold,
        temperature=.2,rounds=3,mode='nonsym',identity_labels_passed_to_algorithm=False,
        parameter_policy='paper TransReID setting' if args.frontend=='transreid' else 'paper CNN/AGW setting for this dataset; no OSNet-specific recommendation claimed')
    dump(args.output/'run-manifest.json',config)
    start=time.monotonic();qnew,gnew=run_author(q,g,threshold,functions)
    distances=1.-qnew.numpy()@gnew.numpy().T
    assert distances.shape==(qn,gn) and np.isfinite(distances).all()
    algorithm_seconds=time.monotonic()-start
    ranks=np.argsort(distances,axis=1,kind='stable').astype(np.int32)+qn
    np.save(args.output/'rankings.npy',ranks)
    metric=evaluate_reid_rankings(ranks,range(qn),data['query_label'],data['query_cam'],
        data['gallery_label'],data['gallery_cam'],qn,max_rank=50,return_per_query=True)
    qp,qc,gp,gc=[data[k].ravel() for k in ['query_label','query_cam','gallery_label','gallery_cam']]
    aps=[];r1=[]
    for i,row in enumerate(ranks):
        local=row.astype(int)-qn;assert np.array_equal(np.sort(local),np.arange(gn))
        keep=(gp[local]!=-1)&~((gp[local]==qp[i])&(gc[local]==qc[i]));hits=gp[local[keep]]==qp[i]
        n=np.sum((gp==qp[i])&(gc!=qc[i])&(gp!=-1));assert n==hits.sum() and n>0
        ap=float(np.mean(np.arange(1,n+1)/(np.flatnonzero(hits)+1)));aps.append(ap);r1.append(int(hits[0]))
        assert abs(ap-metric['per_query'][i]['AP'])<1e-12 and r1[-1]==metric['per_query'][i]['rank1']
    assert abs(np.mean(aps)-metric['mAP'])<1e-12 and np.mean(r1)==metric['rank-1']
    dump(args.output/'metrics.json',metric)
    baseline=json.loads(args.baseline_queries.read_text());baseline={r['query_id']:r for r in baseline.get('per_query',baseline.get('queries'))}
    assert set(baseline)==set(range(qn))
    for i,row in enumerate(metric['per_query']):
        for field in ['pid','camid','valid','positive_count']:assert row[field]==baseline[i][field]
    delta=np.array([[aps[i]-baseline[i]['AP'],r1[i]-baseline[i]['rank1']] for i in range(qn)])
    rng=np.random.default_rng(260626)
    paired={unit:{name:paired_bootstrap(delta[:,j],groups,rng) for j,name in enumerate(['mAP','rank1'])}
            for unit,groups in [('query',np.arange(qn)),('identity',qp)]}
    dump(args.output/'paired-ihyrdp.json',dict(reference='original-IHy',seed=260626,paired=paired,
                                             baseline_sha256=sha(args.baseline_queries)))
    repo=Path(__file__).resolve().parents[3]
    complete=dict(status='full_rankings_independently_verified',scope='full_standard_test',method='cheb-gr',
        **config,metrics={k:v for k,v in metric.items() if k!='per_query'},rankings_sha256=sha(args.output/'rankings.npy'),
        runner_sha256=sha(__file__),execution_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        algorithm_seconds=algorithm_seconds,total_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(args.output/'COMPLETE.json',complete)
    print(json.dumps({k:complete[k] for k in ['status','dataset','frontend','lambda_threshold','algorithm_seconds']}),flush=True)


if __name__=='__main__':main()
