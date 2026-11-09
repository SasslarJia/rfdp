"""One-worker paired CPU timing on explicitly authorized frozen ReID inputs.

Uses the production compute/readback path and the full graph/gallery. This is a
timing record for the listed queries, never a full-population accuracy result.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from . import full, certified, converged
from MODEL.Paper_RFDP import PaperFullIterative as model

TIMING_CANDIDATES = ('two_step_fitting', 'f1_robust_converged')


def timing_sources(cfg):
    # Preserve old F1 timing recipes; new recipes must name their candidate.
    candidate = cfg.get('candidate_source', 'f1_robust_converged')
    if candidate not in TIMING_CANDIDATES:
        raise ValueError('Unsupported paired timing candidate: ' + str(candidate))
    return ('original', candidate)


class ProgressEngine:
    def __init__(self, engine, source, query):
        self.engine,self.source,self.query=engine,source,query

    def rank(self,*args,trace=None,**kwargs):
        start=time.perf_counter()
        def capture(row):
            trace(row)
            if (row['iteration']+1)%10==0:
                print(json.dumps(dict(source=self.source,query=self.query,
                    stages=row['iteration']+1,seconds=time.perf_counter()-start)),flush=True)
        return self.engine.rank(*args,trace=capture,**kwargs)


def run(config_path):
    cfg=json.loads(Path(config_path).read_text())
    sources=timing_sources(cfg)
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    cache=Path(cfg['cache']);qn,gn=cfg['query_count'],cfg['gallery_count']
    queries=cfg['query_ids']
    if not queries or len(set(queries))!=len(queries) or any(type(q) is not int or not 0<=q<qn for q in queries):
        raise ValueError('Explicit unique query IDs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=full.REPO,text=True).strip()
    if cfg.get('expected_commit') and cfg['expected_commit'] != head:
        raise ValueError('Timing checkout differs from expected_commit')
    for name,expected in cfg['cache_sha256'].items():
        if name not in ('graph.npz','features.npy','affinity.npy') or full.sha(cache/name)!=expected:
            raise ValueError('Frozen input SHA mismatch: '+name)
    if set(cfg['cache_sha256'])!={'graph.npz','features.npy','affinity.npy'}:
        raise ValueError('All three frozen input hashes required')
    load_start=time.perf_counter()
    graph=sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr()
    graph.eliminate_zeros();graph.sort_indices()
    affinity=np.load(cache/'affinity.npy',mmap_mode='r')
    features=np.load(cache/'features.npy',mmap_mode='r')
    assert graph.shape==affinity.shape==(qn+gn,qn+gn) and features.shape[0]==qn+gn
    full.GRAPH,full.FULL,full.X=graph,affinity,features
    full.QN,full.GN=qn,gn
    full.INIT_ERROR=None;full.READBACK_KERNEL=None;full.ADMISSION_CAMERAS=None
    full.READBACK_SUMS=np.asarray(affinity.sum(axis=1,dtype=np.float64))
    report=dict(scope='paired-query CPU timing; no population metrics',config=cfg,
        paired_sources=list(sources),
        commit=head,platform=platform.platform(),machine=platform.machine(),python=sys.version,
        numpy=np.__version__,workers=1,blas_threads=1,thread_environment={k:os.environ[k] for k in
        ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        loading_seconds=time.perf_counter()-load_start,runs=[])
    report['source_sha256']={name:full.sha(full.REPO/name) for name in (
        'RFDP/experiments/continuous_fitting/local_timing.py',
        'RFDP/experiments/continuous_fitting/model.py',
        'RFDP/experiments/continuous_fitting/certified.py',
        'RFDP/MODEL/Paper_RFDP/PaperFullIterative.py',
        'RFDP/experiments/continuous_fitting/full.py',
        'RFDP/experiments/continuous_fitting/verify.py',
        'RFDP/experiments/continuous_fitting/converged.py',
        'RFDP/experiments/continuous_fitting/fitting_workspace.py')}
    report['worktree_changes']=subprocess.check_output(
        ['git','status','--porcelain'],cwd=full.REPO,text=True)
    full.dump(output/'timing.json',report)
    for query in queries:
        for source in sources:
            dest=output/f'{query:04d}'/source
            (dest/'queries').mkdir(parents=True);(dest/'accepted').mkdir()
            run_cfg=dict(source=source,alpha=5.,beta=25.,step=5,depth=500,
                relation_model='original',admission_policy='original',retain_initial_state=True,
                trace_retention='sampled_after_readback',retained_queries=[query],
                commit=head,queries=qn,gallery=gn)
            if source == 'f1_robust_converged':
                run_cfg['nu']=1.5
            full.CFG,full.OUT=run_cfg,dest
            full.SIG=hashlib.sha256(json.dumps(dict(config=run_cfg,inputs=cfg['cache_sha256']),
                                                sort_keys=True).encode()).hexdigest()
            previous=full.install_solver(run_cfg)
            try:
                setup=time.perf_counter()
                core=model.FullIHyRDP(graph,affinity,alpha=5.,beta=25.,step=5)
                full.ENGINE=ProgressEngine(core,source,query)
                setup_seconds=time.perf_counter()-setup
                full.dump(dest/'run-manifest.json',dict(config=run_cfg,signature=full.SIG,
                    solver_id=full.solver_id(source),purpose=report['scope']))
                wall_start,cpu_start=time.perf_counter(),time.process_time()
                status=full.compute_query(query)
                wall,cpu=time.perf_counter()-wall_start,time.process_time()-cpu_start
                if 'error' in status:
                    full.dump(dest/'FAILED.json',status)
                    raise RuntimeError(status['error'])
                record=json.loads((dest/'queries'/f'{query:04d}.json').read_text())
                acceptance=json.loads((dest/'accepted'/f'{query:04d}.json').read_text())
                rank=np.load(dest/'queries'/f'{query:04d}.rankings.npy')
                assert rank.shape==(gn,) and np.array_equal(np.sort(rank),np.arange(qn,qn+gn))
                item=dict(source=source,query=query,wall_seconds=wall,cpu_seconds=cpu,
                    setup_seconds=setup_seconds,compute_seconds=record['compute_seconds'],
                    readback_seconds=record['readback_seconds'],diagnostics=record['diagnostics'],
                    full_query_accepted=True,gallery=gn,rankings_sha256=record['rankings_sha256'],
                    prefix_sha256=record['prefix_sha256'],acceptance=acceptance,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
                if source=='original' and cfg.get('baseline_prefix'):
                    saved=np.load(cfg['baseline_prefix'])
                    actual=np.load(dest/'queries'/f'{query:04d}.prefix.npy')
                    item['archived_baseline_prefix_identical']=bool(np.array_equal(saved,actual))
                    assert item['archived_baseline_prefix_identical'],'Baseline dynamic prefix differs'
                report['runs'].append(item)
                if source==sources[1]:
                    base=report['runs'][-2]
                    item['paired_compute_ratio']=item['compute_seconds']/base['compute_seconds']
                    item['paired_compute_and_readback_ratio']=wall/base['wall_seconds']
                full.dump(output/'timing.json',report)
                print(json.dumps({k:v for k,v in item.items() if k not in ('diagnostics','acceptance')},ensure_ascii=False),flush=True)
            finally:
                model.solve_hybrid_sparse=previous;certified.EVENTS.clear()
    report['status']='completed_listed_queries'
    full.dump(output/'timing.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    run(args.config)
