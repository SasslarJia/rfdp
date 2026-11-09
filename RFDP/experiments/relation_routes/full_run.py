"""Three independent official Market-OSNet runs, with resumable acceptance.

prepare binds existing features and builds only the new grouped graph. run
executes the three candidates in order, reusing SHA-verified completed queries.
Baseline methods are never runnable through this entry point.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import contextlib
import fcntl
import json
import math
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from . import protocol
from experiments.continuous_fitting import full


def allocated_cpu_count(cgroup=Path('/sys/fs/cgroup')):
    visible=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    maximum=cgroup/'cpu.max'
    if maximum.exists():text=maximum.read_text().strip()
    else:
        directory=next((cgroup/name for name in ('cpu','cpu,cpuacct')
                        if (cgroup/name/'cpu.cfs_quota_us').exists()),None)
        if directory is None:text='max 100000'
        else:
            quota=int((directory/'cpu.cfs_quota_us').read_text())
            period=int((directory/'cpu.cfs_period_us').read_text())
            text=f'{quota if quota>0 else "max"} {period}'
    quota,period=text.split()
    workers=visible if quota=='max' else min(visible,max(1,math.ceil(int(quota)/int(period))))
    return int(workers),dict(visible_affinity_cpus=visible,cpu_max=text,workers=int(workers),blas_threads=1)


def allocated_memory(cgroup=Path('/sys/fs/cgroup')):
    if (cgroup/'memory.max').exists():
        cap=(cgroup/'memory.max').read_text().strip()
        if cap=='max':raise RuntimeError('A finite memory allocation is required')
        return int(cap),int((cgroup/'memory.current').read_text())
    cap=int((cgroup/'memory/memory.limit_in_bytes').read_text())
    if cap>=2**60:raise RuntimeError('A finite memory allocation is required')
    return cap,int((cgroup/'memory/memory.usage_in_bytes').read_text())


def resource_snapshot(cfg,reserve=False):
    cgroup=Path('/sys/fs/cgroup')
    cap,cur=allocated_memory(cgroup)
    if cur>.7*cap:raise RuntimeError('RAM 70 percent stop')
    workers,cpu=allocated_cpu_count(cgroup)
    if workers!=cfg['workers']:raise RuntimeError('Allocated CPU count changed; rebind before execution')
    out=Path(cfg['output'])
    allocated=sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    budget=cfg['output_budget_bytes']
    if allocated>budget:raise RuntimeError('Output budget exceeded; never truncate evidence/computation')
    disks={}
    for target in ('/','/root/autodl-tmp'):
        d=shutil.disk_usage(target);floor=max(d.total//4,10*2**30)
        if d.free<floor or (reserve and target=='/root/autodl-tmp' and d.free-max(0,budget-allocated)<floor):
            raise RuntimeError('Disk reserve floor; no reduced-depth fallback')
        disks[target]=dict(total=d.total,free=d.free,floor=floor)
    return dict(time=full.now(),memory_current=cur,memory_max=cap,cpu=cpu,
                allocated_bytes=allocated,output_budget_bytes=budget,disks=disks,execution='CPU only')


@contextlib.contextmanager
def lock(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def checkout():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=full.REPO,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=full.REPO,text=True).strip():
        raise ValueError('Use a clean, fixed execution checkout')
    return head


def checked_inputs(base,input_id='market-osnet'):
    cache=Path(base['cache']).resolve();mat=Path(base['mat']).resolve()
    frozen=protocol.input_spec(input_id)
    expected={str(cache/name):value for name,value in frozen['cache_input_hashes'].items()}
    expected[str(mat)]=frozen['feature_sha256']
    for path,value in expected.items():
        if full.sha(path)!=value:raise ValueError('Frozen input SHA mismatch: '+path)
    return cache,mat,expected


def prepare(base_path,root,references_path,input_id='market-osnet'):
    """Required graph construction, not a query experiment or feature extraction."""
    from .csw_grouped import build_grouped_csw_inputs,validate_grouped_inputs
    from .report import load_references
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with lock(root/'.execution.lock'):
        head=checkout();base=json.loads(Path(base_path).read_text())
        frozen=protocol.input_spec(input_id);n=frozen['queries']+frozen['gallery']
        cache,mat,inputs=checked_inputs(base,input_id)
        for previous in [base.get('output'),str(cache),str(mat.parent)]:
            if previous:
                old=Path(previous).resolve()
                if root==old or root in old.parents or old in root.parents:
                    raise ValueError('Execution root overlaps existing input or result directory')
        refs=load_references(references_path,input_id)
        workers,cpu=allocated_cpu_count()
        probe=dict(protocol.recipe('camera_channel',input_id),workers=workers,output=str(root/'runs/camera_channel'))
        resource_snapshot(probe,reserve=True)
        marker=root/'PREPARED.json'
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved['commit']!=head or saved['references_sha256']!=protocol.digest(refs):
                raise ValueError('Prepared root has a different commit/reference binding')
            for name,value in saved['files_sha256'].items():
                if full.sha(root/name)!=value:raise ValueError('Prepared artifact changed: '+name)
            for route in protocol.ROUTES:
                c=json.loads((root/'configs'/f'{route}.json').read_text())
                protocol.validate(c,require_bound=True)
                if c['workers']!=workers:raise ValueError('CPU allocation changed')
            return saved
        assets=root/'inputs';assets.mkdir(exist_ok=True)
        camera_file=assets/'cameras.npy'
        data=loadmat(mat,variable_names=['query_cam','gallery_cam'])
        qcam,gcam=data['query_cam'].ravel(),data['gallery_cam'].ravel()
        if (len(qcam),len(gcam))!=(frozen['queries'],frozen['gallery']):raise ValueError('Official query/gallery camera population differs')
        cameras=np.r_[qcam,gcam]
        if cameras.shape!=(n,) or not np.all(np.isfinite(cameras)) or not np.all(cameras==cameras.astype(np.int64)):
            raise ValueError('Invalid frozen camera order')
        np.save(camera_file,cameras.astype(np.int64));cameras=cameras.astype(np.int64)
        affinity=np.load(cache/'affinity.npy',mmap_mode='r')
        if affinity.shape!=(n,n):raise ValueError('Official complete affinity required')
        asset_marker=assets/'GROUPED_COMPLETE.json';start=time.monotonic()
        if asset_marker.exists():
            gm=json.loads(asset_marker.read_text())
            for name,value in gm['files_sha256'].items():
                if full.sha(assets/name)!=value:raise ValueError('Grouped cache changed')
            graph=sparse.load_npz(assets/'grouped-graph.npz');support=sparse.load_npz(assets/'grouped-support.npz')
            meta=json.loads((assets/'grouped-metadata.json').read_text())
            validate_grouped_inputs(graph,support,meta,cameras=cameras,k_intra=5,k_inter=20)
            if gm['affinity_file_sha256']!=inputs[str(cache/'affinity.npy')]:raise ValueError('Grouped affinity binding changed')
        else:
            graph,support,meta=build_grouped_csw_inputs(affinity,cameras,k_intra=5,k_inter=20,
                progress=lambda item: print(json.dumps(item),flush=True))
            sparse.save_npz(assets/'grouped-graph.npz',graph)
            sparse.save_npz(assets/'grouped-support.npz',support)
            full.dump(assets/'grouped-metadata.json',meta)
            gm=dict(affinity_file_sha256=inputs[str(cache/'affinity.npy')],commit=head,
                seconds=time.monotonic()-start,files_sha256={name:full.sha(assets/name) for name in
                    ('grouped-graph.npz','grouped-support.npz','grouped-metadata.json')})
            full.dump(asset_marker,gm)
        common=dict(inputs);common[str(camera_file)]=full.sha(camera_file)
        configs=root/'configs';configs.mkdir(exist_ok=True)
        for route in protocol.ROUTES:
            cfg=protocol.recipe(route,input_id)
            cfg.update(path_binding='bound',commit=head,cache=str(cache),mat=str(mat),workers=workers,
                cpu_allocation=cpu,camera_ids_file=str(camera_file),input_sha256=dict(common),
                output=str(root/'runs'/route),binding_source='verified_frozen_sha_and_generated_grouped_graph')
            if route=='csw_grouped':
                for key,name in [('grouped_graph_file','grouped-graph.npz'),('support_file','grouped-support.npz'),('grouped_metadata_file','grouped-metadata.json')]:
                    cfg[key]=str(assets/name);cfg['input_sha256'][cfg[key]]=full.sha(assets/name)
            protocol.validate(cfg,require_bound=True)
            path=configs/f'{route}.json'
            if path.exists() and json.loads(path.read_text())!=cfg:
                raise ValueError('Refuse to replace a differently bound configuration')
            full.dump(path,cfg)
        full.dump(root/'references.json',refs)
        files=list(configs.glob('*.json'))+list(assets.glob('*'))+[root/'references.json']
        result=dict(status='prepared_not_started',commit=head,input_id=input_id,routes=list(protocol.ROUTES),
            cpu=cpu,references_sha256=protocol.digest(refs),
            files_sha256={str(p.relative_to(root)):full.sha(p) for p in files if p.is_file()})
        full.dump(marker,result)
        return result


def query_complete(cfg,signature,q):
    """Reuse only an intact earlier complete compute+readback transaction."""
    out=Path(cfg['output']);dest=out/'queries';rp=dest/f'{q:04d}.json'
    if not rp.exists():return False
    row=json.loads(rp.read_text())
    if row.get('signature')!=signature or row.get('query')!=q or row.get('source')!=cfg['source']:
        raise ValueError('Existing query belongs to a different run')
    if 'readback_sha256' not in row:return False
    for path,key in [(dest/f'{q:04d}.prefix.npy','prefix_sha256'),
                     (dest/f'{q:04d}.rankings.npy','rankings_sha256'),
                     (out/'accepted'/f'{q:04d}.json','readback_sha256')]:
        if not path.exists() or full.sha(path)!=row[key]:raise ValueError('Completed query artifact damaged: '+str(path))
    accepted=json.loads((out/'accepted'/f'{q:04d}.json').read_text())
    if accepted['signature']!=signature or accepted['query']!=q or accepted['trace_sha256']!=row['trace_sha256']:
        raise ValueError('Readback binding differs')
    if accepted['max_recomputed_residual']>1e-8:raise ValueError('Readback residual exceeds tolerance')
    trace=dest/f'{q:04d}.trace.jsonl.gz';keep=q in cfg['retained_queries']
    if row.get('trace_retained')!=keep:raise ValueError('Retention setting differs')
    if keep and (not trace.exists() or full.sha(trace)!=row['trace_sha256']):raise ValueError('Retained trace damaged')
    if not keep and trace.exists():raise ValueError('Unexpected trace after completed pruning')
    first=dest/f'{q:04d}.initial.jsonl.gz'
    if full.sha(first)!=row['initial_state_sha256'] or accepted['initial_state_sha256']!=row['initial_state_sha256']:
        raise ValueError('Saved initial state differs')
    return True


def accepted_result(out,cfg=None):
    out=Path(out);path=out/'FULL_ACCEPTANCE.json'
    if not path.exists():return None
    acceptance=json.loads(path.read_text());manifest=json.loads((out/'run-manifest.json').read_text())
    actual=manifest['config'];protocol.validate(actual,require_bound=True)
    if cfg is not None and actual!=cfg:raise ValueError('Accepted run config changed')
    if (acceptance['status']!='accepted_full_population' or acceptance['queries']!=actual['queries'] or
        acceptance['gallery']!=actual['gallery'] or acceptance['signature']!=manifest['signature'] or
        acceptance['signature']!=protocol.digest(actual)):
        raise ValueError('Invalid full-population acceptance')
    for name,key in [('metrics.json','metrics_sha256'),('rankings.npy','rankings_sha256')]:
        if full.sha(out/name)!=acceptance[key]:raise ValueError('Accepted artifact SHA changed: '+name)
    return acceptance


def run_one(config_path,resume=False):
    full.configure(config_path);cfg=full.CFG
    protocol.validate(cfg,require_bound=True);full.sources()
    out=full.OUT;signature=protocol.digest(cfg)
    if accepted_result(out,cfg):return 'reused_complete_run'
    initial=full.resources(reserve=True)
    shared=sum((Path(cfg['cache'])/n).stat().st_size for n in ('affinity.npy','features.npy'))
    total=initial['memory_current']+shared+cfg['workers']*(cfg['worker_address_space_limit_bytes']-shared)+3*2**30
    if total>.7*initial['memory_max']:raise RuntimeError('All-allocated-worker memory reserve exceeds 70 percent')
    if out.exists():
        if not resume:raise ValueError('Existing run requires explicit --resume; no overwrite')
        manifest=json.loads((out/'run-manifest.json').read_text())
        full.check_admission_manifest(cfg,manifest)
        if manifest['signature']!=signature:raise ValueError('Run signature changed')
    else:
        out.mkdir(parents=True);(out/'queries').mkdir();(out/'accepted').mkdir()
        full.dump(out/'run-manifest.json',dict(signature=signature,config=cfg,time=full.now(),
            resources=initial,static_total_bytes=total,shared_mapping_bytes=shared,solver_id=full.solver_id(cfg['source'])))
    failure=out/'FAILED.json'
    if failure.exists():
        with (out/'attempt-history.jsonl').open('a') as stream:stream.write(json.dumps(json.loads(failure.read_text()))+'\n')
        failure.unlink()
    if not (out/'COMPUTE_COMPLETE.json').exists():
        complete=[q for q in range(full.QN) if query_complete(cfg,signature,q)]
        done=set(complete);pending=[q for q in range(full.QN) if q not in done]
        full.dump(out/'resume-plan.json',dict(time=full.now(),complete=len(complete),pending=len(pending)))
        seconds=full.pool_stage('compute',pending,len(complete)) if pending else 0.
        if not all(query_complete(cfg,signature,q) for q in range(full.QN)):
            raise RuntimeError('Full compute population is not accepted')
        metric=full.evaluate();full.sources()
        full.dump(out/'COMPUTE_COMPLETE.json',dict(status='computed_and_independently_evaluated',
            signature=signature,query_count=full.QN,gallery_count=full.GN,
            rankings_sha256=full.sha(out/'rankings.npy'),independent=metric,
            latest_attempt_seconds=seconds,git_commit=cfg['commit'],inline_full_score_readback=True))
        (out/'compute.exit').write_text('0\n')
    complete=json.loads((out/'COMPUTE_COMPLETE.json').read_text())
    if complete['signature']!=signature or (complete['query_count'],complete['gallery_count'])!=(cfg['queries'],cfg['gallery']):
        raise ValueError('Combined computation binding differs')
    seconds=full.aggregate_inline_readback();full.sources()
    rows=[json.loads((out/'accepted'/f'{q:04d}.json').read_text()) for q in range(full.QN)]
    if max(r['max_recomputed_residual'] for r in rows)>1e-8:raise ValueError('Residual acceptance failed')
    full.dump(out/'FULL_ACCEPTANCE.json',dict(status='accepted_full_population',signature=signature,
        time=full.now(),queries=full.QN,gallery=full.GN,
        max_recomputed_residual=max(r['max_recomputed_residual'] for r in rows),
        solver_calls=sum(r['solver_calls'] for r in rows),
        rankings_sha256=full.sha(out/'rankings.npy'),metrics_sha256=full.sha(out/'metrics.json'),
        actual_saved_scores_recomputed=True,all_frontiers_bridges_rankings_rebuilt=True,
        certificate_implementation_shared=True,git_commit=cfg['commit'],seconds=seconds,
        readback_performed_inline=True,retained_queries=cfg['retained_queries'],initial_states_retained=True))
    return 'completed_full_population'


def run_queue(root,resume=False):
    from .report import collect
    root=Path(root).resolve()
    with lock(root/'.execution.lock'):
        prepared=json.loads((root/'PREPARED.json').read_text())
        if prepared.get('input_id','market-osnet')!='market-osnet':
            raise ValueError('Use launch_one and a Market decision for each transfer; no unconditional transfer queue')
        if prepared['routes']!=list(protocol.ROUTES) or prepared['commit']!=checkout():
            raise ValueError('Prepared queue binding changed')
        for name,value in prepared['files_sha256'].items():
            if full.sha(root/name)!=value:raise ValueError('Prepared configuration/input changed: '+name)
        for route in protocol.ROUTES:
            try:
                result=run_one(root/'configs'/f'{route}.json',resume)
                full.dump(root/'QUEUE_STATUS.json',dict(route=route,status=result,time=full.now()))
                collect(root,root/'references.json')
            except BaseException:
                error=dict(route=route,status='failed',time=full.now(),error=traceback.format_exc())
                full.dump(root/'QUEUE_STATUS.json',error)
                out=root/'runs'/route
                if out.exists():full.dump(out/'FAILED.json',error)
                collect(root,root/'references.json')
                raise
        full.dump(root/'QUEUE_COMPLETE.json',dict(status='three_candidates_accepted',time=full.now(),routes=list(protocol.ROUTES)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['recipes','prepare','run','collect'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base-config',type=Path)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--input-id',choices=tuple(protocol.INPUTS),default='market-osnet')
    args=parser.parse_args()
    if args.action=='recipes':
        args.root.mkdir(parents=True,exist_ok=True)
        for route in protocol.ROUTES:
            path=args.root/f'{args.input_id}.{route}.json'
            if path.exists():raise ValueError('Recipe already exists: '+str(path))
            full.dump(path,protocol.recipe(route,args.input_id))
    elif args.action=='prepare':
        if not args.base_config or not args.references:parser.error('prepare requires base-config and references')
        print(json.dumps(prepare(args.base_config,args.root,args.references,args.input_id)))
    elif args.action=='run':run_queue(args.root,args.resume)
    else:
        from .report import collect
        collect(args.root,args.references or args.root/'references.json')


if __name__=='__main__':main()
