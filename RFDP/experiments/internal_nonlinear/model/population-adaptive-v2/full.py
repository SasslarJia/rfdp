"""Resume the identical scientific run with adaptive certified arithmetic."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys,json,time,hashlib,gzip,resource,shutil,subprocess,threading,traceback,multiprocessing as mp,fcntl,signal
from pathlib import Path
import numpy as np
from scipy import sparse
HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/rfdp-graph-routes-20261026')
QUAL=HERE.parent/'retrieval-qualification-v1'
OUT=ROOT/'market-internal-nonlinear-theta0p5-v1'
QN,GN,WORKERS=3368,19732,16
REVISION='adaptive-v2'
BUDGET=40*2**30
WORKER_AS=4*2**30
sys.path.insert(0,str(HERE))
import model_loader as qualified
from verify_artifact import verify_query,read_rows
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def now():
    import datetime
    return datetime.datetime.now().astimezone().isoformat()
def dump(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sources():
    lock=json.loads((HERE/'source-lock.json').read_text())
    for n,h in lock['files'].items():assert sha(HERE/n)==h,n
    for n,h in lock['dependencies'].items():assert sha(HERE/n)==h,n
    refs=json.loads((HERE/'references.json').read_text())
    for n,h in refs['locked_files'].items():assert sha(n)==h,n
    return refs
def resources(reserve=False):
    cg=Path('/sys/fs/cgroup');cur=int((cg/'memory.current').read_text());cap=int((cg/'memory.max').read_text())
    assert cur<=cap*.7,'RAM 70 percent stop'
    allocated=sum(p.stat().st_blocks*512 for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    assert allocated<=BUDGET,'Output budget reached; stop without truncating evidence'
    disks={}
    for n in ['/','/root/autodl-tmp']:
        d=shutil.disk_usage(n);floor=max(d.total//4,10*2**30)
        assert d.free>=floor,'Disk floor'
        if reserve and n=='/root/autodl-tmp':assert d.free-(BUDGET-allocated)>=floor,'Full output reservation'
        disks[n]=dict(total=d.total,free=d.free)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    assert all(int(l.split(',')[0])<=.75*int(l.split(',')[1]) for l in gpu.splitlines())
    return dict(time=now(),memory_current=cur,memory_max=cap,cpu_max=(cg/'cpu.max').read_text().strip(),gpu=gpu,disks=disks,allocated_bytes=allocated,output_budget_bytes=BUDGET)
def guard(stage):
    stop=threading.Event()
    def loop():
        while not stop.wait(20):
            try:dump(OUT/'latest-resources.json',resources())
            except BaseException:
                dump(OUT/'FAILED.json',dict(stage=stage,kind='resource_stop',time=now(),error=traceback.format_exc()))
                os.kill(os.getpid(),15)
    t=threading.Thread(target=loop,daemon=True);t.start();return stop,t
def initialize_worker(stage):
    global ENGINE,PRECISION,EVENTS,GRAPH,FULL,X,REFS,SIG
    resource.setrlimit(resource.RLIMIT_AS,(WORKER_AS,WORKER_AS))
    REFS=json.loads((HERE/'references.json').read_text())
    SIG=json.loads((OUT/'run-manifest.json').read_text())['signature']
    cache=Path(REFS['cache']);GRAPH=sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr();GRAPH.eliminate_zeros();GRAPH.sort_indices()
    FULL=np.load(cache/'affinity.npy',mmap_mode='r');X=np.load(cache/'features.npy',mmap_mode='r')
    assert FULL.shape==(23100,23100) and X.shape[0]==23100
    if stage=='compute':
        qualified.OUT=OUT
        mod,PRECISION,EVENTS=qualified.load_model(REFS,.5)
        ENGINE=mod.FullIHyRDP(GRAPH,FULL,alpha=5,beta=25,step=5)
def init_worker(stage):
    global INIT_ERROR
    INIT_ERROR=None
    try:initialize_worker(stage)
    except BaseException:INIT_ERROR=traceback.format_exc()
def compute_query(query):
    try:
        if INIT_ERROR:raise RuntimeError(INIT_ERROR)
        start=time.monotonic();EVENTS.clear();PRECISION.EVENTS.clear();seen=0
        similarity=X@X[query];dest=OUT/'queries';tf=dest/(f'{query:04d}.trace.jsonl.gz')
        with gzip.open(tf,'wt',encoding='utf-8',compresslevel=6) as f:
            def capture(t):
                nonlocal seen
                ev=None
                if t['scores'] is not None:
                    assert len(EVENTS)==seen+1;ev=EVENTS[seen];seen+=1
                    assert np.asarray(t['unlabeled'])[ev['returned_prefix_local']][:len(t['selected'])].tolist()==t['selected']
                else:assert len(EVENTS)==seen
                row={k:t[k] for k in ['iteration','labeled','direct','indirect','unlabeled','selected','bridges','residual']}
                row['returned_scores']=None if t['scores'] is None else [qualified.adapter.packed(v) for v in t['scores']]
                row['event']=ev
                f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
            prefix,diag=ENGINE.rank([query],nq=500,input_similarity=similarity,numerical_tail=False,trace=capture)
        assert diag['status']=='complete_diffusion' and diag['numerical_tail_count']==0
        assert diag['max_relative_residual']<=1e-8 and len(prefix)==500 and prefix[0]==query and len(set(prefix))==500
        used=np.zeros(QN+GN,dtype=bool);used[prefix]=True;rest=np.flatnonzero(~used & (np.arange(QN+GN)>=QN))
        rank=np.r_[prefix[prefix>=QN],rest[np.lexsort((rest,-similarity[rest]))]].astype(np.int32)
        assert np.array_equal(np.sort(rank),np.arange(QN,QN+GN))
        pp=dest/f'{query:04d}.prefix.npy';rp=dest/f'{query:04d}.rankings.npy';np.save(pp,prefix.astype(np.int32));np.save(rp,rank)
        row=dict(query=query,theta=.5,signature=SIG,execution_revision=REVISION,diagnostics=diag,seconds=time.monotonic()-start,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 trace_sha256=sha(tf),prefix_sha256=sha(pp),rankings_sha256=sha(rp),pid=os.getpid())
        if query in qualified.QUERIES:
            prior=json.loads((QUAL/'attempt-1/theta0p5'/f'{query}.json').read_text())
            assert all(row[k]==prior[k] for k in ['prefix_sha256','rankings_sha256'])
        dump(dest/f'{query:04d}.json',row)
        return {k:row[k] for k in ['query','seconds','peak_rss_bytes','diagnostics','pid']}
    except BaseException:return dict(query=query,error=traceback.format_exc())
def accept_query(query):
    try:
        if INIT_ERROR:raise RuntimeError(INIT_ERROR)
        dest=OUT/'queries';row=json.loads((dest/f'{query:04d}.json').read_text());assert row['signature']==SIG and row['query']==query and row['theta']==.5
        for suffix,key in [('trace.jsonl.gz','trace_sha256'),('prefix.npy','prefix_sha256'),('rankings.npy','rankings_sha256')]:assert sha(dest/f'{query:04d}.{suffix}')==row[key]
        result=verify_query(GRAPH,FULL,X@X[query],query,np.load(dest/f'{query:04d}.prefix.npy'),np.load(dest/f'{query:04d}.rankings.npy'),read_rows(dest/f'{query:04d}.trace.jsonl.gz'),.5,QN,500)
        for k in ['solver_calls','iterations','bridge_events','bridge_edges','max_unlabeled']:assert result[k]==row['diagnostics'][k]
        assert result['max_reported_residual']==row['diagnostics']['max_relative_residual']
        result.update(query=query,signature=SIG,trace_sha256=row['trace_sha256']);dump(OUT/'accepted'/f'{query:04d}.json',result);return result
    except BaseException:return dict(query=query,error=traceback.format_exc())
def pool_stage(stage,pending=None,inherited=0):
    start=time.monotonic();fn=compute_query if stage=='compute' else accept_query
    pool=mp.get_context('spawn').Pool(WORKERS,initializer=init_worker,initargs=(stage,));stop,t=guard(stage)
    def stopped(signum,frame):raise RuntimeError('Stopped by signal '+str(signum))
    old=signal.signal(signal.SIGTERM,stopped)
    try:
        for count,result in enumerate(pool.imap_unordered(fn,range(QN) if pending is None else pending,chunksize=1),1):
            if 'error' in result:dump(OUT/'FAILED.json',dict(stage=stage,time=now(),**result));raise RuntimeError(result['error'])
            dump(OUT/(stage+'-progress.json'),dict(time=now(),stage=stage,completed=inherited+count,total=QN,last=result,seconds=time.monotonic()-start,execution_revision=REVISION,inherited_queries=inherited))
            print(json.dumps(dict(stage=stage,completed=inherited+count,query=result['query'],execution_revision=REVISION)),flush=True)
        pool.close();pool.join()
    finally:stop.set();t.join(21);pool.terminate();pool.join();signal.signal(signal.SIGTERM,old)
    return time.monotonic()-start
def evaluate():
    refs=json.loads((HERE/'analysis-inputs.json').read_text());repo=Path(json.loads((HERE/'references.json').read_text())['repo'])
    sys.path[:0]=[str(repo/'RFDP'),str(repo/'RFDP/experiments/full_reid')]
    from common import load_features
    from EVAL.ReIDEval import evaluate_reid_rankings
    data,nq,ng=load_features(Path(refs['mat']));assert (nq,ng)==(QN,GN)
    ranks=np.stack([np.load(OUT/'queries'/f'{q:04d}.rankings.npy') for q in range(QN)])
    metric=evaluate_reid_rankings(ranks,range(QN),*[data[k] for k in ['query_label','query_cam','gallery_label','gallery_cam']],QN,max_rank=500,return_per_query=True)
    qp,qc,gp,gc=[np.asarray(data[k]).ravel() for k in ['query_label','query_cam','gallery_label','gallery_cam']];aps=[];r1=[]
    for q,rank in enumerate(ranks):
        o=rank-QN;keep=(gp[o]!=-1)&~((gp[o]==qp[q])&(gc[o]==qc[q]));hits=gp[o[keep]]==qp[q]
        positives=np.count_nonzero((gp==qp[q])&(gc!=qc[q])&(gp!=-1));assert hits.sum()==positives
        if positives:aps.append(float(np.mean(np.arange(1,positives+1)/(np.flatnonzero(hits)+1))));r1.append(float(hits[0]))
    assert abs(np.mean(aps)-metric['mAP'])<1e-12 and np.mean(r1)==metric['rank-1']
    np.save(OUT/'rankings.npy',ranks);dump(OUT/'metrics.json',metric)
    return dict(mAP=float(np.mean(aps)),rank1=float(np.mean(r1)),valid_queries=len(aps))
def completed_queries():
    manifest=json.loads((OUT/'run-manifest.json').read_text());sig=manifest['signature'];done=[]
    for p in sorted((OUT/'queries').glob('*.json')):
        row=json.loads(p.read_text());q=row['query']
        assert type(q) is int and 0<=q<QN and p.name==f'{q:04d}.json' and row['signature']==sig
        assert row['theta']==.5 and row['diagnostics']['status']=='complete_diffusion' and row['diagnostics']['numerical_tail_count']==0
        for suffix,key in [('trace.jsonl.gz','trace_sha256'),('prefix.npy','prefix_sha256'),('rankings.npy','rankings_sha256')]:
            assert sha(OUT/'queries'/f'{q:04d}.{suffix}')==row[key],('Completed artifact changed',q,suffix)
        done.append(q)
    assert len(done)==len(set(done))
    return done

def main(stage):
    refs=sources();config=json.loads((HERE/'config.json').read_text())
    if stage=='compute':
        assert OUT.is_dir() and (OUT/'migration-adaptive-v2.json').exists()
        migration=json.loads((OUT/'migration-adaptive-v2.json').read_text());assert migration['user_authorized'] and migration['old_processes_stopped']
        # A relative-path Python argv still has the old cwd. Enumerate both.
        oldcwd=HERE.parent/'population-v1'
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:cwd=os.readlink(proc/'cwd');argv=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
            except (OSError,UnicodeError):continue
            assert not(cwd==str(oldcwd) and ('full.py' in argv or 'multiprocessing.spawn' in argv or 'launch.sh' in argv)),('Old execution still active',proc.name)
        initial=resources(reserve=True)
        cache=Path(refs['cache']);shared=sum(((cache/n).stat().st_size//4096)*4096 for n in ['affinity.npy','features.npy'])
        static_total=initial['memory_current']+shared+WORKERS*(WORKER_AS-shared)+3*2**30
        assert static_total<=.7*initial['memory_max'],('16-worker static address-space budget',static_total)
        manifest=json.loads((OUT/'run-manifest.json').read_text());sig=manifest['signature']
        assert sig==migration['original_signature']
        for key in ['theta','alpha','beta','step','depth','k','gamma','queries','gallery']:
            assert config[key]==manifest['config'][key],key
        done=completed_queries();assert set(done)==set(migration['completed_queries'])
        dump(OUT/'execution-adaptive-v2.json',dict(time=now(),pid=os.getpid(),workers=WORKERS,resources=initial,
            static_total_bytes=static_total,shared_input_mapping_bytes=shared,worker_address_space_limit=WORKER_AS,
            source_lock_sha256=sha(HERE/'source-lock.json'),original_signature=sig,inherited_queries=done,
            precision_schedule=[24,40,80,120],scientific_model_unchanged=True,additional_performance_test=False))
        pending=[q for q in range(QN) if q not in set(done)]
        seconds=pool_stage(stage,pending,len(done)) if pending else 0.
        assert len(completed_queries())==QN
        metric=evaluate();sources()
        dump(OUT/'COMPUTE_COMPLETE.json',dict(status='computed_and_independently_evaluated',signature=sig,query_count=QN,
            gallery_count=GN,rankings_sha256=sha(OUT/'rankings.npy'),independent=metric,seconds=seconds,
            seconds_scope='adaptive_revision_only; original elapsed retained in migration',inherited_queries=len(done),
            execution_revision=REVISION,source_lock_sha256=sha(HERE/'source-lock.json'),resources=resources()))
    elif stage=='acceptance':
        assert (OUT/'compute.exit').read_text().strip()=='0';(OUT/'accepted').mkdir();seconds=pool_stage(stage);sources()
        m=json.loads((OUT/'run-manifest.json').read_text());rows=[json.loads(p.read_text()) for p in (OUT/'accepted').glob('*.json')];assert len(rows)==QN
        dump(OUT/'FULL_ACCEPTANCE.json',dict(status='accepted_full_population',signature=m['signature'],execution_revision=REVISION,source_lock_sha256=sha(HERE/'source-lock.json'),time=now(),queries=QN,solver_calls=sum(r['solver_calls'] for r in rows),max_recomputed_residual=max(r['max_recomputed_residual'] for r in rows),rankings_sha256=sha(OUT/'rankings.npy'),actual_saved_scores_recomputed=True,all_frontiers_bridges_rankings_rebuilt=True,seconds=seconds,resources=resources()))
    else:raise ValueError(stage)
if __name__=='__main__':
    stage=sys.argv[1]
    lockfile=(HERE/'execution.lock').open('a')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main(stage)
    except BaseException:
        if OUT.exists() and not (OUT/'FAILED.json').exists():dump(OUT/'FAILED.json',dict(stage=stage,time=now(),error=traceback.format_exc()))
        raise
