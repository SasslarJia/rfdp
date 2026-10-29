"""Fixed full-population run and saved-score acceptance from a frozen Git checkout."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,gzip,hashlib,json,multiprocessing as mp,resource,shutil,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
from scipy import sparse
from . import certified
from .verify import verify_query,read_rows
from experiments.full_reid.common import load_features,complete_gallery
from EVAL.ReIDEval import evaluate_reid_rankings

REPO=Path(__file__).resolve().parents[3]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def now():
    import datetime
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def dump(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def configure(path):
    global CFG,OUT,QN,GN,WORKERS,BUDGET,WORKER_AS,CONFIG_PATH
    CONFIG_PATH=str(Path(path).resolve());CFG=json.loads(Path(path).read_text());OUT=Path(CFG['output'])
    QN,GN=CFG['queries'],CFG['gallery'];WORKERS=CFG['workers']
    BUDGET=CFG['output_budget_bytes'];WORKER_AS=CFG['worker_address_space_limit_bytes']

def sources():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert head==CFG['commit'],'Git execution revision changed'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip(),'Tracked execution files changed'
    for p,h in CFG['input_sha256'].items():assert sha(p)==h,p

def resources(reserve=False):
    cg=Path('/sys/fs/cgroup');cur=int((cg/'memory.current').read_text());cap=int((cg/'memory.max').read_text())
    assert cur<=cap*.7,'RAM 70 percent stop'
    allocated=sum(p.stat().st_blocks*512 for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    assert allocated<=BUDGET,'Output budget reached; no evidence truncation'
    disks={}
    for name in ['/','/root/autodl-tmp']:
        d=shutil.disk_usage(name);floor=max(d.total//4,10*2**30)
        assert d.free>=floor,'Disk floor'
        if reserve and name=='/root/autodl-tmp':assert d.free-(BUDGET-allocated)>=floor,'Full output reservation'
        disks[name]=dict(total=d.total,free=d.free)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    assert all(int(line.split(',')[0])<=.75*int(line.split(',')[1]) for line in gpu.splitlines())
    return dict(time=now(),memory_current=cur,memory_max=cap,cpu_max=(cg/'cpu.max').read_text().strip(),gpu=gpu,disks=disks,allocated_bytes=allocated,output_budget_bytes=BUDGET)

def guard(stage):
    stop=threading.Event()
    def loop():
        while not stop.wait(20):
            try:dump(OUT/'latest-resources.json',resources())
            except BaseException:
                dump(OUT/'FAILED.json',dict(stage=stage,kind='resource_stop',time=now(),error=traceback.format_exc()))
                os.kill(os.getpid(),signal.SIGTERM)
    thread=threading.Thread(target=loop,daemon=True);thread.start();return stop,thread

def initialize_worker(stage,config_path):
    global ENGINE,GRAPH,FULL,X,SIG,INIT_ERROR
    INIT_ERROR=None
    try:
        configure(config_path);resource.setrlimit(resource.RLIMIT_AS,(WORKER_AS,WORKER_AS))
        cache=Path(CFG['cache']);GRAPH=sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr();GRAPH.eliminate_zeros();GRAPH.sort_indices()
        FULL=np.load(cache/'affinity.npy',mmap_mode='r');X=np.load(cache/'features.npy',mmap_mode='r')
        assert FULL.shape==(QN+GN,QN+GN) and X.shape[0]==QN+GN
        SIG=json.loads((OUT/'run-manifest.json').read_text())['signature']
        if stage=='compute':
            from MODEL.Paper_RFDP import PaperFullIterative as model
            certified.install(CFG['source'])
            ENGINE=model.FullIHyRDP(GRAPH,FULL,alpha=CFG['alpha'],beta=CFG['beta'],step=5)
    except BaseException:INIT_ERROR=traceback.format_exc()

def compute_query(query):
    try:
        if INIT_ERROR:raise RuntimeError(INIT_ERROR)
        start=time.monotonic();certified.EVENTS.clear();seen=0
        similarity=X@X[query];dest=OUT/'queries';trace=dest/f'{query:04d}.trace.jsonl.gz'
        with gzip.open(trace,'wt',encoding='utf-8',compresslevel=6) as stream:
            def capture(t):
                nonlocal seen
                event=None
                if t['scores'] is not None:
                    assert len(certified.EVENTS)==seen+1;event=certified.EVENTS[seen];seen+=1
                    assert np.asarray(t['unlabeled'])[event['returned_prefix_local']][:len(t['selected'])].tolist()==t['selected']
                else:assert len(certified.EVENTS)==seen
                row={k:t[k] for k in ['iteration','labeled','direct','indirect','unlabeled','selected','bridges','residual']}
                row['returned_scores']=None if t['scores'] is None else [certified.packed(v) for v in t['scores']]
                row['event']=event;stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
            prefix,diag=ENGINE.rank([query],nq=CFG['depth'],input_similarity=similarity,numerical_tail=False,trace=capture)
        assert diag['status']=='complete_diffusion' and diag['numerical_tail_count']==0 and len(prefix)==CFG['depth']
        assert diag['max_relative_residual']<=1e-8
        rank=complete_gallery(prefix,similarity,QN)
        pp=dest/f'{query:04d}.prefix.npy';rp=dest/f'{query:04d}.rankings.npy'
        np.save(pp,prefix.astype(np.int32));np.save(rp,rank)
        row=dict(query=query,source=CFG['source'],signature=SIG,diagnostics=diag,seconds=time.monotonic()-start,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 trace_sha256=sha(trace),prefix_sha256=sha(pp),rankings_sha256=sha(rp),pid=os.getpid())
        dump(dest/f'{query:04d}.json',row)
        return {k:row[k] for k in ['query','seconds','peak_rss_bytes','diagnostics','pid']}
    except BaseException:return dict(query=query,error=traceback.format_exc())

def accept_query(query):
    try:
        if INIT_ERROR:raise RuntimeError(INIT_ERROR)
        dest=OUT/'queries';row=json.loads((dest/f'{query:04d}.json').read_text())
        assert row['query']==query and row['signature']==SIG and row['source']==CFG['source']
        for suffix,key in [('trace.jsonl.gz','trace_sha256'),('prefix.npy','prefix_sha256'),('rankings.npy','rankings_sha256')]:assert sha(dest/f'{query:04d}.{suffix}')==row[key]
        result=verify_query(GRAPH,FULL,X@X[query],query,np.load(dest/f'{query:04d}.prefix.npy'),np.load(dest/f'{query:04d}.rankings.npy'),read_rows(dest/f'{query:04d}.trace.jsonl.gz'),CFG['source'],QN,CFG['depth'])
        for k in ['solver_calls','iterations','bridge_events','bridge_edges','max_unlabeled']:assert result[k]==row['diagnostics'][k]
        assert result['max_reported_residual']==row['diagnostics']['max_relative_residual']
        result.update(query=query,signature=SIG,trace_sha256=row['trace_sha256']);dump(OUT/'accepted'/f'{query:04d}.json',result);return result
    except BaseException:return dict(query=query,error=traceback.format_exc())

def pool_stage(stage):
    start=time.monotonic();fn=compute_query if stage=='compute' else accept_query
    pool=mp.get_context('spawn').Pool(WORKERS,initializer=initialize_worker,initargs=(stage,CONFIG_PATH));stop,thread=guard(stage)
    def stopped(signum,frame):raise RuntimeError('Stopped by signal '+str(signum))
    old=signal.signal(signal.SIGTERM,stopped)
    try:
        for count,result in enumerate(pool.imap_unordered(fn,range(QN),chunksize=1),1):
            if 'error' in result:
                dump(OUT/'FAILED.json',dict(stage=stage,time=now(),**result));raise RuntimeError(result['error'])
            dump(OUT/(stage+'-progress.json'),dict(time=now(),stage=stage,completed=count,total=QN,last=result,seconds=time.monotonic()-start))
            print(json.dumps(dict(stage=stage,completed=count,query=result['query'])),flush=True)
        pool.close();pool.join()
    finally:
        stop.set();thread.join(21);pool.terminate();pool.join();signal.signal(signal.SIGTERM,old)
    return time.monotonic()-start

def evaluate():
    data,nq,ng=load_features(Path(CFG['mat']));assert (nq,ng)==(QN,GN)
    ranks=np.stack([np.load(OUT/'queries'/f'{q:04d}.rankings.npy') for q in range(QN)])
    metric=evaluate_reid_rankings(ranks,range(QN),*[data[k] for k in ['query_label','query_cam','gallery_label','gallery_cam']],QN,max_rank=500,return_per_query=True)
    qp,qc,gp,gc=[np.asarray(data[k]).ravel() for k in ['query_label','query_cam','gallery_label','gallery_cam']];aps=[];r1=[]
    for q,rank in enumerate(ranks):
        order=rank-QN;keep=(gp[order]!=-1)&~((gp[order]==qp[q])&(gc[order]==qc[q]));hits=gp[order[keep]]==qp[q]
        positives=np.count_nonzero((gp==qp[q])&(gc!=qc[q])&(gp!=-1));assert hits.sum()==positives
        if positives:aps.append(float(np.mean(np.arange(1,positives+1)/(np.flatnonzero(hits)+1))));r1.append(float(hits[0]))
    assert abs(np.mean(aps)-metric['mAP'])<1e-12 and np.mean(r1)==metric['rank-1']
    np.save(OUT/'rankings.npy',ranks);dump(OUT/'metrics.json',metric)
    return dict(mAP=float(np.mean(aps)),rank1=float(np.mean(r1)),valid_queries=len(aps))

def main(stage,config_path):
    configure(config_path);sources()
    assert CFG['source'] in ('continuous','boundary_odds') and (CFG['alpha'],CFG['beta'],CFG['step'],CFG['depth'],QN,GN,WORKERS)==(5,25,5,500,3368,19732,16)
    if stage=='compute':
        assert not OUT.exists(),'New run only; no automatic overwrite or retry'
        initial=resources(reserve=True)
        shared=sum((Path(CFG['cache'])/n).stat().st_size//4096*4096 for n in ['affinity.npy','features.npy'])
        static_total=initial['memory_current']+shared+WORKERS*(WORKER_AS-shared)+3*2**30
        assert static_total<=.7*initial['memory_max'],('16-worker address-space budget',static_total)
        OUT.mkdir();(OUT/'queries').mkdir()
        scientific={k:CFG[k] for k in ['source','alpha','beta','step','depth','k','gamma','queries','gallery','feature_sha256','graph_sha256']}
        signature=hashlib.sha256(json.dumps(scientific,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        dump(OUT/'run-manifest.json',dict(signature=signature,config=CFG,time=now(),resources=initial,static_total_bytes=static_total,shared_mapping_bytes=shared,solver_id=certified.SOLVER_ID))
        seconds=pool_stage(stage)
        assert len(list((OUT/'queries').glob('*.json')))==QN
        metric=evaluate();sources()
        dump(OUT/'COMPUTE_COMPLETE.json',dict(status='computed_and_independently_evaluated',signature=signature,query_count=QN,gallery_count=GN,rankings_sha256=sha(OUT/'rankings.npy'),independent=metric,seconds=seconds,git_commit=CFG['commit'],resources=resources()))
    elif stage=='acceptance':
        assert (OUT/'compute.exit').read_text().strip()=='0';(OUT/'accepted').mkdir();seconds=pool_stage(stage);sources()
        rows=[json.loads(p.read_text()) for p in (OUT/'accepted').glob('*.json')];assert len(rows)==QN
        manifest=json.loads((OUT/'run-manifest.json').read_text())
        dump(OUT/'FULL_ACCEPTANCE.json',dict(status='accepted_full_population',signature=manifest['signature'],time=now(),queries=QN,solver_calls=sum(r['solver_calls'] for r in rows),max_recomputed_residual=max(r['max_recomputed_residual'] for r in rows),rankings_sha256=sha(OUT/'rankings.npy'),actual_saved_scores_recomputed=True,all_frontiers_bridges_rankings_rebuilt=True,certificate_implementation_shared=True,seconds=seconds,git_commit=CFG['commit'],resources=resources()))
    else:raise ValueError(stage)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['compute','acceptance']);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    lock=args.config.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main(args.stage,args.config)
    except BaseException:
        if 'OUT' in globals() and OUT.exists() and not (OUT/'FAILED.json').exists():dump(OUT/'FAILED.json',dict(stage=args.stage,time=now(),error=traceback.format_exc()))
        raise
