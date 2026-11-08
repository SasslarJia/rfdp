"""Fixed full-population run and saved-score acceptance from a frozen Git checkout."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,gzip,hashlib,io,json,multiprocessing as mp,resource,shutil,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
from scipy import sparse
from . import certified, f1
from .f1_config import validate_f1_config
from . import research_config
from .verify import verify_query,read_rows,WORK_COUNTERS
from experiments.full_reid.common import load_features,complete_gallery
from EVAL.ReIDEval import evaluate_reid_rankings
from MODEL.Paper_RFDP.PaperFullIterative import ADMISSION_POLICIES,ADMISSION_VERSION

REPO=Path(__file__).resolve().parents[3]
READBACK_KERNEL=READBACK_SUMS=None
ADMISSION_CAMERAS=None

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
    CONFIG_PATH=str(Path(path).resolve());CFG=json.loads(Path(path).read_text())
    # Unbound preparation artifacts must stop before data/resource access.
    validate_f1_config(CFG,require_bound=True)
    research_config.validate(CFG,require_bound=True,require_enabled=True)
    OUT=Path(CFG['output'])
    QN,GN=CFG['queries'],CFG['gallery'];WORKERS=CFG['workers']
    BUDGET=CFG['output_budget_bytes'];WORKER_AS=CFG['worker_address_space_limit_bytes']

def admission_binding(cfg):
    """Validate the optional policy and return its scientific input binding."""
    policy=cfg.get('admission_policy','original')
    assert policy in ADMISSION_POLICIES,'Unknown admission policy'
    if policy=='original':return {}
    assert cfg['source']=='original' and cfg.get('relation_model','original')=='original'
    assert (cfg['alpha'],cfg['beta'],cfg['step'])==(5.,25.,5)
    assert cfg['admission_version']==ADMISSION_VERSION
    camera_file=cfg['camera_ids_file']
    assert camera_file in cfg['input_sha256'],'Camera file must be SHA-bound'
    return dict(admission_policy=policy,admission_version=ADMISSION_VERSION,
                camera_ids_sha256=cfg['input_sha256'][camera_file])

def load_admission_cameras(cfg,n):
    binding=admission_binding(cfg)
    if not binding:return None
    assert sha(cfg['camera_ids_file'])==binding['camera_ids_sha256'],'Camera file SHA mismatch'
    cameras=np.load(cfg['camera_ids_file'],allow_pickle=False)
    assert cameras.shape==(n,) and np.issubdtype(cameras.dtype,np.integer)
    return cameras

def check_admission_manifest(cfg,manifest):
    # Also required when final acceptance reuses earlier inline read-backs.
    assert admission_binding(cfg)==admission_binding(manifest['config']),'Admission binding differs from computed run'
    assert fitting_binding(cfg)==fitting_binding(manifest['config']),'Fitting binding differs from computed run'
    if cfg['source'] in research_config.SOURCES:
        assert manifest['solver_id']==solver_id(cfg['source']),'Saved solver implementation differs'
        assert manifest['config']['commit']==cfg['commit'],'Saved execution commit differs'

def fitting_binding(cfg):
    binding=validate_f1_config(cfg,require_bound=True)
    binding.update(research_config.validate(cfg,require_bound=True))
    if cfg['source'] in research_config.SOURCES:
        binding['research_solver_id']=modern_solver(cfg['source']).SOLVER_ID
    if cfg['source'] in f1.SOURCES:
        binding.update(f1_solver_id=f1.SOLVER_ID,f1_weight_rule=f1.WEIGHT_RULE)
    return binding

def modern_solver(source):
    if source in research_config.MAIN:
        from . import converged
        return converged
    if source==research_config.BACKUP:
        from . import diagonal_free
        return diagonal_free
    raise ValueError(source)

def solver_id(source):
    if source in research_config.SOURCES:return modern_solver(source).SOLVER_ID
    return f1.SOLVER_ID if source in f1.SOURCES else certified.SOLVER_ID

def install_solver(cfg):
    if cfg['source'] in research_config.SOURCES:
        return modern_solver(cfg['source']).install(cfg['source'])
    if cfg['source'] in f1.SOURCES:
        return f1.install(cfg['source'],nu=cfg['f1_nu'])
    return certified.install(cfg['source'])

def inline_readback():
    return CFG.get('trace_retention','all')=='sampled_after_readback'

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
    global ENGINE,GRAPH,FULL,X,SIG,INIT_ERROR,READBACK_KERNEL,READBACK_SUMS,ADMISSION_CAMERAS
    INIT_ERROR=None
    try:
        configure(config_path);resource.setrlimit(resource.RLIMIT_AS,(WORKER_AS,WORKER_AS))
        cache=Path(CFG['cache']);GRAPH=sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr();GRAPH.eliminate_zeros();GRAPH.sort_indices()
        FULL=np.load(cache/'affinity.npy',mmap_mode='r');X=np.load(cache/'features.npy',mmap_mode='r')
        assert FULL.shape==(QN+GN,QN+GN) and X.shape[0]==QN+GN
        manifest=json.loads((OUT/'run-manifest.json').read_text())
        check_admission_manifest(CFG,manifest);SIG=manifest['signature']
        READBACK_KERNEL=READBACK_SUMS=None
        ADMISSION_CAMERAS=load_admission_cameras(CFG,QN+GN)
        from MODEL.Paper_RFDP import PaperFullIterative as model
        if CFG.get('relation_model','original')=='csw':
            from experiments.stationary_measure.csw_graph import load_inputs,ReadbackKernel
            from experiments.reliability.reliability_model import ReliabilityIHyRDP
            cameras,support,_=load_inputs(CFG,FULL)
            reader=ReadbackKernel(GRAPH,FULL,cameras,support,CFG['rho'],CFG['support_lambda'])
            READBACK_KERNEL,READBACK_SUMS=reader.block,reader.sums
            if stage=='compute':
                install_solver(CFG)
                # load_inputs already checked full-affinity/camera/support bindings.
                ENGINE=ReliabilityIHyRDP(GRAPH,FULL,cameras,eta=CFG['rho'],support=support,
                    lam=CFG['support_lambda'],alpha=CFG['alpha'],beta=CFG['beta'],step=CFG['step'])
                assert (ENGINE.w-reader.graph).nnz==0
                assert np.array_equal(ENGINE.full_sums,READBACK_SUMS)
            GRAPH=reader.graph
        elif stage=='compute':
            install_solver(CFG)
            ENGINE=model.FullIHyRDP(GRAPH,FULL,alpha=CFG['alpha'],beta=CFG['beta'],step=5,
                admission_policy=CFG.get('admission_policy','original'),camera_ids=ADMISSION_CAMERAS)
        if inline_readback() and READBACK_SUMS is None:
            READBACK_SUMS=np.asarray(FULL.sum(axis=1,dtype=np.float64))
    except BaseException:INIT_ERROR=traceback.format_exc()

def compute_query(query):
    try:
        if INIT_ERROR:raise RuntimeError(INIT_ERROR)
        binding=admission_binding(CFG)
        if binding:assert ENGINE.admission_policy==binding['admission_policy']
        start=time.monotonic();certified.EVENTS.clear();seen=0;linear_systems=0;mm_iterations=0;linear_solve_calls=0;refinement_solve_calls=0;first_state=None
        work_counts=dict.fromkeys(WORK_COUNTERS,0)
        similarity=X@X[query];dest=OUT/'queries';trace=dest/f'{query:04d}.trace.jsonl.gz'
        with gzip.open(trace,'wt',encoding='utf-8',compresslevel=6) as stream:
            def capture(t):
                nonlocal seen,linear_systems,mm_iterations,linear_solve_calls,refinement_solve_calls,first_state
                event=None
                nominated=t['nominated'] if binding else t['selected']
                if t['scores'] is not None:
                    assert len(certified.EVENTS)==seen+1;event=certified.EVENTS[seen];seen+=1
                    linear_systems+=event.get('linear_systems',1)
                    mm_iterations+=event.get('mm_iterations',0)
                    linear_solve_calls+=event.get('linear_solve_calls',event.get('cg_calls',0)+event.get('direct_calls',0)+len(event.get('refinements',[])))
                    refinement_solve_calls+=event.get('refinement_solve_calls',len(event.get('refinements',[])))
                    for key in WORK_COUNTERS:work_counts[key]+=event.get(key,0)
                    expected=np.asarray(t['unlabeled'])[event['returned_prefix_local']][:CFG['depth']-len(t['labeled'])].tolist()
                    assert expected==nominated
                else:
                    assert len(certified.EVENTS)==seen
                    assert t['unlabeled'][:CFG['depth']-len(t['labeled'])]==nominated
                row={k:t[k] for k in ['iteration','labeled','direct','indirect','unlabeled','selected','bridges','residual']}
                if binding:
                    assert t['admission_policy']==binding['admission_policy'] and t['admission_version']==binding['admission_version']
                    row.update({k:t[k] for k in ['nominated','support_camera_counts','admission_policy','admission_version']})
                row['returned_scores']=None if t['scores'] is None else [certified.packed(v) for v in t['scores']]
                row['event']=event
                if t['iteration']==0:first_state=row
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
            prefix,diag=ENGINE.rank([query],nq=CFG['depth'],input_similarity=similarity,numerical_tail=False,trace=capture)
        assert diag['status']=='complete_diffusion' and diag['numerical_tail_count']==0 and len(prefix)==CFG['depth']
        assert diag['max_relative_residual']<=1e-8
        diag['linear_systems']=linear_systems
        diag.update(work_counts)
        diag['mm_iterations']=mm_iterations
        diag['linear_solve_calls']=linear_solve_calls
        diag['refinement_solve_calls']=refinement_solve_calls
        rank=complete_gallery(prefix,similarity,QN)
        pp=dest/f'{query:04d}.prefix.npy';rp=dest/f'{query:04d}.rankings.npy'
        np.save(pp,prefix.astype(np.int32));np.save(rp,rank)
        row=dict(query=query,source=CFG['source'],signature=SIG,diagnostics=diag,seconds=time.monotonic()-start,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                 trace_sha256=sha(trace),prefix_sha256=sha(pp),rankings_sha256=sha(rp),pid=os.getpid())
        row.update(binding)
        if CFG.get('retain_initial_state'):
            assert first_state is not None
            first_path=dest/f'{query:04d}.initial.jsonl.gz'
            with gzip.open(first_path,'wt',encoding='utf-8') as stream:
                stream.write(json.dumps(first_state,separators=(',',':'),allow_nan=False)+'\n')
            row['initial_state_sha256']=sha(first_path)
        dump(dest/f'{query:04d}.json',row)
        if inline_readback():
            before=time.monotonic();accepted=accept_query(query)
            if 'error' in accepted:raise RuntimeError('Saved-score read-back failed: '+accepted['error'])
            row['compute_seconds']=row['seconds']
            row['readback_seconds']=time.monotonic()-before
            row['readback_sha256']=sha(OUT/'accepted'/f'{query:04d}.json')
            row['trace_retained']=query in CFG['retained_queries']
            # A failed read-back leaves all its raw evidence untouched.
            if not row['trace_retained']:trace.unlink()
            row['seconds']=time.monotonic()-start
            dump(dest/f'{query:04d}.json',row)
        return {k:row[k] for k in ['query','seconds','peak_rss_bytes','diagnostics','pid']}
    except BaseException:return dict(query=query,error=traceback.format_exc())

def accept_query(query):
    try:
        if INIT_ERROR:raise RuntimeError(INIT_ERROR)
        binding=admission_binding(CFG)
        dest=OUT/'queries';row=json.loads((dest/f'{query:04d}.json').read_text())
        assert row['query']==query and row['signature']==SIG and row['source']==CFG['source']
        assert row.get('admission_policy','original')==CFG.get('admission_policy','original')
        for key,value in binding.items():assert row[key]==value
        for suffix,key in [('trace.jsonl.gz','trace_sha256'),('prefix.npy','prefix_sha256'),('rankings.npy','rankings_sha256')]:assert sha(dest/f'{query:04d}.{suffix}')==row[key]
        if CFG.get('retain_initial_state'):
            first=dest/f'{query:04d}.initial.jsonl.gz'
            assert sha(first)==row['initial_state_sha256']
            assert list(read_rows(first))==[next(read_rows(dest/f'{query:04d}.trace.jsonl.gz'))]
        result=verify_query(GRAPH,FULL,X@X[query],query,np.load(dest/f'{query:04d}.prefix.npy'),np.load(dest/f'{query:04d}.rankings.npy'),read_rows(dest/f'{query:04d}.trace.jsonl.gz'),CFG['source'],QN,CFG['depth'],kernel=READBACK_KERNEL,full_sums=READBACK_SUMS,alpha=CFG['alpha'],beta=CFG['beta'],admission_policy=CFG.get('admission_policy','original'),camera_ids=ADMISSION_CAMERAS,admission_version=CFG.get('admission_version'),f1_nu=CFG.get('f1_nu',1.5),f1_updates=CFG.get('f1_updates',1))
        for k in ['solver_calls','iterations','bridge_events','bridge_edges','max_unlabeled']:assert result[k]==row['diagnostics'][k]
        assert result['max_reported_residual']==row['diagnostics']['max_relative_residual']
        if 'linear_systems' in row['diagnostics']:
            assert result['linear_systems']==row['diagnostics']['linear_systems']
        for counter in ('mm_iterations','linear_solve_calls','refinement_solve_calls')+WORK_COUNTERS:
            if counter in row['diagnostics']:assert result[counter]==row['diagnostics'][counter]
        if CFG.get('retain_initial_state'):result['initial_state_sha256']=row['initial_state_sha256']
        result.update(query=query,signature=SIG,trace_sha256=row['trace_sha256'],**binding);dump(OUT/'accepted'/f'{query:04d}.json',result);return result
    except BaseException:return dict(query=query,error=traceback.format_exc())

def aggregate_inline_readback():
    """Bind every earlier full read-back to the final complete rankings."""
    start=time.monotonic();complete=json.loads((OUT/'COMPUTE_COMPLETE.json').read_text())
    assert sha(OUT/'rankings.npy')==complete['rankings_sha256']
    ranks=np.load(OUT/'rankings.npy',mmap_mode='r')
    assert ranks.shape==(QN,GN)
    assert len(list((OUT/'accepted').glob('*.json')))==QN
    assert len(list((OUT/'queries').glob('*.trace.jsonl.gz')))==len(CFG['retained_queries'])
    for query in range(QN):
        dest=OUT/'queries';row=json.loads((dest/f'{query:04d}.json').read_text())
        accepted=OUT/'accepted'/f'{query:04d}.json'
        assert sha(accepted)==row['readback_sha256']
        result=json.loads(accepted.read_text())
        assert result['query']==query and result['signature']==complete['signature']==row['signature']
        assert result['trace_sha256']==row['trace_sha256']
        if CFG.get('retain_initial_state'):
            assert result['initial_state_sha256']==row['initial_state_sha256']==sha(dest/f'{query:04d}.initial.jsonl.gz')
        assert sha(dest/f'{query:04d}.prefix.npy')==row['prefix_sha256']
        packed=io.BytesIO();np.save(packed,ranks[query])
        assert hashlib.sha256(packed.getvalue()).hexdigest()==row['rankings_sha256']
        assert row['trace_retained']==(query in CFG['retained_queries'])
        trace=dest/f'{query:04d}.trace.jsonl.gz'
        if row['trace_retained']:assert sha(trace)==row['trace_sha256']
        else:assert not trace.exists()
    # Each row is now checked against the combined array; remove only duplicates.
    for query in range(QN):(OUT/'queries'/f'{query:04d}.rankings.npy').unlink(missing_ok=True)
    return time.monotonic()-start

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
    qp,qc,gp,gc=[np.asarray(data[k]).ravel() for k in ['query_label','query_cam','gallery_label','gallery_cam']];aps=[];r1=[];inps=[]
    for q,rank in enumerate(ranks):
        order=rank-QN;keep=(gp[order]!=-1)&~((gp[order]==qp[q])&(gc[order]==qc[q]));hits=gp[order[keep]]==qp[q]
        positives=np.count_nonzero((gp==qp[q])&(gc!=qc[q])&(gp!=-1));assert hits.sum()==positives
        if positives:
            positions=np.flatnonzero(hits)+1
            aps.append(float(np.mean(np.arange(1,positives+1)/positions)))
            r1.append(float(hits[0]));inps.append(float(positives/positions[-1]))
    assert abs(np.mean(aps)-metric['mAP'])<1e-12 and np.mean(r1)==metric['rank-1']
    assert abs(np.mean(inps)-metric['mINP'])<1e-12
    np.save(OUT/'rankings.npy',ranks);dump(OUT/'metrics.json',metric)
    return dict(mAP=float(np.mean(aps)),rank1=float(np.mean(r1)),mINP=float(np.mean(inps)),valid_queries=len(aps))

def main(stage,config_path):
    configure(config_path);sources()
    binding=admission_binding(CFG);fit_binding=fitting_binding(CFG)
    if stage=='acceptance':check_admission_manifest(CFG,json.loads((OUT/'run-manifest.json').read_text()))
    assert CFG['source'] in ('original','continuous','boundary_odds','two_step_fitting','stationary_measure')+f1.SOURCES+research_config.SOURCES
    assert np.isfinite([CFG['alpha'],CFG['beta']]).all() and min(CFG['alpha'],CFG['beta'])>0
    assert (CFG['step'],CFG['depth'],WORKERS)==(5,500,16)
    assert (QN,GN) in ((3368,19732),(2228,17661)), 'Expected complete Market1501 or DukeMTMC-reID population'
    assert CFG.get('relation_model','original') in ('original','csw')
    if CFG.get('relation_model')=='csw':
        assert (CFG['rho'],CFG['support_lambda'],CFG['support_mode'])==(.5,.25,'ordinary')
        assert CFG['support_file'] in CFG['input_sha256']
    assert CFG.get('trace_retention','all') in ('all','sampled_after_readback')
    if inline_readback():assert CFG['retained_queries']==np.linspace(0,QN-1,min(32,QN),dtype=int).tolist()
    if stage=='compute':
        assert not OUT.exists(),'New run only; no automatic overwrite or retry'
        initial=resources(reserve=True)
        shared=sum((Path(CFG['cache'])/n).stat().st_size//4096*4096 for n in ['affinity.npy','features.npy'])
        static_total=initial['memory_current']+shared+WORKERS*(WORKER_AS-shared)+3*2**30
        assert static_total<=.7*initial['memory_max'],('16-worker address-space budget',static_total)
        OUT.mkdir();(OUT/'queries').mkdir()
        if inline_readback():(OUT/'accepted').mkdir()
        scientific={k:CFG[k] for k in ['source','alpha','beta','step','depth','k','gamma','queries','gallery','feature_sha256','graph_sha256']}
        scientific.update(binding)
        scientific.update(fit_binding)
        if CFG.get('relation_model')=='csw':
            scientific.update({k:CFG[k] for k in ('relation_model','rho','support_lambda','support_mode')})
            scientific['support_sha256']=CFG['input_sha256'][CFG['support_file']]
        signature=hashlib.sha256(json.dumps(scientific,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        dump(OUT/'run-manifest.json',dict(signature=signature,config=CFG,time=now(),resources=initial,static_total_bytes=static_total,shared_mapping_bytes=shared,solver_id=solver_id(CFG['source'])))
        seconds=pool_stage(stage)
        assert len(list((OUT/'queries').glob('*.json')))==QN
        metric=evaluate();sources()
        dump(OUT/'COMPUTE_COMPLETE.json',dict(status='computed_and_independently_evaluated',signature=signature,query_count=QN,gallery_count=GN,rankings_sha256=sha(OUT/'rankings.npy'),independent=metric,seconds=seconds,git_commit=CFG['commit'],resources=resources(),inline_full_score_readback=inline_readback()))
    elif stage=='acceptance':
        assert (OUT/'compute.exit').read_text().strip()=='0'
        if inline_readback():seconds=aggregate_inline_readback()
        else:
            (OUT/'accepted').mkdir();seconds=pool_stage(stage)
        sources()
        rows=[json.loads(p.read_text()) for p in (OUT/'accepted').glob('*.json')];assert len(rows)==QN
        manifest=json.loads((OUT/'run-manifest.json').read_text())
        dump(OUT/'FULL_ACCEPTANCE.json',dict(status='accepted_full_population',signature=manifest['signature'],time=now(),queries=QN,solver_calls=sum(r['solver_calls'] for r in rows),linear_systems=sum(r.get('linear_systems',r['solver_calls']) for r in rows),mm_iterations=sum(r.get('mm_iterations',0) for r in rows),linear_solve_calls=sum(r.get('linear_solve_calls',0) for r in rows),refinement_solve_calls=sum(r.get('refinement_solve_calls',0) for r in rows),initial_states_retained=bool(CFG.get('retain_initial_state')),max_recomputed_residual=max(r['max_recomputed_residual'] for r in rows),rankings_sha256=sha(OUT/'rankings.npy'),actual_saved_scores_recomputed=True,all_frontiers_bridges_rankings_rebuilt=True,certificate_implementation_shared=True,seconds=seconds,git_commit=CFG['commit'],resources=resources(),trace_retention=CFG.get('trace_retention','all'),retained_queries=CFG.get('retained_queries'),all_stage_traces_retained=not inline_readback(),readback_performed_inline=inline_readback(),**{key:sum(r.get(key,0) for r in rows) for key in WORK_COUNTERS}))
    else:raise ValueError(stage)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['compute','acceptance']);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    lock=args.config.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main(args.stage,args.config)
    except BaseException:
        if 'OUT' in globals() and OUT.exists() and not (OUT/'FAILED.json').exists():dump(OUT/'FAILED.json',dict(stage=args.stage,time=now(),error=traceback.format_exc()))
        raise
