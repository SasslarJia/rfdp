"""One frozen configuration; every query and every gallery item is evaluated."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,fcntl,hashlib,json,multiprocessing as mp,sys,time,traceback
from pathlib import Path
import numpy as np
import mpmath
from high_precision_recovery import SOLVER_ID
from scipy import sparse


def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def work(query):
    try:
        folder=OUT/ARGS.method;rec=folder/f'{query:04d}.json';rp=folder/f'{query:04d}.rankings.npy';pp=folder/f'{query:04d}.prefix.npy'
        if rec.exists():
            old=json.loads(rec.read_text())
            if old['signature']!=SIG:raise ValueError('Checkpoint signature changed')
            if rp.exists() and pp.exists() and sha(rp)==old['rankings_sha256'] and sha(pp)==old['prefix_sha256']:
                validate_ranking(np.load(rp),QN,GN)
                return {'query':query,'resumed':True}
        started=time.monotonic();PaperPrecision.EVENTS.clear();similarity=X@X[query];trace=[]
        def capture(r):
            t={'iteration':r['iteration'],'selected':r['selected'],'residual':r['residual'],'size':len(r['unlabeled']),'labeled_count':len(r['labeled']),'direct_count':len(r['direct']),'indirect_count':len(r['indirect']),'bridges':r['bridges']}
            # Complete first three rounds of a fixed predeclared query set; no outcome-based selection.
            if query<10 and r['iteration']<3:
                t.update({k:r[k] for k in ['labeled','direct','indirect','unlabeled']})
                t['scores']=None if r.get('scores') is None else [float(x) for x in r['scores']]
            for event in PaperPrecision.EVENTS:
                if event.get('kind')=='same-float64-equation-high-precision-recovery' and 'query' not in event:
                    event.update(query=query,iteration=int(r['iteration']),unlabeled=list(r['unlabeled']),selected=list(r['selected']))
            trace.append(t)
        prefix,diag=ENGINE.rank([query],nq=ARGS.depth,input_similarity=similarity,numerical_tail=False,trace=capture)
        if len(prefix)!=ARGS.depth or prefix[0]!=query or len(set(prefix))!=ARGS.depth or diag['status']!='complete_diffusion' or diag['numerical_tail_count']!=0 or not np.isfinite(diag['max_relative_residual']) or diag['max_relative_residual']>1e-8:
            raise RuntimeError('Incomplete/numerically invalid diffusion')
        rank=complete_gallery(prefix,similarity,QN);save(rp,rank);save(pp,prefix.astype(np.int32))
        dump(rec,{'signature':SIG,'query':query,'mode':ARGS.method,'rankings_sha256':sha(rp),'prefix_sha256':sha(pp),'diagnostics':diag,'trace':trace,'refinement_events':list(PaperPrecision.EVENTS),'seconds':time.monotonic()-started})
        return {'query':query,'seconds':time.monotonic()-started}
    except Exception:return {'query':query,'error':traceback.format_exc()}


def main():
    global ARGS,OUT,SIG,QN,GN,X,ENGINE,PaperPrecision,sha,dump,save,validate_ranking,complete_gallery
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','mat-path','cache-dir','output-dir','protocol']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--rho',type=float,default=.5);p.add_argument('--lam',type=float,default=0.);p.add_argument('--support-file',type=Path);p.add_argument('--support-mode',choices=['ordinary','cb']);p.add_argument('--depth',type=int,default=500);p.add_argument('--workers',type=int,default=8);p.add_argument('--expected-queries',type=int,required=True);p.add_argument('--expected-gallery',type=int,required=True)
    ARGS=p.parse_args()
    if not 0<ARGS.rho<=1 or not 0<=ARGS.lam<1 or ARGS.workers<1 or '/'in ARGS.method:raise ValueError('Invalid candidate settings')
    sys.path.insert(0,str(ARGS.repo/'RFDP/experiments/full_reid'));sys.path.insert(0,str(ARGS.repo/'RFDP'))
    from common import sha,dump,save,load_features,validate_ranking,complete_gallery
    from MODEL.Paper_RFDP import PaperPrecision
    from MODEL.Paper_RFDP.PaperCamera import CameraIHyRDP
    from EVAL.ReIDEval import evaluate_reid_rankings
    D,QN,GN=load_features(ARGS.mat_path)
    assert (QN,GN)==(ARGS.expected_queries,ARGS.expected_gallery),'Unexpected evaluation population'
    if not 1<=ARGS.depth<=QN+GN:raise ValueError('Invalid depth')
    OUT=ARGS.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sourcepaths=list((ARGS.repo/'RFDP/MODEL').rglob('*.py'))+list((ARGS.repo/'RFDP/EVAL').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    sourcepaths+=[ARGS.repo/'RFDP/experiments/full_reid/common.py',ARGS.repo/'RFDP/experiments/full_reid/build_feature_cache.py',ARGS.repo/'RFDP/Test_RFDP/reid_paper_experiment.py']
    config={'query_count':QN,'gallery_count':GN,'depth':ARGS.depth,'rho':ARGS.rho,'lambda':ARGS.lam,'method':ARGS.method,'support_mode':ARGS.support_mode,'alpha':5,'beta':25,'step':5,'k':20,'gamma':.4,'feature_sha256':sha(ARGS.mat_path),'protocol_sha256':sha(ARGS.protocol),'sources':{str(x.resolve()):sha(x)for x in sorted(set(sourcepaths))},'support_sha256':None if ARGS.support_file is None else sha(ARGS.support_file),'workers':ARGS.workers,'numpy':np.__version__,'mpmath':mpmath.__version__,'numerical_solver':SOLVER_ID,'protocol_path':str(ARGS.protocol.resolve()),'support_file':None if ARGS.support_file is None else str(ARGS.support_file.resolve()),'cache_manifest_path':str((ARGS.cache_dir/'manifest.json').resolve()),'inputs':{str((ARGS.cache_dir/name).resolve()):sha(ARGS.cache_dir/name)for name in ['manifest.json','features.npy','affinity.npy','graph.npz']}}
    SIG=canonical(config);manifest=OUT/'run-manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())['signature']==SIG,'Existing output contract mismatch'
    else:dump(manifest,{'signature':SIG,'signature_scheme':'sha256-json-sort-keys-compact','config':config})
    cached=json.loads((ARGS.cache_dir/'manifest.json').read_text());assert cached['feature_sha256']==config['feature_sha256']
    assert cached['builder_sha256']==sha(ARGS.repo/'RFDP/experiments/full_reid/build_feature_cache.py')
    for rel,h in cached['sources'].items():assert sha(ARGS.repo/rel)==h,'Cache source mismatch'
    for name in ['features.npy','affinity.npy','graph.npz']:assert sha(ARGS.cache_dir/name)==cached['files'][name],'Cache hash mismatch'
    X=np.load(ARGS.cache_dir/'features.npy',mmap_mode='r');assert X.shape[0]==QN+GN
    t=time.monotonic()
    def metrics(r):return evaluate_reid_rankings(r,range(QN),*[D[k]for k in ['query_label','query_cam','gallery_label','gallery_cam']],QN,max_rank=min(500,GN),return_per_query=True)
    if ARGS.method=='original':
        ranks=np.empty((QN,GN),np.int32)
        for q in range(QN):ranks[q]=np.lexsort((np.arange(QN,QN+GN),-(X[QN:]@X[q]))).astype(np.int32)+QN
    else:
        PaperPrecision.install()
        from MODEL.Paper_RFDP import PaperFullIterative
        from high_precision_recovery import install as install_mp
        install_mp(PaperFullIterative,events=PaperPrecision.EVENTS,evidence_dir=OUT/'numerical-evidence')
        full=np.load(ARGS.cache_dir/'affinity.npy',mmap_mode='r');graph=sparse.load_npz(ARGS.cache_dir/'graph.npz');cams=np.r_[D['query_cam'].ravel(),D['gallery_cam'].ravel()]
        if ARGS.lam==0:
            ENGINE=CameraIHyRDP(graph,full,cams,ARGS.rho)
        else:
            if ARGS.support_file is None:raise ValueError('Positive lambda needs support')
            from reliability_model import ReliabilityIHyRDP,load_support,DEFINITION_VERSION
            support,meta=load_support(ARGS.support_file,expected={'mode':ARGS.support_mode,'k':20,'definition_version':DEFINITION_VERSION})
            assert meta['mode']==ARGS.support_mode
            # The metadata builder binds full matrix and cameras; recheck before use.
            ENGINE=ReliabilityIHyRDP(graph,full,cams,ARGS.rho,support=support,lam=ARGS.lam,support_metadata=meta)
        (OUT/ARGS.method).mkdir(exist_ok=True)
        first=work(0)
        if 'error'in first:raise RuntimeError(first['error'])
        pool=mp.get_context('fork').Pool(ARGS.workers)if ARGS.workers>1 else None
        try:
            it=pool.imap_unordered(work,range(1,QN),chunksize=1)if pool else map(work,range(1,QN))
            for count,result in enumerate(it,2):
                if 'error'in result:raise RuntimeError(result['error'])
                if count%25==0 or count==QN:
                    progress={'status':'running','method':ARGS.method,'completed':count,'total':QN,'seconds':time.monotonic()-t}
                    dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        finally:
            if pool:pool.terminate();pool.join()
        ranks=np.stack([np.load(OUT/ARGS.method/f'{q:04d}.rankings.npy')for q in range(QN)])
    save(OUT/f'{ARGS.method}.rankings.npy',ranks);dump(OUT/f'{ARGS.method}.json',metrics(ranks))
    dump(OUT/'COMPUTE_COMPLETE.json',{'status':'computed','signature':SIG,'query_count':QN,'gallery_count':GN,'seconds':time.monotonic()-t,'rankings_sha256':sha(OUT/f'{ARGS.method}.rankings.npy')})
    print('COMPUTED',ARGS.method,flush=True)

if __name__=='__main__':main()
