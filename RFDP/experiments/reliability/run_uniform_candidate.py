"""Local integration draft: full-population uniform off-diagonal mass control.

The engine receives identity support and effective_lambda. The original ordinary
or CB support is only the unlabeled source of the mass-matching contract.
Per-query computation, metric evaluation and MP evidence are unchanged v2 code.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import mpmath
from scipy import sparse
import run_candidate_v2 as core
from high_precision_recovery import SOLVER_ID


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_selection(path, digest, model, mode, source_lambda, rho):
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'Frozen selection SHA mismatch')
    selection = json.loads(Path(path).read_text())
    picked = selection['models'][model][mode]
    require(picked['mode'] == mode and picked['model'] == model, 'Selection model/mode mismatch')
    require(picked['rho'] == rho == .5, 'Uniform controls must pair source rho=.5')
    require(picked['lambda_'] == source_lambda, 'source_lambda differs from frozen selection')
    require(picked['split'] == 'val', 'Selection must originate in development split')
    return {key: picked[key] for key in ('model','mode','rho','lambda_','split','method','verified_sha256')}


def preserve_json(path, value, dump):
    if path.exists():
        require(json.loads(path.read_text()) == value, 'Existing control document changed: ' + str(path))
    else:
        dump(path, value)


def persist_state(path, engine):
    state = dict(data=engine.w.data, indices=engine.w.indices, indptr=engine.w.indptr,
                 shape=np.array(engine.w.shape,dtype=np.int64), full_sums=engine.full_sums)
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            require(set(saved.files)==set(state) and all(np.array_equal(saved[k],v) for k,v in state.items()),
                    'Existing actual engine state changed')
    else:
        with path.open('xb') as stream:
            np.savez_compressed(stream,**state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','mat-path','cache-dir','output-dir','protocol','support-file','selection-file','input-binding-file']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['uniform-mass-ordinary','uniform-mass-cb'],required=True)
    p.add_argument('--support-mode',choices=['ordinary','cb'],required=True)
    p.add_argument('--model',choices=['resnet50','osnet'],required=True)
    p.add_argument('--source-lambda',type=float,required=True)
    p.add_argument('--selection-sha256',required=True)
    p.add_argument('--input-binding-sha256',required=True)
    p.add_argument('--rho',type=float,default=.5)
    p.add_argument('--depth',type=int,default=500)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--expected-queries',type=int,required=True)
    p.add_argument('--expected-gallery',type=int,required=True)
    args=p.parse_args()
    require(args.method=='uniform-mass-'+args.support_mode,'Method/source support mismatch')
    require(np.isfinite(args.source_lambda) and 0<=args.source_lambda<1 and args.workers>0,'Invalid control settings')
    paired=checked_selection(args.selection_file,args.selection_sha256,args.model,args.support_mode,args.source_lambda,args.rho)
    sys.path.insert(0,str(args.repo/'RFDP/experiments/full_reid'))
    sys.path.insert(0,str(args.repo/'RFDP'))
    from common import sha,dump,save,load_features,validate_ranking,complete_gallery
    from MODEL.Paper_RFDP import PaperPrecision
    from EVAL.ReIDEval import evaluate_reid_rankings
    data,nq,ng=load_features(args.mat_path)
    require((nq,ng)==(args.expected_queries,args.expected_gallery),'Unexpected full population')
    require(1<=args.depth<=nq+ng,'Invalid depth')
    from input_binding import validate_input_binding
    input_binding,bound_files=validate_input_binding(args.input_binding_file,args.input_binding_sha256,
        model=args.model,mode=args.support_mode,source_lambda=args.source_lambda,rho=args.rho,
        selection_sha256=args.selection_sha256,mat=args.mat_path,cache=args.cache_dir,support=args.support_file,
        query_count=nq,gallery_count=ng,depth=args.depth)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=args.cache_dir.resolve();cached=json.loads((cache/'manifest.json').read_text())
    require(cached['feature_sha256']==sha(args.mat_path),'Cache feature SHA mismatch')
    require(cached['builder_sha256']==sha(args.repo/'RFDP/experiments/full_reid/build_feature_cache.py'),'Cache builder mismatch')
    for relative,digest in cached['sources'].items():
        require(sha(args.repo/relative)==digest,'Cache source mismatch')
    for name in ['features.npy','affinity.npy','graph.npz']:
        require(sha(cache/name)==cached['files'][name],'Cache file mismatch')
    x=np.load(cache/'features.npy',mmap_mode='r');require(x.shape[0]==nq+ng,'Cache sample count')
    full=np.load(cache/'affinity.npy',mmap_mode='r');graph=sparse.load_npz(cache/'graph.npz')
    cameras=np.r_[data['query_cam'].ravel(),data['gallery_cam'].ravel()]
    # Preserve production module identity and patch order before importing engine.
    PaperPrecision.install()
    from MODEL.Paper_RFDP import PaperFullIterative
    from high_precision_recovery import install as install_mp
    install_mp(PaperFullIterative,events=PaperPrecision.EVENTS,evidence_dir=out/'numerical-evidence')
    from reliability_model import load_support,DEFINITION_VERSION
    from uniform_control import UniformMassMatchedIHyRDP
    q,meta=load_support(args.support_file,expected={'mode':args.support_mode,'k':20,'definition_version':DEFINITION_VERSION})
    engine=UniformMassMatchedIHyRDP(graph,full,cameras,rho=args.rho,lam=args.source_lambda,
                                  source_support=q,source_support_metadata=meta)
    contract=engine.control_metadata
    contract_path=out/'control-contract.json';state_path=out/'control-engine-state.npz'
    preserve_json(contract_path,contract,dump);persist_state(state_path,engine)
    sources=list((args.repo/'RFDP/MODEL').rglob('*.py'))+list((args.repo/'RFDP/EVAL').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    sources += [args.repo/'RFDP/experiments/full_reid/common.py',args.repo/'RFDP/experiments/full_reid/build_feature_cache.py',args.repo/'RFDP/Test_RFDP/reid_paper_experiment.py']
    paths=[cache/name for name in ['manifest.json','features.npy','affinity.npy','graph.npz']]
    paths += [args.mat_path,args.protocol,args.support_file,args.selection_file,contract_path,state_path]
    paths += [Path(path)for path in bound_files]
    config=dict(query_count=nq,gallery_count=ng,depth=args.depth,rho=args.rho,
                source_lambda=args.source_lambda,effective_lambda=engine.lam,
                method=args.method,model=args.model,support_mode='identity-uniform-control',
                source_support_mode=args.support_mode,control_support='identity Q; q_ii=1, q_ij=0 for i!=j',
                alpha=5,beta=25,step=5,k=20,gamma=.4,numerical_tail=False,
                feature_sha256=sha(args.mat_path),protocol_sha256=sha(args.protocol),
                support_sha256=sha(args.support_file),selection_sha256=args.selection_sha256,
                paired_selection=paired,input_binding_file=str(args.input_binding_file.resolve()),
                input_binding_sha256=args.input_binding_sha256,input_scope=input_binding['scope'],workers=args.workers,numpy=np.__version__,mpmath=mpmath.__version__,
                numerical_solver=SOLVER_ID,repo_path=str(args.repo.resolve()),mat_path=str(args.mat_path.resolve()),
                protocol_path=str(args.protocol.resolve()),support_file=str(args.support_file.resolve()),
                selection_file=str(args.selection_file.resolve()),cache_dir=str(cache),cache_manifest_path=str(cache/'manifest.json'),
                control_contract_path=str(contract_path),control_state_path=str(state_path),
                control_contract_sha256=contract['contract_sha256'],control_contract_file_sha256=sha(contract_path),
                # Entire canonical contract is in the signature. A JSON string prevents
                # the unmodified v2 verifier mistaking its symbolic "inputs" for paths.
                control_contract_json=json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False),
                sources={str(path.resolve()):sha(path)for path in sorted(set(sources))},
                inputs={str(path.resolve()):sha(path)for path in paths})
    signature=core.canonical(config)
    preserve_json(out/'run-manifest.json',dict(signature=signature,signature_scheme='sha256-json-sort-keys-compact',config=config),dump)
    # No query may start until the complete contract and every input/source are signed.
    core.ARGS=args;core.OUT=out;core.SIG=signature;core.QN=nq;core.GN=ng;core.X=x;core.ENGINE=engine
    core.PaperPrecision=PaperPrecision;core.sha=sha;core.dump=dump;core.save=save
    core.validate_ranking=validate_ranking;core.complete_gallery=complete_gallery
    (out/args.method).mkdir(exist_ok=True);started=time.monotonic()
    first=core.work(0)
    require('error'not in first,first.get('error',''))
    pool=mp.get_context('fork').Pool(args.workers)if args.workers>1 else None
    try:
        it=pool.imap_unordered(core.work,range(1,nq),chunksize=1)if pool else map(core.work,range(1,nq))
        for count,result in enumerate(it,2):
            require('error'not in result,result.get('error',''))
            if count%25==0 or count==nq:
                progress=dict(status='running',method=args.method,completed=count,total=nq,seconds=time.monotonic()-started)
                dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    finally:
        if pool:pool.terminate();pool.join()
    ranks=np.stack([np.load(out/args.method/f'{q:04d}.rankings.npy')for q in range(nq)])
    metrics=evaluate_reid_rankings(ranks,range(nq),*[data[k]for k in ['query_label','query_cam','gallery_label','gallery_cam']],nq,max_rank=min(500,ng),return_per_query=True)
    save(out/f'{args.method}.rankings.npy',ranks);dump(out/f'{args.method}.json',metrics)
    dump(out/'COMPUTE_COMPLETE.json',dict(status='computed',signature=signature,query_count=nq,gallery_count=ng,
         seconds=time.monotonic()-started,rankings_sha256=sha(out/f'{args.method}.rankings.npy')))
    print('COMPUTED',args.method,flush=True)

if __name__=='__main__':main()
