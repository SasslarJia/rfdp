"""Fixed theta0 then theta.5; one process, full original outer loop, no AP."""
from runtime import *
import importlib.util,types,time,gzip
import numpy as np
from scipy import sparse
sys.path.insert(0,str(HERE.parent/'runner-interface-v1'))
import internal_adapter as adapter

def load_model(refs,theta):
    repo=Path(refs['repo']);source=repo/'RFDP/MODEL/Paper_RFDP'
    sys.path[:0]=[str(ROOT),str(repo/'RFDP/experiments/reliability')]
    import high_precision_recovery as recovery
    import accurate_hybrid_ties as original
    assert original.SOLVER_ID=='original-formula-certified-exact-ties-v3'
    name='internal_qualification_'+str(theta).replace('.','p')
    pkg=types.ModuleType(name);pkg.__path__=[str(source)];sys.modules[name]=pkg;mods=[]
    for leaf in ['PaperFullIterative','PaperPrecision']:
        spec=importlib.util.spec_from_file_location(name+'.'+leaf,source/(leaf+'.py'))
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        setattr(pkg,leaf,mod);mods.append(mod)
    mod,precision=mods;precision.install()
    recovery.install(mod,events=precision.EVENTS,evidence_dir=OUT/'numerical-evidence')
    original.install(mod,precision.EVENTS);events=[];adapter.install(mod,events,theta)
    return mod,precision,events

def main():
    # Fail before creating attempt artifacts if prerequisites/resources are absent.
    pre=prerequisite();initial=resources(admission=True);refs=check_sources()
    assert refs['queries']==QUERIES and refs['thetas']==[0.,.5]
    OUT.mkdir(exist_ok=False);rc=1;stop,thread=start_guard('compute');started=time.monotonic()
    try:
        resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
        cache=Path(refs['cache']);graph=sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr()
        graph.eliminate_zeros();graph.sort_indices()
        X=np.load(cache/'features.npy',mmap_mode='r');full=np.load(cache/'affinity.npy',mmap_mode='r')
        assert X.shape[0]==23100 and full.shape==(23100,23100) and (graph-graph.T).nnz==0
        definition=dict(queries=QUERIES,thetas=[0.,.5],alpha=5,beta=25,step=5,depth=500,
                        source_lock_sha256=sha(HERE/'source-lock.json'),references_sha256=sha(HERE/'references.json'),
                        labels_loaded=False,cameras_loaded=False,AP_evaluated=False,prerequisite=pre)
        signature=canonical(definition);dump(OUT/'definition.json',definition)
        dump(OUT/'STARTED.json',dict(time=now(),pid=os.getpid(),signature=signature,resources=initial))
        records=[]
        # Observe existing explicit sparse-LU objects without replacing their solve.
        # Candidate scipy.spsolve factors are opaque; their memory is covered by RSS.
        lu_original=adapter.sc.splu
        lu_stats={}
        def observed_lu(*args,**kwargs):
            obj=lu_original(*args,**kwargs)
            lu_stats['calls']=lu_stats.get('calls',0)+1
            lu_stats['max_L_plus_U_nnz']=max(lu_stats.get('max_L_plus_U_nnz',0),obj.L.nnz+obj.U.nnz)
            return obj
        adapter.sc.splu=observed_lu
        for arm,theta in ARMS:
            mod,precision,events=load_model(refs,theta)
            engine=mod.FullIHyRDP(graph,full,alpha=5,beta=25,step=5)
            dest=OUT/arm;dest.mkdir()
            for query in QUERIES:
                resources();start=time.monotonic();events.clear();precision.EVENTS.clear();seen=0;lu_stats.clear()
                similarity=X@X[query];tracefile=dest/(str(query)+'.trace.jsonl.gz')
                with gzip.open(tracefile,'wt',encoding='utf-8',compresslevel=6) as f:
                    def capture(t):
                        nonlocal seen
                        ev=None
                        if t['scores'] is not None:
                            assert len(events)==seen+1;ev=events[seen];seen+=1
                            assert np.asarray(t['unlabeled'])[ev['returned_prefix_local']][:len(t['selected'])].tolist()==t['selected']
                        else:assert len(events)==seen
                        row={k:t[k] for k in ['iteration','labeled','direct','indirect','unlabeled','selected','bridges','residual']}
                        row['returned_scores']=None if t['scores'] is None else [adapter.packed(v) for v in t['scores']]
                        row['event']=ev
                        f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
                    prefix,diag=engine.rank([query],nq=500,input_similarity=similarity,numerical_tail=False,trace=capture)
                assert diag['status']=='complete_diffusion' and diag['numerical_tail_count']==0
                assert len(prefix)==500 and prefix[0]==query and len(set(prefix))==500
                used=np.zeros(23100,dtype=bool);used[prefix]=True
                rest=np.flatnonzero(~used & (np.arange(23100)>=3368))
                rank=np.r_[prefix[prefix>=3368],rest[np.lexsort((rest,-similarity[rest]))]].astype(np.int32)
                pp=dest/(str(query)+'.prefix.npy');rp=dest/(str(query)+'.rankings.npy')
                np.save(pp,prefix.astype(np.int32));np.save(rp,rank)
                row=dict(arm=arm,theta=theta,query=query,signature=signature,diagnostics=diag,
                         seconds=time.monotonic()-start,resources=resources(),
                         trace_sha256=sha(tracefile),prefix_sha256=sha(pp),rankings_sha256=sha(rp),
                         explicit_refinement_LU=dict(lu_stats),candidate_spsolve_fill_not_observable=True)
                if theta==0:
                    expected=refs['neutral_references'][str(query)]
                    assert all(row[k]==expected[k] for k in ['prefix_sha256','rankings_sha256'])
                dump(dest/(str(query)+'.json'),row);records.append(row)
                dump(OUT/'progress.json',dict(time=now(),completed=len(records),total=18,arm=arm,query=query))
                print(json.dumps(dict(arm=arm,query=query,seconds=row['seconds'])),flush=True)
        check_sources()
        dump(OUT/'COMPUTE_COMPLETE.json',dict(signature=signature,records=records,seconds=time.monotonic()-started,resources=resources()))
        rc=0
    except BaseException:
        dump(OUT/'FAILED.json',dict(time=now(),stage='compute',error=traceback.format_exc()));raise
    finally:
        stop.set();thread.join(timeout=11);(OUT/'compute.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
