"""Full saved-score read-back; reconstruct all frontiers, bridges and rankings."""
from runtime import *
import time
import numpy as np
from scipy import sparse
from verify_artifact import verify_query,read_rows

def require_complete(out):
    assert (out/'compute.exit').read_text().strip()=='0'
    assert (out/'COMPUTE_COMPLETE.json').is_file() and not (out/'FAILED.json').exists()
    assert not (out/'COMPLETE.json').exists()
def main():
    require_complete(OUT);pre=prerequisite();initial=resources(admission=True);refs=check_sources()
    rc=1;stop,thread=start_guard('acceptance');started=time.monotonic()
    try:
        resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
        definition=json.loads((OUT/'definition.json').read_text());comp=json.loads((OUT/'COMPUTE_COMPLETE.json').read_text())
        signature=canonical(definition)
        assert comp['signature']==signature and len(comp['records'])==18
        assert definition['source_lock_sha256']==sha(HERE/'source-lock.json')
        assert definition['references_sha256']==sha(HERE/'references.json')
        assert definition['queries']==QUERIES and definition['thetas']==[0.,.5]
        assert definition['prerequisite']==pre
        cache=Path(refs['cache']);graph=sparse.load_npz(cache/'graph.npz').astype(np.float64).tocsr()
        graph.eliminate_zeros();graph.sort_indices()
        full=np.load(cache/'affinity.npy',mmap_mode='r');X=np.load(cache/'features.npy',mmap_mode='r')
        accepted=[];hashes={}
        for arm,theta in ARMS:
            for query in QUERIES:
                dest=OUT/arm;row=json.loads((dest/(str(query)+'.json')).read_text())
                assert row['signature']==signature and row['theta']==theta and row['query']==query and row['arm']==arm
                index=[v for v in comp['records'] if v['arm']==arm and v['query']==query]
                assert len(index)==1 and index[0]==row
                for suffix,key in [('trace.jsonl.gz','trace_sha256'),('prefix.npy','prefix_sha256'),('rankings.npy','rankings_sha256')]:
                    p=dest/(str(query)+'.'+suffix);assert sha(p)==row[key];hashes[str(p.relative_to(OUT))]=row[key]
                prefix=np.load(dest/(str(query)+'.prefix.npy'),allow_pickle=False)
                rank=np.load(dest/(str(query)+'.rankings.npy'),allow_pickle=False)
                assert prefix.dtype==rank.dtype==np.int32
                result=verify_query(graph,full,X@X[query],query,prefix,rank,read_rows(dest/(str(query)+'.trace.jsonl.gz')),theta,3368,500,resources)
                diag=row['diagnostics']
                for k in ['solver_calls','iterations','bridge_events','bridge_edges','max_unlabeled']:assert result[k]==diag[k]
                assert result['max_reported_residual']==diag['max_relative_residual']
                assert diag['status']=='complete_diffusion' and diag['numerical_tail_count']==diag['tail_count']==0
                if theta==0:
                    assert all(row[k]==refs['neutral_references'][str(query)][k] for k in ['prefix_sha256','rankings_sha256'])
                accepted.append(dict(arm=arm,query=query,**result))
                dump(OUT/'acceptance-progress.json',dict(completed=len(accepted),total=18,last=accepted[-1],time=now()))
        check_sources()
        dump(OUT/'COMPLETE.json',dict(status='accepted_fixed_queries_not_accuracy',signature=signature,
             rows=accepted,hashes=hashes,time=now(),seconds=time.monotonic()-started,resources=resources(),
             initial_resources=initial,actual_scores_recomputed=True,all_frontiers_bridges_and_rankings_rebuilt=True,
             shared_mathematical_enclosure_utilities=True,AP_evaluated=False,full_run_queued=False))
        rc=0
    except BaseException:
        dump(OUT/'ACCEPTANCE_FAILED.json',dict(time=now(),error=traceback.format_exc()));raise
    finally:
        stop.set();thread.join(timeout=11);(OUT/'acceptance.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
