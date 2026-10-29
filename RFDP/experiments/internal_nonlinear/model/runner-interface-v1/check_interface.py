import json,time,datetime,traceback,importlib.util,sys,types,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy import sparse
from internal_adapter import install,packed,score_hash,solve as inner_solve,InternalFailure
from scalable_certificate import scalar
from certificate import certify as small_cert,independent_nonlinear_oracle
P=Path(__file__).resolve().parent;E=P.parents[2]
R=E.parent/'reliability-market-20261015/repo/RFDP/MODEL/Paper_RFDP'
sys.path.insert(0,str(E));sys.path.insert(0,str(E.parent/'reliability-market-20261015/v2/source'))
import accurate_hybrid_ties as arithmetic
import high_precision_recovery as recovery
refs={R/'PaperFullIterative.py':'14e8903d75a925c53c07dbbfd0b5888c20ab9378fbbb98bb9c17d83d5f64c965',
      R/'PaperPrecision.py':'7cd9c025f0d0314229334b32a4f1b86b2d14b6e25c902b3826ed1cc4c313c99d',
      Path(arithmetic.__file__):'591328e10fb5cf2b4aaf871cb1779579957da17856dff8e0022b5525ed89d3e0',
      Path(recovery.__file__):'ebcd958e1eb62f4ef11fbac2fbad8e064c82bde2da481fb9933d701fc7dc8fa6'}
for p,sha in refs.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha

def load(name,theta):
    pkg=types.ModuleType(name);pkg.__path__=[str(R)];sys.modules[name]=pkg
    modules=[]
    for leaf in ['PaperFullIterative','PaperPrecision']:
        spec=importlib.util.spec_from_file_location(name+'.'+leaf,R/(leaf+'.py'))
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        setattr(pkg,leaf,m);modules.append(m)
    mod,precision=modules
    precision.install();recovery.install(mod,events=precision.EVENTS);arithmetic.install(mod,precision.EVENTS)
    events=[]
    if theta is not None:install(mod,events,theta)
    return mod,events,precision.EVENTS

def graph(n,rng):
    w=np.eye(n)
    for i in range(n-1):w[i,i+1]=w[i+1,i]=rng.uniform(.2,1.7)
    for i in range(n-3):w[i,i+3]=w[i+3,i]=rng.uniform(.1,.8)
    return w

rng=np.random.default_rng(26062819)
fixtures=[('connected',sparse.csr_matrix(graph(37,rng)),13,31),
          ('bridge_and_small_frontier',sparse.block_diag([graph(4,rng),graph(13,rng),graph(18,rng)],format='csr'),2,33),
          ('exact_ties',sparse.csr_matrix(np.ones((19,19))),9,18)]
near=np.ones((19,19));near[0,7]=near[7,0]=np.nextafter(1.,2.)
fixtures.append(('one_ulp_near_ties',sparse.csr_matrix(near),7,18))
rows=[];raw=[];started=time.monotonic()
try:
 for case,(name,w,seed,nq) in enumerate(fixtures):
    w.eliminate_zeros();w.sort_indices()
    x=rng.uniform(.1,1.,(w.shape[0],9));full=x@x.T;sim=full[:,seed];arms=[]
    for arm,theta in [('original',None),('neutral',0.),('candidate',.5)]:
        mod,events,original_events=load('fixture_'+str(case)+'_'+arm,theta)
        before=w.data.tobytes();trace=[];seen=0;oracle_checked=False
        boundary=np.asarray(w[[seed]].sum(axis=0)).ravel()
        def capture(t):
            global seen,oracle_checked
            uu=np.asarray(t['unlabeled']);selected=t['selected']
            if t['scores'] is not None:
                ev=events[seen:] if theta is not None else []
                if theta is not None:
                    assert len(ev)==1
                    pred=uu[ev[0]['returned_prefix_local']].tolist()
                    assert pred[:len(selected)]==selected
                    assert score_hash(t['scores'])==ev[0]['returned_scores_sha256']
                if arm=='candidate' and not oracle_checked:
                    with mp.workdps(80):
                        wu=w[uu][:,uu]; ee=boundary[uu].copy()
                        for node,anchor,value in t['bridges']:
                            assert anchor in t['labeled']
                            ee[np.flatnonzero(uu==node)[0]]+=value
                        h=[sum((scalar(v) for v in wu.data[wu.indptr[i]:wu.indptr[i+1]]),mp.mpf(0)) for i in range(len(uu))]
                        zz=[s/mp.sqrt(v) for s,v in zip(t['scores'],h)]
                        cc=small_cert(wu,ee,zz)
                        oracle=independent_nonlinear_oracle(cc,zz)
                        assert all(lo<=v<=hi for (lo,hi),v in zip(cc['intervals'],oracle))
                        # No approximate tie handling: only exact returned equal scores may use ID.
                        expected=sorted(range(len(uu)),key=lambda i:(oracle[i],int(uu[i])))
                        # All comparisons for actually selected rows get independent certificate coverage.
                        for idx in ev[0]['returned_prefix_local']:
                            assert any(lo<=oracle[idx]<=hi for lo,hi in [cc['intervals'][idx]])
                    oracle_checked=True
                seen=len(events)
            else:assert len(events)==seen
            trace.append({k:t[k] for k in ['iteration','labeled','direct','indirect','unlabeled','selected','bridges','residual']})
            trace[-1]['returned_scores']=None if t['scores'] is None else [packed(v) for v in t['scores']]
            for node in selected:
                begin,end=w.indptr[node:node+2]
                boundary[w.indices[begin:end]]+=w.data[begin:end]
        engine=mod.FullIHyRDP(w,full,alpha=5,beta=25,step=5)
        prefix,diag=engine.rank([seed],nq=nq,input_similarity=sim,numerical_tail=False,trace=capture)
        assert diag['status']=='complete_diffusion' and diag['numerical_tail_count']==0
        assert len(prefix)==nq and len(set(prefix.tolist()))==nq and w.data.tobytes()==before
        assert len(events)==diag['solver_calls'] if theta is not None else True
        remaining=np.array([i for i in range(len(sim)) if i not in set(prefix.tolist())])
        rank=np.r_[prefix,remaining[np.lexsort((remaining,-sim[remaining]))]]
        assert len(set(rank.tolist()))==len(sim)
        arms.append(dict(arm=arm,prefix=prefix.tolist(),full_ranking=rank.tolist(),trace=trace,events=events,
                         diag={k:v for k,v in diag.items() if k!='elapsed_seconds'}))
    assert arms[0]['prefix']==arms[1]['prefix'] and arms[0]['full_ranking']==arms[1]['full_ranking']
    assert arms[0]['trace']==arms[1]['trace'] and arms[0]['diag']==arms[1]['diag']
    if name=='bridge_and_small_frontier':
        assert all(a['diag']['bridge_events']>0 and any(t['returned_scores'] is None for t in a['trace']) for a in arms)
    if name=='exact_ties':
        assert any(ev['exact_tie_groups'] for ev in arms[2]['events'])
    raw.append(dict(fixture=name,n=w.shape[0],seed=seed,nq=nq,arms=arms))
    rows.append(dict(fixture=name,n=w.shape[0],seed=seed,nq=nq,
         neutral_prefix_rank_trace_and_actual_scores_identical=True,
         candidate_all_actual_prefixes_match_certificate=True,
         candidate_solver_calls=arms[2]['diag']['solver_calls'],
         candidate_small_frontier_batches=sum(t['returned_scores'] is None for t in arms[2]['trace']),
         candidate_bridge_events=arms[2]['diag']['bridge_events'],
         independent_MP_first_solve_checked=oracle_checked,
         candidate_vs_original_prefix_changed=arms[0]['prefix']!=arms[2]['prefix']))
    print(json.dumps(rows[-1]),flush=True)
 # Input domain bound is checked before any matrix or solution allocation.
 try:inner_solve(sparse.eye(8193,format='csr'),np.ones(8193))
 except InternalFailure:cap=True
 else:raise AssertionError('Cap ignored')
 for p,sha in refs.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 (P/'synthetic-traces.json').write_text(json.dumps(raw,indent=2)+'\n')
 out=dict(time=datetime.datetime.now().astimezone().isoformat(),status='original_outer_synthetic_interface_accepted_not_real_query_qualified',
          fixtures=rows,local_domain_cap_rejected_without_truncation=cap,
          original_source_unchanged={str(p):sha for p,sha in refs.items()},
          traces_sha256=hashlib.sha256((P/'synthetic-traces.json').read_bytes()).hexdigest(),
          seconds=time.monotonic()-started,AP_evaluated=False,real_data_loaded=False,server_job_started=False)
 (P/'qualification.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out))
except Exception as exc:
 (P/('FAILED-'+datetime.datetime.now().strftime('%H%M%S')+'.json')).write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),'completed':rows},indent=2)+'\n')
 raise

