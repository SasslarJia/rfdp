"""Rebuild full trajectories from saved points, without running a solver.

Frontier, boundary, bridge and full-ranking reconstruction derives from the
previous verified reader. The new source uses its own equation certificate.
Producer and reader share certificate utilities; this is independent artifact
read-back, not two independent implementations of interval arithmetic.
"""
import json,gzip,hashlib
import numpy as np
from scipy.sparse.csgraph import connected_components
from .certified import unpack,certify_saved as certify_scores
from MODEL.Paper_RFDP.PaperFullIterative import ADMISSION_POLICIES,ADMISSION_VERSION

WORK_COUNTERS = ('additional_factorizations', 'reference_certifications', 'mp_mm_iterations')

def bridge_edges(graph,full,labels,active,sums,kernel=None):
    inside=np.flatnonzero(np.isin(labels,list(active)));choices=[]
    for comp in range(int(labels.max())+1):
        if comp in active:continue
        outside=np.flatnonzero(labels==comp);best=None
        for start in range(0,len(outside),128):
            ids=outside[start:start+128]
            raw=np.asarray(full[np.ix_(ids,inside)],dtype=np.float64) if kernel is None else kernel(ids,inside)
            block=raw/sums[ids,None]
            i,j=np.unravel_index(np.argmax(block),block.shape)
            candidate=(float(block[i,j]),int(ids[i]),int(inside[j]))
            if best is None or (-candidate[0],candidate[1:])<(-best[0],best[1:]):best=candidate
        if best is not None and best[0]>0:choices.append(best)
    choices.sort(key=lambda t:(-t[0],t[1],t[2]))
    return [[i,j,float(full[i,j]) if kernel is None else float(kernel([i],[j])[0,0])] for _,i,j in choices[:5]]

def verify_query(graph,full,similarity,query,prefix,rank,rows,source,qn,depth,check=lambda:None,*,kernel=None,full_sums=None,alpha=5.,beta=25.,admission_policy='original',camera_ids=None,admission_version=None,f1_nu=1.5,f1_updates=1,stage_certifier=None,score_decoder=None,promotion_schedule='fixed5'):
    from . import f1, research_config
    from experiments.reid_adaptation.schedule import promotion_step
    promotion_step(promotion_schedule,0)
    if promotion_schedule!='fixed5':assert admission_policy=='original' and source=='original'
    if source in research_config.SOURCES:
        assert admission_policy=='original' and kernel is None
    if source in f1.SOURCES:
        assert f1_updates==1 and admission_policy=='original' and kernel is None
    n=graph.shape[0];prefix=list(map(int,prefix));rank=list(map(int,rank))
    assert admission_policy in ADMISSION_POLICIES
    if admission_policy!='original':
        assert source=='original' and kernel is None
        assert admission_version==ADMISSION_VERSION
        camera_ids=np.asarray(camera_ids)
        assert camera_ids.shape==(n,) and np.issubdtype(camera_ids.dtype,np.integer)
    assert len(prefix)==depth and prefix[0]==query and len(set(prefix))==depth
    assert all(0<=i<n for i in prefix)
    nc,labels=connected_components(graph,directed=False);sizes=np.bincount(labels)
    sums=np.asarray(full.sum(axis=1,dtype=np.float64)) if full_sums is None else full_sums
    labeled=[query];mask=np.zeros(n,dtype=bool);mask[query]=True
    boundary=np.asarray(graph[[query]].sum(axis=0)).ravel();active={int(labels[query])}
    calls=batches=no_solve=bridges=edges=linear_systems=mm_iterations=linear_solve_calls=refinement_solve_calls=0;maximum=0.;reported=0.;max_size=0
    work_counts=dict.fromkeys(WORK_COUNTERS,0)
    for t in rows:
        check();assert len(labeled)<depth and t['iteration']==batches
        assert t['labeled']==labeled
        current_step=promotion_step(promotion_schedule,len(labeled)-1)
        if promotion_schedule!='fixed5' or 'promotion_schedule' in t or 'promotion_step' in t:
            assert t['promotion_schedule']==promotion_schedule and t['promotion_step']==current_step
        expected_bridges=[]
        if len(labeled)==int(sizes[list(active)].sum()) and len(labeled)<n:
            expected_bridges=bridge_edges(graph,full,labels,active,sums,kernel)
            assert expected_bridges
        assert t['bridges']==expected_bridges
        ew=boundary.copy()
        for i,j,v in expected_bridges:assert not mask[i] and mask[j];ew[i]+=v
        a=(ew>0)&~mask
        # Reachability from direct rows, independent of producer's sparse matvec.
        b=np.zeros(n,dtype=bool)
        for node in np.flatnonzero(a):b[graph.indices[graph.indptr[node]:graph.indptr[node+1]]]=True
        b&=~mask&~a;u=np.flatnonzero(a|b)
        # u was independently rebuilt from the complete graph. A node-count
        # cap is not an accuracy check; retain the actual graph population.
        assert 0<len(u)<=n-len(labeled)
        max_size=max(max_size,len(u))
        assert t['direct']==np.flatnonzero(a).tolist() and t['indirect']==np.flatnonzero(b).tolist()
        assert t['unlabeled']==u.tolist()
        residual=float(t['residual']);assert np.isfinite(residual) and 0<=residual<=1e-8
        reported=max(reported,residual)
        if len(u)<=current_step:
            assert t['returned_scores'] is None and t['event'] is None and residual==0
            nominated=u[:depth-len(labeled)].tolist();no_solve+=1
        else:
            # Hybrid costs remain strictly positive by default. Independent
            # relationship models may supply a signed decoder plus their own
            # equation/projection certificate; this does not relax HyRDP.
            if score_decoder is not None:
                assert stage_certifier is not None
                values=score_decoder(t['returned_scores'])
            else:
                values=unpack(t['returned_scores'])
            ev=t['event']
            digest=hashlib.sha256(json.dumps(t['returned_scores'],separators=(',',':')).encode()).hexdigest()
            assert ev['returned_scores_sha256']==digest and ev['size']==len(u) and ev['source']==source
            if stage_certifier is not None:
                c=stage_certifier(graph[u][:,u],ew[u],values,ev,
                    alpha=alpha,beta=beta,source=source,indices=u,
                    targets=tuple(labeled),bridges=tuple(expected_bridges))
            elif source in research_config.SOURCES:
                from .full import modern_solver
                c=modern_solver(source).certify_saved(graph[u][:,u],ew[u],values,ev,alpha=alpha,beta=beta,source=source)
            elif source in f1.SOURCES:
                c=f1.certify_saved(graph[u][:,u],ew[u],values,ev,alpha=alpha,beta=beta,source=source,nu=f1_nu)
            else:
                c=certify_scores(graph[u][:,u],ew[u],values,alpha=alpha,beta=beta,source=source)
            linear_systems+=c.get('linear_systems',1)
            mm_iterations+=c.get('mm_iterations',0)
            # Runtime counters are trace records, not reconstructed timings.
            linear_solve_calls+=ev.get('linear_solve_calls',ev.get('cg_calls',0)+ev.get('direct_calls',0)+len(ev.get('refinements',[])))
            refinement_solve_calls+=ev.get('refinement_solve_calls',len(ev.get('refinements',[])))
            for key in WORK_COUNTERS:
                value=ev.get(key,0);assert type(value) is int and value>=0
                work_counts[key]+=value
            nominated=u[c['prefix']][:min(current_step,depth-len(labeled))].tolist()
            assert c['prefix']==ev['returned_prefix_local']
            maximum=max(maximum,c['relative']);calls+=1
        selected=nominated
        if admission_policy!='original':
            assert t['admission_policy']==admission_policy and t['admission_version']==ADMISSION_VERSION
            assert t['nominated']==nominated
            # Independently read candidate rows against the old labeled mask;
            # do not reuse the producer's incremental bitsets or gate logic.
            counts=[]
            for node in nominated:
                begin,end=graph.indptr[node:node+2]
                neighbors=graph.indices[begin:end]
                supporting=neighbors[mask[neighbors] & (graph.data[begin:end]>0)]
                cameras=set(camera_ids[supporting].tolist())
                for outside,inside,value in expected_bridges:
                    if outside==node and value>0:cameras.add(int(camera_ids[inside]))
                counts.append(len(cameras))
            assert t['support_camera_counts']==counts
            admitted=[node for k,node in enumerate(nominated) if k==0 or counts[k]>=2]
            selected=admitted if admission_policy=='camera_gate' else nominated[:len(admitted)]
        else:
            assert t.get('admission_policy','original')=='original'
        assert selected==t['selected']==prefix[len(labeled):len(labeled)+len(selected)]
        mask[selected]=True;labeled.extend(selected)
        for node in selected:
            begin,end=graph.indptr[node:node+2]
            boundary[graph.indices[begin:end]]+=graph.data[begin:end]
        active.update(map(int,labels[selected]));batches+=1;bridges+=bool(expected_bridges);edges+=len(expected_bridges)
    assert labeled==prefix and batches==calls+no_solve
    used=set(prefix);expected=[i for i in prefix if i>=qn]
    expected+=sorted((i for i in range(qn,n) if i not in used),key=lambda i:(-float(similarity[i]),i))
    assert rank==expected and sorted(rank)==list(range(qn,n))
    return dict(solver_calls=calls,linear_systems=linear_systems,mm_iterations=mm_iterations,linear_solve_calls=linear_solve_calls,refinement_solve_calls=refinement_solve_calls,iterations=batches,no_solve_batches=no_solve,
                bridge_events=bridges,bridge_edges=edges,max_recomputed_residual=maximum,
                max_reported_residual=reported,max_unlabeled=max_size,**work_counts)

def read_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
