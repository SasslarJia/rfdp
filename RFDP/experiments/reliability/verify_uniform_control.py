"""Independently reconstruct the uniform contract, graph, sums and bridge evidence.

Run after the unchanged v2 standard and exact-numerical verifiers. This adds no
metric definition. It reads cameras, never identity labels or test metrics, for
support reconstruction. Every MP system is rebound to the reconstructed graph.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def independent_bridges(full,cameras,sums,components,active,rho,gamma):
    inside=np.flatnonzero(np.isin(components,list(active)));choices=[]
    for comp in np.unique(components):
        if comp in active:continue
        outside=np.flatnonzero(components==comp);best=None
        for start in range(0,len(outside),128):
            ids=outside[start:start+128]
            raw=np.asarray(full[np.ix_(ids,inside)],dtype=np.float64)
            raw*=np.where(cameras[ids,None]==cameras[inside][None,:],rho,1.)
            raw*=gamma
            values=raw/sums[ids,None]
            i,j=np.unravel_index(np.argmax(values),values.shape)
            candidate=(float(values[i,j]),int(ids[i]),int(inside[j]))
            if best is None or candidate[0]>best[0] or (candidate[0]==best[0] and candidate[1:]<best[1:]):best=candidate
        if best is not None and best[0]>0:choices.append(best)
    choices.sort(key=lambda value:(-value[0],value[1],value[2]))
    return [[i,j,float(full[i,j])*(rho if cameras[i]==cameras[j] else 1.)*gamma]
            for _,i,j in choices[:5]]


def verify(root,method):
    root=Path(root).resolve();manifest_path=root/'run-manifest.json'
    manifest=json.loads(manifest_path.read_text());cfg=manifest['config']
    require(canonical(cfg)==manifest['signature'],'Manifest signature mismatch')
    require(cfg['method']==method=='uniform-mass-'+cfg['source_support_mode'],'Wrong control type')
    require(cfg['support_mode']=='identity-uniform-control' and 'lambda' not in cfg,'Ambiguous gate semantics')
    require(cfg['rho']==.5 and cfg['alpha']==5 and cfg['beta']==25 and cfg['step']==5 and cfg['k']==20 and cfg['gamma']==.4,'Changed production parameters')
    require(cfg['numerical_tail'] is False,'Numerical tail forbidden')
    checked={}
    for group in ['inputs','sources']:
        for name,digest in cfg[group].items():
            require(sha(name)==digest,'Changed declared '+group+': '+name);checked[name]=digest
    require(str(manifest_path) not in checked,'Self-referential manifest')
    checked[str(manifest_path)]=sha(manifest_path)
    standard_path=root/f'verified-{method}.json';numerical_path=root/'verified-numerical-evidence.json'
    standard=json.loads(standard_path.read_text());numerical=json.loads(numerical_path.read_text())
    require(standard['status']==numerical['status']=='verified','Prior audits not verified')
    require(standard['method']==numerical['method']==method,'Prior audit method mismatch')
    require(numerical['standard_report_sha256']==sha(standard_path),'Numerical audit not bound to standard report')
    require(standard['manifest']['signature']==manifest['signature'],'Standard audit belongs to another manifest')
    require(standard['checks']['full_permutation_queries']==standard['checks']['diffusion_checkpoints']==cfg['query_count'],'Incomplete standard coverage')
    require(standard['metrics']['gallery_count']==cfg['gallery_count'] and standard['checks']['diffusion_depth']==cfg['depth'],'Coverage/depth differs')
    require(numerical['query_checkpoints']==cfg['query_count'],'Incomplete numerical audit')
    for name,record in standard['files'].items():
        require(sha(name)==record['sha256'],'Evidence changed since standard audit: '+name);checked[name]=record['sha256']
    for name,digest in numerical['files'].items():
        require(sha(name)==digest,'Numerical evidence changed');checked[name]=digest
    require(standard['verifier_sha256']==sha(Path(__file__).with_name('verify_experiment.py')),'Wrong standard verifier')
    require(numerical['verifier_sha256']==sha(Path(__file__).with_name('verify_numerics.py')),'Wrong numeric verifier')
    require(numerical['recovery_sha256']==sha(Path(__file__).with_name('high_precision_recovery.py')),'Wrong recovery verifier')
    repo=Path(cfg['repo_path']);sys.path.insert(0,str(repo/'RFDP'))
    import reliability_model as rm
    from uniform_control import prepare_uniform_control
    from run_uniform_candidate import checked_selection
    from high_precision_recovery import system_sha256,SOLVER_ID
    require(cfg['numerical_solver']==SOLVER_ID,'Changed numerical solver')
    paired=checked_selection(cfg['selection_file'],cfg['selection_sha256'],cfg['model'],cfg['source_support_mode'],cfg['source_lambda'],cfg['rho'])
    require(paired==cfg['paired_selection'],'Selection projection changed')
    from input_binding import validate_input_binding
    binding,bound_files=validate_input_binding(cfg['input_binding_file'],cfg['input_binding_sha256'],
        model=cfg['model'],mode=cfg['source_support_mode'],source_lambda=cfg['source_lambda'],rho=cfg['rho'],
        selection_sha256=cfg['selection_sha256'],mat=cfg['mat_path'],cache=cfg['cache_dir'],support=cfg['support_file'],
        query_count=cfg['query_count'],gallery_count=cfg['gallery_count'],depth=cfg['depth'])
    require(binding['scope']==cfg['input_scope'],'Input scope differs')
    require(all(cfg['inputs'].get(path)==digest for path,digest in bound_files.items()),'Input binding evidence missing from signed manifest')
    cache=Path(cfg['cache_dir']);full=np.load(cache/'affinity.npy',mmap_mode='r')
    graph=rm._canonical_csr(sparse.load_npz(cache/'graph.npz'))
    cameras_mat=loadmat(cfg['mat_path'],variable_names=['query_cam','gallery_cam'])
    cameras=np.r_[cameras_mat['query_cam'].ravel(),cameras_mat['gallery_cam'].ravel()]
    n=len(cameras);require(n==cfg['query_count']+cfg['gallery_count'],'Camera population mismatch')
    q,meta=rm.load_support(cfg['support_file'],expected={'mode':cfg['source_support_mode'],'k':20,'definition_version':rm.DEFINITION_VERSION})
    # Recompute from full affinity + camera IDs, not the saved support claims.
    rebuilt,rebuilt_meta=rm.build_support(full,cameras,k=20,mode=cfg['source_support_mode'])
    require(rm.csr_sha256(rebuilt)==rm.csr_sha256(q),'Source support not reproduced from unlabeled input')
    for key,value in rebuilt_meta.items():require(meta[key]==value,'Source metadata differs: '+key)
    contract=json.loads(Path(cfg['control_contract_path']).read_text())
    require(sha(cfg['control_contract_path'])==cfg['control_contract_file_sha256'],'Control document changed')
    require(json.loads(cfg['control_contract_json'])==contract,'Embedded complete contract differs')
    _,expected=prepare_uniform_control(graph,full,cameras,rho=cfg['rho'],lam=cfg['source_lambda'],source_support=rebuilt,source_support_metadata=meta)
    require(contract==expected and contract['contract_sha256']==cfg['control_contract_sha256'],'Control contract cannot be reproduced')
    # Independent formula: count every graph edge in denominator, including Q=0.
    rows=np.repeat(np.arange(n),np.diff(graph.indptr));cols=graph.indices;off=rows!=cols
    csw=graph.copy();csw.data*=np.where(cameras[rows]==cameras[cols],cfg['rho'],1.)
    denominator=math.fsum(float(v) for v in csw.data[off])
    numerator=math.fsum(float(w)*float(q[i,j]) for i,j,w in zip(rows[off],cols[off],csw.data[off]))
    qbar=numerator/denominator;effective=cfg['source_lambda']*(1-qbar);gamma=1-effective
    require(qbar==contract['statistics']['qbar'] and effective==cfg['effective_lambda'],'Independent qbar/effective_lambda differs')
    require(contract['parameters']==dict(rho=cfg['rho'],source_lambda=cfg['source_lambda'],lambda_effective=effective,gamma=gamma),'Contract parameters differ')
    weighted=csw.copy();weighted.data[off]*=gamma
    with np.load(cfg['control_state_path'],allow_pickle=False) as state:
        actual=sparse.csr_matrix((state['data'],state['indices'],state['indptr']),shape=tuple(state['shape']))
        sums=state['full_sums'].copy()
    require(rm.csr_sha256(weighted)==rm.csr_sha256(actual),'Actual weighted graph differs')
    require(np.array_equal(actual.diagonal(),csw.diagonal()),'CSW diagonal rho was altered')
    max_sum_relative=0.
    for begin in range(0,n,128):
        end=min(begin+128,n);ids=np.arange(begin,end)
        block=np.asarray(full[begin:end],dtype=np.float64).copy()
        block*=np.where(cameras[ids,None]==cameras[None,:],cfg['rho'],1.)
        block*=gamma
        block[np.arange(end-begin),ids]=np.asarray(full[ids,ids],dtype=np.float64)*cfg['rho']
        expected_sums=block.sum(axis=1,dtype=np.float64)
        relative=float(np.max(np.abs(sums[begin:end]-expected_sums)/expected_sums))
        max_sum_relative=max(max_sum_relative,relative)
        require(relative<=2e-12,'Full-kernel sums do not apply the same gate')
    nc,components=connected_components(weighted,directed=False);sizes=np.bincount(components)
    bridge_cache={};bridge_events=0;reconstructed_mp=0;frontiers=0
    for query in range(cfg['query_count']):
        record=json.loads((root/method/f'{query:04d}.json').read_text())
        mask=np.zeros(n,dtype=bool);mask[query]=True
        e=np.asarray(weighted[[query]].sum(axis=0)).ravel();active={int(components[query])};labeled=[query]
        mp_events={event['iteration']:event for event in record['refinement_events']
                   if event.get('kind')=='same-float64-equation-high-precision-recovery'}
        for trace in record['trace']:
            should_bridge=len(labeled)==int(sizes[list(active)].sum()) and len(labeled)<n
            require(bool(trace['bridges'])==should_bridge,'Bridge activation differs from actual components')
            ew=e
            if should_bridge:
                key=tuple(sorted(active))
                if key not in bridge_cache:bridge_cache[key]=independent_bridges(full,cameras,sums,components,active,cfg['rho'],gamma)
                require(trace['bridges']==bridge_cache[key],'Bridge ordering/raw weight differs from full control gate')
                ew=e.copy()
                for i,j,value in trace['bridges']:
                    require(not mask[i] and mask[j] and i!=j,'Bridge endpoint invariant');ew[i]+=value
                bridge_events+=1
            event=mp_events.get(trace['iteration'])
            if 'unlabeled' in trace or event is not None:
                a=(ew>0)&~mask;b=(np.asarray(weighted@a.astype(np.float64)).ravel()>0)&~mask&~a
                u=np.flatnonzero(a|b);frontiers+=1
                require(len(u)==trace['size'],'Reconstructed frontier size differs')
                if 'unlabeled' in trace:
                    require(u.tolist()==trace['unlabeled'] and np.flatnonzero(a).tolist()==trace['direct'] and np.flatnonzero(b).tolist()==trace['indirect'] and labeled==trace['labeled'],'Detailed frontier differs')
                if event is not None:
                    require(u.tolist()==event['unlabeled'],'MP frontier differs from controlled graph')
                    sub=weighted[u][:,u];boundary=ew[u];h=np.asarray(sub.sum(axis=1)).ravel()
                    inv=1/np.sqrt(h)
                    matrix=(sparse.diags(1+5*boundary/(h+boundary))-sparse.diags(inv)@sub@sparse.diags(inv)).tocsr()
                    rhs=np.where(boundary>0,5.,25.)
                    require(system_sha256(matrix,rhs)==event['certificate']['system_sha256'],'MP equation differs from controlled graph/boundary')
                    reconstructed_mp+=1
            chosen=trace['selected'];require(all(not mask[i] for i in chosen),'Duplicate selected node')
            mask[chosen]=True;labeled.extend(chosen)
            for node in chosen:
                start,end=weighted.indptr[node:node+2];e[weighted.indices[start:end]]+=weighted.data[start:end]
            active.update(components[chosen].tolist())
    require(reconstructed_mp==numerical['arbitrary_precision_events'],'Not all MP events rebound to graph')
    # Recheck inputs/sources after the potentially long rebuild to reject mutation.
    for name,digest in checked.items():require(sha(name)==digest,'Evidence changed during control audit: '+name)
    return dict(status='verified',method=method,scope='uniform mechanism-control software/provenance audit; no performance claim',
                signature=manifest['signature'],control_contract_sha256=contract['contract_sha256'],
                selection_sha256=cfg['selection_sha256'],source_support_mode=cfg['source_support_mode'],
                source_lambda=cfg['source_lambda'],effective_lambda=effective,qbar=qbar,
                queries=cfg['query_count'],gallery=cfg['gallery_count'],source_support_reconstructed=True,input_binding_sha256=cfg['input_binding_sha256'],input_scope=cfg['input_scope'],
                graph_exact=True,diagonal_preserved=True,full_sums_max_relative_error=max_sum_relative,
                independently_checked_bridge_events=bridge_events,reconstructed_frontiers=frontiers,
                mp_systems_rebound_to_controlled_graph=reconstructed_mp,
                standard_report_sha256=sha(standard_path),numerical_report_sha256=sha(numerical_path),
                verifier_sha256=sha(__file__),files=checked)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--method',required=True)
    args=parser.parse_args();report=verify(args.output_dir,args.method)
    path=args.output_dir/'verified-uniform-control.json'
    if path.exists():require(json.loads(path.read_text())==report,'Existing uniform audit differs')
    else:path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report[key]for key in ['status','method','queries','gallery','source_lambda','effective_lambda','qbar','independently_checked_bridge_events','mp_systems_rebound_to_controlled_graph']}),flush=True)

if __name__=='__main__':main()
