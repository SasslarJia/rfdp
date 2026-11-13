"""Prepare or run ONE complete official input; no automatic experiment queue."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import platform
import shutil
import tarfile
import time
import traceback
from pathlib import Path
import numpy as np
from experiments.continuous_fitting import full
from experiments.relation_routes.full_run import query_complete
from . import protocol, resources, graph
from .io import checkout, labels_from_manifest, lock

LEGACY_EARLY5 = {
 'market-osnet':'6cd5770a2422a76054a3e3b7a1261cb8ddfe1b37260706ebb541ef8b57829751',
 'duke-osnet':'176fd0efeee8814f4024972198f77c2e78c3ee4c80cd3e637795d6ca53f0d4cd',
}


def unit_id(cfg):
    return cfg['adaptation_route']+'--'+cfg['input_id']


def load_node(path):
    node=json.loads(Path(path).read_text())
    if node.get('profile')!='wsl2-cpu' or node.get('cpu_workers_cap')!=12:
        raise ValueError('Expected the recorded WSL CPU profile')
    for key in ('workspace','archive_root'):
        if not Path(node[key]).is_absolute():raise ValueError('Bind absolute node paths: '+key)
    resources.native_path(node['workspace'])
    if not node['archive_root'].startswith('/mnt/d/'):
        raise ValueError('Archive destination must be on the requested D drive')
    if node.get('host_storage_roots')!=['/mnt/c','/mnt/d']:
        raise ValueError('Check the current VHD backing C drive and archive D drive')
    return node


def check_graph(cache,cfg,known_hashes=None):
    meta=json.loads((cache/'manifest.json').read_text())
    if (meta['status']!='complete' or meta['builder_id']!=graph.BUILDER_ID or
        meta['parameters']!=dict(kd=cfg['kd'],ks=cfg['ks'],gamma=cfg['gamma']) or
        meta['n']!=cfg['queries']+cfg['gallery'] or meta['features_renormalized']):
        raise ValueError('P1 graph provenance differs')
    if set(meta['files'])!={'features.npy','affinity.npy','graph.npz'}:
        raise ValueError('P1 requires the complete frozen-Z/full-A/sparse-G bundle')
    for name,h in meta['files'].items():
        actual=full.sha(cache/name) if known_hashes is None else known_hashes.get(str(cache/name))
        if actual!=h:raise ValueError('P1 cache SHA differs: '+name)
    if meta['files']['features.npy']!=cfg['expected_cache_input_hashes']['features.npy']:
        raise ValueError('P1 graph uses different features')
    return meta


def check_bound_inputs(cfg):
    """Recheck preparation semantics; self-declared file hashes alone are insufficient."""
    if cfg['k']==30:
        if cfg['graph_builder_id']!=graph.BUILDER_ID:raise ValueError('P1 builder ID differs')
        check_graph(Path(cfg['cache']),cfg,known_hashes=cfg['input_sha256'])
    expected=labels_from_manifest(cfg['label_manifest'],cfg['queries'],cfg['gallery'])
    from .io import load_labels
    saved=load_labels(cfg['labels_file'],cfg['queries'],cfg['gallery'])
    if any(not np.array_equal(saved[k],v) for k,v in expected.items()):raise ValueError('Evaluation labels differ from fixed image order')
    cameras=np.concatenate([expected['query_cam'],expected['gallery_cam']])
    if not np.array_equal(np.load(cfg['camera_ids_file'],allow_pickle=False),cameras):
        raise ValueError('Camera IDs differ from fixed image order')


def graph_storage_guard(node,n,feature_bytes):
    # Full A + temporary D + worst-case dense CSR, including ties; no truncation.
    required=20*n*n+feature_bytes+2*2**30
    for path in (node['workspace'],*node['host_storage_roots']):
        d=shutil.disk_usage(path)
        if d.free-required<max(d.total//4,10*2**30):
            raise RuntimeError('Full graph construction reserve unavailable on '+path)
    return required


def graph_memory_guard():
    state=resources.memory_allocation()
    if state['current']>.7*state['cap']:
        raise RuntimeError('Graph construction reached the 70 percent memory line')


def guarded_graph_progress():
    last=0.
    def report(state):
        nonlocal last
        current=time.monotonic()
        if current-last>=10:
            graph_memory_guard();last=current
            print(json.dumps(state),flush=True)
    return report


def prepare(recipe_path,node_path):
    cfg=json.loads(Path(recipe_path).read_text());protocol.validate(cfg)
    if cfg['path_binding']!='required':raise ValueError('Prepare expects an unbound scientific recipe')
    if platform.system()!='Linux':raise RuntimeError('Preparation binds the Linux execution node, not the Mac')
    node=load_node(node_path);workspace=resources.native_path(node['workspace'])
    workspace.mkdir(parents=True,exist_ok=True)
    with lock(workspace/'.execution.lock'):
        head=checkout(full.REPO);source=node['inputs'][cfg['input_id']]
        cache=resources.native_path(source['cache']);label=Path(source['label_manifest']).resolve()
        # Verify before graph construction. These are already extracted frozen Z/A/G.
        for name,h in cfg['expected_cache_input_hashes'].items():
            if full.sha(cache/name)!=h:raise ValueError('Frozen cache differs: '+name)
        if full.sha(label)!=cfg['expected_label_sha256']:raise ValueError('Official label order differs')
        labels=labels_from_manifest(label,cfg['queries'],cfg['gallery'])
        x=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
        if x.shape[0]!=cfg['queries']+cfg['gallery'] or x.dtype!=np.float32:raise ValueError('Frozen features population/dtype differs')
        if cfg['k']==30:
            newcache=workspace/'graphs'/cfg['input_id']/'kd30-ks30-gamma0.4'
            if not newcache.exists():
                graph_memory_guard()
                graph_storage_guard(node,len(x),(cache/'features.npy').stat().st_size)
                graph.build_graph_cache(cache/'features.npy',newcache,kd=30,ks=30,gamma=.4,
                    expected_features_sha256=cfg['expected_cache_input_hashes']['features.npy'],
                    progress=guarded_graph_progress())
            check_graph(newcache,cfg);cache=newcache
            cfg['graph_builder_id']=graph.BUILDER_ID
        metadata=workspace/'metadata'/cfg['input_id'];metadata.mkdir(parents=True,exist_ok=True)
        cameras=np.concatenate([labels['query_cam'],labels['gallery_cam']])
        camera_file=metadata/'cameras.npy';labels_file=metadata/'labels.npz'
        if camera_file.exists():
            if not np.array_equal(np.load(camera_file,allow_pickle=False),cameras):raise ValueError('Existing camera order differs')
        else:np.save(camera_file,cameras)
        if labels_file.exists():
            with np.load(labels_file,allow_pickle=False) as z:
                if set(z.files)!=set(labels) or any(not np.array_equal(z[k],v) for k,v in labels.items()):
                    raise ValueError('Existing evaluation labels differ')
        else:np.savez(labels_file,**labels)
        inputs=[cache/n for n in ('features.npy','affinity.npy','graph.npz')]+[camera_file,labels_file,label]
        if cfg['k']==30:inputs.append(cache/'manifest.json')
        cpu=resources.cpu_allocation()
        shared=sum((cache/n).stat().st_size for n in ('features.npy','affinity.npy'))
        cfg.update(path_binding='bound',commit=head,workspace=str(workspace),archive_root=node['archive_root'],
            host_storage_roots=node['host_storage_roots'],workers=cpu['workers'],cpu_allocation=cpu,
            cache=str(cache),label_manifest=str(label),camera_ids_file=str(camera_file),labels_file=str(labels_file),
            output=str(workspace/'runs'/unit_id(cfg)),input_sha256={str(p):full.sha(p) for p in inputs},
            graph_sha256=full.sha(cache/'graph.npz'),worker_address_space_limit_bytes=shared+cfg['worker_private_budget_bytes'])
        protocol.validate(cfg,require_bound=True)
        # Check actual resources; this is not a stress test or a retrieval run.
        state=resources.snapshot(cfg,reserve=True);reserved=resources.reservation(cfg,state)
        configs=workspace/'configs';configs.mkdir(exist_ok=True);path=configs/(unit_id(cfg)+'.json')
        if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Do not overwrite a differently bound configuration')
        if not path.exists():full.dump(path,cfg)
        result=dict(status='prepared_not_started',config=str(path),config_sha256=full.sha(path),
                    commit=head,resources=state,reservation=reserved)
        full.dump(path.with_suffix('.prepared.json'),result)
        return result


def accepted_result(out,cfg=None):
    out=Path(out);path=out/'FULL_ACCEPTANCE.json'
    if not path.exists():return None
    a=json.loads(path.read_text());m=json.loads((out/'run-manifest.json').read_text())
    actual=m['config'];protocol.validate(actual,require_bound=True)
    if cfg is not None and actual!=cfg:raise ValueError('Accepted configuration differs')
    if (a['status']!='accepted_full_population' or a['queries']!=actual['queries'] or
        a['gallery']!=actual['gallery'] or a['signature']!=m['signature'] or
        a['signature']!=protocol.digest(actual) or a['max_recomputed_residual']>1e-8):
        raise ValueError('Full-population acceptance differs')
    full.check_admission_manifest(actual,m)
    for name,key in [('metrics.json','metrics_sha256'),('rankings.npy','rankings_sha256')]:
        if full.sha(out/name)!=a[key]:raise ValueError('Accepted artifact differs: '+name)
    complete=json.loads((out/'COMPUTE_COMPLETE.json').read_text())
    if complete['signature']!=a['signature'] or complete['metrics_sha256']!=a['metrics_sha256']:
        raise ValueError('Accepted metrics differ from computed metrics')
    full.aggregate_inline_readback(configuration=actual,output=out,remove_duplicates=False)
    return a


def decision_evidence(decision):
    keys=set()
    for ref in decision.get('evidence',[]):
        path=Path(ref['path'])
        if full.sha(path)!=ref['sha256']:raise ValueError('Decision evidence SHA differs')
        if ref.get('kind')=='legacy_early5':
            if ref['sha256']!=LEGACY_EARLY5.get(ref['input_id']):raise ValueError('Unregistered historical early5 evidence')
            keys.add(('early5',ref['input_id']))
        else:
            if path.name!='FULL_ACCEPTANCE.json' or not accepted_result(path.parent):
                raise ValueError('Decision requires a complete accepted input')
            c=json.loads((path.parent/'run-manifest.json').read_text())['config']
            keys.add((c['adaptation_route'],c['input_id']))
    return keys


def check_decision(cfg,path):
    if not cfg['requires_decision']:return
    if path is None:raise ValueError('This conditional/transfer input requires an evidence-bound decision')
    d=json.loads(Path(path).read_text())
    if d.get('verdict')!='continue' or not d.get('rationale') or unit_id(cfg) not in d.get('allowed_units',[]):
        raise ValueError('Decision does not authorize this unit')
    keys=decision_evidence(d)
    if cfg['adaptation_route']=='p1_graph':
        required={('p1_graph','market-transreid'),('p1_graph','duke-transreid')}
    elif cfg['adaptation_route']=='p1_early5':
        required={('p1_graph',cfg['input_id']),('early5',cfg['input_id'])}
    else:
        required={('early5',cfg['input_id'])}
        if len(d['allowed_units'])!=1:raise ValueError('The late5 control is limited to one selected input')
        marker=Path(cfg['workspace'])/'late5-selection.json'
        selection=dict(input_id=cfg['input_id'],decision_sha256=full.sha(path))
        if marker.exists() and json.loads(marker.read_text())!=selection:raise ValueError('Late5 input already frozen')
        if not required.issubset(keys):raise ValueError('Missing complete early5 evidence')
        if not marker.exists():full.dump(marker,selection)
    if not required.issubset(keys):raise ValueError('Missing complete predecessor evidence')


def _run(config_path,resume=False):
    full.configure(config_path);cfg=full.CFG;full.sources();check_bound_inputs(cfg)
    out=full.OUT;signature=protocol.digest(cfg)
    if accepted_result(out,cfg):return 'reused_complete_run'
    state=resources.snapshot(cfg,reserve=True);reserved=resources.reservation(cfg,state)
    if out.exists():
        if not resume:raise ValueError('Existing output requires explicit --resume; no overwrite')
        m=json.loads((out/'run-manifest.json').read_text());full.check_admission_manifest(cfg,m)
        if m['signature']!=signature:raise ValueError('Run signature differs')
    else:
        out.mkdir(parents=True);(out/'queries').mkdir();(out/'accepted').mkdir()
        full.dump(out/'run-manifest.json',dict(signature=signature,config=cfg,time=full.now(),
            resources=state,**reserved,solver_id=full.configured_solver_id(cfg),execution_commit=full.execution_revision()))
    failure=out/'FAILED.json'
    if failure.exists():
        with (out/'attempt-history.jsonl').open('a') as f:f.write(json.dumps(json.loads(failure.read_text()))+'\n')
        failure.unlink()
    if not (out/'COMPUTE_COMPLETE.json').exists():
        done={q for q in range(full.QN) if query_complete(cfg,signature,q)}
        pending=[q for q in range(full.QN) if q not in done]
        full.dump(out/'resume-plan.json',dict(time=full.now(),complete=len(done),pending=len(pending)))
        seconds=full.pool_stage('compute',pending,len(done)) if pending else 0.
        if not all(query_complete(cfg,signature,q) for q in range(full.QN)):raise RuntimeError('Query transaction missing')
        start=time.monotonic();metric=full.evaluate();evaluation_seconds=time.monotonic()-start;full.sources()
        full.dump(out/'COMPUTE_COMPLETE.json',dict(status='computed_and_independently_evaluated',signature=signature,
            query_count=full.QN,gallery_count=full.GN,rankings_sha256=full.sha(out/'rankings.npy'),
            metrics_sha256=full.sha(out/'metrics.json'),
            independent=metric,latest_attempt_seconds=seconds,evaluation_seconds=evaluation_seconds,
            git_commit=full.execution_revision(),inline_full_score_readback=True))
        (out/'compute.exit').write_text('0\n')
    c=json.loads((out/'COMPUTE_COMPLETE.json').read_text())
    if c['signature']!=signature or (c['query_count'],c['gallery_count'])!=(full.QN,full.GN):raise ValueError('Compute population differs')
    if full.sha(out/'metrics.json')!=c['metrics_sha256']:raise ValueError('Computed metrics changed before final acceptance')
    seconds=full.aggregate_inline_readback();full.sources()
    rows=[json.loads((out/'accepted'/f'{q:04d}.json').read_text()) for q in range(full.QN)]
    records=[json.loads((out/'queries'/f'{q:04d}.json').read_text()) for q in range(full.QN)]
    if max(r['max_recomputed_residual'] for r in rows)>1e-8:raise ValueError('Residual acceptance failed')
    timing={key:sum(r[key] for r in records) for key in ('compute_seconds','saving_seconds','readback_seconds')}
    timing.update(final_acceptance_seconds=seconds,evaluation_seconds=c['evaluation_seconds'],
                  scope='per-query sums are worker CPU-wall totals, not elapsed parallel run time; compute includes online certification')
    full.dump(out/'FULL_ACCEPTANCE.json',dict(status='accepted_full_population',signature=signature,time=full.now(),
        queries=full.QN,gallery=full.GN,max_recomputed_residual=max(r['max_recomputed_residual'] for r in rows),
        solver_calls=sum(r['solver_calls'] for r in rows),rankings_sha256=full.sha(out/'rankings.npy'),
        metrics_sha256=full.sha(out/'metrics.json'),actual_saved_scores_recomputed=True,
        all_frontiers_bridges_rankings_rebuilt=True,certificate_implementation_shared=True,
        git_commit=full.execution_revision(),readback_performed_inline=True,
        retained_queries=cfg['retained_queries'],initial_states_retained=True,timing=timing))
    return 'completed_full_population'


def run(config_path,decision=None,resume=False):
    cfg=json.loads(Path(config_path).read_text());protocol.validate(cfg,require_bound=True)
    with lock(Path(cfg['workspace'])/'.execution.lock'):
        check_decision(cfg,decision)
        try:return _run(config_path,resume)
        except BaseException:
            out=Path(cfg['output'])
            if out.exists() and not (out/'FAILED.json').exists():
                full.dump(out/'FAILED.json',dict(time=full.now(),error=traceback.format_exc()))
            raise


def collect(config_path,destination):
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output'])
    if not accepted_result(out,cfg):raise ValueError('Only full accepted results may be collected')
    dest=Path(destination)/unit_id(cfg);dest.mkdir(parents=True,exist_ok=True)
    names=['metrics.json','run-manifest.json','COMPUTE_COMPLETE.json','FULL_ACCEPTANCE.json']
    for name in names:shutil.copyfile(out/name,dest/name)
    full.dump(dest/'config.json',cfg);names.append('config.json')
    # Per-query evaluation lives in metrics; runtime and certificates stay separate.
    runtime=[json.loads((out/'queries'/f'{q:04d}.json').read_text()) for q in range(cfg['queries'])]
    full.dump(dest/'query-runtime.json',runtime);names.append('query-runtime.json')
    full.dump(dest/'SHA256.json',{name:full.sha(dest/name) for name in names})
    return dict(status='collected_complete_evidence',path=str(dest))


def archive(config_path):
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output'])
    with lock(Path(cfg['workspace'])/'.execution.lock'):
        if not accepted_result(out,cfg):raise ValueError('Archive complete accepted runs only')
        dest=Path(cfg['archive_root'])/unit_id(cfg);dest.mkdir(parents=True,exist_ok=True)
        collect(config_path,dest/'evidence')
        target=dest/'complete-run.tar'
        if target.exists():raise ValueError('Archive exists; do not overwrite')
        budget=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+64*2**20
        d=shutil.disk_usage(dest)
        if d.free-budget<max(d.total//4,10*2**30):raise RuntimeError('D archive disk reserve unavailable')
        tmp=target.with_suffix('.tar.partial')
        with tarfile.open(tmp,'w') as tar:
            tar.add(out,arcname=unit_id(cfg));tar.add(config_path,arcname='config.json')
        # Verify each source file against bytes read from the archive before publication.
        import hashlib
        with tarfile.open(tmp,'r') as tar:
            for member in tar:
                if member.isfile():
                    source=Path(config_path) if member.name=='config.json' else out/Path(member.name).relative_to(unit_id(cfg))
                    digest=hashlib.sha256()
                    with tar.extractfile(member) as stream:
                        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
                    if digest.hexdigest()!=full.sha(source):raise ValueError('Archive content differs: '+member.name)
        tmp.replace(target)
        full.dump(dest/'archive.json',dict(path=str(target),sha256=full.sha(target),source=str(out),source_retained=True))
        return dict(status='archived_and_verified',path=str(target),source_retained=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['recipes','prepare','run','collect','archive'])
    p.add_argument('--recipe',type=Path);p.add_argument('--node',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--decision',type=Path);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='recipes':
        if not a.output:p.error('recipes requires --output')
        result=protocol.write_recipes(a.output)
    elif a.action=='prepare':
        if not a.recipe or not a.node:p.error('prepare requires --recipe and --node')
        result=prepare(a.recipe,a.node)
    else:
        if not a.config:p.error('this action requires --config')
        if a.action=='run':result=run(a.config,a.decision,a.resume)
        elif a.action=='archive':result=archive(a.config)
        else:
            if not a.output:p.error('collect requires --output')
            result=collect(a.config,a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
