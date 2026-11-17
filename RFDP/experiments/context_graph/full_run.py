"""Prepare/run a single fixed complete input, using the existing certified runner."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import platform
import shutil
import traceback

import numpy as np

from experiments.continuous_fitting import full
from experiments.reid_adaptation import full_run as shared
from experiments.reid_adaptation import resources
from experiments.reid_adaptation.io import checkout, labels_from_manifest, lock
from . import graph, protocol


def check_graph(cache,cfg,known_hashes=None):
    meta=json.loads((cache/'manifest.json').read_text())
    if (meta['status']!='complete' or meta['builder_id']!=graph.BUILDER_ID or
        meta['variant']!=cfg['context_variant'] or meta['source_files']!=cfg['expected_cache_input_hashes'] or
        meta['parameters']!=dict(kd=20,ks=20,gamma=.4)):
        raise ValueError('Context graph scientific provenance differs')
    if meta['calibration']['nodes']!=cfg['queries']+cfg['gallery']:raise ValueError('Context population differs')
    for key in ('labels_used','cameras_used'):
        if meta['calibration'][key]:raise ValueError('Context builder may not use labels/cameras')
    if not meta['calibration']['topology_unchanged'] or not meta['calibration']['diagonal_unchanged']:
        raise ValueError('Context support/diagonal differs')
    if set(meta['files'])!={'features.npy','affinity.npy','graph.npz'}:raise ValueError('Complete cache bundle required')
    for name,h in meta['files'].items():
        actual=full.sha(cache/name) if known_hashes is None else known_hashes.get(str(cache/name))
        if actual!=h:raise ValueError('Context input SHA differs: '+name)
    for name in ('features.npy','affinity.npy'):
        if meta['files'][name]!=cfg['expected_cache_input_hashes'][name]:raise ValueError('Frozen Z/dense A changed')
    for key,field in [('context_kappa','kappa'),('context_scale','scale')]:
        if key in cfg and cfg[key]!=meta['calibration'][field]:raise ValueError('Actual calibration coefficient differs')
    if cfg['context_variant']=='weight_permutation':
        p=meta['permutation']
        if not meta['calibration']['degenerate'] and (p['seed']!=graph.SEED or p['bins']!=graph.BINS or
            not p['exact_global_weight_multiset'] or not p['exact_stratum_weight_multisets']):
            raise ValueError('Fixed placebo differs')
        if 'context_parent_graph_sha256' in cfg and cfg['context_parent_graph_sha256']!=meta['parent_graph_sha256']:
            raise ValueError('Placebo parent differs')
    return meta


def prepare(recipe_path,node_path,*,plan=protocol):
    cfg=json.loads(Path(recipe_path).read_text());plan.validate(cfg)
    if cfg['path_binding']!='required':raise ValueError('Prepare expects an unbound recipe')
    if platform.system()!='Linux':raise RuntimeError('Bind full execution on the Linux node')
    node=shared.load_node(node_path);workspace=resources.native_path(node['workspace'])
    workspace.mkdir(parents=True,exist_ok=True)
    with lock(workspace/'.execution.lock'):
        head=checkout(full.REPO);source=node['inputs'][cfg['input_id']]
        original=resources.native_path(source['cache']);label=Path(source['label_manifest']).resolve()
        shared.verify_source_cache(original,cfg)
        if full.sha(label)!=cfg['expected_label_sha256']:raise ValueError('Official input order differs')
        labels=labels_from_manifest(label,cfg['queries'],cfg['gallery'])
        x=np.load(original/'features.npy',mmap_mode='r',allow_pickle=False)
        if x.shape[0]!=cfg['queries']+cfg['gallery'] or x.dtype!=np.float32:raise ValueError('Frozen features differ')
        base=workspace/'graphs/context-calibration-v1'/cfg['input_id']
        parent=base/'context'
        if not parent.exists():
            shared.graph_memory_guard()
            # Original large arrays will be hardlinked in native storage; no dense rebuild.
            for p in (workspace,*node['host_storage_roots']):
                d=shutil.disk_usage(p)
                if d.free-512*2**20<max(d.total//4,10*2**30):raise RuntimeError('Graph storage reserve unavailable')
            graph.build_cache(original,parent)
        check_graph(parent,dict(cfg,context_variant='context'))
        cache=parent
        if cfg['context_variant']=='weight_permutation':
            cache=base/f'weight-permutation-seed{graph.SEED}'
            if not cache.exists():
                shared.graph_memory_guard();graph.build_cache(original,cache,variant='weight_permutation',parent=parent)
        meta=check_graph(cache,cfg)
        metadata=workspace/'metadata'/cfg['input_id'];metadata.mkdir(parents=True,exist_ok=True)
        cameras=np.r_[labels['query_cam'],labels['gallery_cam']]
        camera_file=metadata/'cameras.npy';labels_file=metadata/'labels.npz'
        if camera_file.exists():
            if not np.array_equal(np.load(camera_file,allow_pickle=False),cameras):raise ValueError('Existing camera order differs')
        else:np.save(camera_file,cameras)
        if labels_file.exists():
            with np.load(labels_file,allow_pickle=False) as z:
                if set(z.files)!=set(labels) or any(not np.array_equal(z[k],v) for k,v in labels.items()):raise ValueError('Evaluation metadata differs')
        else:np.savez(labels_file,**labels)
        paths=[original/n for n in ('features.npy','affinity.npy','graph.npz')]
        paths += [cache/n for n in ('features.npy','affinity.npy','graph.npz','manifest.json')]
        paths += [camera_file,labels_file,label]
        if cfg['context_variant']=='weight_permutation':paths += [parent/'graph.npz',parent/'manifest.json']
        cpu=resources.cpu_allocation();shared_bytes=sum((cache/n).stat().st_size for n in ('features.npy','affinity.npy'))
        cfg.update(path_binding='bound',commit=head,workspace=str(workspace),archive_root=node['archive_root'],
            host_storage_roots=node['host_storage_roots'],workers=cpu['workers'],cpu_allocation=cpu,
            original_cache=str(original),cache=str(cache),label_manifest=str(label),camera_ids_file=str(camera_file),labels_file=str(labels_file),
            output=str(workspace/'runs'/shared.unit_id(cfg)),input_sha256={str(p):full.sha(p) for p in paths},
            graph_sha256=meta['files']['graph.npz'],context_kappa=meta['calibration']['kappa'],context_scale=meta['calibration']['scale'],
            context_parent_graph_sha256=meta['parent_graph_sha256'],worker_address_space_limit_bytes=shared_bytes+cfg['worker_private_budget_bytes'])
        plan.validate(cfg,require_bound=True);shared.check_bound_inputs(cfg)
        state=resources.snapshot(cfg,reserve=True);reserved=resources.reservation(cfg,state)
        configs=workspace/'configs';configs.mkdir(exist_ok=True);path=configs/(shared.unit_id(cfg)+'.json')
        if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Do not overwrite a differently bound config')
        if not path.exists():full.dump(path,cfg)
        result=dict(status='prepared_not_started',config=str(path),config_sha256=full.sha(path),commit=head,
                    calibration=meta['calibration'],resources=state,reservation=reserved)
        full.dump(path.with_suffix('.prepared.json'),result)
        return result


def check_decision(cfg,path):
    if not cfg['requires_decision']:return
    if path is None:raise ValueError('Full predecessor evidence and a recorded research decision are required')
    decision=json.loads(Path(path).read_text());phase=cfg['phase']
    expected={'control':'continue_fixed_control','transfer':'continue_transfer','combination':'continue_early5_combination'}
    if decision.get('schema')!=protocol.SCHEMA or decision.get('verdict')!=expected[phase] or not decision.get('rationale'):
        raise ValueError('Wrong or missing context decision')
    if shared.unit_id(cfg) not in decision.get('allowed_units',[]):raise ValueError('Unit not allowed by this decision')
    found=set()
    for row in decision.get('evidence',[]):
        out=Path(row['output'])
        if not out.is_absolute() or full.sha(out/'FULL_ACCEPTANCE.json')!=row['acceptance_sha256']:raise ValueError('Predecessor SHA differs')
        shared.accepted_result(out)
        old=json.loads((out/'run-manifest.json').read_text())['config']
        if old['adaptation_schema']!=protocol.SCHEMA:raise ValueError('Wrong predecessor method')
        found.add((old['adaptation_route'],old['input_id']))
    required={('context_graph','duke-osnet')}
    if phase in ('transfer','combination'):required.add(('context_graph_permuted','duke-osnet'))
    if phase=='combination':required.update(('context_graph',key) for key in protocol.ORDER)
    if not required.issubset(found):raise ValueError('Missing accepted full predecessor inputs')


def run(config_path,decision=None,resume=False):
    cfg=json.loads(Path(config_path).read_text());protocol.validate(cfg,require_bound=True)
    with lock(Path(cfg['workspace'])/'.execution.lock'):
        check_decision(cfg,decision)
        try:return shared._run(config_path,resume)
        except BaseException:
            out=Path(cfg['output'])
            if out.exists() and not (out/'FAILED.json').exists():full.dump(out/'FAILED.json',dict(time=full.now(),error=traceback.format_exc()))
            raise


def collect(config_path,destination):
    result=shared.collect(config_path,destination)
    cfg=json.loads(Path(config_path).read_text());dest=Path(result['path'])
    shutil.copyfile(Path(cfg['cache'])/'manifest.json',dest/'context-graph.json')
    hashes=json.loads((dest/'SHA256.json').read_text())
    hashes['context-graph.json']=full.sha(dest/'context-graph.json')
    full.dump(dest/'SHA256.json',hashes)
    return result


def archive(config_path):
    result=shared.archive(config_path)
    cfg=json.loads(Path(config_path).read_text())
    collect(config_path,Path(cfg['archive_root'])/shared.unit_id(cfg)/'evidence')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['recipes','prepare','run','collect','archive'])
    for key in ('recipe','node','config','output','decision'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='recipes':result=protocol.write_recipes(args.output)
    elif args.action=='prepare':result=prepare(args.recipe,args.node)
    elif args.action=='run':result=run(args.config,args.decision,args.resume)
    elif args.action=='collect':result=collect(args.config,args.output)
    else:result=archive(args.config)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
