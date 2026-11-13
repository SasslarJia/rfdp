"""Frozen scientific recipes; preparation binds files but never launches runs."""
import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.relation_routes.protocol import INPUTS

SCHEMA = 'reid-adaptation-v1'
ROUTES = {
    'p1_graph': dict(k=30, kd=30, ks=30, promotion_schedule='fixed5', boundary_rho=1.),
    'early5': dict(k=20, kd=20, ks=20, promotion_schedule='early5', boundary_rho=1.),
    'camera_boundary': dict(k=20, kd=20, ks=20, promotion_schedule='fixed5', boundary_rho=.5),
    'p1_early5': dict(k=30, kd=30, ks=30, promotion_schedule='early5', boundary_rho=1.),
    'late5': dict(k=20, kd=20, ks=20, promotion_schedule='late5', boundary_rho=1.),
}
FIRST_STAGE = [('p1_graph', x) for x in ('market-transreid','duke-transreid','market-osnet','duke-osnet')]
FIRST_STAGE += [('early5', x) for x in ('market-transreid','duke-transreid')]
FIRST_STAGE += [('camera_boundary', x) for x in ('duke-osnet','market-osnet')]
CONDITIONAL = [('p1_early5', x) for x in INPUTS]
LABEL_SHA = {
    'market': '728b40e8730a96c5061cbce15d5d26149b2cadda467293878a146c57d588af81',
    'duke': 'efd96a1be0ee89f0853e3b00e20dba485903d4e9cb6403e8769c834dc2f561af',
}
FIXED = dict(source='original', relation_model='original', admission_policy='original',
             alpha=5, beta=25, step=5, depth=500, gamma=.4, bridge_budget=5,
             support_lambda=0., blas_threads=1, auto_start=False,
             trace_retention='sampled_after_readback', retain_initial_state=True,
             runner='experiments.reid_adaptation.full_run', resource_profile='wsl2-cpu',
             worker_cap=12, memory_limit_bytes=24*2**30,
             worker_private_budget_bytes=768*2**20, output_budget_bytes=3*2**30)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def recipe(route, input_id, *, conditional=False):
    if route not in ROUTES or input_id not in INPUTS:
        raise ValueError('Unknown route or official frozen input')
    is_conditional = (route, input_id) not in FIRST_STAGE
    if not is_conditional and conditional:
        raise ValueError('First-stage runs have a fixed phase')
    if is_conditional and route not in ('p1_early5','late5'):
        raise ValueError('This input/route must be reused or is outside this plan')
    frozen=INPUTS[input_id]
    # Match the historical np.linspace(..., dtype=int) floor convention.
    retained=[i*(frozen['queries']-1)//31 for i in range(32)]
    return dict(FIXED, **ROUTES[route], adaptation_schema=SCHEMA,
                adaptation_route=route,input_id=input_id,
                queries=frozen['queries'],gallery=frozen['gallery'],
                feature_sha256=frozen['feature_sha256'],
                expected_cache_input_hashes=copy.deepcopy(frozen['cache_input_hashes']),
                expected_label_sha256=LABEL_SHA[input_id.split('-')[0]],
                retained_queries=retained,path_binding='required',
                phase='conditional' if is_conditional else 'first_stage',
                requires_decision=is_conditional or (route=='p1_graph' and input_id.endswith('osnet')))


def validate(cfg, require_bound=False):
    if cfg.get('adaptation_schema') != SCHEMA:
        raise ValueError('Explicit adaptation schema required')
    expected=recipe(cfg.get('adaptation_route'),cfg.get('input_id'))
    for key,value in expected.items():
        if key=='path_binding':continue
        if cfg.get(key)!=value or isinstance(cfg.get(key),bool)!=isinstance(value,bool):
            raise ValueError('Fixed adaptation setting differs: '+key)
    if any(key in cfg for key in ('relation_schema','f1_config_schema','research_schema','theta','rho','support_file','original_solver')):
        raise ValueError('Unplanned model/solver combination')
    state=cfg.get('path_binding')
    if state=='required':
        if require_bound:raise ValueError('Prepare and bind actual data before execution')
        if any(k in cfg for k in ('commit','cache','output','workers','input_sha256')):
            raise ValueError('Unbound recipe may not claim checked runtime files')
    elif state=='bound':
        if not re.fullmatch('[0-9a-f]{40}',str(cfg.get('commit',''))):raise ValueError('Fixed commit required')
        if type(cfg.get('workers')) is not int or not 1<=cfg['workers']<=12:raise ValueError('WSL worker cap is 12')
        for key in ('cache','output','camera_ids_file','labels_file','workspace','archive_root','label_manifest'):
            p=Path(cfg.get(key,''))
            if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute runtime path required: '+key)
        for key in ('cache','output','workspace'):
            if str(Path(cfg[key])).startswith('/mnt/'):
                raise ValueError('Active files must be in native Linux storage: '+key)
        hashes=cfg.get('input_sha256',{})
        if not hashes or any(not Path(p).is_absolute() or not re.fullmatch('[0-9a-f]{64}',str(h)) for p,h in hashes.items()):
            raise ValueError('Invalid input SHA map')
        for key in ('camera_ids_file','labels_file','label_manifest'):
            if cfg[key] not in hashes:raise ValueError('Unbound metadata: '+key)
        if hashes[cfg['label_manifest']]!=cfg['expected_label_sha256']:raise ValueError('Label order source differs')
        cache=Path(cfg['cache']);old=cfg['expected_cache_input_hashes']
        for name in ('features.npy','affinity.npy','graph.npz'):
            if str(cache/name) not in hashes:raise ValueError('Missing cache binding: '+name)
        if hashes[str(cache/'features.npy')]!=old['features.npy']:raise ValueError('Frozen features changed')
        if cfg['k']==20:
            for name,h in old.items():
                if hashes[str(cache/name)]!=h:raise ValueError('Original cache changed: '+name)
        else:
            meta=str(cache/'manifest.json')
            if meta not in hashes or cfg.get('graph_builder_id') is None:raise ValueError('New full graph/affinity need provenance')
        if cfg.get('graph_sha256')!=hashes[str(cache/'graph.npz')]:raise ValueError('Actual graph file not bound')
        out=Path(cfg['output'])
        if out==cache or out in cache.parents or cache in out.parents:raise ValueError('Input/output overlap')
        if cfg.get('worker_address_space_limit_bytes',0)<=0:raise ValueError('Worker memory limit required')
        if not cfg.get('host_storage_roots'):raise ValueError('Host C/D disk guards required for WSL VHD')
    else:raise ValueError('Unknown path binding')
    return {k:cfg[k] for k in ('adaptation_schema','adaptation_route','input_id','kd','ks','gamma','promotion_schedule','boundary_rho','bridge_budget','support_lambda')}


def write_recipes(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    entries=[]
    for route,input_id in FIRST_STAGE+CONDITIONAL:
        cfg=recipe(route,input_id);validate(cfg)
        name=f'{route}--{input_id}.json'
        (directory/name).write_text(json.dumps(cfg,indent=2)+'\n')
        entries.append(dict(recipe=name,phase=cfg['phase'],auto_start=False))
    # Exactly one late control is selected after an accepted beneficial input;
    # this selector is not a runnable config and creates no extra run entries.
    manifest=dict(schema=SCHEMA,first_stage_count=8,conditional_combination_max=4,
                  conditional_late5_max=1,maximum_full_runs=13,
                  early5_osnet='reuse archived complete results',runs=entries,
                  late5=dict(input_id=None,window_new_nodes=[101,105],requires_decision=True),
                  auto_start=False)
    (directory/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
