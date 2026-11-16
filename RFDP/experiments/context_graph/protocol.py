"""Frozen context calibration and a single distribution-matched placebo."""
import copy
import json
import math
from pathlib import Path
import re

from experiments.reid_adaptation.protocol import FIXED, INPUTS, LABEL_SHA
from .graph import BUILDER_ID, SEED, BINS

SCHEMA='context-graph-v1'
ROUTES=('context_graph','context_graph_permuted','context_graph_early5')
ORDER=('duke-osnet','market-osnet','market-transreid','duke-transreid')
UNITS=[('context_graph','duke-osnet'),('context_graph_permuted','duke-osnet')]
UNITS += [('context_graph',key) for key in ORDER[1:]]
UNITS += [('context_graph_early5',key) for key in ORDER]


def recipe(route,input_id):
    if (route,input_id) not in UNITS:raise ValueError('Outside the fixed context experiment plan')
    frozen=INPUTS[input_id];permuted=route=='context_graph_permuted'
    phase=('main' if (route,input_id)==UNITS[0] else 'control' if permuted else
           'combination' if route=='context_graph_early5' else 'transfer')
    return dict(FIXED,runner='experiments.context_graph.full_run',numerical_solver='stable_lu_certified',
        adaptation_schema=SCHEMA,adaptation_route=route,input_id=input_id,
        k=20,kd=20,ks=20,promotion_schedule='early5' if route=='context_graph_early5' else 'fixed5',
        boundary_rho=1.,context_builder_id=BUILDER_ID,
        context_variant='weight_permutation' if permuted else 'context',
        context_rule='endpoint-removed-Laplace-Jaccard',kappa_rule='unweighted-log-std-ddof0',
        context_mass_rule='total-nonself-mass-original',context_diagonal='unchanged',context_bridges='original-full-A',
        permutation_seed=SEED if permuted else None,permutation_bins=BINS if permuted else None,
        queries=frozen['queries'],gallery=frozen['gallery'],feature_sha256=frozen['feature_sha256'],
        expected_cache_input_hashes=copy.deepcopy(frozen['cache_input_hashes']),
        expected_label_sha256=LABEL_SHA[input_id.split('-')[0]],
        retained_queries=[i*(frozen['queries']-1)//31 for i in range(32)],
        path_binding='required',phase=phase,requires_decision=phase!='main')


def validate(cfg,require_bound=False):
    expected=recipe(cfg.get('adaptation_route'),cfg.get('input_id'))
    for key,value in expected.items():
        if key in ('path_binding','numerical_solver'):continue
        if cfg.get(key)!=value or isinstance(cfg.get(key),bool)!=isinstance(value,bool):
            raise ValueError('Fixed context setting differs: '+key)
    if any(key in cfg for key in ('relation_schema','f1_config_schema','research_schema','theta','rho','support_file','original_solver')):
        raise ValueError('Unplanned context model combination')
    if cfg.get('numerical_solver','warm_certified') not in ('warm_certified','stable_lu_certified'):
        raise ValueError('Unknown context numerical solver')
    if cfg.get('path_binding')=='required':
        if require_bound:raise ValueError('Context input needs preparation')
        if any(k in cfg for k in ('commit','cache','output','workers','input_sha256')):raise ValueError('Unbound context recipe claims runtime data')
    elif cfg.get('path_binding')=='bound':
        if not re.fullmatch('[0-9a-f]{40}',str(cfg.get('commit',''))):raise ValueError('Fixed execution commit required')
        if type(cfg.get('workers')) is not int or not 1<=cfg['workers']<=12:raise ValueError('WSL worker limit differs')
        for key in ('cache','original_cache','output','camera_ids_file','labels_file','workspace','archive_root','label_manifest'):
            p=Path(cfg.get(key,''))
            if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute context path required: '+key)
        for key in ('cache','original_cache','output','workspace'):
            if str(Path(cfg[key])).startswith('/mnt/'):raise ValueError('Native active paths required')
        hashes=cfg.get('input_sha256',{})
        if not hashes or any(not Path(p).is_absolute() or not re.fullmatch('[0-9a-f]{64}',str(h)) for p,h in hashes.items()):raise ValueError('Invalid input binding')
        old=cfg['expected_cache_input_hashes'];cache=Path(cfg['cache']);original=Path(cfg['original_cache'])
        for name,h in old.items():
            if hashes.get(str(original/name))!=h:raise ValueError('Original frozen input differs: '+name)
        for name in ('features.npy','affinity.npy'):
            if hashes.get(str(cache/name))!=old[name]:raise ValueError('Context changed frozen Z or dense A')
        for name in ('graph.npz','manifest.json'):
            if str(cache/name) not in hashes:raise ValueError('Context graph/manifest missing')
        if cfg.get('graph_sha256')!=hashes[str(cache/'graph.npz')]:raise ValueError('Actual context graph not bound')
        for key in ('camera_ids_file','labels_file','label_manifest'):
            if cfg[key] not in hashes:raise ValueError('Unbound evaluation metadata')
        if hashes[cfg['label_manifest']]!=cfg['expected_label_sha256']:raise ValueError('Official label order differs')
        for key in ('context_kappa','context_scale'):
            v=cfg.get(key)
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v):raise ValueError('Invalid computed calibration coefficient')
        if cfg['context_kappa']<0 or cfg['context_scale']<=0:raise ValueError('Invalid calibration range')
        parent=cfg.get('context_parent_graph_sha256')
        if cfg['context_variant']=='weight_permutation':
            if not re.fullmatch('[0-9a-f]{64}',str(parent)) or parent not in hashes.values():raise ValueError('Unbound placebo parent')
        elif parent is not None:raise ValueError('Unexpected context parent')
        if cfg.get('worker_address_space_limit_bytes',0)<=0 or cfg.get('host_storage_roots')!=['/mnt/c','/mnt/d']:raise ValueError('Resource binding missing')
        out=Path(cfg['output'])
        if out==cache or out in cache.parents or cache in out.parents:raise ValueError('Input/output overlap')
        if out==original or out in original.parents or original in out.parents:raise ValueError('Original input/output overlap')
    else:raise ValueError('Unknown context binding state')
    binding={k:cfg[k] for k in ('adaptation_schema','adaptation_route','input_id','context_variant',
        'kappa_rule','context_mass_rule','context_diagonal','context_bridges','permutation_seed','permutation_bins','promotion_schedule')}
    if 'numerical_solver' in cfg:binding['numerical_solver']=cfg['numerical_solver']
    return binding


def write_recipes(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for route,key in UNITS:
        cfg=recipe(route,key);validate(cfg);name=route+'--'+key+'.json'
        (directory/name).write_text(json.dumps(cfg,indent=2)+'\n')
        records.append(dict(recipe=name,phase=cfg['phase'],auto_start=False))
    plan=dict(schema=SCHEMA,initial_full_runs=2,conditional_transfer=3,conditional_combinations=4,
              maximum_full_runs=9,parameter_search=False,extra_kappa_points=False,auto_start=False,runs=records)
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan
