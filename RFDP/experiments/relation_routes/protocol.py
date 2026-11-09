"""Fixed Market-OSNet full-population protocol; recipes never launch work."""
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from experiments.continuous_fitting.f1_config import FROZEN_INPUTS

SCHEMA = 'relation-routes-full-v1'
ROUTES = ('camera_channel', 'csw_grouped', 'th_target')
FROZEN = FROZEN_INPUTS['market-osnet']
FIXED = dict(input_id='market-osnet', queries=3368, gallery=19732,
             alpha=5, beta=25, step=5, depth=500, k=20, gamma=.4,
             admission_policy='original', blas_threads=1, auto_start=False,
             retain_initial_state=True, trace_retention='sampled_after_readback',
             worker_policy='all_allocated_cpu',
             runner='experiments.relation_routes.full_run')
PARAMETERS = {
    'camera_channel': dict(source='camera_channel', relation_model='original'),
    'csw_grouped': dict(source='original', relation_model='csw_grouped',
                       k_intra=5, k_inter=20, rho=.5, support_lambda=.25, support_mode='ordinary'),
    'th_target': dict(source='th_target', relation_model='original', theta=.5),
}
REFERENCE_METHODS = ['I-HyRDP', 'CSW+I-HyRDP']


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def recipe(route):
    if route not in ROUTES:
        raise ValueError('Only the three active routes may enter the full queue')
    cfg = dict(FIXED, **PARAMETERS[route], relation_schema=SCHEMA, route=route,
               path_binding='required', run_role='candidate',
               reference_methods_reuse=REFERENCE_METHODS.copy(),
               expected_cache_input_hashes=copy.deepcopy(FROZEN['cache_input_hashes']),
               feature_sha256=FROZEN['feature_sha256'], graph_sha256=FROZEN['graph_sha256'],
               output_budget_bytes=3*2**30, worker_address_space_limit_bytes=int(3.5*2**30),
               retained_queries=np.linspace(0,3367,32,dtype=int).tolist())
    validate(cfg)
    return cfg


def validate(cfg, require_bound=False):
    if cfg.get('relation_schema') != SCHEMA or cfg.get('route') not in ROUTES:
        raise ValueError('Explicit active relation route/schema required')
    expected = dict(FIXED, **PARAMETERS[cfg['route']])
    for key, value in expected.items():
        if cfg.get(key) != value or (type(value) is int and type(cfg.get(key)) is not int):
            raise ValueError('Fixed full-run setting differs: '+key)
    if cfg.get('reference_methods_reuse') != REFERENCE_METHODS or cfg.get('run_role') != 'candidate':
        raise ValueError('Both baseline results must be reused, never scheduled')
    for key in ('feature_sha256','graph_sha256'):
        if cfg.get(key) != FROZEN[key]:raise ValueError('Frozen input identity differs: '+key)
    if cfg.get('expected_cache_input_hashes') != FROZEN['cache_input_hashes']:
        raise ValueError('Frozen cache hashes differ')
    if cfg.get('retained_queries') != np.linspace(0,3367,32,dtype=int).tolist():
        raise ValueError('Keep the existing 32 full traces after complete per-query readback')
    for key in ('output_budget_bytes','worker_address_space_limit_bytes'):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:raise ValueError('Missing resource budget')
    if cfg['route'] != 'csw_grouped' and any(k in cfg for k in ('rho','support_lambda','k_intra','k_inter')):
        raise ValueError('Do not silently combine camera graph and internal routes')
    if cfg['route'] != 'th_target' and 'theta' in cfg:
        raise ValueError('TH strength belongs only to the TH route')
    if cfg.get('path_binding') == 'required':
        if require_bound:raise ValueError('Prepare and SHA-bind runtime paths before execution')
        if any(k in cfg for k in ('commit','input_sha256','cache','mat','output','workers')):
            raise ValueError('Unbound recipe must not pretend runtime paths were checked')
    elif cfg.get('path_binding') == 'bound':
        if not re.fullmatch('[0-9a-f]{40}',str(cfg.get('commit',''))):raise ValueError('Execution commit required')
        if type(cfg.get('workers')) is not int or cfg['workers'] < 1:raise ValueError('Invalid allocated CPU count')
        paths = ['cache','mat','output','camera_ids_file']
        if cfg['route']=='csw_grouped':paths += ['grouped_graph_file','support_file','grouped_metadata_file']
        for key in paths:
            path=Path(cfg.get(key,''))
            if not path.is_absolute() or '..' in path.parts:raise ValueError('Absolute runtime path required: '+key)
        hashes=cfg.get('input_sha256',{})
        for name,value in hashes.items():
            if not Path(name).is_absolute() or not re.fullmatch('[0-9a-f]{64}',str(value)):
                raise ValueError('Invalid input SHA binding')
        for key in paths:
            if key not in ('cache','output') and cfg[key] not in hashes:raise ValueError('Unbound input: '+key)
        for name,value in FROZEN['cache_input_hashes'].items():
            if hashes.get(str(Path(cfg['cache'])/name)) != value:raise ValueError('Wrong frozen cache: '+name)
        if hashes[cfg['mat']] != FROZEN['feature_sha256']:raise ValueError('Wrong feature MAT')
        # A run may never overwrite inputs or an existing baseline tree.
        out=Path(cfg['output']);cache=Path(cfg['cache'])
        if out==cache or out in cache.parents or cache in out.parents:
            raise ValueError('Output overlaps frozen cache')
    else:raise ValueError('Unknown path binding state')
    return {key:cfg[key] for key in ('relation_schema','route','input_id','source','relation_model',
        'alpha','beta','step','depth','feature_sha256','graph_sha256')+tuple(
        k for k in PARAMETERS[cfg['route']] if k not in ('source','relation_model'))}
