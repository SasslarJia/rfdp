"""Offline recipes for converged fitting and the disabled diagonal-free reserve."""
import argparse
import copy
import json
from pathlib import Path
from . import f1_config as legacy

SCHEMA = 'converged-and-reserve-v1'
MAIN = ('f1_robust_converged', 'f1_uniform_converged')
BACKUP = 'diagonal_free'
SOURCES = MAIN + (BACKUP,)
FIXED = dict(alpha=5, beta=25, step=5, depth=500, k=20, gamma=.4,
             relation_model='original', admission_policy='original')


def recipe(input_id, source):
    if source not in SOURCES:
        raise ValueError('Unknown research source')
    cfg = legacy.scientific_config(input_id, 'original')
    for key in ('f1_config_schema', 'f1_input_id', 'f1_nu', 'f1_updates'):
        cfg.pop(key)
    cfg.update(research_schema=SCHEMA, input_id=input_id, source=source,
               run_role='reserve' if source == BACKUP else 'candidate',
               enabled=source != BACKUP, retain_initial_state=True,
               solver_mode='diagonal_free' if source == BACKUP else 'nonlinear_converged')
    if source in MAIN:
        cfg.update(nu=1.5, relative_scale=1., score_tolerance=1e-8)
    validate(cfg)
    return cfg


def _legacy_view(cfg):
    view = copy.deepcopy(cfg)
    view.update(source='original', f1_config_schema=legacy.SCHEMA,
                f1_input_id=cfg['input_id'], f1_nu=1.5, f1_updates=1,
                run_role='reference_reuse')
    return view


def validate(cfg, require_bound=False, require_enabled=False):
    if 'research_schema' not in cfg and cfg.get('source') not in SOURCES:
        return {}
    if cfg.get('research_schema') != SCHEMA or cfg.get('source') not in SOURCES:
        raise ValueError('New research source requires its own schema')
    source = cfg['source']
    if cfg.get('input_id') not in legacy.FROZEN_INPUTS:
        raise ValueError('Unknown official input')
    expected = dict(FIXED, solver_mode='diagonal_free' if source == BACKUP else 'nonlinear_converged',
                    run_role='reserve' if source == BACKUP else 'candidate', retain_initial_state=True)
    if source in MAIN:
        expected.update(nu=1.5, relative_scale=1., score_tolerance=1e-8)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Frozen research setting differs: ' + key)
    if any(key in cfg for key in ('f1_config_schema','f1_updates','f1_nu')):
        raise ValueError('Legacy one-update fields cannot identify a converged run')
    if type(cfg.get('enabled')) is not bool:
        raise ValueError('Explicit enabled boolean required')
    if require_enabled and not cfg['enabled']:
        raise ValueError('Reserve is disabled; explicitly enable only after the main results are recorded')
    legacy.validate_f1_config(_legacy_view(cfg), require_bound=require_bound)
    return dict(research_schema=SCHEMA, input_id=cfg['input_id'], source=source,
                **{k:v for k,v in expected.items() if k not in ('run_role','retain_initial_state')})


def bind(cfg, base, *, output_root, commit, enable_backup=False):
    validate(cfg)
    old = legacy.bind_config(_legacy_view(cfg), base, output_root=output_root, commit=commit)
    result = copy.deepcopy(cfg)
    for key in legacy.RUNTIME_FIELDS:
        if key in old:
            result[key] = old[key]
    output = Path(output_root) / cfg['input_id'] / cfg['source']
    previous = Path(base['output'])
    if output == previous or output in previous.parents or previous in output.parents:
        raise ValueError('New output overlaps existing results')
    result.update(path_binding='bound', binding_source=old['binding_source'],
                  cache_hash_binding=old['cache_hash_binding'], commit=commit, output=str(output))
    if cfg['source'] == BACKUP:
        result['enabled'] = bool(enable_backup)
    validate(result, require_bound=True)
    return result


def write(configs, directory):
    directory = Path(directory)
    if any((directory/name).exists() for name in configs):
        raise ValueError('Refuse to overwrite existing configurations')
    directory.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        validate(cfg)
        (directory/name).write_text(json.dumps(cfg, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','bind'))
    parser.add_argument('--group', choices=('main','backup'), required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--input-id', choices=tuple(legacy.FROZEN_INPUTS))
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--output-root')
    parser.add_argument('--commit')
    parser.add_argument('--enable-backup', action='store_true')
    args = parser.parse_args()
    sources = MAIN if args.group == 'main' else (BACKUP,)
    ids = [args.input_id] if args.input_id else list(legacy.FROZEN_INPUTS)
    configs = {i+'.'+s+'.json': recipe(i,s) for i in ids for s in sources}
    if args.action == 'bind':
        if not all((args.input_id,args.base_config,args.output_root,args.commit)):
            parser.error('bind requires input-id, base-config, output-root and commit')
        base = json.loads(args.base_config.read_text())
        configs = {name: bind(cfg,base,output_root=args.output_root,commit=args.commit,
                             enable_backup=args.enable_backup) for name,cfg in configs.items()}
    elif args.enable_backup:
        parser.error('Preparation cannot enable the reserve')
    write(configs,args.destination)
    print(json.dumps(dict(configurations=len(configs),execution_started=False)))


if __name__ == '__main__':
    main()
