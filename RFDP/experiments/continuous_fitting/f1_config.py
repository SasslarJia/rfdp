"""Prepare and bind F1 configurations locally; this module never starts a run."""
import argparse
import copy
import json
import re
from pathlib import Path, PurePosixPath

SCHEMA = 'f1-scientific-v1'
SOURCES = ('original', 'f1_uniform', 'f1_robust')
RUNNER = 'experiments.continuous_fitting.full'
FIXED = dict(alpha=5, beta=25, step=5, depth=500, k=20, gamma=.4,
             relation_model='original', admission_policy='original',
             f1_nu=1.5, f1_updates=1)
# Copied from research/evidence/输入与归档索引.json, not computed from new data.
# graph_sha256 identifies the archived run graph, not cache/graph.npz's file SHA.
FROZEN_INPUTS = {
    'market-osnet': dict(
        queries=3368, gallery=19732,
        feature_sha256='9f603d7149c99e5f4ba44934829d03d3fa1e25f737924c9bb6fa81f37c2d9c31',
        graph_sha256='39057cf989217c8404da364fb0e58121249c107789daecdfc1ab5eed754555f9',
        cache_input_hashes={
            'features.npy': '8b290bcd696e0925dc627382f8f74630ec1cbe0a8186d63e3cc11945013f3aa1',
            'affinity.npy': '6a1ae9a25cccff26997877431f72055c23445aa0b898a745efffbfe7a6ba43da',
            'graph.npz': '3464ff8e0b6dc0d821fc7ef92366066d486bcd1588a427eb4787a38a41ca0617'}),
    'market-transreid': dict(
        queries=3368, gallery=19732,
        feature_sha256='ed8925dc55a02818304dcd89b40ec3a21ed7d23a5d56c62a9279b826c4a25967',
        graph_sha256='5755f6f586cec6d5883082e81516625ac9f5955dbeeb3534a3a539c229a16335',
        cache_input_hashes={
            'features.npy': 'ce153531fa2c1908e3210b5f2b60281ad18064ff41f7f91a23c03360758176fb',
            'affinity.npy': 'd3dd1f9983b20e994dbb64538c432a5b1a113ee86b59fb899ec8f8ab96805bd4',
            'graph.npz': '82ce6eb3deae872020308395b85b5d9fe779eed5990fe94906a64b3254f2848c'}),
    'duke-osnet': dict(
        queries=2228, gallery=17661,
        feature_sha256='be2c3de789facfbca13e603bec674441ea5a16d5ba5f5f31f181135a7b5a197e',
        graph_sha256='7b4d632fe3f50400263b3306f5195350b3de72d0cad15c94c5bf9f24164cefb0',
        cache_input_hashes={}),
    'duke-transreid': dict(
        queries=2228, gallery=17661,
        feature_sha256='fdf0edc6e826bd83160244f9926537428624619231cab17f03a83d7a2a39fa54',
        graph_sha256='6f619617852ec278c890f3d8a4b79373be35a67be6792b8c5da89e72c29a8e1a',
        cache_input_hashes={
            'features.npy': '40c6b383ec4e2ecefc4bc65546f564f73d7318b0017e424e6b7015db859ffd8d',
            'affinity.npy': '6b65f7f7b6d132aa0a29f9853c72902821aab7379ada71e899d93a7d2e3dd6ae',
            'graph.npz': '3aff78f678e9c58557e8b405be62afed879ea82834bff5660e9bb026976dfbf5'}),
}
CACHE_FILES = ('features.npy', 'affinity.npy', 'graph.npz', 'manifest.json')
RUNTIME_FIELDS = (
    'cache', 'mat', 'baseline_queries', 'baseline_summary', 'workers',
    'blas_threads', 'output_budget_bytes', 'worker_address_space_limit_bytes',
    'input_sha256', 'trace_retention', 'retained_queries', 'comparisons')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _absolute(value, name):
    _require(isinstance(value, str) and value.startswith('/') and
             '..' not in PurePosixPath(value).parts, name + ' must be an absolute path')
    return PurePosixPath(value)


def _validate_inputs(cfg, frozen):
    for key in ('queries', 'gallery', 'feature_sha256', 'graph_sha256'):
        _require(cfg.get(key) == frozen[key], 'Frozen input mismatch: ' + key)
    _require(cfg.get('expected_cache_input_hashes') == frozen['cache_input_hashes'],
             'Frozen cache declarations differ')


def _validate_bound(cfg, frozen):
    _require(_digest(cfg.get('commit'), 40), 'An explicit execution commit is required')
    paths = {key: _absolute(cfg.get(key), key) for key in
             ('cache', 'mat', 'output', 'baseline_queries', 'baseline_summary')}
    _require(cfg.get('workers') == 16 and cfg.get('blas_threads') == 1,
             'Keep the existing 16-worker, one-BLAS-thread resource line')
    for key in ('output_budget_bytes', 'worker_address_space_limit_bytes'):
        _require(type(cfg.get(key)) is int and cfg[key] > 0, 'Missing resource line: ' + key)
    hashes = cfg.get('input_sha256')
    _require(isinstance(hashes, dict) and bool(hashes), 'Existing input SHA bindings are required')
    for path, digest in hashes.items():
        _absolute(path, 'input_sha256 key')
        _require(_digest(digest), 'Invalid input SHA256: ' + path)
    for path in [paths['mat'], paths['baseline_queries'], paths['baseline_summary']] + [
            paths['cache'] / name for name in CACHE_FILES]:
        _require(str(path) in hashes, 'Missing checked-base SHA256: ' + str(path))
    _require(hashes[str(paths['mat'])] == frozen['feature_sha256'], 'MAT SHA256 mismatch')
    for name, expected in frozen['cache_input_hashes'].items():
        _require(hashes[str(paths['cache'] / name)] == expected, 'Cache SHA256 mismatch: ' + name)
    origin = 'frozen_index' if frozen['cache_input_hashes'] else 'checked_base_config'
    _require(cfg.get('cache_hash_binding') == origin, 'Cache SHA provenance is not bound')
    _require(cfg.get('binding_source') == 'checked_base_config', 'Missing checked base provenance')
    for ref in cfg.get('comparisons', []):
        _require(isinstance(ref, dict), 'Invalid comparison binding')
        for key in ('queries', 'summary'):
            _require(ref.get(key) in hashes, 'Comparison path is not SHA-bound')


def validate_f1_config(cfg, require_bound=False):
    """Validate an F1 recipe/run config and return its scientific signature fields.

    Legacy sources without this schema retain their existing runner validation.
    Neither this function nor binding opens data files or verifies remote paths.
    """
    _require(isinstance(cfg, dict), 'Configuration must be a JSON object')
    source = cfg.get('source')
    schema = cfg.get('f1_config_schema')
    if schema is None and source not in ('f1_uniform', 'f1_robust'):
        return {}
    _require(schema == SCHEMA, 'F1 source requires f1_config_schema=' + SCHEMA)
    _require(source in SOURCES, 'Unknown F1 comparison source')
    input_id = cfg.get('f1_input_id')
    _require(input_id in FROZEN_INPUTS, 'Unknown frozen input')
    frozen = FROZEN_INPUTS[input_id]
    for key, expected in FIXED.items():
        _require(type(cfg.get(key)) is not bool and cfg.get(key) == expected,
                 'Fixed F1 setting mismatch: ' + key)
    _validate_inputs(cfg, frozen)
    _require(cfg.get('runner') == RUNNER, 'Use the continuous_fitting full runner')
    _require(cfg.get('auto_start') is False, 'F1 configuration preparation never auto-starts')
    role = 'reference_reuse' if source == 'original' else 'candidate'
    _require(cfg.get('run_role') == role, 'Incorrect baseline/candidate role')
    state = cfg.get('path_binding')
    _require(state in ('required', 'bound'), 'Unknown path binding state')
    if state == 'required':
        _require(not require_bound, 'F1 paths are unbound; bind a checked same-input base config first')
        _require(not any(key in cfg for key in ('cache', 'mat', 'output', 'commit', 'input_sha256')),
                 'Unbound recipes must not contain purported runtime paths or hashes')
        origin = 'frozen_index' if frozen['cache_input_hashes'] else 'checked_base_required'
        _require(cfg.get('cache_hash_binding') == origin, 'Incorrect unbound cache provenance')
    else:
        _validate_bound(cfg, frozen)
    return dict(f1_config_schema=SCHEMA, f1_input_id=input_id, source=source,
                relation_model='original', admission_policy='original',
                f1_nu=1.5, f1_updates=1)


def scientific_config(input_id, source):
    """Return one concrete path-unbound recipe using the frozen evidence index."""
    _require(input_id in FROZEN_INPUTS and source in SOURCES, 'Unknown input/source combination')
    frozen = FROZEN_INPUTS[input_id]
    cfg = dict(f1_config_schema=SCHEMA, f1_input_id=input_id, source=source,
               path_binding='required', runner=RUNNER, auto_start=False,
               run_role='reference_reuse' if source == 'original' else 'candidate',
               cache_hash_binding='frozen_index' if frozen['cache_input_hashes'] else 'checked_base_required',
               expected_cache_input_hashes=copy.deepcopy(frozen['cache_input_hashes']),
               **FIXED)
    cfg.update({key: frozen[key] for key in ('queries', 'gallery', 'feature_sha256', 'graph_sha256')})
    validate_f1_config(cfg)
    return cfg


def prepare_configs(input_index):
    """Cross-check the supplied small evidence index and build all twelve recipes."""
    records = input_index.get('frozen_inputs', {})
    _require(set(records) == set(FROZEN_INPUTS), 'Expected the four official frozen inputs')
    for input_id, frozen in FROZEN_INPUTS.items():
        record = records[input_id]
        for key in ('feature_sha256', 'graph_sha256', 'cache_input_hashes'):
            _require(record.get(key) == frozen[key], 'Evidence index differs: ' + input_id + '/' + key)
        _require(record.get('parameters') == {key: FIXED[key] for key in
                                              ('alpha', 'beta', 'step', 'depth', 'k', 'gamma')},
                 'Evidence parameters differ: ' + input_id)
    return {input_id + '.' + source + '.json': scientific_config(input_id, source)
            for input_id in FROZEN_INPUTS for source in SOURCES}


def bind_config(recipe, base, *, output_root, commit):
    """Bind a checked same-input base without reading or re-hashing any data.

    For Duke-OSNet, missing cache hashes come only from that checked base and
    remain marked as such. A fresh file hash is never substituted as evidence.
    The caller supplies the execution commit explicitly; the base commit is not
    inherited. Only existing runtime fields and SHA/resource bindings are kept.
    """
    validate_f1_config(recipe)
    _require(recipe.get('path_binding') == 'required', 'Bind an unbound scientific recipe')
    _require(isinstance(base, dict), 'Base configuration must be an object')
    frozen = FROZEN_INPUTS[recipe['f1_input_id']]
    for key in ('queries', 'gallery', 'feature_sha256', 'graph_sha256'):
        _require(base.get(key) == frozen[key], 'Base input differs: ' + key)
    for key in ('alpha', 'beta', 'step', 'depth', 'k', 'gamma'):
        _require(base.get(key) == FIXED[key], 'Base scientific setting differs: ' + key)
    _require(base.get('relation_model', 'original') == 'original' and
             base.get('admission_policy', 'original') == 'original',
             'Bind only the original graph and admission policy')
    root = _absolute(str(output_root), 'output_root')
    output = root / recipe['f1_input_id'] / recipe['source']
    previous = _absolute(base.get('output'), 'base output')
    _require(output != previous and previous not in output.parents and output not in previous.parents,
             'New output must not replace or contain the base run')
    cfg = copy.deepcopy(recipe)
    cfg.update({key: copy.deepcopy(base[key]) for key in RUNTIME_FIELDS if key in base})
    cfg.update(path_binding='bound', binding_source='checked_base_config',
               cache_hash_binding='frozen_index' if frozen['cache_input_hashes'] else 'checked_base_config',
               commit=commit, output=str(output))
    validate_f1_config(cfg, require_bound=True)
    return cfg


def write_configs(configs, destination):
    """Write JSON configuration artifacts only, never overwrite an existing one."""
    destination = Path(destination)
    for name, cfg in configs.items():
        _require(Path(name).name == name, 'Configuration filename must be a basename')
        validate_f1_config(cfg)
        _require(not (destination / name).exists(), 'Configuration already exists: ' + name)
    destination.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        with (destination / name).open('x', encoding='utf-8') as stream:
            json.dump(cfg, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare', help='Write twelve unbound science configurations')
    prepare.add_argument('--input-index', type=Path, required=True)
    prepare.add_argument('--destination', type=Path, required=True)
    bind = sub.add_parser('bind', help='Bind three recipes to a checked same-input base; do not run')
    bind.add_argument('--input-id', choices=tuple(FROZEN_INPUTS), required=True)
    bind.add_argument('--base-config', type=Path, required=True)
    bind.add_argument('--output-root', required=True, help='New absolute run root on the execution host')
    bind.add_argument('--commit', required=True, help='Explicit 40-character execution commit, not the historical commit')
    bind.add_argument('--destination', type=Path, required=True, help='Local directory for three bound JSON files')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        configs = prepare_configs(json.loads(args.input_index.read_text()))
    else:
        base = json.loads(args.base_config.read_text())
        configs = {args.input_id + '.' + source + '.json':
                   bind_config(scientific_config(args.input_id, source), base,
                               output_root=args.output_root, commit=args.commit)
                   for source in SOURCES}
    write_configs(configs, args.destination)
    print(json.dumps(dict(configurations=len(configs), destination=str(args.destination),
                          execution_started=False)))


if __name__ == '__main__':
    main()
