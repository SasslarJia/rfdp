"""F1 configuration identity/binding tests; no feature, graph, or solver access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.continuous_fitting.f1_config import (
    FIXED, FROZEN_INPUTS, SOURCES, bind_config, prepare_configs, scientific_config,
    validate_f1_config, write_configs)


def checked_base(input_id):
    frozen = FROZEN_INPUTS[input_id]
    cache = '/checked/cache/' + input_id
    mat = '/checked/features/' + input_id + '.mat'
    baseline = '/checked/baselines/' + input_id + '.json'
    hashes = {cache + '/' + name: frozen['cache_input_hashes'].get(name, 'a' * 64)
              for name in ('features.npy', 'affinity.npy', 'graph.npz', 'manifest.json')}
    hashes.update({mat: frozen['feature_sha256'], baseline: 'b' * 64})
    base = dict(output='/checked/old/' + input_id, cache=cache, mat=mat,
                baseline_queries=baseline, baseline_summary=baseline,
                source='original', commit='c' * 40, workers=16, blas_threads=1,
                output_budget_bytes=17179869184,
                worker_address_space_limit_bytes=3758096384,
                input_sha256=hashes, trace_retention='all', **FIXED)
    base.update({key: frozen[key] for key in ('queries', 'gallery', 'feature_sha256', 'graph_sha256')})
    return base


class F1ConfigTests(unittest.TestCase):
    def test_twelve_committed_scientific_combinations_are_concrete_and_unbound(self):
        root = Path(__file__).resolve().parents[2] / 'configs' / 'f1'
        files = sorted(root.glob('*.json'))
        self.assertEqual(len(files), 12)
        seen = set()
        for path in files:
            cfg = json.loads(path.read_text())
            seen.add((cfg['f1_input_id'], cfg['source']))
            self.assertEqual(cfg, scientific_config(cfg['f1_input_id'], cfg['source']))
            self.assertEqual(validate_f1_config(cfg)['f1_updates'], 1)
            with self.assertRaisesRegex(ValueError, 'unbound'):
                validate_f1_config(cfg, require_bound=True)
        self.assertEqual(seen, {(key, source) for key in FROZEN_INPUTS for source in SOURCES})

    def test_binding_preserves_checked_inputs_resources_and_never_reuses_old_commit(self):
        for input_id in FROZEN_INPUTS:
            base = checked_base(input_id)
            untouched = copy.deepcopy(base)
            outputs = set()
            for source in SOURCES:
                cfg = bind_config(scientific_config(input_id, source), base,
                                  output_root='/new/f1', commit='d' * 40)
                outputs.add(cfg['output'])
                self.assertEqual(cfg['input_sha256'], base['input_sha256'])
                for key in ('workers', 'blas_threads', 'output_budget_bytes',
                            'worker_address_space_limit_bytes', 'trace_retention'):
                    self.assertEqual(cfg[key], base[key])
                self.assertEqual(cfg['commit'], 'd' * 40)
                self.assertFalse(cfg['auto_start'])
                self.assertEqual(cfg['run_role'], 'reference_reuse' if source == 'original' else 'candidate')
                self.assertEqual(validate_f1_config(cfg, require_bound=True)['source'], source)
            self.assertEqual(len(outputs), 3)
            self.assertEqual(base, untouched)

    def test_wrong_backbone_or_cache_bytes_cannot_be_rebound(self):
        recipe = scientific_config('market-osnet', 'f1_robust')
        base = checked_base('market-transreid')
        with self.assertRaisesRegex(ValueError, 'Base input differs'):
            bind_config(recipe, base, output_root='/new/f1', commit='d' * 40)
        base = checked_base('market-osnet')
        base['input_sha256'][base['cache'] + '/graph.npz'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'Cache SHA256 mismatch'):
            bind_config(recipe, base, output_root='/new/f1', commit='d' * 40)

    def test_duke_osnet_missing_cache_hashes_remain_explicit_until_checked_base_binding(self):
        recipe = scientific_config('duke-osnet', 'f1_uniform')
        self.assertEqual(recipe['expected_cache_input_hashes'], {})
        self.assertEqual(recipe['cache_hash_binding'], 'checked_base_required')
        base = checked_base('duke-osnet')
        cfg = bind_config(recipe, base, output_root='/new/f1', commit='d' * 40)
        self.assertEqual(cfg['cache_hash_binding'], 'checked_base_config')
        del base['input_sha256'][base['cache'] + '/affinity.npy']
        with self.assertRaisesRegex(ValueError, 'Missing checked-base SHA256'):
            bind_config(recipe, base, output_root='/new/f1', commit='d' * 40)

    def test_fixed_algorithm_and_original_relation_cannot_drift(self):
        for key, value in [('f1_nu', 2), ('f1_updates', 3), ('alpha', 4),
                           ('relation_model', 'csw'), ('admission_policy', 'camera_gate')]:
            cfg = scientific_config('duke-transreid', 'f1_robust')
            cfg[key] = value
            with self.assertRaisesRegex(ValueError, 'Fixed F1 setting mismatch'):
                validate_f1_config(cfg)
        base = checked_base('market-osnet')
        base['admission_policy'] = 'camera_gate'
        with self.assertRaisesRegex(ValueError, 'original graph and admission'):
            bind_config(scientific_config('market-osnet', 'f1_uniform'), base,
                        output_root='/new/f1', commit='d' * 40)

    def test_explicit_commit_and_full_existing_hash_bindings_are_required(self):
        recipe = scientific_config('market-osnet', 'original')
        base = checked_base('market-osnet')
        for commit in (None, 'test', 'REPLACE_WITH_EXECUTION_GIT_COMMIT'):
            with self.assertRaisesRegex(ValueError, 'execution commit'):
                bind_config(recipe, base, output_root='/new/f1', commit=commit)
        del base['input_sha256'][base['mat']]
        with self.assertRaisesRegex(ValueError, 'Missing checked-base SHA256'):
            bind_config(recipe, base, output_root='/new/f1', commit='d' * 40)

    def test_legacy_baselines_are_compatible_but_f1_requires_schema(self):
        self.assertEqual(validate_f1_config({'source': 'original'}, require_bound=True), {})
        for source in ('f1_uniform', 'f1_robust'):
            with self.assertRaisesRegex(ValueError, 'requires f1_config_schema'):
                validate_f1_config({'source': source}, require_bound=True)

    def test_prepare_checks_evidence_and_write_only_creates_config_files(self):
        index = dict(frozen_inputs={key: dict(
            feature_sha256=value['feature_sha256'], graph_sha256=value['graph_sha256'],
            cache_input_hashes=copy.deepcopy(value['cache_input_hashes']),
            parameters={name: FIXED[name] for name in ('alpha', 'beta', 'step', 'depth', 'k', 'gamma')})
            for key, value in FROZEN_INPUTS.items()})
        configs = prepare_configs(index)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'configs'
            write_configs(configs, target)
            self.assertEqual({p.suffix for p in target.iterdir()}, {'.json'})
            self.assertEqual(len(list(target.iterdir())), 12)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                write_configs(configs, target)
        index['frozen_inputs']['duke-osnet']['feature_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Evidence index differs'):
            prepare_configs(index)


if __name__ == '__main__':
    unittest.main()
