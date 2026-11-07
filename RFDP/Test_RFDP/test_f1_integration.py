"""Local synthetic integration: saved scores, expansion, bridges and full tail."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from MODEL.Paper_RFDP import PaperFullIterative as model
from experiments.continuous_fitting import certified, f1, full as runner
from experiments.continuous_fitting.f1_config import bind_config, scientific_config
from experiments.continuous_fitting.verify import read_rows, verify_query
from Test_RFDP.test_f1_config import checked_base


class F1IntegrationTests(unittest.TestCase):
    def trajectory(self, source):
        rng = np.random.default_rng(2041)
        n = 17
        w = np.eye(n)
        # Two connected components force the existing full-affinity bridge rule.
        for ids in (range(8), range(8, n)):
            for i in ids:
                for j in ids:
                    if i < j:
                        w[i, j] = w[j, i] = rng.uniform(.05, .8)
        affinity = rng.uniform(.05, .95, (n, n))
        affinity = (affinity + affinity.T) / 2
        np.fill_diagonal(affinity, 1.)
        graph = sparse.csr_matrix(w)
        # Unit features are sufficient for deterministic completion of the tail;
        # the affinity is an independently supplied synthetic graph input.
        features = np.eye(n)
        cfg = dict(source=source, alpha=5., beta=25., depth=16, f1_nu=1.5,
                   f1_updates=1, trace_retention='all')
        old = runner.install_solver(cfg)
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                (out / 'queries').mkdir()
                (out / 'accepted').mkdir()
                with patch.multiple(runner, create=True, OUT=out, QN=1, GN=n-1,
                                    CFG=cfg, SIG='synthetic-f1', GRAPH=graph,
                                    FULL=affinity, X=features, INIT_ERROR=None,
                                    ADMISSION_CAMERAS=None, READBACK_KERNEL=None,
                                    READBACK_SUMS=None,
                                    ENGINE=model.FullIHyRDP(graph, affinity)):
                    row = runner.compute_query(0)
                    self.assertNotIn('error', row, row.get('error'))
                    # Acceptance must reconstruct the saved chain, not rerun it.
                    with patch.object(f1, 'solve', side_effect=AssertionError('readback solve')), \
                         patch.object(certified, 'solve', side_effect=AssertionError('readback solve')), \
                         patch.object(f1.WeightedBlock, 'solve', side_effect=AssertionError('readback refine')):
                        accepted = runner.accept_query(0)
                    self.assertNotIn('error', accepted, accepted.get('error'))
                    prefix = np.load(out / 'queries/0000.prefix.npy')
                    rank = np.load(out / 'queries/0000.rankings.npy')
                    rows = list(read_rows(out / 'queries/0000.trace.jsonl.gz'))
        finally:
            model.solve_hybrid_sparse = old
            certified.EVENTS.clear()
        args = (graph, affinity, features @ features[0], 0, prefix, rank, rows,
                source, 1, 16)
        return args, row['diagnostics'], accepted

    def test_both_candidates_preserve_dynamic_expansion_bridges_and_full_ranking(self):
        for source in f1.SOURCES:
            with self.subTest(source=source):
                args, diag, accepted = self.trajectory(source)
                self.assertGreater(accepted['bridge_events'], 0)
                self.assertGreater(accepted['no_solve_batches'], 0)
                self.assertGreater(accepted['solver_calls'], 1)
                self.assertEqual(accepted['linear_systems'], 2 * accepted['solver_calls'])
                self.assertEqual(accepted['linear_systems'], diag['linear_systems'])
                self.assertEqual(diag['numerical_tail_count'], 0)
                self.assertEqual(sorted(args[5]), list(range(1, 17)))
                self.assertEqual(len(args[4]), 16)
                for stage in args[6]:
                    if stage['event'] is not None:
                        self.assertEqual(stage['event']['updates'], 1)
                        self.assertEqual(len(stage['event']['initial_scores']),
                                         len(stage['unlabeled']))

    def test_saved_selection_and_algorithm_changes_are_rejected(self):
        args, _, _ = self.trajectory('f1_robust')
        damaged = copy.deepcopy(args[6])
        damaged[0]['selected'][0] = 16
        with self.assertRaises(AssertionError):
            verify_query(*args[:6], damaged, *args[7:])
        with self.assertRaises(AssertionError):
            verify_query(*args[:6], args[6][:-1], *args[7:])
        for options in (dict(f1_nu=2.), dict(f1_updates=2)):
            with self.subTest(options=options), self.assertRaises((AssertionError, ValueError)):
                verify_query(*args, **options)

    def test_legacy_original_still_uses_one_system_per_scoring_stage(self):
        args, diag, accepted = self.trajectory('original')
        self.assertEqual(accepted['linear_systems'], accepted['solver_calls'])
        self.assertEqual(diag['linear_systems'], diag['solver_calls'])
        self.assertEqual(verify_query(*args)['solver_calls'], diag['solver_calls'])

    def test_unbound_recipes_stop_before_any_runtime_or_data_access(self):
        root = Path(__file__).resolve().parents[2] / 'configs/f1'
        with patch.object(runner, 'sources', side_effect=AssertionError('data access')), \
             patch.object(runner, 'resources', side_effect=AssertionError('resource access')):
            for path in root.glob('*.json'):
                with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, 'unbound'):
                    runner.main('compute', path)

    def test_new_algorithm_binding_and_legacy_manifest_compatibility(self):
        configs = [bind_config(scientific_config('market-osnet', source),
                               checked_base('market-osnet'), output_root='/new/f1',
                               commit='d' * 40) for source in ('original', *f1.SOURCES)]
        bindings = [runner.fitting_binding(cfg) for cfg in configs]
        self.assertEqual(len({json.dumps(b, sort_keys=True) for b in bindings}), 3)
        for cfg in configs:
            runner.check_admission_manifest(cfg, {'config': copy.deepcopy(cfg)})
        with self.assertRaisesRegex(AssertionError, 'Fitting binding'):
            runner.check_admission_manifest(configs[1], {'config': configs[2]})
        runner.check_admission_manifest({'source': 'original'},
                                        {'config': {'source': 'original'}})


if __name__ == '__main__':
    unittest.main()
