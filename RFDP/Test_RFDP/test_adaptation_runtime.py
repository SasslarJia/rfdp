"""Synthetic trajectory, schedule and independent boundary read-back checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
from experiments.continuous_fitting import certified
from experiments.continuous_fitting.original_solver import WarmOriginalSolver
from experiments.continuous_fitting.verify import verify_query
from experiments.reid_adaptation import runtime
from experiments.reid_adaptation.schedule import promotion_step


class AdaptationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        certified.EVENTS.clear()

    def tearDown(self):
        certified.EVENTS.clear()
        self.temp.cleanup()

    def fixture(self, sizes=(8, 7, 6)):
        rng = np.random.default_rng(152)
        n = sum(sizes)
        affinity = rng.uniform(.05, .9, (n, n))
        affinity = (affinity + affinity.T) / 2
        np.fill_diagonal(affinity, 1.)
        graph = np.zeros((n, n))
        start = 0
        for size in sizes:
            stop = start + size
            graph[start:stop, start:stop] = affinity[start:stop, start:stop]
            start = stop
        cameras = np.arange(n) % 3
        path = self.root / 'cameras.npy'
        np.save(path, cameras)
        return sparse.csr_matrix(graph), affinity, cameras, path

    def config(self, route, path, rho=.5):
        schedule = ('early5' if route in ('early5', 'p1_early5') else
                    'late5' if route == 'late5' else 'fixed5')
        return dict(adaptation_route=route, source='original', promotion_schedule=schedule,
                    camera_ids_file=str(path), boundary_rho=rho if route == 'camera_boundary' else 1.,
                    alpha=5., beta=25., step=5, depth=500)

    def capture(self, engine, affinity, query=0, depth=None):
        certified.EVENTS.clear()
        rows = []
        seen = 0
        def trace(record):
            nonlocal seen
            row = copy.deepcopy(record)
            values = row.pop('scores')
            row['returned_scores'] = None if values is None else [certified.packed(v) for v in values]
            if values is None:
                self.assertEqual(len(certified.EVENTS), seen)
                row['event'] = None
            else:
                self.assertEqual(len(certified.EVENTS), seen + 1)
                row['event'] = copy.deepcopy(certified.EVENTS[seen])
                seen += 1
            rows.append(json.loads(json.dumps(row, allow_nan=False)))
        depth = len(affinity) if depth is None else depth
        prefix, diag = engine.rank([query], nq=depth, input_similarity=affinity[query], trace=trace)
        used = set(prefix.tolist())
        rank = [int(i) for i in prefix if i >= 2]
        rank += sorted((i for i in range(2, len(affinity)) if i not in used),
                       key=lambda i: (-float(affinity[query, i]), i))
        return prefix, rank, rows, diag

    def verify(self, cfg, graph, affinity, output, query=0):
        prefix, rank, rows, _ = output
        engine, read_graph, kernel, sums, certifier = runtime.initialize(cfg, graph, affinity, compute=False)
        self.assertIsNone(engine)
        self.assertIsNone(kernel)
        return verify_query(read_graph, affinity, affinity[query], query, prefix, rank, rows,
                            'original', 2, len(prefix), full_sums=sums, stage_certifier=certifier,
                            promotion_schedule=cfg['promotion_schedule'])

    def test_schedule_boundaries_and_late_cap(self):
        self.assertEqual([promotion_step('early5', i) for i in range(7)], [1]*5 + [5]*2)
        self.assertEqual([promotion_step('late5', i) for i in (95, 97, 98, 99, 100, 104, 105)],
                         [5, 3, 2, 1, 1, 1, 5])
        for bad in ('early10', 'late1'):
            with self.assertRaises(ValueError):
                promotion_step(bad, 0)

    def test_early_small_frontier_solves_and_fixed5_does_not(self):
        graph, affinity, _, path = self.fixture((4,))
        outputs = {}
        for route in ('p1_graph', 'early5'):
            cfg = self.config(route, path)
            engine, *_ = runtime.initialize(cfg, graph, affinity)
            outputs[route] = self.capture(engine, affinity)
            self.verify(cfg, graph, affinity, outputs[route])
        self.assertIsNone(outputs['p1_graph'][2][0]['returned_scores'])
        early = outputs['early5'][2][0]
        self.assertIsNotNone(early['returned_scores'])
        self.assertEqual(len(early['event']['returned_prefix_local']), 3)
        self.assertEqual(len(early['selected']), 1)
        altered = copy.deepcopy(outputs['p1_graph'])
        altered[2][0]['promotion_step'] = 1
        with self.assertRaises(AssertionError):
            self.verify(self.config('p1_graph', path), graph, affinity, altered)

    def test_bridge_budget_stays_five_during_single_admissions(self):
        graph, affinity, _, path = self.fixture((1,)*9)
        cfg = self.config('early5', path)
        engine, *_ = runtime.initialize(cfg, graph, affinity)
        out = self.capture(engine, affinity, depth=8)
        self.assertEqual(out[2][0]['promotion_step'], 1)
        self.assertEqual(len(out[2][0]['bridges']), 5)
        self.assertEqual(len(out[2][0]['selected']), 1)
        self.verify(cfg, graph, affinity, out)

    def test_complete_synthetic_routes_and_query_resets(self):
        graph, affinity, _, path = self.fixture()
        for route in ('p1_graph', 'early5', 'p1_early5', 'camera_boundary'):
            cfg = self.config(route, path)
            engine, returned, kernel, sums, _ = runtime.initialize(cfg, graph, affinity)
            self.assertEqual((returned - graph).nnz, 0)
            self.assertIsNone(kernel)
            np.testing.assert_array_equal(sums, affinity.sum(axis=1, dtype=np.float64))
            for query in (0, 1):
                out = self.capture(engine, affinity, query=query)
                checked = self.verify(cfg, graph, affinity, out, query=query)
                self.assertGreater(checked['bridge_events'], 0)
                solved = [row for row in out[2] if row['event'] is not None]
                self.assertEqual(solved[0]['event']['warm_reused_nodes'], 0)
                self.assertEqual(solved[0]['event']['adaptation_route'], route)

    def test_late_window_real_trajectory_and_readback(self):
        graph, affinity, _, path = self.fixture((113,))
        cfg = self.config('late5', path)
        engine, *_ = runtime.initialize(cfg, graph, affinity)
        out = self.capture(engine, affinity, depth=111)
        singles = [len(t['labeled'])-1 for t in out[2] if t['promotion_step'] == 1]
        self.assertEqual(singles, [100, 101, 102, 103, 104])
        self.verify(cfg, graph, affinity, out)

    def test_rho_one_matches_original_complete_trajectory(self):
        graph, affinity, _, path = self.fixture()
        baseline = FullIHyRDP(graph, affinity, stage_solver=WarmOriginalSolver())
        old = self.capture(baseline, affinity)
        cfg = self.config('camera_boundary', path, rho=1.)
        engine, *_ = runtime.initialize(cfg, graph, affinity)
        new = self.capture(engine, affinity)
        np.testing.assert_array_equal(old[0], new[0])
        self.assertEqual(old[1], new[1])
        for before, after in zip(old[2], new[2]):
            for key in ('selected', 'bridges', 'unlabeled', 'returned_scores'):
                self.assertEqual(before[key], after[key])
        self.verify(cfg, graph, affinity, new)

    def test_camera_formula_accumulation_reader_and_temporary_bridges(self):
        graph, _, cameras, _ = self.fixture((9,))
        acc = runtime.CameraBoundaryAccumulator(graph, cameras, .5)
        ids = np.array([2, 3, 4, 5, 6, 7, 8])
        targets = (1, 0)
        bridges = ((8, 1, .125),)
        acc.stage(ids, targets[:1], ())
        actual = acc.stage(ids, targets, bridges)
        expected = np.zeros(len(ids))
        for target in targets:
            expected += graph[ids, target].toarray().ravel() * np.where(cameras[ids] == cameras[target], .5, 1.)
        expected[-1] += .125 * (.5 if cameras[8] == cameras[1] else 1.)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, runtime.rebuild_camera_boundary(graph, cameras, .5, ids, targets, bridges))
        np.testing.assert_array_equal(acc.stage(ids, targets, ()), runtime.rebuild_camera_boundary(graph, cameras, .5, ids, targets, ()))
        with self.assertRaises(ValueError):
            acc.stage(ids, (0,), ())

    def test_camera_scores_match_independently_assembled_original_formula(self):
        graph, affinity, cameras, path = self.fixture((9,))
        ids = np.arange(2, 9)
        targets = (0, 1)
        local = graph[ids][:, ids]
        raw_e = np.asarray(graph[list(targets)].sum(axis=0)).ravel()[ids]
        expected_e = np.zeros(len(ids))
        for k in targets:
            expected_e += np.asarray(graph[ids, k].toarray()).ravel() * np.where(cameras[ids] == cameras[k], .5, 1.)
        h = np.asarray(local.sum(axis=1)).ravel()
        p = expected_e / (h + expected_e)
        dense = np.diag(1 + 5*p) - local.toarray() / np.sqrt(h[:, None]*h[None, :])
        expected = np.linalg.solve(dense, np.where(expected_e > 0, 5., 25.))
        solver = runtime.AdaptationSolver('camera_boundary', 'fixed5', graph, cameras, .5)
        values, _ = solver(local, raw_e, 5., 25., diagnostics={}, indices=ids, targets=targets, bridges=())
        np.testing.assert_allclose(np.asarray(values, dtype=float), expected, rtol=1e-11, atol=0.)
        _, _, _, _, reader = runtime.initialize(self.config('camera_boundary', path), graph, affinity, compute=False)
        result = reader(local, raw_e, values, certified.EVENTS[-1], alpha=5., beta=25., source='original',
                        indices=ids, targets=targets, bridges=())
        self.assertEqual(result['prefix'], certified.EVENTS[-1]['returned_prefix_local'])

    def test_reader_rejects_boundary_schedule_and_score_tampering(self):
        graph, affinity, _, path = self.fixture((10,))
        cfg = self.config('camera_boundary', path)
        engine, *_ = runtime.initialize(cfg, graph, affinity)
        original = self.capture(engine, affinity)
        # Reader never consults the producer's incremental boundary cache.
        with patch.object(runtime.CameraBoundaryAccumulator, 'stage', side_effect=AssertionError('producer used')):
            self.verify(cfg, graph, affinity, original)
        for field, value in (('fitting_boundary_sha256', '0'*64), ('fitting_rho', .75),
                             ('adaptation_route', 'p1_graph')):
            altered = copy.deepcopy(original)
            altered[2][0]['event'][field] = value
            with self.assertRaises(AssertionError):
                self.verify(cfg, graph, affinity, altered)
        altered = copy.deepcopy(original)
        scores = certified.unpack(altered[2][0]['returned_scores'])
        scores[0] *= 2
        altered[2][0]['returned_scores'] = [certified.packed(v) for v in scores]
        altered[2][0]['event']['returned_scores_sha256'] = certified.digest_scores(scores)
        with self.assertRaises((AssertionError, np.linalg.LinAlgError)):
            self.verify(cfg, graph, affinity, altered)
        early_cfg = self.config('early5', path)
        engine, *_ = runtime.initialize(early_cfg, graph, affinity)
        altered = self.capture(engine, affinity)
        altered[2][0]['promotion_step'] = 5
        with self.assertRaises(AssertionError):
            self.verify(early_cfg, graph, affinity, altered)


if __name__ == '__main__':
    unittest.main()
