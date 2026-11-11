"""Tests of paired timing mechanics; no frozen ReID input or benchmark is run."""
import contextlib
import copy
import gzip
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse

from experiments.continuous_fitting import solver_timing as timing


def fixture_row():
    return dict(iteration=0, labeled=[0], direct=[1, 2, 3, 4, 5, 6], indirect=[],
                unlabeled=[1, 2, 3, 4, 5, 6], selected=[2, 1, 3, 4, 5],
                bridges=[], residual=0., scores=[2., 1., 3., 4., 5., 6.])


def fixture_event():
    return dict(source='original', size=6, returned_prefix_local=[1, 0, 2, 3, 4])


class TimingContractTests(unittest.TestCase):
    def config(self, directory):
        return dict(cache=str(directory / 'cache'), output=str(directory / 'out'),
                    cache_sha256={name: '0' * 64 for name in timing.INPUTS},
                    query_count=10, gallery_count=490, query_ids=[3, 7, 2])

    def test_fixed_protocol_and_explicit_query_list(self):
        cfg = self.config(Path('/tmp/timing-test'))
        self.assertEqual(timing.validate_config(cfg)['query_ids'], [3, 7, 2])
        for queries in ([], [3, 3], [True], [-1], [10]):
            with self.assertRaises(ValueError):
                timing.validate_config(dict(cfg, query_ids=queries))
        with self.assertRaises(ValueError):
            timing.validate_config(dict(cfg, parameters=dict(alpha=5, beta=25, step=5, depth=100)))
        with self.assertRaises(ValueError):
            timing.validate_config(dict(cfg, cache_sha256={'graph.npz': '0' * 64}))
        with self.assertRaises(ValueError):
            timing.validate_config(dict(cfg, blas_threads=2))
        self.assertEqual(timing.route_order(0), ('original', 'optimized'))
        self.assertEqual(timing.route_order(1), ('optimized', 'original'))

    def test_capture_retains_producer_objects_without_packing(self):
        row, event = fixture_row(), fixture_event()
        capture = timing.TraceCapture([event], 500)
        with mock.patch.object(timing.certified, 'packed', side_effect=AssertionError('packing during compute')):
            capture(row)
        self.assertIs(capture.rows[0][0], row)
        self.assertIs(capture.rows[0][1], event)
        self.assertEqual(capture.seen, 1)
        bad = fixture_row()
        bad['selected'][:2] = bad['selected'][:2][::-1]
        capture = timing.TraceCapture([fixture_event()], 500)
        with self.assertRaisesRegex(AssertionError, 'Ordered selection'):
            capture(bad)
        self.assertEqual(len(capture.rows), 1)  # Preserve mismatch evidence.

    def test_saved_comparison_rejects_same_set_different_order_and_event_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'a', Path(directory) / 'b'
            left.mkdir(); right.mkdir()
            row, event = fixture_row(), fixture_event()
            prefix = np.array([0, 2, 1, 3, 4, 5])
            rank = np.array([2, 1, 3, 4, 5, 6])
            timing.save_artifacts(left, [(row, event)], [event], prefix, rank)
            timing.save_artifacts(right, [(row, event)], [event], prefix, rank)
            self.assertTrue(timing.compare_saved(left, right)['complete_gallery_identical'])
            changed = copy.deepcopy(row)
            changed['selected'][:2] = changed['selected'][:2][::-1]
            timing.save_artifacts(right, [(changed, event)], [event], prefix, rank)
            with self.assertRaisesRegex(AssertionError, 'selected'):
                timing.compare_saved(left, right)
            altered_event = copy.deepcopy(event)
            altered_event['returned_prefix_local'][-2:] = altered_event['returned_prefix_local'][-2:][::-1]
            timing.save_artifacts(right, [(row, altered_event)], [altered_event], prefix, rank)
            with self.assertRaisesRegex(AssertionError, 'returned_prefix_local'):
                timing.compare_saved(left, right)
            timing.save_artifacts(right, [], [], prefix, rank)
            with self.assertRaisesRegex(AssertionError, 'Trace length'):
                timing.compare_saved(left, right)

    def test_compute_failure_retains_completed_trace(self):
        class FailingEngine:
            def __init__(self, *args, **kwargs):
                self.full_sums = np.ones(3)

            def rank(self, *args, trace, **kwargs):
                trace(dict(iteration=0, labeled=[0], direct=[1, 2], indirect=[],
                           unlabeled=[1, 2], selected=[1, 2], bridges=[], residual=0., scores=None))
                raise RuntimeError('deliberate later-stage failure')
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                timing, 'FullIHyRDP', FailingEngine), mock.patch.object(timing, 'make_stage_solver'):
            dest = Path(directory) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'later-stage'):
                timing.execute_query('original', 0, sparse.eye(3), np.eye(3), np.eye(3), 1, dest, depth=3)
            failure = json.loads((dest / 'FAILED.json').read_text())
            self.assertEqual(failure['phase'], 'compute')
            self.assertEqual(len(list(timing.read_rows(dest / 'trace.jsonl.gz'))), 1)
            self.assertFalse((dest / 'arrays.npz').exists())

    def test_archive_checks_complete_order_not_set(self):
        with tempfile.TemporaryDirectory() as directory:
            dest, archive = Path(directory) / 'run', Path(directory) / 'archive'
            dest.mkdir(); (archive / 'queries').mkdir(parents=True)
            np.savez_compressed(dest / 'arrays.npz', prefix=[0, 2, 1], rankings=[2, 1, 3])
            np.save(archive / 'queries/0000.prefix.npy', [0, 2, 1])
            np.save(archive / 'rankings.npy', [[2, 1, 3]])
            self.assertTrue(timing.compare_archive(dest, archive, 0)['complete_gallery_identical'])
            np.save(archive / 'rankings.npy', [[1, 2, 3]])
            with self.assertRaisesRegex(AssertionError, 'complete gallery'):
                timing.compare_archive(dest, archive, 0)

    def test_run_alternates_and_finishes_declared_queries_after_failure(self):
        calls = []
        fake_threads = types.SimpleNamespace(threadpool_limits=lambda **kwargs: contextlib.nullcontext(),
                                              threadpool_info=lambda: [])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = self.config(base)
            archive = base / 'archive'
            archive.mkdir()
            cfg['archive_run'] = str(archive)
            config = base / 'config.json'
            config.write_text(json.dumps(cfg))
            def execute(route, query, graph, affinity, features, qn, dest):
                calls.append((query, route))
                dest.mkdir(parents=True)
                if query == 3 and route == 'original':
                    raise RuntimeError('one recorded failure')
                return dict(query=query, route=route, timings={
                    phase: dict(wall_seconds=1., cpu_seconds=1.)
                    for phase in ('preparation', 'compute', 'save', 'verify')})
            def load(path, **kwargs):
                return np.eye(500) if Path(path).name == 'affinity.npy' else np.ones((500, 1))
            def compare_archive(dest, archive, query):
                if query == 7:
                    raise AssertionError('historical tail order differs')
                return {'prefix_identical': True, 'complete_gallery_identical': True}
            with mock.patch.dict('sys.modules', threadpoolctl=fake_threads), mock.patch.object(
                    timing, 'sha', return_value='0' * 64), mock.patch.object(
                    timing.platform, 'platform', return_value='synthetic-test-platform'), mock.patch.object(
                    timing.subprocess, 'check_output', return_value='test-commit\n'), mock.patch.object(
                    timing.sparse, 'load_npz', return_value=sparse.eye(500, format='csr')), mock.patch.object(
                    timing.np, 'load', side_effect=load), mock.patch.object(
                    timing, 'execute_query', side_effect=execute), mock.patch.object(
                    timing, 'compare_archive', side_effect=compare_archive), mock.patch.object(
                    timing, 'compare_saved', return_value={'complete_gallery_identical': True}), mock.patch('builtins.print'):
                report = timing.run(config)
            self.assertEqual(calls, [(3, 'original'), (3, 'optimized'), (7, 'optimized'),
                                     (7, 'original'), (2, 'original'), (2, 'optimized')])
            self.assertEqual(report['status'], 'completed_with_failures')
            self.assertEqual([p['status'] for p in report['pairs']], ['incomplete', 'accepted', 'accepted'])
            self.assertTrue((base / 'out/0003/original/RUN_FAILED.json').exists())
            self.assertFalse(report['native_thread_limits_verified'])
            self.assertEqual(report['native_backend_detection'], 'no_supported_backend_detected')
            accepted = next(row for row in report['runs'] if row['query'] == 7 and row['route'] == 'original')
            self.assertEqual(accepted['archive_comparison_status'], 'failed')
            self.assertTrue((base / 'out/0007/original/ARCHIVE_FAILED.json').exists())
            self.assertEqual(len(report['runs']), 5)
            self.assertEqual(len(report['failures']), 2)


class TinyTrajectoryReadbackTests(unittest.TestCase):
    def test_real_rank_bridge_short_batch_and_saved_original_certificate(self):
        # A small synthetic integration check, never a timed production query.
        rng = np.random.default_rng(2041)
        n = 17
        w = np.zeros((n, n))
        for ids in (range(8), range(8, n)):
            for i in ids:
                w[i, i] = 1.
                for j in ids:
                    if i < j:
                        w[i, j] = w[j, i] = float(rng.uniform(.05, .8))
        affinity = rng.uniform(.05, .95, (n, n))
        affinity = (affinity + affinity.T) / 2
        np.fill_diagonal(affinity, 1.)
        graph = sparse.csr_matrix(w)
        features = np.eye(n)
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'original'
            result = timing.execute_query('original', 0, graph, affinity, features, 1, dest, depth=16)
            optimized = Path(directory) / 'optimized'
            warm = timing.execute_query('optimized', 0, graph, affinity, features, 1, optimized, depth=16)
            self.assertEqual(warm['status'], 'accepted')
            self.assertTrue(timing.compare_saved(dest, optimized)['ordered_trajectories_identical'])
            self.assertEqual(set(result['timings']), {'preparation', 'compute', 'save', 'verify'})
            self.assertEqual(result['gallery_count'], n - 1)
            self.assertGreater(result['acceptance']['bridge_events'], 0)
            self.assertGreater(result['acceptance']['no_solve_batches'], 0)
            self.assertEqual(len(list(timing.read_rows(dest / 'trace.jsonl.gz'))), result['trace_rows'])
            with np.load(dest / 'arrays.npz') as arrays:
                self.assertEqual(len(arrays['prefix']), 16)
                np.testing.assert_array_equal(np.sort(arrays['rankings']), np.arange(1, n))
            damaged = list(timing.read_rows(dest / 'trace.jsonl.gz'))
            first = next(row for row in damaged if row['returned_scores'] is not None)
            first['returned_scores'][0][2] += 1
            with gzip.open(dest / 'trace.jsonl.gz', 'wt') as stream:
                for row in damaged:
                    stream.write(json.dumps(row) + '\n')
            hashes = dict(result['files_sha256'])
            hashes['trace.jsonl.gz'] = timing.sha(dest / 'trace.jsonl.gz')
            with self.assertRaises((AssertionError, np.linalg.LinAlgError)):
                timing.verify_artifacts(dest, graph, affinity, features, 0, 1, 16,
                                        affinity.sum(axis=1), hashes)


if __name__ == '__main__':
    unittest.main()
