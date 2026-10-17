"""Local mathematical/structural tests only; no benchmark data or remote calls.

Run with the repository's RFDP directory on PYTHONPATH:
    python -B -m unittest discover -s <this directory> -p test_reliability_model.py -v
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

from reliability_model import (
    CameraIHyRDP, FullIHyRDP, DEFINITION_VERSION, ReliabilityIHyRDP, array_sha256, build_reciprocal_graph,
    build_support, build_supports, csr_sha256, load_support, save_support,
    support_from_reciprocal, support_stats,
)


def dense_support_reference(full, cameras, k, mode):
    n = len(full)
    neighbors = []
    for i in range(n):
        ids = [j for j in range(n) if i != j and full[i, j] > 0]
        ids.sort(key=lambda j: (-float(full[i, j]), j))
        neighbors.append(set(ids[:k]))
    reciprocal = [{j for j in neighbors[i] if i in neighbors[j]} for i in range(n)]
    result = np.eye(n)
    for i in range(n):
        for j in range(i):
            a, b = reciprocal[i] - {j}, reciprocal[j] - {i}
            if mode == 'ordinary':
                value = len(a & b) / len(a | b) if a | b else 0.0
            else:
                values = []
                for camera in np.unique(cameras):
                    ac = {v for v in a if cameras[v] == camera}
                    bc = {v for v in b if cameras[v] == camera}
                    if ac | bc:
                        values.append(len(ac & bc) / len(ac | bc))
                value = np.mean(values) if values else 0.0
            result[i, j] = result[j, i] = value
    return result


def fixture(n=24, dtype=np.float32):
    rng = np.random.default_rng(20261015)
    raw = rng.uniform(.02, 1., (n, n))
    full = np.asarray((raw + raw.T) / 2, dtype=dtype)
    np.fill_diagonal(full, 1.)
    graph = full.copy()
    for i in range(n):
        for j in range(n):
            if i == j or i // 8 != j // 8:
                graph[i, j] = 0
    cameras = np.arange(n) % 3
    return sparse.csr_matrix(graph), full, cameras


class SupportTests(unittest.TestCase):
    def test_sparse_support_matches_independent_sets_and_hashes(self):
        rng = np.random.default_rng(517)
        full = rng.integers(0, 5, (13, 13)).astype(np.float32)
        full = full + full.T  # Includes ties and positive diagonal values.
        cameras = np.arange(13) % 4
        for k in (1, 4, 20):
            progress = []
            supports, metadata = build_supports(full, cameras, k=k, block_rows=3, progress=progress.append)
            self.assertEqual(sum(p['stage'] == 'reciprocal_topk' for p in progress), 5)
            for mode in ('ordinary', 'cb'):
                np.testing.assert_allclose(supports[mode].toarray(), dense_support_reference(full, cameras, k, mode), atol=1e-15, rtol=0)
                self.assertEqual(metadata[mode]['full_sha256'], array_sha256(full, block_rows=4))
                self.assertEqual(metadata[mode]['cameras_sha256'], array_sha256(cameras))
                self.assertEqual(metadata[mode]['support_sha256'], csr_sha256(supports[mode]))
                self.assertEqual((supports[mode] != supports[mode].T).nnz, 0)

    def test_empty_and_bipartite_witnesses(self):
        for n in (1, 6):
            full = np.eye(n)
            for mode in ('ordinary', 'cb'):
                q, _ = build_support(full, np.arange(n) % 2, mode=mode)
                np.testing.assert_array_equal(q.toarray(), np.eye(n))
        full = np.zeros((6, 6))
        full[:3, 3:] = full[3:, :3] = 1
        for mode in ('ordinary', 'cb'):
            q, _ = build_support(full, np.array([0, 0, 0, 1, 1, 1]), k=3, mode=mode)
            self.assertEqual(np.max(q.toarray()[full > 0]), 0)

    def test_sparse_candidate_bound(self):
        _, full, cameras = fixture(n=48)
        reciprocal, _ = build_reciprocal_graph(full, k=4)
        degrees = np.asarray(reciprocal.sum(axis=1)).ravel()
        self.assertLessEqual(degrees.max(), 4)
        for mode in ('ordinary', 'cb'):
            q = support_from_reciprocal(reciprocal, cameras, mode)
            self.assertLessEqual(q.nnz - len(full), np.sum(degrees ** 2))

    def test_archive_roundtrip_hash_binding_and_tamper_rejection(self):
        graph, full, cameras = fixture()
        q, meta = build_support(full, cameras, k=5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'support.npz'
            save_support(path, q, meta)
            loaded, info = load_support(path, expected={'mode': 'cb', 'k': 5,
                                                       'definition_version': DEFINITION_VERSION})
            np.testing.assert_array_equal(loaded.toarray(), q.toarray())
            self.assertEqual(info['full_sha256'], array_sha256(full))
            with self.assertRaisesRegex(ValueError, 'expectation mismatch'):
                load_support(path, expected={'k': 20})
            model = ReliabilityIHyRDP(graph, full, cameras, support=loaded, lam=.5, support_metadata=info)
            self.assertEqual(model.input_hashes['full_sha256'], meta['full_sha256'])
            modified = full.copy()
            modified[0, 1] += .01
            with self.assertRaisesRegex(ValueError, 'Full affinity'):
                ReliabilityIHyRDP(graph, modified, cameras, support=loaded, lam=.5, support_metadata=info)
            with self.assertRaisesRegex(ValueError, 'Camera vector'):
                ReliabilityIHyRDP(graph, full, np.roll(cameras, 1), support=loaded, lam=.5, support_metadata=info)
            # Corrupt values while retaining JSON; content hash must reject it.
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            first_offdiag = np.flatnonzero(q.tocoo().row != q.tocoo().col)[0]
            arrays['data'][first_offdiag] *= .5
            np.savez_compressed(path, **arrays)
            with self.assertRaises(ValueError):
                load_support(path)

    def test_stats_do_not_count_implicit_zero_as_positive(self):
        graph, full, cameras = fixture()
        q, _ = build_support(full, cameras, k=4)
        stats = support_stats(q, graph, cameras)
        dense = q.toarray()
        off = ~np.eye(len(full), dtype=bool)
        self.assertAlmostEqual(stats['offdiag_mean_including_zeros'], dense[off].mean())
        self.assertAlmostEqual(stats['offdiag_positive_fraction'], np.mean(dense[off] > 0))
        graph_mask = graph.toarray() > 0
        self.assertAlmostEqual(stats['graph_q_mean'], dense[graph_mask].mean())
        self.assertIn('cross_camera', stats)


class ModelTests(unittest.TestCase):
    def assert_same_trace(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assertEqual(a[key], b[key], key)

    def test_lambda_zero_is_exact_production_regression(self):
        graph, full, cameras = fixture()
        q, _ = build_support(full, cameras, k=5)
        baseline = CameraIHyRDP(graph, full, cameras, eta=.5)
        for supplied in (None, q):
            model = ReliabilityIHyRDP(graph, full, cameras, eta=.5, support=supplied, lam=0)
            np.testing.assert_array_equal(model.w.data, baseline.w.data)
            np.testing.assert_array_equal(model.w.indices, baseline.w.indices)
            np.testing.assert_array_equal(model.w.indptr, baseline.w.indptr)
            np.testing.assert_array_equal(model.full_sums, baseline.full_sums)
            for seed in (0, 9):
                lt, rt = [], []
                lr, _ = baseline.rank([seed], nq=len(full), trace=lt.append)
                rr, diagnostics = model.rank([seed], nq=len(full), trace=rt.append)
                np.testing.assert_array_equal(lr, rr)
                self.assert_same_trace(lt, rt)
                self.assertGreater(diagnostics['solver_calls'], 0)
                self.assertGreater(diagnostics['bridge_events'], 0)

    def test_all_one_support_is_exact_identity(self):
        graph, full, cameras = fixture()
        baseline = CameraIHyRDP(graph, full, cameras, eta=.5)
        q = sparse.csr_matrix(np.ones(full.shape))
        model = ReliabilityIHyRDP(graph, full, cameras, eta=.5, support=q, lam=.73)
        np.testing.assert_array_equal(model.w.toarray(), baseline.w.toarray())
        np.testing.assert_array_equal(model.full_sums, baseline.full_sums)
        self.assertEqual(model.bridges({0}), baseline.bridges({0}))
        np.testing.assert_array_equal(model.rank([0], nq=len(full))[0], baseline.rank([0], nq=len(full))[0])

    def test_full_graph_bridge_and_ranking_match_dense_reference(self):
        graph, full, cameras = fixture()
        supports, _ = build_supports(full, cameras, k=6)
        for mode, q in supports.items():
            for eta in (.5, 1.):
                for lam in (.25, .5, .95):
                    model = ReliabilityIHyRDP(graph, full, cameras, eta=eta, support=q, lam=lam)
                    factor = np.where(cameras[:, None] == cameras[None, :], eta, 1.)
                    reliability = 1 - lam + lam * q.toarray()
                    weighted_full = full.astype(np.float64) * factor * reliability
                    weighted_graph = graph.toarray().astype(np.float64) * factor * reliability
                    reference = FullIHyRDP(sparse.csr_matrix(weighted_graph), weighted_full)
                    np.testing.assert_array_equal(model.w.toarray(), weighted_graph)
                    np.testing.assert_allclose(model.full_sums, weighted_full.sum(axis=1), atol=1e-13, rtol=1e-14)
                    np.testing.assert_array_equal(model.full_kernel_block([1, 9], [0, 8, 17]), weighted_full[np.ix_([1, 9], [0, 8, 17])])
                    np.testing.assert_array_equal(model.w.toarray() > 0, graph.toarray() > 0)
                    np.testing.assert_array_equal(model.components, reference.components)
                    for active in ({0}, {1}, {0, 1}):
                        actual, expected = model.bridges(active), reference.bridges(active)
                        self.assertEqual([(i, j) for i, j, _ in actual], [(i, j) for i, j, _ in expected], mode)
                        np.testing.assert_allclose([v for _, _, v in actual], [v for _, _, v in expected], rtol=1e-15, atol=0)
                    # Fixed bridge selections are checked above. A connected
                    # graph guarantees positive internal degrees throughout
                    # full ranking, an explicit production-solver condition.
                    connected = full.copy()
                    np.fill_diagonal(connected, 0)
                    connected_model = ReliabilityIHyRDP(sparse.csr_matrix(connected), full, cameras,
                                                       eta=eta, support=q, lam=lam)
                    connected_reference = FullIHyRDP(sparse.csr_matrix(connected * factor * reliability), weighted_full)
                    actual, diagnostics = connected_model.rank([0], nq=len(full))
                    expected, _ = connected_reference.rank([0], nq=len(full))
                    np.testing.assert_array_equal(actual, expected)
                    self.assertLessEqual(diagnostics['max_relative_residual'], 1e-8)

    def test_production_precision_patch_uses_same_solver_module(self):
        module = sys.modules[FullIHyRDP.__module__]
        precision_name = FullIHyRDP.__module__.rsplit('.', 1)[0] + '.PaperPrecision'
        precision = sys.modules.get(precision_name)
        if precision is not None:
            self.assertIs(module.solve_hybrid_sparse, precision.refined)
        self.assertIs(ReliabilityIHyRDP.rank, FullIHyRDP.rank)
        self.assertIs(CameraIHyRDP.rank, FullIHyRDP.rank)

    def test_gate_validation_and_input_ownership(self):
        graph, full, cameras = fixture()
        q, _ = build_support(full, cameras, k=4)
        original_graph, original_full, original_q = graph.copy(), full.copy(), q.copy()
        model = ReliabilityIHyRDP(graph, full, cameras, support=q, lam=.5)
        np.testing.assert_array_equal(graph.toarray(), original_graph.toarray())
        np.testing.assert_array_equal(full, original_full)
        np.testing.assert_array_equal(q.toarray(), original_q.toarray())
        self.assertFalse(model.full.flags.writeable)
        self.assertFalse(model.support.data.flags.writeable)
        for lam in (-.1, 1., np.nan):
            with self.assertRaises(ValueError):
                ReliabilityIHyRDP(graph, full, cameras, support=q, lam=lam)
        with self.assertRaisesRegex(ValueError, 'required'):
            ReliabilityIHyRDP(graph, full, cameras, lam=.5)
        bad = q.toarray()
        bad[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'diagonal'):
            ReliabilityIHyRDP(graph, full, cameras, support=sparse.csr_matrix(bad), lam=.5)
        bad[0, 0] = 1
        bad[0, 1], bad[1, 0] = .9, .1
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            ReliabilityIHyRDP(graph, full, cameras, support=sparse.csr_matrix(bad), lam=.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
