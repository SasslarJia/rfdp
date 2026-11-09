"""Deterministic input/metadata checks, not retrieval-effect experiments."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

RFDP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RFDP))

from experiments.relation_routes.csw_grouped import (
    build_grouped_csw_inputs, validate_grouped_inputs,
)
from experiments.reliability.reliability_model import (
    ReliabilityIHyRDP, array_sha256, csr_sha256, load_support, save_support,
    support_from_reciprocal,
)


def dense_oracle(full, cameras, k_intra, k_inter):
    """Small exhaustive oracle: sort groups, then compare explicit neighbor sets."""
    n = len(full)
    selected = []
    for i in range(n):
        groups = []
        for same, budget in ((True, k_intra), (False, k_inter)):
            candidates = [j for j in range(n) if j != i and full[i, j] > 0
                          and bool(cameras[j] == cameras[i]) == same]
            groups.extend(sorted(candidates, key=lambda j: (-float(full[i, j]), j))[:budget])
        selected.append(set(groups))
    neighbors = [{j for j in selected[i] if i in selected[j]} for i in range(n)]
    graph = np.diag(np.diag(full)).astype(float)
    support = np.eye(n)
    for i in range(n):
        for j in neighbors[i]:
            graph[i, j] = full[i, j]
        for j in range(n):
            if i == j:
                continue
            left, right = neighbors[i] - {j}, neighbors[j] - {i}
            union = left | right
            support[i, j] = len(left & right) / len(union) if union else 0.
    return graph, support


class TestGroupedCSW(unittest.TestCase):
    def test_groupwise_mutual_selection_and_full_support_match_dense_oracle(self):
        rng = np.random.default_rng(20261109)
        full = rng.integers(0, 5, (11, 11)).astype(np.float32) / 5
        full = np.maximum(full, full.T)
        np.fill_diagonal(full, np.linspace(.5, 1, len(full)))
        cameras = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4])
        snapshot = full.copy()
        graph, support, metadata = build_grouped_csw_inputs(
            full, cameras, k_intra=2, k_inter=3, block_rows=3)
        expected_graph, expected_support = dense_oracle(full, cameras, 2, 3)
        np.testing.assert_array_equal(graph.toarray(), expected_graph)
        np.testing.assert_array_equal(support.toarray(), expected_support)
        np.testing.assert_array_equal(full, snapshot)
        reciprocal = (graph > 0).astype(float)
        reciprocal.setdiag(0); reciprocal.eliminate_zeros()
        np.testing.assert_array_equal(support.toarray(), support_from_reciprocal(
            reciprocal, cameras, mode='ordinary').toarray())
        self.assertEqual(metadata['full_sha256'], array_sha256(full))
        self.assertEqual(metadata['raw_graph_sha256'], csr_sha256(graph))
        self.assertEqual(metadata['support_sha256'], csr_sha256(support))
        self.assertTrue(validate_grouped_inputs(graph, support, metadata,
                        full_sha256=array_sha256(full), cameras=cameras,
                        k_intra=2, k_inter=3))

    def test_cutoff_ties_and_nonedge_witnesses_are_kept(self):
        full = np.ones((6, 6), dtype=np.float32)
        cameras = np.array([0, 0, 0, 1, 1, 1])
        graph, support, _ = build_grouped_csw_inputs(
            full, cameras, k_intra=1, k_inter=1, block_rows=1)
        expected = np.eye(6)
        for i, j in ((0, 1), (0, 3), (3, 4)):
            expected[i, j] = expected[j, i] = 1
        np.testing.assert_array_equal(graph.toarray(), expected)
        self.assertEqual(graph[1, 3], 0)
        self.assertEqual(support[1, 3], .5)
        self.assertEqual(support[0, 4], .5)
        self.assertEqual(support[2, 2], 1)

    def test_missing_groups_and_zero_affinities_do_not_borrow_budget(self):
        full = np.array([[.7, .8, .5, 0], [.8, .9, .6, .4],
                         [.5, .6, 1., .9], [0, .4, .9, .8]])
        cameras = np.zeros(4, dtype=int)
        graph, support, _ = build_grouped_csw_inputs(
            full, cameras, k_intra=1, k_inter=20)
        expected_graph, expected_support = dense_oracle(full, cameras, 1, 20)
        np.testing.assert_array_equal(graph.toarray(), expected_graph)
        np.testing.assert_array_equal(support.toarray(), expected_support)
        self.assertEqual(graph[1, 2], 0)

    def test_readonly_memmap_and_block_size_do_not_change_outputs(self):
        full = np.ones((7, 7), dtype=np.float32)
        cameras = np.arange(7) % 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'full.npy'
            np.save(path, full)
            mapped = np.load(path, mmap_mode='r')
            a, q, ma = build_grouped_csw_inputs(mapped, cameras, block_rows=1)
            b, r, mb = build_grouped_csw_inputs(mapped, cameras, block_rows=4)
            self.assertEqual(csr_sha256(a), csr_sha256(b))
            self.assertEqual(csr_sha256(q), csr_sha256(r))
            self.assertEqual(ma['cache_key_sha256'], mb['cache_key_sha256'])
            self.assertFalse(mapped.flags.writeable)
            np.testing.assert_array_equal(mapped, full)

    def test_singleton_retains_raw_diagonal(self):
        graph, support, metadata = build_grouped_csw_inputs(
            np.array([[.6]], dtype=np.float32), np.array([2]))
        self.assertEqual(graph[0, 0], float(np.float32(.6)))
        self.assertEqual(support[0, 0], 1)
        self.assertEqual(metadata['reciprocal_nnz'], 0)

    def test_cache_content_input_and_configuration_are_bound(self):
        full = np.ones((6, 6), dtype=np.float32)
        cameras = np.array([0, 0, 0, 1, 1, 1])
        graph, support, metadata = build_grouped_csw_inputs(
            full, cameras, k_intra=1, k_inter=1)
        changed_graph = graph.copy(); changed_graph.data[0] *= .9
        changed_support = support.copy(); changed_support.data[0] *= .9
        changed_metadata = dict(metadata, k_inter=2)
        for g, q, m in ((changed_graph, support, metadata),
                        (graph, changed_support, metadata),
                        (graph, support, changed_metadata)):
            with self.assertRaises(ValueError):
                validate_grouped_inputs(g, q, m)
        for expectations in (dict(full_sha256='0' * 64), dict(cameras=1-cameras),
                             dict(k_intra=2), dict(k_inter=2)):
            with self.assertRaises(ValueError):
                validate_grouped_inputs(graph, support, metadata, **expectations)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'support.npz'
            save_support(path, support, metadata)
            loaded, info = load_support(path, expected={
                'neighbor_definition_version': metadata['neighbor_definition_version'],
                'k_intra': 1, 'k_inter': 1, 'mode': 'ordinary'})
            np.testing.assert_array_equal(loaded.toarray(), support.toarray())
            self.assertEqual(info['full_sha256'], metadata['full_sha256'])
            self.assertTrue(validate_grouped_inputs(graph, loaded, info))

    def test_existing_csw_postprocess_uses_raw_full_kernel_and_new_support(self):
        full = np.ones((6, 6), dtype=np.float32)
        cameras = np.array([0, 0, 0, 1, 1, 1])
        graph, support, metadata = build_grouped_csw_inputs(
            full, cameras, k_intra=1, k_inter=1)
        engine = ReliabilityIHyRDP(graph, full, cameras, eta=.5, lam=.25,
                                  support=support, support_metadata=metadata)
        scale = np.where(cameras[:, None] == cameras[None, :], .5, 1.)
        scale *= .75 + .25 * support.toarray()
        expected_full = full * scale
        np.testing.assert_array_equal(engine.w.toarray(), graph.toarray() * scale)
        np.testing.assert_allclose(engine.full_sums, expected_full.sum(axis=1), rtol=1e-15)
        np.testing.assert_array_equal(engine.full_kernel_block(np.arange(6), np.arange(6)),
                                      expected_full)
        # Support on nonedges affects full row sums; it cannot be discarded.
        self.assertGreater(expected_full[1, 3], .75)

    def test_invalid_inputs_rejected_without_silent_repair(self):
        full = np.ones((3, 3))
        cams = np.array([0, 0, 1])
        for key, value in (('k_intra', 0), ('k_inter', -1), ('k_intra', True),
                           ('k_inter', 1.5), ('block_rows', 0), ('block_rows', True)):
            with self.assertRaises(ValueError):
                build_grouped_csw_inputs(full, cams, **{key: value})
        for matrix in (np.ones((2, 3)), np.ones((0, 0)), full.astype(complex)):
            with self.assertRaises(ValueError):
                build_grouped_csw_inputs(matrix, cams)
        for value in (-.2, np.nan, np.inf):
            broken = full.copy(); broken[0, 1] = value
            with self.assertRaises(ValueError):
                build_grouped_csw_inputs(broken, cams)
        broken = full.copy(); broken[0, 0] = 0
        with self.assertRaises(ValueError):
            build_grouped_csw_inputs(broken, cams)
        broken = full.copy(); broken[0, 1] = .5
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            build_grouped_csw_inputs(broken, cams)
        with self.assertRaises(ValueError):
            build_grouped_csw_inputs(full, cams[:-1])


if __name__ == '__main__':
    unittest.main()
