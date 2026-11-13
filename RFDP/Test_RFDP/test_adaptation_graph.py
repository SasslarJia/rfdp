"""Synthetic graph-builder oracles; never extract features or rank queries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from experiments.reid_adaptation.graph import (
    _symmetrize_distance, _write_mutual_graph, build_graph_cache, sha256,
)
from MODEL.Aff_RFDP.Dist2Aff import Dist2Aff
from MODEL.Aff_RFDP.KnnRstct import KnnRstct
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy


def frozen_features(n=23, dimensions=9, seed=73):
    values = np.random.default_rng(seed).normal(size=(n, dimensions))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values.astype(np.float32)


def historical_oracle(features, kd, ks, gamma, block_rows):
    # The old pairwise_l2_distance arithmetic, starting AFTER its one-time
    # normalization/cast. Calling that function would normalize frozen Z again.
    n = len(features)
    distance = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, block_rows):
        similarity = features[start:start + block_rows].dot(features.T)
        block = np.maximum(2.0 - 2.0 * similarity, 0.0)
        np.sqrt(block, out=block)
        distance[start:start + block_rows] = block
    np.fill_diagonal(distance, 0.)
    assistant = OperAsstNumpy()
    affinity = Dist2Aff(kd, gamma, assistant)(distance)
    graph = KnnRstct(ks, asst_rela=assistant)(affinity).astype(np.float32)
    return affinity, graph


class AdaptationGraphTests(unittest.TestCase):
    def test_complete_cache_matches_historical_float32_oracle(self):
        # Include Kd=Ks=20 and P1 Kd=Ks=30, unequal internal Kd/Ks, a
        # one-column reduction, and a final one-column tile.
        cases = [(23, 3, 3, 1), (23, 20, 20, 11),
                 (37, 30, 30, 6), (17, 3, 5, 4)]
        for n, kd, ks, rows in cases:
            with self.subTest(n=n, kd=kd, ks=ks, rows=rows), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # Preserve an existing Fortran-ordered .npy without conversion.
                features = np.asfortranarray(frozen_features(n))
                source = root / 'source.npy'
                np.save(source, features)
                expected_hash = sha256(source)
                expected_a, expected_g = historical_oracle(features, kd, ks, .4, rows)
                progress = []
                result = build_graph_cache(source, root / 'cache', kd=kd, ks=ks,
                                           block_rows=rows, expected_features_sha256=expected_hash,
                                           progress=progress.append)
                affinity = np.load(root / 'cache/affinity.npy')
                graph = sparse.load_npz(root / 'cache/graph.npz')
                np.testing.assert_array_equal(affinity, expected_a)
                np.testing.assert_array_equal(graph.toarray(), expected_g)
                np.testing.assert_array_equal(graph.toarray(), graph.toarray().T)
                np.testing.assert_array_equal(graph.diagonal(), np.ones(n, dtype=np.float32))
                self.assertEqual(affinity.dtype, np.dtype('float32'))
                self.assertEqual(graph.dtype, np.dtype('float32'))
                self.assertEqual(sha256(source), expected_hash)
                self.assertEqual(sha256(root / 'cache/features.npy'), expected_hash)
                self.assertFalse(result['features_renormalized'])
                self.assertEqual(result['parameters'], dict(kd=kd, ks=ks, gamma=.4))
                self.assertEqual(result['nnz'], graph.nnz)
                for name, digest in result['files'].items():
                    self.assertEqual(sha256(root / 'cache' / name), digest)
                self.assertEqual(json.loads((root / 'cache/manifest.json').read_text()), result)
                self.assertEqual({p.name for p in (root / 'cache').iterdir()},
                                 {'features.npy', 'affinity.npy', 'graph.npz', 'manifest.json'})
                self.assertEqual(progress[-1]['phase'], 'graph_write')

    def test_distance_symmetry_uses_original_float32_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            distance = np.lib.format.open_memmap(Path(tmp) / 'distance.npy', mode='w+',
                                                dtype=np.float32, shape=(5, 5))
            original = np.random.default_rng(8).uniform(.1, 2., size=(5, 5)).astype(np.float32)
            np.fill_diagonal(original, 0.)
            distance[:] = original
            _symmetrize_distance(distance, block_rows=2)
            np.testing.assert_array_equal(distance, (original + original.T) * .5)
            distance._mmap.close()

    def test_mutual_threshold_retains_all_ties_and_zero_values_stay_absent(self):
        # Six equally close off-axis neighbors exceed Ks=2; the mask is
        # mutual >=, never argpartition's arbitrary fixed-size neighbor list.
        features = np.r_[np.eye(4), -np.eye(4)].astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'features.npy', features)
            result = build_graph_cache(root / 'features.npy', root / 'cache',
                                       kd=2, ks=2, block_rows=3)
            expected_a, expected_g = historical_oracle(features, 2, 2, .4, 3)
            actual = sparse.load_npz(root / 'cache/graph.npz')
            np.testing.assert_array_equal(np.load(root / 'cache/affinity.npy'), expected_a)
            np.testing.assert_array_equal(actual.toarray(), expected_g)
            self.assertTrue(np.all(np.diff(actual.indptr) == 7))
            self.assertEqual(result['graph']['rows_exceeding_ks_due_to_ties'], 8)
            self.assertEqual(result['graph']['max_nonself_neighbors'], 6)

            # With zero affinity threshold, dense zeros remain zeros and CSR
            # need not store them. Check the second AffXfrm symmetry as well.
            affinity = np.array([[1., .4, 0.], [.4000001, 1., 0.], [0., 0., 1.]], dtype=np.float32)
            _write_mutual_graph(affinity, root / 'zero.npz', ks=2, block_rows=2)
            expected = KnnRstct(2, asst_rela=OperAsstNumpy())(affinity).astype(np.float32)
            actual = sparse.load_npz(root / 'zero.npz')
            np.testing.assert_array_equal(actual.toarray(), expected)
            self.assertTrue(np.all(actual.data > 0))

    def test_gamma_changes_weights_without_changing_normal_range_neighbor_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'features.npy', frozen_features())
            graphs, affinities = [], []
            for index, gamma in enumerate((.35, .65)):
                output = root / str(index)
                build_graph_cache(root / 'features.npy', output, kd=3, ks=3,
                                  gamma=gamma, block_rows=4)
                graphs.append(sparse.load_npz(output / 'graph.npz'))
                affinities.append(np.load(output / 'affinity.npy'))
            np.testing.assert_array_equal(graphs[0].indptr, graphs[1].indptr)
            np.testing.assert_array_equal(graphs[0].indices, graphs[1].indices)
            self.assertFalse(np.array_equal(graphs[0].data, graphs[1].data))
            self.assertFalse(np.array_equal(affinities[0], affinities[1]))

    def test_copy_fallback_preserves_feature_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Tiny scaling detects any silently repeated normalization.
            features = frozen_features(7) * np.float32(1.000001)
            np.save(root / 'features.npy', features)
            with mock.patch('experiments.reid_adaptation.graph.os.link', side_effect=OSError('cross-device')):
                result = build_graph_cache(root / 'features.npy', root / 'cache', kd=2, ks=2)
            self.assertEqual(result['features_storage'], 'byte_copy')
            self.assertEqual(sha256(root / 'features.npy'), sha256(root / 'cache/features.npy'))
            expected_a, _ = historical_oracle(features, 2, 2, .4, 128)
            np.testing.assert_array_equal(np.load(root / 'cache/affinity.npy'), expected_a)

    def test_zero_local_scale_fails_without_changing_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'features.npy', np.tile(np.array([1., 0.], dtype=np.float32), (5, 1)))
            with self.assertRaisesRegex(ValueError, 'no epsilon floor'):
                build_graph_cache(root / 'features.npy', root / 'cache', kd=2, ks=2)
            self.assertTrue((root / 'cache/FAILED.json').is_file())
            self.assertFalse((root / 'cache/manifest.json').exists())

    def test_source_sha_and_dtype_must_match_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'features.npy'
            np.save(source, frozen_features(5))
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                build_graph_cache(source, root / 'cache', kd=2, ks=2,
                                  expected_features_sha256='0' * 64)
            self.assertFalse((root / 'cache').exists())
            np.save(source, frozen_features(5).astype(np.float64))
            with self.assertRaisesRegex(ValueError, 'float32'):
                build_graph_cache(source, root / 'cache', kd=2, ks=2)
            self.assertFalse((root / 'cache').exists())


if __name__ == '__main__':
    unittest.main()
