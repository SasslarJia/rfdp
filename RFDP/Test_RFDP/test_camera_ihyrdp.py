"""Camera prior and full-experiment regressions independent of test-set labels."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

RFDP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RFDP))
sys.path.insert(0, str(RFDP / 'experiments/full_reid'))
from MODEL.Paper_RFDP.PaperCamera import CameraIHyRDP, camera_graph
from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
from MODEL.Paper_RFDP import PaperPrecision
from EVAL.ReIDEval import evaluate_reid_rankings
from common import complete_gallery, validate_ranking
from verify_full import evaluate


class TestCameraIHyRDP(unittest.TestCase):
    def graph(self):
        rng = np.random.default_rng(20261015)
        a = rng.uniform(.2, 1, (24, 24))
        full = ((a + a.T) / 2).astype(np.float32)
        graph = full.copy()
        graph[np.arange(24)[:, None] // 8 != np.arange(24)[None, :] // 8] = 0
        return graph, full, np.arange(24) % 4

    def test_same_camera_prior_matches_dense_equation_without_mutation(self):
        graph, _, cams = self.graph()
        snapshot = graph.copy()
        expected = graph.astype(np.float64) * np.where(cams[:, None] == cams[None, :], .5, 1.)
        np.testing.assert_array_equal(camera_graph(graph, cams).toarray(), expected)
        np.testing.assert_array_equal(graph, snapshot)

    def test_unit_weight_is_original_dynamic_model_including_bridges(self):
        graph, full, cams = self.graph()
        original, camera = FullIHyRDP(graph, full), CameraIHyRDP(graph, full, cams, eta=1)
        for query in [0, 10, 20]:
            a, da = original.rank([query])
            b, db = camera.rank([query])
            np.testing.assert_array_equal(a, b)
            self.assertGreater(db['bridge_events'], 0)
            self.assertEqual(da['solver_calls'], db['solver_calls'])

    def test_weighted_dynamic_model_matches_explicit_dense_prior_oracle(self):
        graph, full, cams = self.graph()
        prior = np.where(cams[:, None] == cams[None, :], .5, 1.)
        oracle = FullIHyRDP(graph.astype(np.float64) * prior, full.astype(np.float64) * prior)
        camera = CameraIHyRDP(graph, full, cams)
        np.testing.assert_allclose(camera.full_sums, oracle.full_sums, rtol=1e-14)
        for query in [0, 10, 20]:
            np.testing.assert_array_equal(oracle.rank([query])[0], camera.rank([query])[0])
            for active in [{0}, {1}, {0, 2}]:
                self.assertEqual(oracle.bridges(active), camera.bridges(active))

    def test_invalid_camera_parameters_rejected(self):
        graph, full, cams = self.graph()
        for eta in [0, -.5, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                CameraIHyRDP(graph, full, cams, eta)
        for bad in [cams[:-1], cams[:, None]]:
            with self.assertRaises(ValueError):
                CameraIHyRDP(graph, full, bad)

    def test_refinement_keeps_same_linear_equation(self):
        w = sparse.csr_matrix([[1., .2, .3], [.2, 1, .4], [.3, .4, 1.]])
        e = np.array([.5, .1, 0.])
        h = np.asarray(w.sum(1)).ravel()
        dh = np.diag(1 / np.sqrt(h))
        matrix = np.diag(1 + 5 * e / (h + e)) - dh @ w.toarray() @ dh
        expected = np.linalg.solve(matrix, np.where(e > 0, 5, 25))
        with patch.object(PaperPrecision, 'old', side_effect=np.linalg.LinAlgError('forced residual check')):
            actual, residual = PaperPrecision.refined(w, e)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        self.assertLessEqual(residual, 1e-8)

    def test_programming_error_does_not_trigger_numerical_refinement(self):
        with patch.object(PaperPrecision, 'old', side_effect=ValueError('invalid input')):
            with self.assertRaises(ValueError):
                PaperPrecision.refined(sparse.eye(2), np.ones(2))

    def test_original_tail_preserves_complete_gallery_order(self):
        actual = complete_gallery([0, 4, 1], [1., .6, .2, .5, .4, .7], 2)
        np.testing.assert_array_equal(actual, [4, 5, 3, 2])
        with self.assertRaises(ValueError):
            validate_ranking([2, 2, 4, 5], 2, 4)

    def test_independent_metrics_exclude_junk_and_same_identity_camera(self):
        ranks = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]], dtype=np.int32)
        qp, qc = np.array([1, 99]), np.array([1, 1])
        gp, gc = np.array([1, -1, 2, 1, 1]), np.array([1, 3, 2, 2, 3])
        independent, rows = evaluate(ranks, qp, qc, gp, gc)
        regular = evaluate_reid_rankings(ranks, [0, 1], qp, qc, gp, gc, 2)
        self.assertAlmostEqual(independent['mAP'], (1/2 + 2/3)/2)
        self.assertEqual(independent['valid_queries'], 1)
        self.assertEqual(rows[1]['valid'], False)
        self.assertAlmostEqual(independent['mAP'], regular['mAP'])
        np.testing.assert_array_equal(independent['CMC'], regular['cmc'])


if __name__ == '__main__':
    unittest.main()
