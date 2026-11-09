"""Equation and read-back tests, not retrieval-effect or performance trials."""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.relation_routes.quaternary import (
    QuaternaryCertificationError, certify_saved, solve_quaternary_sparse,
)


def normalized(w):
    h = np.sum(w, axis=1)
    return w / np.sqrt(h[:, None] * h[None, :])


class TestQuaternaryRoute(unittest.TestCase):
    def setUp(self):
        self.u = np.array([[1., .3, 0.], [.3, 1., .15], [0., .15, 1.]])
        self.l = np.array([[1., .5, 0.], [.5, 1., .2], [0., .2, 1.]])
        self.y = np.array([[.8, 0., .1], [0., .4, .3], [.15, .2, 0.]])

    def test_full_four_index_solution_matches_independent_dense_oracle(self):
        for diffusion in (.4, .99):
            with self.subTest(diffusion=diffusion):
                coefficient = np.eye(9) - diffusion * np.kron(
                    normalized(self.u), normalized(self.l))
                expected = np.linalg.solve(coefficient,
                                           ((1. - diffusion) * self.y).ravel()).reshape(3, 3)
                stats = {}
                scores, residual, x = solve_quaternary_sparse(
                    self.u, self.y, self.l, diffusion=diffusion,
                    diagnostics=stats, return_relations=True)
                np.testing.assert_allclose(x, expected, rtol=2e-11, atol=2e-13)
                np.testing.assert_allclose(scores, -expected.mean(axis=1), atol=2e-13)
                self.assertLess(residual, 1e-8)
                self.assertEqual(residual, stats['relative_residual_upper'])
                self.assertGreaterEqual(residual, stats['relative_residual_native'])
                self.assertTrue(stats['ranking_certified'])
                self.assertEqual(stats['relation_variables'], 9)

    def test_second_endpoint_propagation_is_real_not_a_renamed_vector_solver(self):
        _, _, full = solve_quaternary_sparse(self.u, self.y, self.l, return_relations=True)
        _, _, no_target_edges = solve_quaternary_sparse(
            self.u, self.y, np.eye(3), return_relations=True)
        self.assertGreater(np.max(np.abs(full - no_target_edges)), 1e-4)
        self.assertGreater(np.max(np.abs(full.mean(axis=1) - no_target_edges.mean(axis=1))), 1e-5)

    def test_one_target_naturally_reduces_to_one_sided_rdp(self):
        y = self.y[:, :1]
        scores, _, x = solve_quaternary_sparse(self.u, y, np.ones((1, 1)),
                                              return_relations=True)
        expected = np.linalg.solve(np.eye(3) - .99 * normalized(self.u), .01 * y[:, 0])
        np.testing.assert_allclose(x[:, 0], expected, rtol=2e-11)
        np.testing.assert_allclose(scores, -expected, rtol=2e-11)

    def test_missing_direct_boundary_is_propagated_without_discarding_row(self):
        y = self.y.copy()
        y[2] = 0.
        _, _, x = solve_quaternary_sparse(self.u, y, self.l, return_relations=True)
        self.assertEqual(x.shape, (3, 3))
        self.assertTrue(np.all(x[2] > 0))

    def test_no_kronecker_matrix_is_materialized(self):
        with patch('numpy.kron', side_effect=AssertionError('dense product forbidden')), \
                patch('scipy.sparse.kron', side_effect=AssertionError('sparse product forbidden')):
            stats = {}
            solve_quaternary_sparse(sparse.csr_matrix(self.u), sparse.csr_matrix(self.y),
                                    sparse.csr_matrix(self.l), diagnostics=stats)
            self.assertTrue(stats['matrix_free'])

    def test_saved_projection_alone_cannot_certify_relation_equation(self):
        scores, _, x = solve_quaternary_sparse(self.u, self.y, self.l, return_relations=True)
        stats = certify_saved(self.u, self.y, self.l, x.copy(), scores.copy())
        self.assertTrue(stats['saved_projection_verified'])
        with self.assertRaises(ValueError):
            certify_saved(self.u, self.y, self.l, None, scores)
        # Preserve every row mean while corrupting the four-index solution.
        damaged = x.copy()
        damaged[0, 0] += .1
        damaged[0, 1] -= .1
        with self.assertRaises(QuaternaryCertificationError):
            certify_saved(self.u, self.y, self.l, damaged, -damaged.mean(axis=1))
        with self.assertRaises(ValueError):
            certify_saved(self.u, self.y, self.l, x, scores + .01)

    def test_tied_uncertified_scores_are_not_silently_accepted(self):
        with self.assertRaises(QuaternaryCertificationError):
            solve_quaternary_sparse(np.eye(6), np.ones((6, 1)), np.ones((1, 1)))
        stats = {}
        solve_quaternary_sparse(np.eye(6), np.ones((6, 1)), np.ones((1, 1)),
                                require_rank_certificate=False, diagnostics=stats)
        self.assertTrue(stats['equation_certified'])
        self.assertFalse(stats['ranking_certified'])
        self.assertTrue(stats['native_precision_audit'])

    def test_zero_rhs_has_exact_lexicographic_ties(self):
        stats = {}
        scores, _ = solve_quaternary_sparse(self.u, np.zeros((3, 3)), self.l,
                                           ids=np.array([8, 3, 1]), diagnostics=stats)
        np.testing.assert_array_equal(scores, np.zeros(3))
        self.assertTrue(stats['ranking_certified'])
        self.assertEqual(stats['certificate_kind'], 'zero_rhs_exact_lexicographic')

    def test_unconverged_iterate_is_not_accepted(self):
        with self.assertRaises(QuaternaryCertificationError):
            solve_quaternary_sparse(self.u, self.y, self.l, maxiter=1)

    def test_tolerance_boundary_uses_residual_bound_not_native_norm(self):
        # An accurately recomputed native residual just below the threshold is
        # insufficient when its rounding allowance crosses the threshold.
        diffusion = .4
        matrix = np.eye(9) - diffusion * np.kron(
            normalized(self.u), normalized(self.l))
        rhs = ((1. - diffusion) * self.y).ravel()
        x = np.linalg.solve(matrix, rhs).reshape(3, 3)
        perturbation = (1e-8 - 2e-15) * np.linalg.norm(rhs) / np.linalg.norm(matrix[:, 0])
        x[0, 0] += perturbation
        with self.assertRaises(QuaternaryCertificationError) as rejected:
            certify_saved(self.u, self.y, self.l, x, -x.mean(axis=1),
                          diffusion=diffusion)
        stats = rejected.exception.diagnostics
        self.assertLess(stats['relative_residual_native'], 1e-8)
        self.assertGreater(stats['relative_residual_upper'], 1e-8)
        self.assertFalse(stats['equation_certified'])

    def test_invalid_graph_and_boundary_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_quaternary_sparse(self.u, self.y, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            solve_quaternary_sparse(self.u, self.y[:, :2], self.l)
        for diffusion in (1., -1., np.nan):
            with self.subTest(diffusion=diffusion), self.assertRaises(ValueError):
                solve_quaternary_sparse(self.u, self.y, self.l, diffusion=diffusion)
        asymmetric = self.u.copy()
        asymmetric[0, 1] += 1e-9
        with self.assertRaises(ValueError):
            solve_quaternary_sparse(asymmetric, self.y, self.l)


if __name__ == '__main__':
    unittest.main()
