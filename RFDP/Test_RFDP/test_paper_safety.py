"""Numerical safety regressions for locally truncated diffusion graphs."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MODEL.Paper_RFDP.PaperUtils import build_paper_problem, solve_linear
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm, HyRdpIterative


class TestPaperSafety(unittest.TestCase):
    def test_disconnected_component_is_excluded_in_original_order(self):
        weights = np.array([[1, .5, 0, 0], [.5, 1, 0, 0],
                            [0, 0, 1, .8], [0, 0, .8, 1]], dtype=np.float32)
        problem = build_paper_problem(weights, [3, 1, 0, 2], [0])
        self.assertEqual(problem.sub_ids, [1, 0])
        self.assertEqual(problem.unlabeled_ids, [1])
        np.testing.assert_allclose(HyRdpClosedForm().solve(problem, 5, 25), [3.0])

    def test_reachable_from_either_query_is_retained(self):
        weights = np.array([[1, .5, 0, 0, 0], [.5, 1, 0, 0, 0],
                            [0, 0, 1, .8, 0], [0, 0, .8, 1, 0],
                            [0, 0, 0, 0, 1]], dtype=np.float32)
        problem = build_paper_problem(weights, [4, 3, 2, 1, 0], [0, 2])
        self.assertEqual(problem.sub_ids, [3, 2, 1, 0])
        self.assertEqual(problem.query_ids, [0, 2])

    def test_directed_node_must_have_path_to_query(self):
        weights = np.array([[1, .5], [0, 1]], dtype=np.float32)
        problem = build_paper_problem(weights, [0, 1], [0])
        self.assertEqual(problem.sub_ids, [0])
        self.assertTrue(problem.empty_unlabeled())

    def test_degree_accumulation_is_float64_and_input_is_unchanged(self):
        weights = np.array([[1, .5, 0], [.5, 1e8, 1], [0, 1, 1]], dtype=np.float32)
        original = weights.copy()
        problem = build_paper_problem(weights, [0, 1, 2], [0])
        self.assertEqual(problem.w_sub.dtype, np.dtype('float64'))
        self.assertEqual(problem.d_u[0], 100000001.5)
        self.assertEqual(problem.h_u[0], 100000001.0)
        np.testing.assert_array_equal(weights, original)

    def test_invalid_graph_weights_are_rejected(self):
        for bad in (-.1, np.nan, np.inf):
            with self.subTest(weight=bad), self.assertRaises(ValueError):
                build_paper_problem(np.array([[1, bad], [bad, 1]]), [0, 1], [0])

    def test_missing_query_and_duplicates_are_rejected(self):
        weights = np.eye(2)
        for subset, query in (([0, 1], []), ([0], [1]), ([0, 0], [0]), ([0, 1], [0, 0])):
            with self.subTest(subset=subset, query=query), self.assertRaises(ValueError):
                build_paper_problem(weights, subset, query)

    def test_singular_equation_is_not_silently_least_squares(self):
        with self.assertRaises(np.linalg.LinAlgError):
            solve_linear(np.array([[1, -1], [-1, 1.0]]), np.ones(2))

    def test_nonfinite_system_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_linear(np.array([[np.nan]]), np.ones(1))

    def test_general_nonsymmetric_system_preserves_solution(self):
        # GMFPT does not have a symmetric coefficient, so cannot use Cholesky.
        np.testing.assert_allclose(solve_linear(np.array([[2, -1], [0, 3]]),
                                               np.array([1, 3])), [1, 1])

    def test_iterative_series_exhaustion_is_not_a_valid_solution(self):
        weights = np.array([[1, .5, 0], [.5, 1, .5], [0, .5, 1]])
        problem = build_paper_problem(weights, [0, 1, 2], [0])
        with self.assertRaises(np.linalg.LinAlgError):
            HyRdpIterative(max_terms=1).solve(problem, 5, 25)
        closed = HyRdpClosedForm().solve(problem, 5, 25)
        iterative = HyRdpIterative(max_terms=1000).solve(problem, 5, 25)
        np.testing.assert_allclose(iterative, closed, rtol=1e-8, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
