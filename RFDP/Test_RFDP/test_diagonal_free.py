"""D3 engineering contracts on synthetic matrices; no retrieval inputs."""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified as base
from experiments.continuous_fitting import diagonal_free as d3


def exact(value):
    if isinstance(value, mp.mpf):
        return value
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator)/denominator


def oracle(w, e, alpha=5., beta=25.):
    """Independently assemble the frozen D3 definition at 100 decimal digits."""
    with mp.workdps(100):
        dense = sparse.csr_matrix(w).toarray()
        n = len(e)
        ww = [[mp.mpf(0) if i == j else exact(dense[i, j])
               for j in range(n)] for i in range(n)]
        h = [mp.fsum(row) for row in ww]
        ee = [exact(v) for v in e]
        a, b = exact(alpha), exact(beta)
        matrix = mp.matrix(n)
        rhs = mp.matrix([a if v > 0 else b for v in ee])
        for i in range(n):
            if h[i] == 0:
                if ee[i] == 0:
                    raise ValueError('Unanchored analytic node')
                matrix[i, i] = a
            else:
                matrix[i, i] = 1+a*ee[i]/(h[i]+ee[i])
                for j in range(n):
                    if ww[i][j]:
                        matrix[i, j] = -ww[i][j]/mp.sqrt(h[i]*h[j])
        return mp.lu_solve(matrix, rhs), matrix, rhs


def order(values):
    return sorted(range(len(values)), key=lambda i: (values[i], i))[:5]


def fixture():
    w = np.diag([1., 2., .5, 8., .25, 4., 1., 10.])
    for i, j, weight in ((0, 1, .75), (1, 2, .5), (2, 3, .25),
                         (3, 4, .8), (4, 5, .4), (5, 0, .125), (1, 4, .2)):
        w[i, j] = w[j, i] = weight
    return sparse.csr_matrix(w), np.array([2., .01, 0., .2, 0., .5, .001, 100.])


def unit_star(leaf_boundary=15.):
    # h_center=4, h_leaf=1. For alpha=5, e_center=18 and e_leaf=15,
    # the exact positive-degree solution is (11/8, 1, 1, 1, 1).
    w = np.eye(9)
    for i in range(1, 5):
        w[0, i] = w[i, 0] = 1.
    return sparse.csr_matrix(w), np.array([18.] + [leaf_boundary]*4 + [.001, 2., 7., 100.])


class DiagonalFreeTests(unittest.TestCase):
    def check(self, w, e, alpha=5., beta=25., expected_prefix=None):
        expected, matrix, rhs = oracle(w, e, alpha, beta)
        expected_prefix = order(expected) if expected_prefix is None else expected_prefix
        values, residual, event = d3.solve(w, e, alpha, beta)
        with mp.workdps(100):
            actual = mp.matrix([exact(v) for v in values])
            relative_error = max(abs(a-b)/b for a, b in zip(actual, expected))
            eta = max(abs(r)/b for r, b in zip(matrix*actual-rhs, rhs))
            self.assertLessEqual(relative_error, mp.mpf('1e-8'))
            self.assertLessEqual(eta, mp.mpf('1e-8'))
        self.assertLessEqual(residual, 1e-8)
        self.assertEqual(order(values), expected_prefix)
        self.assertEqual(event['returned_prefix_local'], expected_prefix)
        checked = d3.certify_saved(w, e, values, event, alpha, beta)
        self.assertEqual(checked['prefix'], expected_prefix)
        self.assertEqual(checked['linear_systems'], event['linear_systems'])
        self.assertLessEqual(checked['relative'], 1e-8)
        return values, event, expected

    def test_independent_ideal_equation_with_mixed_components(self):
        w, e = fixture()
        values, event, _ = self.check(w, e)
        self.assertEqual(event['analytic_nodes_local'], [6, 7])
        self.assertEqual(values[6], 1)
        self.assertEqual(values[7], 1)
        self.assertEqual(event['linear_systems'], 1)
        self.assertEqual(event['positive_degree_count'], 6)

    def test_recomputes_degree_probability_and_preserves_inputs(self):
        w, e = fixture()
        before = (w.data.copy(), w.indices.copy(), w.indptr.copy(), e.copy())
        values, _, _ = self.check(w, e)
        for actual, saved in zip((w.data, w.indices, w.indptr, e), before):
            np.testing.assert_array_equal(actual, saved)
        original_values, _, _ = base.solve(w, e, source='original')
        self.assertGreater(max(abs(float(a)-float(b)) for a, b in zip(values, original_values)), .01)

    def test_diagonal_values_do_not_change_scores(self):
        w, e = fixture()
        changed = w.copy()
        changed.setdiag([.125, 32., 0., 1., 400., 2., 3., 0.])
        values, _, event = d3.solve(w, e)
        other, _, other_event = d3.solve(changed, e)
        self.assertEqual(list(values), list(other))
        self.assertNotEqual(event['input_sha256'], other_event['input_sha256'])
        self.assertEqual(event['scoring_input_sha256'], other_event['scoring_input_sha256'])
        with self.assertRaises(ValueError):
            d3.certify_saved(changed, e, values, event)

    def test_all_analytic_nodes_with_different_boundaries_tie_exactly(self):
        w = sparse.diags([1., 0., 3., .25, 8., 0., 1.], format='csr')
        e = np.array([np.nextafter(0., 1.), .001, 1., 2., 8., 100., 1e200])
        with patch.object(d3, 'cg', side_effect=AssertionError('No analytic CG')), \
                patch.object(d3, 'splu', side_effect=AssertionError('No analytic LU')), \
                patch.object(base.Block, 'solve', side_effect=AssertionError('No analytic refinement')):
            values, residual, event = d3.solve(w, e)
            checked = d3.certify_saved(w, e, values, event)
        self.assertTrue(all(value == 1 for value in values))
        self.assertEqual(event['linear_systems'], 0)
        self.assertEqual(event['candidate_method'], 'analytic')
        self.assertEqual(event['returned_prefix_local'], [0, 1, 2, 3, 4])
        self.assertEqual(checked['prefix'], [0, 1, 2, 3, 4])
        self.assertEqual(residual, 0.)

    def test_positive_degree_exact_one_ties_with_analytic_nodes(self):
        w, e = unit_star()
        values, _, event = d3.solve(w, e)
        self.assertEqual(values[0], mp.mpf(11)/8)
        self.assertTrue(all(value == 1 for value in values[1:]))
        self.assertEqual(event['exact_one_nodes_local'], list(range(1, 9)))
        self.assertEqual(event['returned_prefix_local'], [1, 2, 3, 4, 5])
        self.assertEqual(d3.certify_saved(w, e, values, event)['prefix'], [1, 2, 3, 4, 5])

    def test_positive_degree_one_ulp_below_one_is_not_merged(self):
        w, e = unit_star(np.nextafter(15., 16.))
        # Exact leaf transposition symmetry defines the tie order; the MP LU
        # oracle's final rounding bits do not define an ordering inside a tie.
        values, event, expected = self.check(w, e, expected_prefix=[1, 2, 3, 4, 5])
        self.assertTrue(all(expected[i] < 1 for i in range(1, 5)))
        self.assertTrue(all(values[i] < 1 for i in range(1, 5)))
        self.assertEqual(event['exact_one_nodes_local'], [5, 6, 7, 8])
        self.assertEqual(event['returned_prefix_local'], [1, 2, 3, 4, 5])
        self.assertGreater(event['dps'], 0)

    def test_positive_degree_one_ulp_above_one_orders_after_analytic_nodes(self):
        w, e = unit_star(np.nextafter(15., 0.))
        values, event, expected = self.check(w, e, expected_prefix=[5, 6, 7, 8, 1])
        self.assertTrue(all(expected[i] > 1 for i in range(1, 5)))
        self.assertTrue(all(values[i] > 1 for i in range(1, 5)))
        self.assertEqual(event['exact_one_nodes_local'], [5, 6, 7, 8])
        self.assertEqual(event['returned_prefix_local'], [5, 6, 7, 8, 1])

    def test_nondefault_alpha_beta_and_wrong_readback_parameters(self):
        w, e = fixture()
        values, event, _ = self.check(w, e, alpha=2.5, beta=9.5)
        for parameters in (dict(alpha=5., beta=9.5), dict(alpha=2.5, beta=25.)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                d3.certify_saved(w, e, values, event, **parameters)

    def test_zero_degree_without_boundary_fails(self):
        with self.assertRaisesRegex(np.linalg.LinAlgError, 'Zero internal degree'):
            d3.solve(sparse.eye(3, format='csr'), np.array([1., 0., 2.]))

    def test_positive_degree_unanchored_component_fails(self):
        w = sparse.block_diag(([[1., .5], [.5, 1.]], [[1., .25], [.25, 1.]]), format='csr')
        with self.assertRaisesRegex(np.linalg.LinAlgError, 'Unanchored'):
            d3.solve(w, np.array([1., 0., 0., 0.]))

    def test_invalid_source_asymmetry_and_duplicate_coordinates_fail(self):
        w, e = fixture()
        with self.assertRaises(ValueError):
            d3.solve(w, e, source='original')
        bad = w.copy().tolil()
        bad[0, 1] = .5
        with self.assertRaisesRegex(np.linalg.LinAlgError, 'symmetry'):
            d3.solve(bad.tocsr(), e)
        duplicated = sparse.csr_matrix((np.array([1., .25, .25, .5, 1.]),
                                       np.array([0, 1, 1, 0, 1]), np.array([0, 3, 5])),
                                      shape=(2, 2))
        with self.assertRaisesRegex(np.linalg.LinAlgError, 'Duplicate'):
            d3.solve(duplicated, np.ones(2))

    def test_saved_analytic_point_must_be_exact_one_even_after_rehash(self):
        w, e = fixture()
        values, _, event = d3.solve(w, e)
        changed = list(values)
        changed[6] = mp.mpf(np.nextafter(1., 2.))
        damaged = copy.deepcopy(event)
        damaged['returned_scores_sha256'] = base.digest_scores(changed)
        with self.assertRaises(np.linalg.LinAlgError):
            d3.certify_saved(w, e, changed, damaged)

    def test_saved_scores_and_prefix_are_checked_independently_of_digest(self):
        w, e = fixture()
        values, _, event = d3.solve(w, e)
        changed = list(values)
        changed[0] *= mp.mpf('1.01')
        damaged = copy.deepcopy(event)
        damaged['returned_scores_sha256'] = base.digest_scores(changed)
        with self.assertRaises(np.linalg.LinAlgError):
            d3.certify_saved(w, e, changed, damaged)
        damaged = copy.deepcopy(event)
        damaged['returned_prefix_local'] = list(reversed(damaged['returned_prefix_local']))
        with self.assertRaises(ValueError):
            d3.certify_saved(w, e, values, damaged)

    def test_saved_system_count_and_exact_one_declaration_are_reconstructed(self):
        w, e = unit_star()
        values, _, event = d3.solve(w, e)
        for key, value in (('linear_systems', 0), ('exact_one_nodes_local', [5, 6, 7, 8]),
                           ('positive_degree_count', 4), ('components', 1)):
            damaged = copy.deepcopy(event)
            damaged[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d3.certify_saved(w, e, values, damaged)
        w, e = sparse.eye(7, format='csr'), np.arange(1., 8.)
        values, _, event = d3.solve(w, e)
        self.assertEqual(d3.certify_saved(w, e, values, event)['linear_systems'], 0)
        event['linear_systems'] = 1
        with self.assertRaises(ValueError):
            d3.certify_saved(w, e, values, event)

    def test_saved_readback_never_solves_or_modifies_points(self):
        w, e = unit_star()
        values, _, event = d3.solve(w, e)
        digest = base.digest_scores(values)
        forbidden = AssertionError('Readback cannot solve or refine')
        with ExitStack() as stack:
            for owner, name in ((d3, 'solve'), (d3, 'cg'), (d3, 'splu'),
                                (base, 'solve'), (base.Block, 'solve'),
                                (mp, 'lu_solve'), (np.linalg, 'solve')):
                stack.enter_context(patch.object(owner, name, side_effect=forbidden))
            checked = d3.certify_saved(w, e, values, event)
        self.assertEqual(checked['prefix'], event['returned_prefix_local'])
        self.assertEqual(base.digest_scores(values), digest)

    def test_large_positive_block_stays_sparse_and_merges_global_prefix(self):
        count = 65
        positive = sparse.kron(sparse.eye(count, format='csr'),
                               sparse.csr_matrix([[1., .5], [.5, 1.]]), format='csr')
        w = sparse.block_diag((positive, sparse.eye(3, format='csr')), format='csr')
        boundaries = .25+np.arange(count)/64.
        e = np.r_[np.repeat(boundaries, 2), [.001, 1., 100.]]
        expected = np.r_[np.repeat(1.+.5/boundaries, 2), np.ones(3)]
        with patch.object(d3, 'cg', wraps=d3.cg) as cg_call:
            values, residual, event = d3.solve(w, e)
        self.assertTrue(cg_call.called)
        matrix = cg_call.call_args.args[0]
        self.assertTrue(sparse.issparse(matrix))
        self.assertEqual(matrix.shape, (130, 130))
        self.assertLessEqual(matrix.nnz, 4*count)
        self.assertEqual(event['candidate_method'], 'cg')
        self.assertEqual(event['linear_systems'], 1)
        np.testing.assert_allclose(np.asarray(values, dtype=float), expected, rtol=1e-8, atol=0.)
        self.assertEqual(event['returned_prefix_local'], [130, 131, 132, 128, 129])
        self.assertLessEqual(residual, 1e-8)
        with patch.object(d3, 'cg', side_effect=AssertionError('Readback CG')), \
                patch.object(d3, 'splu', side_effect=AssertionError('Readback LU')):
            checked = d3.certify_saved(w, e, values, event)
        self.assertEqual(checked['prefix'], event['returned_prefix_local'])

    def test_install_emits_one_event_for_full_scoring_call(self):
        from MODEL.Paper_RFDP import PaperFullIterative as model
        old_solver, old_events = model.solve_hybrid_sparse, list(base.EVENTS)
        base.EVENTS.clear()
        try:
            self.assertIs(d3.install(), old_solver)
            w, e = fixture()
            diagnostics = {}
            values, residual = model.solve_hybrid_sparse(w, e, diagnostics=diagnostics)
            self.assertEqual(len(base.EVENTS), 1)
            self.assertEqual(base.EVENTS[0]['solver_id'], d3.SOLVER_ID)
            self.assertEqual(base.EVENTS[0]['size'], len(e))
            self.assertEqual(len(values), len(e))
            self.assertLessEqual(residual, 1e-8)
            self.assertEqual(diagnostics['uncertified_cg'], 0)
        finally:
            model.solve_hybrid_sparse = old_solver
            base.EVENTS[:] = old_events


if __name__ == '__main__':
    unittest.main()
