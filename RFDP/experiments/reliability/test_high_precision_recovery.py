"""Same-equation numerical regression tests; no full ReID experiment."""
from pathlib import Path
import types
import tempfile
import unittest

import mpmath as mp
import numpy as np
from scipy import sparse

from high_precision_recovery import (
    RecoveryFailure, assemble_hybrid, certify_solution, install,
    solve_fixed_system, system_sha256, save_evidence, verify_evidence,
)


class RecoveryTests(unittest.TestCase):
    def test_near_singular_spd_matches_independent_mp_lu(self):
        eps = np.spacing(1.)
        matrix = sparse.csr_matrix([[1. + 2 * eps, -1.], [-1., 1. + 3 * eps]])
        rhs = np.array([5., 25.])
        before = system_sha256(matrix, rhs)
        scores, residual, report = solve_fixed_system(matrix, rhs)
        self.assertEqual(system_sha256(matrix, rhs), before)
        self.assertLessEqual(residual, 1e-8)
        self.assertEqual(report['certificate']['matrix_class'], 'SPD-certified')
        with mp.workdps(120):
            oracle = mp.lu_solve(mp.matrix([[mp.mpf(float(v)) for v in row] for row in matrix.toarray()]), mp.matrix(rhs.tolist()))
            for actual, expected in zip(scores, oracle):
                if isinstance(actual, mp.mpf):
                    lifted = actual
                else:
                    n, d = actual.as_integer_ratio()
                    lifted = mp.mpf(n) / d
                self.assertLess(abs((lifted - expected) / expected), mp.mpf('1e-15'))

    def test_nonsymmetric_rounded_z_matrix_is_not_called_spd(self):
        matrix = sparse.csr_matrix([[1.1, -1.], [-np.nextafter(1., 0.), 1.2]])
        scores, residual, report = solve_fixed_system(matrix, np.array([5., 25.]))
        self.assertLessEqual(residual, 1e-8)
        self.assertFalse(report['certificate']['exact_symmetric'])
        self.assertEqual(report['certificate']['matrix_class'], 'nonsymmetric-M-matrix-certified')
        self.assertEqual(report['certificate']['spd_status'], 'not-symmetric-rounded-matrix')

    def test_singular_and_indefinite_rounded_systems_are_rejected(self):
        for matrix in ([[1., -1.], [-1., 1.]], [[1., -2.], [-2., 1.]]):
            with self.assertRaises(RecoveryFailure):
                solve_fixed_system(sparse.csr_matrix(matrix), np.array([5., 25.]), max_refinement=2)

    def test_actual_output_not_precast_vector_is_certified(self):
        eps = np.spacing(1.)
        matrix = sparse.csr_matrix([[1. + eps, -1.], [-1., 1. + 2 * eps]])
        rhs = np.array([3., 7.])
        scores, residual, report = solve_fixed_system(matrix, rhs)
        certificate = certify_solution(matrix, rhs, scores)
        self.assertEqual(residual, certificate['relative_residual'])
        self.assertTrue(certificate['exact_residual_pass'])
        self.assertTrue(certificate['exact_positive_scaling_certificate'])
        # A deliberately insufficient representation must not inherit the MP residual.
        rounded = np.array([float(value) for value in scores], dtype=np.float64)
        poor = certify_solution(matrix, rhs, rounded)
        self.assertFalse(poor['exact_residual_pass'])

    def test_mpf_scores_sort_and_trace_serialization_at_low_global_precision(self):
        with mp.workdps(100):
            values = np.asarray([mp.mpf(1) + mp.mpf('1e-60'), mp.mpf(1), mp.mpf(2)], dtype=object)
        with mp.workdps(15):
            np.testing.assert_array_equal(np.lexsort((np.array([0, 1, 2]), values)), [1, 0, 2])
        import json
        encoded = json.dumps({'scores': [float(value) for value in values]}, allow_nan=False)
        self.assertIn('scores', encoded)

    def test_runtime_wrapper_preserves_original_success_and_same_equation(self):
        calls = []
        def previous(w, e, alpha=5., beta=25., backend='auto', diagnostics=None):
            calls.append(True)
            raise np.linalg.LinAlgError('forced numerical failure')
        module = types.SimpleNamespace(solve_hybrid_sparse=previous)
        events = []
        installed = install(module, events=events)
        self.assertIs(install(module, events=events), installed)
        w = sparse.csr_matrix([[1., .3], [.3, 1.]])
        e = np.array([.4, 0.])
        stats = {}
        scores, residual = module.solve_hybrid_sparse(w, e, diagnostics=stats)
        matrix, rhs = assemble_hybrid(w, e)
        self.assertTrue(certify_solution(matrix, rhs, scores)['exact_residual_pass'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(stats['direct_calls'], 1)
        self.assertLessEqual(residual, 1e-8)

    def test_captured_query143_system(self):
        path = Path(__file__).with_name('query-0143-failed-system.npz')
        if not path.exists():
            self.skipTest('Captured diagnostic system not present')
        saved = np.load(path)
        matrix = sparse.csr_matrix((saved['m_data'], saved['m_indices'], saved['m_indptr']), shape=tuple(saved['m_shape']))
        w = sparse.csr_matrix((saved['w_data'], saved['w_indices'], saved['w_indptr']), shape=tuple(saved['w_shape']))
        assembled, rhs = assemble_hybrid(w, saved['e'], float(saved['alpha']), float(saved['beta']))
        self.assertEqual((matrix != assembled).nnz, 0)
        np.testing.assert_array_equal(rhs, saved['b'])
        scores, residual, report = solve_fixed_system(matrix, rhs)
        self.assertLessEqual(residual, 1e-8)
        self.assertEqual(report['components'], 5)
        self.assertTrue(report['certificate']['exact_positive_scaling_certificate'])
        self.assertFalse(report['certificate']['exact_symmetric'])
        self.assertTrue(np.all(np.asarray([float(value) for value in scores]) > 0))

    def test_persisted_equation_and_exact_output_are_independently_verifiable(self):
        matrix = sparse.csr_matrix([[1.1, -1.], [-1., 1.2]])
        rhs = np.array([5., 25.])
        scores, residual, report = solve_fixed_system(matrix, rhs)
        with tempfile.TemporaryDirectory() as directory:
            evidence = save_evidence(matrix, rhs, scores, report, directory)
            self.assertEqual(verify_evidence(evidence['npz']), report['certificate'])
            repeated = save_evidence(matrix, rhs, scores, report, directory)
            self.assertNotEqual(repeated['npz'], evidence['npz'])
            self.assertEqual(verify_evidence(evidence['npz']), report['certificate'])
            self.assertEqual(verify_evidence(repeated['npz']), report['certificate'])
            path = Path(evidence['npz'])
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(RecoveryFailure, 'hash mismatch'):
                verify_evidence(path)

    def test_duplicate_coordinate_input_is_rejected_without_summing(self):
        matrix = sparse.csr_matrix((np.array([1., 1., -1., -1., 3.]),
                                    np.array([0, 0, 1, 0, 1]), np.array([0, 3, 5])), shape=(2, 2))
        self.assertFalse(matrix.has_canonical_format)
        with self.assertRaisesRegex(RecoveryFailure, 'Noncanonical'):
            solve_fixed_system(matrix, np.array([5., 25.]))

    def test_unsorted_coordinates_are_sorted_without_changing_equation(self):
        matrix = sparse.csr_matrix((np.array([-1., 2., 3., -1.]),
                                    np.array([1, 0, 1, 0]), np.array([0, 2, 4])), shape=(2, 2))
        self.assertFalse(matrix.has_sorted_indices)
        scores, residual, report = solve_fixed_system(matrix, np.array([5., 25.]))
        self.assertTrue(report['certificate']['exact_residual_pass'])
        self.assertFalse(matrix.has_sorted_indices)  # Caller input is untouched.


if __name__ == '__main__':
    unittest.main(verbosity=2)
