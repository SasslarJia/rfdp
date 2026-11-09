"""Independent formula, order and actual-score tests; no retrieval experiment."""
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import sparse

from experiments.continuous_fitting.model import assemble_system as baseline
from experiments.relation_routes import smoothing as routes


def oracle(w, e, alpha, beta, source, cameras=None, target_count=None, theta=.5):
    """Dense MP energy assembled independently, used only for tiny unit cases."""
    n = len(e)
    ww = [[mp.mpf(float(v)) for v in row] for row in w]
    ee = [mp.mpf(float(v)) for v in e]
    h = [mp.fsum(row) for row in ww]
    c = mp.matrix(n)
    if source == 'camera_channel':
        d = [mp.fsum(ww[i][j] for j in range(n) if j != i) for i in range(n)]
        budgets = [[mp.fsum(ww[i][j] for j in range(n)
                           if j != i and int(cameras[i] == cameras[j]) == t)
                    for t in (0, 1)] for i in range(n)]
        omega = mp.matrix(n)
        for i in range(n):
            count = sum(v > 0 for v in budgets[i])
            for j in range(n):
                if i != j and ww[i][j]:
                    omega[i, j] = d[i]*ww[i][j]/(count*budgets[i][int(cameras[i] == cameras[j])])
        for i in range(n):
            for j in range(n):
                c[i, j] = (omega[i, j]+omega[j, i])/2
    else:
        m = [v/target_count for v in ee]
        s = [mp.sqrt(1-v*v) for v in m]
        for i in range(n):
            for j in range(n):
                if i != j:
                    c[i, j] = ((1-mp.mpf(theta))*ww[i][j] +
                               mp.mpf(theta)*(m[i]*m[j]+s[i]*ww[i][j]*s[j]))
    matrix = mp.matrix(n)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = -c[i, j]/mp.sqrt(h[i]*h[j])
        matrix[i, i] = mp.fsum(c[i, j] for j in range(n))/h[i]+mp.mpf(alpha)*ee[i]/(h[i]+ee[i])
    rhs = mp.matrix([alpha if v > 0 else beta for v in e])
    return mp.lu_solve(matrix, rhs), matrix, c


class SmoothingRouteTests(unittest.TestCase):
    def setUp(self):
        self.w = np.eye(6)
        for i, j, v in ((0, 1, .4), (0, 2, .2), (1, 2, .1), (1, 3, .3),
                        (2, 4, .5), (3, 4, .15), (4, 5, .25)):
            self.w[i, j] = self.w[j, i] = v
        self.e = np.array([.4, 0., .06, .01, 0., .2])
        self.cameras = np.array([1, 1, 2, 2, 3, 1])

    def options(self, source):
        return dict(cameras=self.cameras) if source == 'camera_channel' else dict(target_count=3, theta=.5)

    def test_independent_energy_oracle_and_actual_score_intervals(self):
        for source in routes.SOURCES:
            with self.subTest(source=source), mp.workdps(90):
                kw = self.options(source)
                actual, residual, event = routes.solve(self.w, self.e, source=source, **kw)
                expected, matrix, _ = oracle(self.w, self.e, 5., 25., source, **kw)
                system = routes.assemble_system(self.w, self.e, source=source, **kw)
                cert = routes.Block(system).certify(actual, 80)
                self.assertTrue(cert['accepted'])
                self.assertLessEqual(residual, 1e-8)
                for value, (lo, hi) in zip(expected, cert['intervals']):
                    self.assertTrue(lo <= value <= hi)
                fast = routes.fast_certificate(system, actual)
                self.assertIsNotNone(fast)
                for value, (lo, hi) in zip(expected, fast['intervals']):
                    self.assertTrue(mp.mpf(float(lo)) <= value <= mp.mpf(float(hi)))
                float_matrix = np.array(matrix.tolist(), dtype=float)
                np.testing.assert_allclose(system.matrix@np.eye(6), float_matrix, rtol=2e-14, atol=2e-15)
                np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float).ravel(), rtol=1e-11)
                self.assertTrue(np.all(np.asarray(actual, float) > 0))
                self.assertGreater(np.linalg.eigvalsh(float_matrix).min(), 0.)
                got = routes.certify_saved(self.w, self.e, actual, source=source, **kw)
                self.assertEqual(got['prefix'], event['returned_prefix_local'])

    def test_camera_budget_conservation_and_psd_bound(self):
        with mp.workdps(60):
            _, matrix, c = oracle(self.w, self.e, 5., 25., 'camera_channel', cameras=self.cameras)
            total = sum(self.w[i, j] for i in range(6) for j in range(6) if i != j)
            self.assertAlmostEqual(float(mp.fsum(c)), total, places=13)
            for i in range(6):
                for j in range(6):
                    if i != j:
                        self.assertGreaterEqual(c[i, j], mp.mpf(float(self.w[i, j]))/2)
            old = baseline(self.w, self.e, source='original').matrix.toarray()
            delta = np.array(matrix.tolist(), dtype=float)-old/2
            self.assertGreater(np.linalg.eigvalsh(delta).min(), 0.)

    def test_original_limits_and_nondefault_parameters(self):
        old = baseline(self.w, self.e, 2.5, 12.5, 'original').matrix.toarray()
        same = routes.assemble_system(self.w, self.e, 2.5, 12.5, 'camera_channel', cameras=np.ones(6, dtype=int))
        zero = routes.assemble_system(self.w, self.e, 2.5, 12.5, 'th_target', target_count=3, theta=0.)
        np.testing.assert_allclose(same.matrix@np.eye(6), old, atol=2e-15)
        np.testing.assert_allclose(zero.matrix@np.eye(6), old, atol=2e-15)
        for source in routes.SOURCES:
            x, _, _ = routes.solve(self.w, self.e, 2.5, 12.5, source, **self.options(source))
            routes.certify_saved(self.w, self.e, x, 2.5, 12.5, source, **self.options(source))

    def test_th_endpoint_keeps_original_relay_and_sparse_storage(self):
        w = np.array([[1., .4, 0.], [.4, 1., .2], [0., .2, 1.]])
        e = np.array([1., 0., 0.])
        system = routes.assemble_system(w, e, source='th_target', target_count=1, theta=.5)
        dense = system.matrix@np.eye(3)
        self.assertLess(dense[0, 1], 0.)
        self.assertLess(dense[1, 2], 0.)
        x, _, _ = routes.solve(w, e, source='th_target', target_count=1)
        routes.certify_saved(w, e, x, source='th_target', target_count=1)
        n = 150
        w = sparse.diags([np.full(n-1, .1), np.ones(n), np.full(n-1, .1)], [-1, 0, 1], format='csr')
        system = routes.assemble_system(w, np.linspace(.01, .3, n), source='th_target', target_count=5)
        self.assertLessEqual(system.base.nnz, w.nnz+n)
        self.assertEqual(system.vector.shape, (n,))
        # Sherman-Morrison uses one factorization, including multiple RHS.
        rhs = np.column_stack((system.rhs, np.arange(n, dtype=float)))
        solution = system.factorize().solve(rhs)
        np.testing.assert_allclose(system.matrix@solution, rhs, rtol=1e-11, atol=1e-10)

    def test_saved_scores_never_repaired_and_wrong_context_rejected(self):
        for source in routes.SOURCES:
            kw = self.options(source)
            x, _, _ = routes.solve(self.w, self.e, source=source, **kw)
            damaged = list(x); damaged[0] *= mp.mpf('1.01')
            with patch.object(routes.Block, 'solve', side_effect=AssertionError('readback must not solve')):
                with self.assertRaises(routes.ArithmeticFailure):
                    routes.certify_saved(self.w, self.e, damaged, source=source, **kw)
                wrong = dict(cameras=np.array([1, 2, 2, 3, 1, 1])) if source == 'camera_channel' else dict(target_count=1, theta=.5)
                with self.assertRaises(routes.ArithmeticFailure):
                    routes.certify_saved(self.w, self.e, x, source=source, **wrong)

    def test_exact_ties_and_non_tie_one_ulp(self):
        w = np.array([[1., .8, .8], [.8, 1., 0.], [.8, 0., 1.]])
        e = np.array([.2, 0., 0.])
        for source in routes.SOURCES:
            kw = dict(cameras=np.ones(3, dtype=int)) if source == 'camera_channel' else dict(target_count=1)
            x, _, event = routes.solve(w, e, source=source, **kw)
            self.assertEqual(x[1], x[2])
            self.assertEqual(event['returned_prefix_local'], [0, 1, 2])
            changed = w.copy(); changed[0, 2] = changed[2, 0] = np.nextafter(.8, 1.)
            x, _, event = routes.solve(changed, e, source=source, **kw)
            self.assertNotEqual(x[1], x[2])
            self.assertGreater(event['dps'], 0)
            routes.certify_saved(changed, e, x, source=source, **kw)

    def test_high_precision_sparse_refinement_without_dense_fallback(self):
        n = 270
        w = sparse.diags([np.full(n-1, .3), np.ones(n), np.full(n-1, .3)], [-1, 0, 1], format='csr')
        e = np.zeros(n); e[::3] = np.linspace(.001, .4, len(e[::3]))
        for source in routes.SOURCES:
            kw = dict(cameras=(np.arange(n)//2 % 3)) if source == 'camera_channel' else dict(target_count=5)
            with mp.workdps(50):
                system = routes.assemble_system(w, e, source=source, **kw)
                block = routes.Block(system)
                x, report = block.solve(None, 50)
                self.assertEqual(report['method'], 'accurate-residual-sparse-rank-one-refinement')
                self.assertTrue(block.certify(x, 50)['accepted'])

    def test_invalid_inputs_fail_instead_of_changing_the_model(self):
        with self.assertRaises(routes.ArithmeticFailure):
            routes.solve(self.w, np.zeros(6), source='camera_channel', cameras=self.cameras)
        for kw in (dict(target_count=0), dict(target_count=1.5), dict(target_count=1, theta=1.),
                   dict(target_count=1, theta=-.1)):
            with self.assertRaises(ValueError):
                routes.assemble_system(self.w, self.e, source='th_target', **kw)
        with self.assertRaises(ValueError):
            routes.assemble_system(self.w, self.e, source='camera_channel', cameras=None)
        with self.assertRaises(ValueError):
            routes.assemble_system(self.w, np.full(6, 2.), source='th_target', target_count=1)


if __name__ == '__main__':
    unittest.main()
