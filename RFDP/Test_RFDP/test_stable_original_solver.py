"""Equation/ranking equivalence and mandatory fallback, not retrieval trials."""
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified
from experiments.continuous_fitting import stable_original_solver as stable


class StableOriginalSolverTests(unittest.TestCase):
    def graph(self, n=12, scale=1.):
        edge = np.linspace(.03, .16, n - 1) * scale
        w = sparse.diags([edge, np.ones(n), edge], [-1, 0, 1], format='csr')
        e = np.linspace(.003, .09, n) * scale
        e[3::4] = 0.
        return w, e

    def exact(self, w, e):
        with mp.workdps(80):
            a = w.toarray()
            h = [mp.fsum(mp.mpf(float(v)) for v in row) for row in a]
            root = [mp.sqrt(v) for v in h]
            lap = mp.matrix(len(e))
            for i in range(len(e)):
                for j in range(len(e)):
                    if i != j:
                        lap[i, j] = -mp.mpf(float(a[i, j]))
                lap[i, i] = mp.fsum(-lap[i, j] for j in range(len(e)) if j != i) + 5*h[i]*mp.mpf(float(e[i]))/(h[i]+mp.mpf(float(e[i])))
            z = mp.lu_solve(lap, mp.matrix([r*(5 if t > 0 else 25) for r, t in zip(root, e)]))
            return [r*v for r, v in zip(root, z)]

    def test_same_exact_equation_at_ordinary_and_self_dominated_scales(self):
        for scale in (1., 1e-8, 1e-14):
            w, e = self.graph(scale=scale)
            expected = self.exact(w, e)
            x, residual, ev = stable.solve(w, e)
            self.assertEqual(ev['returned_prefix_local'], sorted(range(len(e)), key=lambda i: expected[i])[:5])
            self.assertLessEqual(residual, certified.TOL)
            self.assertEqual(certified.certify_saved(w, e, x, source='original')['prefix'], ev['returned_prefix_local'])
            self.assertEqual(ev['direct_calls'], 1)
            self.assertEqual(ev['cg_calls'], 0)

    def test_mp_fallback_reuses_factor_and_keeps_one_ulp_order(self):
        w = sparse.eye(7, format='csr')
        e = np.ones(7)
        e[-1] = np.nextafter(1., 2.)
        with patch.object(stable, 'splu', wraps=stable.splu) as factor:
            x, residual, ev = stable.solve(w, e)
        self.assertEqual(factor.call_count, 1)
        self.assertGreater(ev['dps'], 0)
        self.assertEqual(ev['returned_prefix_local'], [6, 0, 1, 2, 3])
        self.assertLessEqual(residual, certified.TOL)
        self.assertEqual(certified.certify_saved(w, e, x, source='original')['prefix'], ev['returned_prefix_local'])

    def test_rejected_fast_paths_require_unchanged_mp_certificate(self):
        w, e = self.graph()
        with patch.object(certified, 'fast_certificate', return_value=None), patch.object(stable, 'splu', wraps=stable.splu) as factor:
            x, residual, ev = stable.solve(w, e)
        self.assertEqual(factor.call_count, 1)
        self.assertEqual(ev['certificate_backend'], 'outward-mp-row-equation')
        self.assertLessEqual(residual, certified.TOL)
        certified.certify_saved(w, e, x, source='original')

    def test_wide_certificate_does_not_silently_round_mp_points(self):
        w = sparse.eye(2, format='csr')
        with mp.workdps(50):
            points = [mp.mpf(2) + mp.mpf('1e-30'), mp.mpf(2)]
            self.assertIsNone(certified.fast_certificate(w, np.ones(2), 5., 25., points, 'original', wide=True))

    @unittest.skipUnless(np.finfo(np.longdouble).nmant > 53, 'extended float unavailable')
    def test_wide_intervals_enclose_independent_high_precision_solution(self):
        w, e = self.graph(scale=1e-8)
        expected = self.exact(w, e)
        x = [mp.mpf(float(v)) for v in expected]
        cert = certified.fast_certificate(w, e, 5., 25., x, 'original', wide=True)
        self.assertIsNotNone(cert)
        with mp.workdps(80):
            for exact, (lo, hi) in zip(expected, cert['intervals']):
                la, lb = lo.as_integer_ratio()
                ua, ub = hi.as_integer_ratio()
                self.assertLessEqual(mp.mpf(la)/lb, exact)
                self.assertLessEqual(exact, mp.mpf(ua)/ub)

    def test_complete_synthetic_bridged_admission_and_rank_match(self):
        from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
        from experiments.continuous_fitting.original_solver import WarmOriginalSolver
        rng = np.random.default_rng(1203)
        n = 17
        graph = np.zeros((n, n))
        for ids in (range(8), range(8, n)):
            for i in ids:
                graph[i, i] = 1.
                for j in ids:
                    if i < j:
                        graph[i, j] = graph[j, i] = rng.uniform(.01, .3)
        graph = sparse.csr_matrix(graph)
        affinity = rng.uniform(.03, .8, (n, n))
        affinity = (affinity + affinity.T)/2
        np.fill_diagonal(affinity, 1.)
        all_results = []
        for solver in (WarmOriginalSolver(events=[]), stable.StableOriginalSolver(events=[])):
            engine = FullIHyRDP(graph, affinity, stage_solver=solver)
            all_results.append([engine.rank([q], nq=16, input_similarity=affinity[q])[0].tolist() for q in (0, 2)])
        self.assertEqual(all_results[0], all_results[1])


if __name__ == '__main__':
    unittest.main()
