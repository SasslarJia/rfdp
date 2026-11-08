"""Numerical and reuse contracts; fixed algebraic systems, no retrieval data."""
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified, fitting_workspace as ws


def stats():
    return dict(cg_iterations=0, cg_calls=0, linear_solve_calls=0,
                direct_calls=0, additional_factorizations=0)


class FittingWorkspaceTests(unittest.TestCase):
    def work(self, w, e):
        w, e, labels, nc = certified.original.validate(w, e, 5., 25.)
        return ws.Workspace(w, e, 5., 25., labels, nc)

    def test_cached_assembly_matches_declared_weighted_equation(self):
        w = sparse.csr_matrix([[2., .25, .125], [.25, 1., .5], [.125, .5, 3.]])
        e = np.array([.5, 0., .25]); weights = [.3, 1., .7]
        work = self.work(w,e); system = work.system(weights, stats())
        h = np.asarray(w.sum(axis=1)).ravel()
        expected = np.eye(3)-w.toarray()/np.sqrt(h[:,None]*h[None,:])
        expected += np.diag(5*e/(h+e)*weights)
        np.testing.assert_allclose(system.matrix.toarray(), expected, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(system.rhs, [1.5,25.,3.5])

    def test_lu_factor_is_reused_for_correction_of_same_system(self):
        work = self.work(sparse.csr_matrix([[1.,.25],[.25,2.]]),np.array([.5,0.]))
        counters = stats(); system = work.system([.4,1.], counters)
        x = system.candidate(None, 1e-12)
        with mp.workdps(40):
            repaired, _ = system.block().solve([mp.mpf(float(v))*mp.mpf('1.01') for v in x],40)
        self.assertEqual(counters['additional_factorizations'],1)
        expected = np.linalg.solve(system.matrix.toarray(),system.rhs)
        np.testing.assert_allclose(np.asarray(repaired,float),expected,rtol=1e-12)
        changed = work.system([.2,1.],counters)
        changed.candidate(x,1e-12)
        self.assertEqual(counters['additional_factorizations'],2)

    def test_already_accurate_point_needs_no_factorization(self):
        work = self.work(sparse.eye(1,format='csr'),np.array([1.]))
        system = work.system([1.],stats())
        with mp.workdps(40), patch.object(system,'factorize',side_effect=AssertionError('unneeded LU')):
            x, report = system.block().solve([mp.mpf(2)],40)
        self.assertEqual(x,[2]); self.assertEqual(report['iterations'],0)

    def test_sparse_cg_uses_previous_point_and_spd_preconditioning(self):
        n=130
        w=sparse.diags([np.full(n-1,.25),np.ones(n),np.full(n-1,.25)],[-1,0,1],format='csr')
        e=np.full(n,.5);work=self.work(w,e);counters=stats()
        system=work.system(np.linspace(.4,.8,n),counters)
        previous=np.full(n,3.)
        with patch.object(ws,'cg',wraps=ws.cg) as solve:
            x=system.candidate(previous,1e-12)
        call=solve.call_args.kwargs
        np.testing.assert_array_equal(call['x0'],previous)
        self.assertTrue(np.all(call['M'].diagonal()>0))
        expected=np.linalg.solve(system.matrix.toarray(),system.rhs)
        np.testing.assert_allclose(x,expected,rtol=1e-10,atol=1e-12)
        self.assertEqual(counters['cg_calls'],1)


if __name__=='__main__':
    unittest.main()
