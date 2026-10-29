"""Small mathematical checks; no retrieval or runtime qualification."""
import unittest

import numpy as np
from scipy import sparse

from experiments.continuous_fitting.model import assemble_system
from MODEL.Paper_RFDP.PaperFullIterative import solve_hybrid_sparse


class ContinuousFittingTests(unittest.TestCase):
    def setUp(self):
        self.w = np.array([[1., 2., 0., .25], [2., 1., .5, 0.],
                           [0., .5, 1., 1.], [.25, 0., 1., 1.]])
        self.e = np.array([2., .001, 0., .2])

    def solve(self, e=None, source='continuous', **kwargs):
        system = assemble_system(self.w, self.e if e is None else e,
                                 source=source, **kwargs)
        return np.linalg.solve(system.matrix.toarray(), system.rhs)

    def test_matrix_and_original_solution_match_existing_implementation(self):
        original = assemble_system(self.w, self.e, source='original')
        candidate = assemble_system(self.w, self.e)
        h = self.w.sum(axis=1)
        inv_h = sparse.diags(1. / np.sqrt(h))
        matrix = (sparse.diags(1. + 5. * self.e / (h + self.e))
                  - inv_h @ sparse.csr_matrix(self.w) @ inv_h).toarray()
        np.testing.assert_array_equal(candidate.matrix.toarray(), matrix)
        np.testing.assert_array_equal(original.matrix.toarray(), matrix)
        scores, _ = solve_hybrid_sparse(self.w, self.e, backend='direct')
        np.testing.assert_allclose(self.solve(source='original'), scores, rtol=1e-13)

    def test_independent_energy_gradient(self):
        system = assemble_system(self.w, self.e)
        h = self.w.sum(axis=1)
        p = self.e / (h + self.e)
        def energy(v):
            z = v / np.sqrt(h)
            smooth = .25 * np.sum(self.w * (z[:, None] - z[None, :]) ** 2)
            return smooth + 2.5 * np.sum(p * (v - 1.) ** 2) - 25. * np.sum((1. - p) * v)
        v = np.array([1.25, 4., 2., .75])
        eps = 1e-4
        numerical = np.array([(energy(v + eps * row) - energy(v - eps * row))
                              / (2. * eps) for row in np.eye(4)])
        np.testing.assert_allclose(numerical, system.matrix @ v - system.rhs,
                                   rtol=1e-9, atol=1e-9)
        solution = self.solve()
        self.assertTrue(np.all(solution > 0))
        self.assertGreater(np.linalg.eigvalsh(system.matrix.toarray()).min(), 0.)
        for row in np.eye(4):
            self.assertGreater(energy(solution + row), energy(solution))

    def test_weak_boundary_limit_removes_original_jump(self):
        e0 = np.array([1., 0., 0., 0.])  # An anchor survives in this fixed component.
        target = self.solve(e0)
        errors = []
        for eps in [1e-4, 1e-6, 1e-8]:
            e = e0.copy()
            e[1] = eps
            errors.append(np.linalg.norm(self.solve(e) - target))
        self.assertLess(errors[1], errors[0] / 50.)
        self.assertLess(errors[2], errors[1] / 50.)
        self.assertGreater(np.linalg.norm(self.solve(e, source='original')
                                         - self.solve(e0, source='original')), 1.)

    def test_equal_parameters_recover_original(self):
        np.testing.assert_array_equal(self.solve(alpha=5., beta=5.),
                                      self.solve(source='original', alpha=5., beta=5.))

    def test_uniform_direct_boundary_changes_only_scale(self):
        e = self.w.sum(axis=1) / 4.  # p=.2 for every node.
        original = self.solve(e, source='original')
        candidate = self.solve(e)
        np.testing.assert_allclose(candidate, original * (5. * .2 + 25. * .8) / 5.,
                                   rtol=1e-13)

    def test_common_weight_scale_invariance(self):
        a = assemble_system(self.w, self.e)
        b = assemble_system(8. * self.w, 8. * self.e)
        # Mathematical scale invariance; sqrt rounds differently after scaling.
        np.testing.assert_allclose(a.matrix.toarray(), b.matrix.toarray(), rtol=1e-14, atol=0.)
        np.testing.assert_array_equal(a.rhs, b.rhs)

    def test_nonuniform_boundary_can_change_ranking(self):
        np.testing.assert_array_equal(np.argsort(self.solve(source='original')), [0, 1, 3, 2])
        np.testing.assert_array_equal(np.argsort(self.solve()), [0, 3, 1, 2])

    def test_regular_graph_all_direct_is_affine_equivalent(self):
        w = np.array([[1., 1., 0., 1.], [1., 1., 1., 0.],
                      [0., 1., 1., 1.], [1., 0., 1., 1.]])
        e = np.array([.01, .1, 2., 7.])  # Unequal p, equal h, no B nodes.
        original = assemble_system(w, e, source='original')
        candidate = assemble_system(w, e)
        x0 = np.linalg.solve(original.matrix.toarray(), original.rhs)
        xc = np.linalg.solve(candidate.matrix.toarray(), candidate.rhs)
        np.testing.assert_allclose(xc, 5. * x0 - 4., rtol=1e-13)
        np.testing.assert_array_equal(np.argsort(xc), np.argsort(x0))

    def test_unanchored_component_rejected_without_ridge(self):
        w = sparse.block_diag([self.w, self.w])
        with self.assertRaisesRegex(ValueError, 'anchor'):
            assemble_system(w, np.r_[self.e, np.zeros(4)])
        with self.assertRaisesRegex(ValueError, 'anchor'):
            assemble_system(self.w, np.zeros(4))

    def test_invalid_graph_rejected(self):
        w = self.w.copy()
        w[0, 1] += .1
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            assemble_system(w, self.e)
        with self.assertRaises(ValueError):
            assemble_system(self.w, [1., -1., 0., 0.])
        with self.assertRaises(ValueError):
            assemble_system(self.w, self.e, beta=0.)


class CertifiedFittingTests(unittest.TestCase):
    def test_mp_oracle_and_saved_score_damage_rejection(self):
        import mpmath as mp
        from experiments.continuous_fitting.certified import solve, certify_saved, Block
        w = sparse.csr_matrix([[1., 2., 0., .25], [2., 1., .5, 0.],
                               [0., .5, 1., 1.], [.25, 0., 1., 1.]])
        e = np.array([2., .001, 0., .2])
        values, _, event = solve(w, e)
        with mp.workdps(90):
            h = [mp.fsum(mp.mpf(float(v)) for v in row) for row in w.toarray()]
            p = [mp.mpf(float(v))/(hh+mp.mpf(float(v))) for v, hh in zip(e, h)]
            m = mp.matrix([[int(i==j)*(1+5*p[i])-mp.mpf(float(w[i,j]))/mp.sqrt(h[i]*h[j])
                            for j in range(4)] for i in range(4)])
            b = mp.matrix([5*v+25*(1-v) for v in p]); exact = mp.lu_solve(m, b)
            cert = Block(w,e,5.,25.,'continuous').certify(values,80)
            self.assertTrue(cert['accepted'])
            for v, (lo, hi) in zip(exact, cert['intervals']):self.assertTrue(lo <= v <= hi)
        self.assertEqual(certify_saved(w,e,values)['prefix'],event['returned_prefix_local'])
        damaged = list(values); damaged[0] *= 1.01
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(w,e,damaged)

    def test_exact_ties_and_one_ulp_difference(self):
        from experiments.continuous_fitting.certified import solve, certify_saved
        w = sparse.eye(7, format='csr'); e = np.ones(7)
        values, _, event = solve(w,e)
        self.assertTrue(all(v == values[0] for v in values))
        self.assertEqual(event['returned_prefix_local'],[0,1,2,3,4])
        self.assertEqual(certify_saved(w,e,values)['prefix'],[0,1,2,3,4])
        e[-1] = np.nextafter(1.,2.)
        values, _, event = solve(w,e)
        self.assertLess(values[-1],values[0])
        self.assertEqual(event['returned_prefix_local'],[6,0,1,2,3])
        self.assertGreater(event['dps'],0)

    def test_actual_rank_and_independent_trajectory_readback(self):
        import copy
        import json
        from MODEL.Paper_RFDP import PaperFullIterative as mod
        from experiments.continuous_fitting import certified as cert
        from experiments.continuous_fitting.verify import verify_query
        # Two components; step five includes skipped solves, a bridge and final truncation.
        rng=np.random.default_rng(2041); n=17
        w=np.zeros((n,n))
        for ids in [list(range(8)),list(range(8,n))]:
            for i in ids:
                w[i,i]=1.
                for j in ids:
                    if i<j:w[i,j]=w[j,i]=float(rng.uniform(.05,.8))
        full=rng.uniform(.05,.95,(n,n));full=(full+full.T)/2;np.fill_diagonal(full,1.)
        import tempfile
        from pathlib import Path
        from experiments.continuous_fitting import full as runner
        from experiments.continuous_fitting.verify import read_rows
        graph=sparse.csr_matrix(w); similarity=full[0]; cert.EVENTS.clear()
        old=cert.install()
        try:
            with tempfile.TemporaryDirectory() as td:
                out=Path(td);(out/'queries').mkdir();(out/'accepted').mkdir()
                runner.OUT=out;runner.QN=1;runner.GN=n-1
                runner.CFG={'source':'continuous','depth':16};runner.SIG='synthetic'
                runner.GRAPH=graph;runner.FULL=full
                # Exact chosen similarity row without needing an affinity=XX^T fixture.
                runner.X=np.eye(n);runner.INIT_ERROR=None
                runner.ENGINE=mod.FullIHyRDP(graph,full)
                row=runner.compute_query(0)
                self.assertNotIn('error',row,row.get('error'))
                accepted=runner.accept_query(0)
                self.assertNotIn('error',accepted,accepted.get('error'))
                prefix=np.load(out/'queries/0000.prefix.npy')
                rank=np.load(out/'queries/0000.rankings.npy')
                rows=list(read_rows(out/'queries/0000.trace.jsonl.gz'))
                similarity=runner.X@runner.X[0]
                diag=row['diagnostics']
        finally:mod.solve_hybrid_sparse=old
        result=verify_query(graph,full,similarity,0,prefix,rank,rows,'continuous',1,16)
        self.assertGreater(result['bridge_events'],0)
        self.assertEqual(result['solver_calls'],diag['solver_calls'])
        damaged=copy.deepcopy(rows);damaged[0]['selected'][0]=16
        with self.assertRaises(AssertionError):verify_query(graph,full,similarity,0,prefix,rank,damaged,'continuous',1,16)
        with self.assertRaises(AssertionError):verify_query(graph,full,similarity,0,prefix,rank,rows[:-1],'continuous',1,16)


if __name__ == '__main__':
    unittest.main()
