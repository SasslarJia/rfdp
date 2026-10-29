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


if __name__ == '__main__':
    unittest.main()
