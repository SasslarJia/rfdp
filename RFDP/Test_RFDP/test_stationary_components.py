"""Independent dense checks of the new score decomposition and GMFPT control."""
import unittest
import numpy as np
from experiments.stationary_measure.score_components import components


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.w = np.array([[1.,2.,.5,0.],[2.,1.,0.,.7],[.5,0.,1.,.3],[0.,.7,.3,1.]])
        self.h = self.w.sum(1)

    def test_decomposition_matches_independent_original_and_gmfpt_equations(self):
        e = np.array([1., .2, 0., 0.]); p = e/(self.h+e)
        c = components(self.w, e)
        walk = self.w/self.h[:, None]
        g = np.linalg.inv(np.eye(4)-walk+5*np.diag(p))
        f = 5*g@np.diag(p)
        np.testing.assert_allclose(f.sum(1), 1., atol=1e-13)
        self.assertGreaterEqual(f.min(), 0.)
        np.testing.assert_allclose(c['u'], f[:, :2]@(1/p[:2]), rtol=1e-13)
        np.testing.assert_allclose(c['v'], g[:, 2:].sum(1), rtol=1e-13)
        root = np.sqrt(self.h)
        m = np.eye(4)-self.w/root[:,None]/root[None,:]+5*np.diag(p)
        np.testing.assert_allclose(c['original'], np.linalg.solve(m,np.where(e>0,5.,25.)), rtol=1e-13)
        np.testing.assert_allclose(c['gmfpt'], np.linalg.solve(np.eye(4)-self.w/(self.h+e)[:,None],np.ones(4)), rtol=1e-13)
        self.assertGreater(np.max(np.abs(c['original_mass']-1)), .01)

    def test_common_boundary_target_with_indirect_nodes(self):
        p = np.array([.2, .2, 0., 0.]); e = self.h*p/(1-p)
        c = components(self.w, e)
        np.testing.assert_allclose(c['u'], 5., atol=1e-12)
        np.testing.assert_allclose(c['gmfpt_u'], 5., atol=1e-12)
        np.testing.assert_allclose(c['score'], 5.+25*c['v'], atol=1e-12)
        self.assertGreater(np.ptp(c['score']), 1.)
        np.testing.assert_allclose(c['gmfpt_mass'], 1., atol=1e-12)

    def test_all_direct_common_target_and_invalid_anchor(self):
        c = components(self.w, self.h*.25)
        np.testing.assert_allclose(c['score'], 5., atol=1e-12)
        np.testing.assert_allclose(c['gmfpt'], 5., atol=1e-12)
        np.testing.assert_allclose(c['v'], 0.)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            components(self.w, np.zeros(4))


if __name__ == '__main__':
    unittest.main()
