"""New response identity and local-margin semantics, not a retrieval pilot."""
import unittest
import numpy as np
from scipy import sparse
from experiments.continuous_fitting.model import assemble_system
from experiments.stationary_measure.diagnose import degree_response, local_margin, summarize


class ResponseTests(unittest.TestCase):
    def test_degree_response_and_pair_margin_identity(self):
        w = sparse.csr_matrix([[1.,2.,.5,0.],[2.,1.,0.,.7],[.5,0.,1.,.3],[0.,.7,.3,1.]])
        e = np.array([1.,.2,0.,0.]); h = np.asarray(w.sum(1)).ravel()
        old = assemble_system(w,e,source='original'); new = assemble_system(w,e,source='stationary_measure')
        psi = np.linalg.solve(old.matrix.toarray(),old.rhs); x = np.linalg.solve(new.matrix.toarray(),new.rhs)
        p=e/(h+e); P=w.toarray()/h[:,None]; S=w.toarray()/np.sqrt(h[:,None]*h[None,:]); N=np.diag(1+5*p)-P
        delta=np.linalg.solve(N,(P-S)@psi)
        np.testing.assert_allclose(x-psi,delta,rtol=1e-11,atol=1e-12)
        self.assertAlmostEqual(x[2]-x[0],psi[2]-psi[0]+delta[2]-delta[0],places=11)
        stats,_=degree_response(w,e,psi,x)
        self.assertLess(stats['relative_response_closure'],1e-13)
        damaged,_=degree_response(w,e,psi,x+np.array([.1,0,0,0]))
        self.assertGreater(damaged['relative_response_closure'],1e-5)

    def test_regular_degree_and_gallery_margin(self):
        w=sparse.csr_matrix([[1.,.5,.5],[.5,1.,.5],[.5,.5,1.]])
        e=np.array([1.,0.,.2]); a=assemble_system(w,e,source='original')
        psi=np.linalg.solve(a.matrix.toarray(),a.rhs)
        stats,_=degree_response(w,e,psi,psi)
        self.assertEqual(stats['log_degree_std'],0.)
        self.assertLess(stats['relative_response_closure'],1e-14)
        self.assertEqual(local_margin(np.array([4.,1.,3.]),np.array([True,False,False]),np.array([False,True,True])),-3.)
        self.assertIsNone(local_margin(np.ones(3),np.zeros(3,dtype=bool),np.ones(3,dtype=bool)))
        self.assertEqual(summarize([])['queries'],0)


if __name__=='__main__': unittest.main()
