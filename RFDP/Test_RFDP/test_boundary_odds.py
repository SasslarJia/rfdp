"""New-target algebra, exact-score certificate and unchanged rank interface."""
import unittest
import numpy as np
import mpmath as mp
from scipy import sparse
from experiments.continuous_fitting.model import assemble_system
from experiments.continuous_fitting.certified import solve, certify_saved, Block


class BoundaryOddsTests(unittest.TestCase):
    def setUp(self):
        self.w=sparse.csr_matrix([[1.,2.,0.,.25],[2.,1.,.5,0.],[0.,.5,1.,1.],[.25,0.,1.,1.]])
        self.e=np.array([2.,.001,0.,.2]);self.h=np.asarray(self.w.sum(1)).ravel()

    def test_target_energy_and_response_identity(self):
        a=assemble_system(self.w,self.e,source='boundary_odds');b=assemble_system(self.w,self.e,source='original')
        p=self.e/(self.h+self.e);r=self.e/self.h;direct=p>0;t=np.zeros(4);t[direct]=1/p[direct]
        def energy(x):
            z=x/np.sqrt(self.h)
            return .25*np.sum(self.w.toarray()*(z[:,None]-z[None,:])**2)+2.5*np.sum(r[direct]*(x[direct]-t[direct])**2)-25*np.sum(x[~direct])
        x=np.array([1.,2.,3.,4.]);eps=1e-4
        numerical=np.array([(energy(x+eps*v)-energy(x-eps*v))/(2*eps) for v in np.eye(4)])
        np.testing.assert_allclose(numerical,a.matrix@x-a.rhs,rtol=1e-6,atol=1e-5)
        x0=np.linalg.solve(b.matrix.toarray(),b.rhs);x1=np.linalg.solve(a.matrix.toarray(),a.rhs)
        correction=5*p*p/(1-p)*(t-x0)
        np.testing.assert_allclose(a.matrix@(x1-x0),correction,rtol=1e-10,atol=1e-11)
        np.testing.assert_allclose(a.rhs[direct]/(5*r[direct]),t[direct])
        self.assertTrue(np.all(x1>0));self.assertGreater(np.linalg.eigvalsh(a.matrix.toarray()).min(),0)
        self.assertTrue(np.all(np.diag((a.matrix-b.matrix).toarray())>=0))

    def test_scale_invariance_and_anchor_refusal(self):
        a=assemble_system(self.w,self.e,source='boundary_odds');b=assemble_system(self.w*8,self.e*8,source='boundary_odds')
        np.testing.assert_allclose(a.matrix.toarray(),b.matrix.toarray(),rtol=1e-14,atol=1e-15)
        np.testing.assert_array_equal(a.rhs,b.rhs)
        with self.assertRaises(ValueError):assemble_system(self.w,np.zeros(4),source='boundary_odds')

    def test_independent_90_digit_oracle_and_wrong_equation_rejected(self):
        values,_,ev=solve(self.w,self.e,source='boundary_odds')
        with mp.workdps(90):
            w=[[mp.mpf(float(v)) for v in row] for row in self.w.toarray()]
            h=[mp.fsum(row) for row in w];e=list(map(lambda v:mp.mpf(float(v)),self.e))
            m=mp.matrix([[int(i==j)*(1+5*e[i]/h[i])-w[i][j]/mp.sqrt(h[i]*h[j]) for j in range(4)] for i in range(4)])
            rhs=mp.matrix([5*(h[i]+e[i])/h[i] if e[i] else 25 for i in range(4)]);exact=mp.lu_solve(m,rhs)
            block=Block(self.w,self.e,5.,25.,'boundary_odds');cert=block.certify(values,80)
            for x,(lo,hi) in zip(exact,cert['intervals']):self.assertTrue(lo<=x<=hi)
            repaired,report=block.solve([v*mp.mpf('1.01') for v in values],80)
            self.assertLess(max(abs(x-y)/x for x,y in zip(exact,repaired)),mp.mpf('1e-38'))
        self.assertEqual(certify_saved(self.w,self.e,values,source='boundary_odds')['prefix'],ev['returned_prefix_local'])
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(self.w,self.e,values,source='original')
        damaged=list(values);damaged[0]*=1.01
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(self.w,self.e,damaged,source='boundary_odds')

    def test_exact_ties_and_one_ulp(self):
        w=sparse.eye(7,format='csr');e=np.ones(7)
        values,_,ev=solve(w,e,source='boundary_odds')
        self.assertTrue(all(v==values[0] for v in values));self.assertEqual(ev['returned_prefix_local'],list(range(5)))
        e[-1]=np.nextafter(1.,2.);values,_,ev=solve(w,e,source='boundary_odds')
        self.assertLess(values[-1],values[0]);self.assertEqual(ev['returned_prefix_local'],[6,0,1,2,3])
        self.assertGreater(ev['dps'],0)

    def test_large_boundary_and_sparse_refinement(self):
        # Exercises the sparse correction matrix override above the dense cap.
        n=270;w=sparse.diags([np.full(n-1,.3),np.ones(n),np.full(n-1,.3)],[-1,0,1],format='csr')
        e=np.zeros(n);e[[0,80,160,269]]=[1000.,.001,2.,.1]
        with mp.workdps(50):
            block=Block(w,e,5.,25.,'boundary_odds');values,report=block.solve(None,50)
            self.assertTrue(block.certify(values,50)['accepted'])
            self.assertEqual(report['method'],'accurate-residual-sparse-lu-refinement')


if __name__=='__main__':unittest.main()
