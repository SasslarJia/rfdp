"""Independent energy/Bellman/oracle and returned-score checks; no retrieval pilot."""
import unittest
import numpy as np
import mpmath as mp
from scipy import sparse
from experiments.continuous_fitting.model import assemble_system
from experiments.continuous_fitting.certified import solve, certify_saved, Block

class StationaryMeasureTests(unittest.TestCase):
    def setUp(self):
        self.w=np.array([[1.,2.,.5,0.],[2.,1.,0.,.7],[.5,0.,1.,.3],[0.,.7,.3,1.]])
        self.e=np.array([1.,.2,0.,0.]);self.h=self.w.sum(1)

    def test_energy_and_independent_killed_walk(self):
        a=assemble_system(self.w,self.e,source='stationary_measure')
        h=self.h;p=self.e/(h+self.e);b=np.where(self.e>0,5.,25.)
        def energy(x):
            return .25*np.sum(self.w*(x[:,None]-x[None,:])**2)+2.5*np.dot(h*p,x*x)-np.dot(h*b,x)
        x=np.array([1.25,4.,2.,.75]);eps=1e-4
        gradient=np.array([(energy(x+eps*v)-energy(x-eps*v))/(2*eps) for v in np.eye(4)])
        np.testing.assert_allclose(gradient,a.matrix@x-a.rhs,atol=1e-8,rtol=1e-9)
        transition=(self.w/h[:,None])/(1+5*p[:,None])
        costs=b/(1+5*p)
        expected=np.linalg.solve(np.eye(4)-transition,costs)
        np.testing.assert_allclose(expected,np.linalg.solve(a.matrix.toarray(),a.rhs),rtol=1e-12)
        self.assertTrue(np.all(expected>0));self.assertGreater(np.linalg.eigvalsh(a.matrix.toarray()).min(),0)

    def test_scale_regular_degree_and_direct_target(self):
        a=assemble_system(self.w,self.e,source='stationary_measure')
        s=assemble_system(8*self.w,8*self.e,source='stationary_measure')
        np.testing.assert_allclose(s.matrix.toarray(),8*a.matrix.toarray(),rtol=1e-14)
        np.testing.assert_allclose(s.rhs,8*a.rhs,rtol=1e-14)
        w=np.array([[1.,.4,.4],[.4,1.,.4],[.4,.4,1.]])
        a=assemble_system(w,[1.,.2,0.],source='stationary_measure');old=assemble_system(w,[1.,.2,0.],source='original')
        np.testing.assert_allclose(np.linalg.solve(a.matrix.toarray(),a.rhs),np.linalg.solve(old.matrix.toarray(),old.rhs),rtol=1e-13)
        w=np.diag([1.,4.]);e=np.array([1.125,4.]);a=assemble_system(w,e,source='stationary_measure')
        target=(w.sum(1)+e)/e
        np.testing.assert_allclose(np.linalg.solve(a.matrix.toarray(),a.rhs),target,rtol=1e-14)
        self.assertLess(target[0],target[1]);self.assertGreater(target[0],target[1]/2)
        with self.assertRaisesRegex(ValueError,'anchor'):assemble_system(self.w,np.zeros(4),source='stationary_measure')

    def test_independent_ninety_digit_oracle_actual_values_and_correction(self):
        w=sparse.csr_matrix(self.w);values,_,event=solve(w,self.e,source='stationary_measure')
        with mp.workdps(90):
            ww=[[mp.mpf(float(v)) for v in row] for row in self.w]
            h=list(map(mp.fsum,ww));e=list(map(lambda v:mp.mpf(float(v)),self.e));p=[ee/(hh+ee) for hh,ee in zip(h,e)]
            a=mp.matrix([[int(i==j)*(h[i]+5*h[i]*p[i])-ww[i][j] for j in range(4)] for i in range(4)])
            exact=mp.lu_solve(a,mp.matrix([h[i]*(5 if e[i] else 25) for i in range(4)]))
            block=Block(w,self.e,5.,25.,'stationary_measure');cert=block.certify(values,80)
            self.assertTrue(cert['accepted'])
            for v,(lo,hi) in zip(exact,cert['intervals']):self.assertTrue(lo<=v<=hi)
            repaired,_=block.solve([v*mp.mpf('1.01') for v in values],80)
            self.assertLess(max(abs(v-t)/t for v,t in zip(repaired,exact)),mp.mpf('1e-38'))
        self.assertEqual(certify_saved(w,self.e,values,source='stationary_measure')['prefix'],event['returned_prefix_local'])
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(w,self.e,values,source='original')
        broken=list(values);broken[-1]*=1.01
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(w,self.e,broken,source='stationary_measure')

    def test_exact_symmetry_and_ulp_are_not_merged(self):
        w=np.array([[1.,1.,1.],[1.,1.,0.],[1.,0.,1.]]);e=np.array([1.,0.,0.])
        v,_,r=solve(sparse.csr_matrix(w),e,source='stationary_measure')
        self.assertEqual(v[1],v[2]);self.assertEqual(r['returned_prefix_local'],[0,1,2])
        w[0,2]=w[2,0]=np.nextafter(1.,2.)
        v,_,r=solve(sparse.csr_matrix(w),e,source='stationary_measure')
        self.assertNotEqual(v[1],v[2]);self.assertGreater(r['dps'],0)

    def test_sparse_correction_above_dense_limit(self):
        n=270;w=sparse.diags([np.full(n-1,.3),np.ones(n),np.full(n-1,.3)],[-1,0,1],format='csr')
        e=np.zeros(n);e[::3]=np.linspace(.001,3.,len(e[::3]))
        with mp.workdps(50):
            b=Block(w,e,5.,25.,'stationary_measure');v,r=b.solve(None,50)
            self.assertTrue(b.certify(v,50)['accepted']);self.assertEqual(r['method'],'accurate-residual-sparse-lu-refinement')

if __name__=='__main__':unittest.main()
