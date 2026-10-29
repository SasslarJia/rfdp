"""Independent renewal, energy and actual-score checks; no performance task."""
import unittest
import numpy as np
import mpmath as mp
from scipy import sparse
from experiments.continuous_fitting.model import assemble_system
from experiments.continuous_fitting.certified import solve, certify_saved, Block


class TwoStepFittingTests(unittest.TestCase):
    def setUp(self):
        self.w=np.array([[1.,2.,.5,0.],[2.,1.,0.,.7],[.5,0.,1.,.3],[0.,.7,.3,1.]])
        self.e=np.array([1.,.2,0.,0.]);self.h=self.w.sum(1)

    def test_renewal_mean_via_independent_phase_chain(self):
        p=self.e/(self.h+self.e);q=self.w@p/(self.h+self.e)
        for i in [2,3]:
            # Phase 0 starts at i; phase 1 remembers first-step destination.
            # Absorption occurs on the next step; all failures reset to i.
            transient=np.zeros((5,5));transient[0,1:]=self.w[i]/self.h[i]
            transient[1:,0]=1-p
            expected=np.linalg.solve(np.eye(5)-transient,np.ones(5))[0]
            self.assertAlmostEqual(expected,2/q[i],places=10)

    def test_energy_response_and_componentwise_comparison(self):
        a=assemble_system(self.w,self.e,source='two_step_fitting')
        old=assemble_system(self.w,self.e,source='original')
        p=self.e/(self.h+self.e);q=self.w@p/(self.h+self.e)
        k=np.where(self.e>0,5*p,25*q/2);b=np.where(self.e>0,5.,25.)
        def energy(v):
            z=v/np.sqrt(self.h)
            return .25*np.sum(self.w*(z[:,None]-z[None,:])**2)+.5*np.dot(k,v*v)-b@v
        x=np.array([1.25,4.,2.,.75]);eps=1e-4
        gradient=np.array([(energy(x+eps*v)-energy(x-eps*v))/(2*eps) for v in np.eye(4)])
        np.testing.assert_allclose(gradient,a.matrix@x-a.rhs,atol=1e-9,rtol=1e-9)
        x0=np.linalg.solve(old.matrix.toarray(),old.rhs);x1=np.linalg.solve(a.matrix.toarray(),a.rhs)
        d=np.where(self.e==0,25*q/2,0)
        np.testing.assert_allclose(a.matrix@(x1-x0),-d*x0,rtol=1e-12,atol=1e-11)
        self.assertTrue(np.all(x1<=x0));self.assertTrue(np.all(x1>0))
        self.assertGreater(np.linalg.eigvalsh(a.matrix.toarray()).min(),0)

    def test_scale_all_direct_and_zero_two_step_limit(self):
        a=assemble_system(self.w,self.e,source='two_step_fitting')
        b=assemble_system(8*self.w,8*self.e,source='two_step_fitting')
        np.testing.assert_allclose(a.matrix.toarray(),b.matrix.toarray(),rtol=1e-14,atol=1e-15)
        np.testing.assert_array_equal(a.rhs,b.rhs)
        a=assemble_system(self.w,np.ones(4),source='two_step_fitting')
        b=assemble_system(self.w,np.ones(4),source='original')
        np.testing.assert_array_equal(a.matrix.toarray(),b.matrix.toarray())
        w=sparse.diags([np.ones(3),np.ones(4),np.ones(3)],[-1,0,1],format='csr');e=[1.,0.,0.,0.]
        a=assemble_system(w,e,source='two_step_fitting');b=assemble_system(w,e,source='original')
        np.testing.assert_array_equal(a.matrix.toarray()[2:],b.matrix.toarray()[2:])
        with self.assertRaisesRegex(ValueError,'anchor'):assemble_system(w,np.zeros(4),source='two_step_fitting')

    def test_independent_mp_oracle_refinement_and_wrong_target_refusal(self):
        w=sparse.csr_matrix(self.w);values,_,event=solve(w,self.e,source='two_step_fitting')
        with mp.workdps(90):
            ww=[[mp.mpf(float(v)) for v in row] for row in self.w]
            h=list(map(mp.fsum,ww));e=[mp.mpf(float(v)) for v in self.e]
            p=[v/(hh+v) for v,hh in zip(e,h)]
            q=[mp.fsum(ww[i][j]*p[j] for j in range(4))/(h[i]+e[i]) for i in range(4)]
            k=[5*p[i] if e[i]>0 else 25*q[i]/2 for i in range(4)]
            m=mp.matrix([[int(i==j)*(1+k[i])-ww[i][j]/mp.sqrt(h[i]*h[j]) for j in range(4)] for i in range(4)])
            exact=mp.lu_solve(m,mp.matrix([5 if v else 25 for v in e]))
            block=Block(w,self.e,5.,25.,'two_step_fitting');cert=block.certify(values,80)
            self.assertTrue(cert['accepted'])
            for v,(lo,hi) in zip(exact,cert['intervals']):self.assertTrue(lo<=v<=hi)
            repaired,_=block.solve([v*mp.mpf('1.01') for v in values],80)
            self.assertLess(max(abs(v-x)/x for v,x in zip(repaired,exact)),mp.mpf('1e-38'))
        self.assertEqual(certify_saved(w,self.e,values,source='two_step_fitting')['prefix'],event['returned_prefix_local'])
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(w,self.e,values,source='original')
        damaged=list(values);damaged[-1]*=1.01
        with self.assertRaises(np.linalg.LinAlgError):certify_saved(w,self.e,damaged,source='two_step_fitting')

    def test_indirect_exact_tie_and_one_ulp(self):
        w=np.array([[1.,1.,1.],[1.,1.,0.],[1.,0.,1.]]);e=np.array([1.,0.,0.])
        values,_,event=solve(sparse.csr_matrix(w),e,source='two_step_fitting')
        self.assertEqual(values[1],values[2]);self.assertEqual(event['returned_prefix_local'],[0,1,2])
        w[0,2]=w[2,0]=np.nextafter(1.,2.)
        values,_,event=solve(sparse.csr_matrix(w),e,source='two_step_fitting')
        self.assertNotEqual(values[1],values[2]);self.assertGreater(event['dps'],0)

    def test_sparse_refinement_above_dense_limit(self):
        n=270;w=sparse.diags([np.full(n-1,.3),np.ones(n),np.full(n-1,.3)],[-1,0,1],format='csr')
        e=np.zeros(n);e[::3]=np.linspace(.001,3.,len(e[::3]))
        with mp.workdps(50):
            block=Block(w,e,5.,25.,'two_step_fitting');values,report=block.solve(None,50)
            self.assertTrue(block.certify(values,50)['accepted'])
            self.assertEqual(report['method'],'accurate-residual-sparse-lu-refinement')


if __name__=='__main__':unittest.main()
