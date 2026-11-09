"""Stage hooks must preserve original discovery and supply actual target context."""
import unittest
import numpy as np
from scipy import sparse
from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP, solve_hybrid_sparse
from experiments.relation_routes.local_timing import boundary_matrix


class StageContextTests(unittest.TestCase):
    def test_default_and_contextual_original_are_identical(self):
        rng = np.random.default_rng(23)
        full = rng.uniform(.1, .9, (14,14))
        full = (full+full.T)/2
        np.fill_diagonal(full, 1.)
        graph = sparse.csr_matrix(full)
        contexts = []
        def hook(w,e,alpha,beta,*,diagnostics,indices,targets,bridges):
            self.assertTrue(set(indices).isdisjoint(targets))
            np.testing.assert_allclose(e, np.asarray(graph[indices][:,list(targets)].sum(axis=1)).ravel(),rtol=0,atol=1e-14)
            contexts.append((targets,indices.copy()))
            return solve_hybrid_sparse(w,e,alpha,beta,diagnostics=diagnostics)
        a, da = FullIHyRDP(graph,full).rank([0],nq=12)
        b, db = FullIHyRDP(graph,full,stage_solver=hook).rank([0],nq=12)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(da['iterations'],db['iterations'])
        self.assertEqual(contexts[0][0],(0,))
        self.assertEqual(len(contexts[1][0]),6)

    def test_pair_boundary_includes_temporary_bridge_once(self):
        graph=sparse.csr_matrix(np.array([[1.,.2,0],[.2,1.,0],[0,0,1.]]))
        result=boundary_matrix(graph,np.array([1,2]),(0,),[(2,0,.7)])
        np.testing.assert_array_equal(result.toarray(),[[.2],[.7]])
        self.assertEqual(graph[2,0],0.)


if __name__ == '__main__':unittest.main()
