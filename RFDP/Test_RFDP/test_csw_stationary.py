"""Actual CSW/SC ranking, bridge read-back, and bounded trace retention."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.io import savemat
from MODEL.Paper_RFDP import PaperFullIterative as model
from experiments.continuous_fitting import full, certified
from experiments.reliability.reliability_model import ReliabilityIHyRDP, build_support
from experiments.stationary_measure.csw_graph import ReadbackKernel


class CombinedTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2041);n=17
        a=rng.uniform(.1,1.,(n,n));self.aff=(a+a.T)/2;np.fill_diagonal(self.aff,1.)
        graph=np.zeros((n,n))
        for ids in (list(range(8)),list(range(8,n))):graph[np.ix_(ids,ids)]=self.aff[np.ix_(ids,ids)]
        self.graph=sparse.csr_matrix(graph);self.cameras=np.arange(n)%3
        self.support,_=build_support(self.aff,self.cameras,k=3,mode='ordinary')
        self.x=rng.normal(size=(n,5));self.x/=np.linalg.norm(self.x,axis=1)[:,None]
        self.old_solver=model.solve_hyrdp_closed_form

    def tearDown(self):
        model.solve_hyrdp_closed_form=self.old_solver

    def test_weighted_graph_kernel_and_bridge_match_independent_dense_formula(self):
        engine=ReliabilityIHyRDP(self.graph,self.aff,self.cameras,eta=.5,support=self.support,lam=.25)
        reader=ReadbackKernel(self.graph,self.aff,self.cameras,self.support,.5,.25)
        camera=np.where(self.cameras[:,None]==self.cameras[None,:],.5,1.)
        expected=self.aff*camera*(.75+.25*self.support.toarray())
        np.testing.assert_allclose(reader.block(np.arange(17),np.arange(17)),expected,rtol=0,atol=0)
        np.testing.assert_allclose(reader.sums,expected.sum(1),rtol=1e-14)
        np.testing.assert_array_equal(reader.sums,engine.full_sums)
        np.testing.assert_array_equal(reader.graph.toarray(),engine.w.toarray())
        from experiments.continuous_fitting.verify import bridge_edges
        edges=bridge_edges(reader.graph,self.aff,engine.components,{0},reader.sums,reader.block)
        self.assertEqual(edges,[list(x) for x in engine.bridges({0})])

    def test_disabling_csw_recovers_actual_sc_trajectory(self):
        certified.install('stationary_measure')
        plain=model.FullIHyRDP(self.graph,self.aff)
        combined=ReliabilityIHyRDP(self.graph,self.aff,self.cameras,eta=1.,support=self.support,lam=0.)
        a,_=plain.rank([0],nq=13,input_similarity=self.x@self.x[0],numerical_tail=False)
        b,_=combined.rank([0],nq=13,input_similarity=self.x@self.x[0],numerical_tail=False)
        np.testing.assert_array_equal(a,b)

    def fixture(self, directory, source='stationary_measure'):
        out=Path(directory);(out/'queries').mkdir();(out/'accepted').mkdir()
        mat=out/'input.mat'
        savemat(mat,dict(query_f=self.x[:2],gallery_f=self.x[2:],query_cam=self.cameras[:2],
            gallery_cam=self.cameras[2:],query_label=np.arange(2),gallery_label=np.arange(15)%2))
        certified.install(source)
        engine=ReliabilityIHyRDP(self.graph,self.aff,self.cameras,eta=.5,support=self.support,lam=.25)
        reader=ReadbackKernel(self.graph,self.aff,self.cameras,self.support,.5,.25)
        cfg=dict(source=source,mat=str(mat),depth=13,trace_retention='sampled_after_readback',retained_queries=[0])
        return patch.multiple(full,create=True,OUT=out,CFG=cfg,QN=2,GN=15,ENGINE=engine,
            GRAPH=reader.graph,FULL=self.aff,X=self.x,SIG='synthetic',INIT_ERROR=None,
            READBACK_KERNEL=reader.block,READBACK_SUMS=reader.sums)

    def complete_fixture(self, out):
        for q in range(2):
            result=full.compute_query(q);self.assertNotIn('error',result,result.get('error'))
        full.evaluate()
        full.dump(out/'COMPUTE_COMPLETE.json',dict(signature='synthetic',rankings_sha256=full.sha(out/'rankings.npy')))

    def test_both_sources_read_every_score_before_pruning_and_keep_final_rankings(self):
        for source in ('original','stationary_measure'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                out=Path(directory)
                with self.fixture(out,source):
                    self.complete_fixture(out)
                    self.assertTrue((out/'queries/0000.trace.jsonl.gz').exists())
                    self.assertFalse((out/'queries/0001.trace.jsonl.gz').exists())
                    self.assertEqual(len(list((out/'accepted').glob('*.json'))),2)
                    full.aggregate_inline_readback()
                    self.assertFalse(list((out/'queries').glob('*.rankings.npy')))
                    self.assertEqual(np.load(out/'rankings.npy').shape,(2,15))

    def test_readback_failure_preserves_raw_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with self.fixture(out), patch.object(full,'verify_query',side_effect=ValueError('damaged score')):
                result=full.compute_query(1)
                self.assertIn('error',result)
                self.assertTrue((out/'queries/0001.trace.jsonl.gz').exists())
                self.assertFalse((out/'accepted/0001.json').exists())

    def test_wrong_combined_ranking_rejected_before_duplicate_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with self.fixture(out):
                self.complete_fixture(out)
                ranks=np.load(out/'rankings.npy');ranks[0]=ranks[0][::-1];np.save(out/'rankings.npy',ranks)
                full.dump(out/'COMPUTE_COMPLETE.json',dict(signature='synthetic',rankings_sha256=full.sha(out/'rankings.npy')))
                with self.assertRaises(AssertionError):full.aggregate_inline_readback()
                self.assertTrue((out/'queries/0001.rankings.npy').exists())


if __name__=='__main__':unittest.main()
