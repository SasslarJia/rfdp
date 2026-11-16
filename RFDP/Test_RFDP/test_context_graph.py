"""Synthetic equation/input invariants; no real-query pilots or effect selection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from experiments.context_graph import graph, protocol
from experiments.context_graph.full_run import check_decision
from experiments.reid_adaptation import protocol as original_protocol, runtime


def fixture():
    rng=np.random.default_rng(83);w=np.eye(9)
    for nodes in (range(5),range(5,9)):
        for i in nodes:
            for j in nodes:
                if i<j:w[i,j]=w[j,i]=rng.uniform(.02,.7)
    for i,j in ((4,5),(0,5),(3,6)):w[i,j]=w[j,i]=rng.uniform(.01,.1)
    return sparse.csr_matrix(w)


class ContextGraphTests(unittest.TestCase):
    def test_weights_match_independent_set_formula_and_power_threshold(self):
        original=fixture();actual,meta=graph.calibrate(original);w=original.toarray()
        ii,jj=np.nonzero(np.triu(w,1));a=w[ii,jj];q=[]
        neighbors=[set(np.flatnonzero(row))-{i} for i,row in enumerate(w)]
        for i,j in zip(ii,jj):
            x=neighbors[i]-{j};y=neighbors[j]-{i}
            q.append((len(x&y)+1)/(len(x|y)+2))
        q=np.array(q);k=np.std(np.log(a))/np.std(np.log(q));s=a.sum()/np.sum(a*q**k)
        expected=s*a*q**k
        np.testing.assert_allclose(np.asarray(actual[ii,jj]).ravel(),expected,rtol=2e-14,atol=0)
        self.assertAlmostEqual(meta['kappa'],k,places=13)
        np.testing.assert_array_equal(actual.indices,original.indices)
        np.testing.assert_array_equal(actual.indptr,original.indptr)
        np.testing.assert_array_equal(actual.diagonal(),original.diagonal())
        self.assertAlmostEqual(float(actual.sum()),float(original.sum()),places=12)
        threshold=(np.sum(a*q**k)/a.sum())**(1/k)
        np.testing.assert_array_equal(expected>a,q>threshold)

    def test_constant_context_and_isolated_graph_reduce_to_original(self):
        rng=np.random.default_rng(4);x=rng.uniform(.1,.9,(6,6));x=(x+x.T)/2;np.fill_diagonal(x,1.)
        for g in (sparse.csr_matrix(x),sparse.eye(5,format='csr')):
            result,meta=graph.calibrate(g)
            np.testing.assert_array_equal(result.toarray(),g.toarray())
            self.assertEqual((meta['kappa'],meta['scale']),(0.,1.))
        g=fixture();g.data[:]=.2;g.setdiag(1.)
        result,meta=graph.calibrate(g)
        np.testing.assert_array_equal(result.toarray(),g.toarray())
        self.assertEqual(meta['degenerate'],'constant_affinity')

    def test_placebo_exact_weight_multiset_symmetric_and_reproducible(self):
        g=fixture();main,_=graph.calibrate(g)
        one,info=graph.permute_weights(g,main);two,_=graph.permute_weights(g,main)
        np.testing.assert_array_equal(one.toarray(),two.toarray())
        np.testing.assert_array_equal(np.sort(graph.edges(one)[2]),np.sort(graph.edges(main)[2]))
        self.assertEqual((one-one.T).nnz,0)
        np.testing.assert_array_equal(one.diagonal(),g.diagonal())
        self.assertTrue(info['exact_stratum_weight_multisets'])
        self.assertGreater(info['fraction_edge_positions_changed'],0)
        a=graph.edges(g)[2];cuts=np.quantile(a,np.arange(1,10)/10);groups=np.searchsorted(cuts,a,side='right')
        for b in range(10):
            np.testing.assert_array_equal(np.sort(graph.edges(one)[2][groups==b]),np.sort(graph.edges(main)[2][groups==b]))

    def test_invalid_graphs_rejected(self):
        for mutate in ('asymmetric','negative','missing_diagonal','nonfinite'):
            a=fixture().toarray()
            if mutate=='asymmetric':a[0,1]*=2
            elif mutate=='negative':a[0,1]=a[1,0]=-.1
            elif mutate=='missing_diagonal':a[0,0]=0
            else:a[0,0]=np.nan
            with self.assertRaises(ValueError):graph.calibrate(a)

    def test_complete_cache_retains_Z_and_A_and_parent_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();g=fixture()
            np.save(source/'features.npy',np.zeros((9,3),dtype=np.float32))
            np.save(source/'affinity.npy',np.eye(9,dtype=np.float32))
            sparse.save_npz(source/'graph.npz',g)
            initial={n:graph.sha(source/n) for n in ('features.npy','affinity.npy','graph.npz')}
            main=graph.build_cache(source,root/'main');control=graph.build_cache(source,root/'control',variant='weight_permutation',parent=root/'main')
            self.assertEqual(main['source_files'],initial)
            for n in ('features.npy','affinity.npy'):
                self.assertEqual(main['files'][n],initial[n]);self.assertEqual(control['files'][n],initial[n])
            self.assertEqual(control['parent_graph_sha256'],main['files']['graph.npz'])
            (root/'main/graph.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):graph.build_cache(source,root/'bad-control',variant='weight_permutation',parent=root/'main')
            self.assertTrue((root/'bad-control/FAILED.json').exists())
            for n,h in initial.items():self.assertEqual(graph.sha(source/n),h)

    def test_closed_recipes_and_validation_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=protocol.write_recipes(tmp)
            self.assertEqual(len(plan['runs']),9)
            for row in plan['runs']:
                cfg=json.loads((Path(tmp)/row['recipe']).read_text())
                self.assertEqual(original_protocol.validate(cfg),protocol.validate(cfg))
                self.assertEqual(cfg['queries'],3368 if cfg['input_id'].startswith('market') else 2228)
            cfg=protocol.recipe('context_graph','duke-osnet')
            for key,value in [('alpha',6),('boundary_rho',.5),('support_lambda',.25),('k',30),('context_bridges','weighted')]:
                changed=copy.deepcopy(cfg);changed[key]=value
                with self.assertRaises(ValueError):original_protocol.validate(changed)
            with self.assertRaises(ValueError):protocol.recipe('context_graph_permuted','market-osnet')
            control=protocol.recipe('context_graph_permuted','duke-osnet')
            with self.assertRaises(ValueError):check_decision(control,None)
            p=Path(tmp)/'decision.json';p.write_text(json.dumps(dict(schema=protocol.SCHEMA,verdict='continue_fixed_control',rationale='test',allowed_units=['context_graph_permuted--duke-osnet'],evidence=[])))
            with self.assertRaises(ValueError):check_decision(control,p)

    def test_original_equation_and_full_readback_use_actual_new_graph(self):
        from Test_RFDP.test_adaptation_runtime import AdaptationRuntimeTests
        from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
        from experiments.continuous_fitting.original_solver import WarmOriginalSolver
        helper=AdaptationRuntimeTests();helper.setUp()
        try:
            old,affinity,_,path=helper.fixture()
            main,_=graph.calibrate(old);placebo,_=graph.permute_weights(old,main)
            for route,g,schedule in [('context_graph',main,'fixed5'),('context_graph_permuted',placebo,'fixed5'),('context_graph_early5',main,'early5')]:
                cfg=helper.config('p1_graph',path);cfg.update(adaptation_route=route,promotion_schedule=schedule)
                engine,returned,kernel,sums,_=runtime.initialize(cfg,g,affinity)
                np.testing.assert_array_equal(returned.toarray(),g.toarray());self.assertIsNone(kernel)
                np.testing.assert_array_equal(sums,affinity.sum(axis=1,dtype=np.float64))
                out=helper.capture(engine,affinity)
                verified=helper.verify(cfg,g,affinity,out)
                self.assertLessEqual(verified['max_recomputed_residual'],1e-8)
                self.assertGreater(verified['bridge_events'],0)
                if schedule=='fixed5':
                    baseline=helper.capture(FullIHyRDP(g,affinity,stage_solver=WarmOriginalSolver()),affinity)
                    np.testing.assert_array_equal(out[0],baseline[0]);self.assertEqual(out[1],baseline[1])
        finally:helper.tearDown()


if __name__=='__main__':unittest.main()
