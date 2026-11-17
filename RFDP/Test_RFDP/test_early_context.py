"""Stage boundary, independent dense oracle, and exact first-gallery stopping."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from Test_RFDP.test_adaptation_runtime import AdaptationRuntimeTests
from experiments.continuous_fitting import certified
from experiments.continuous_fitting.verify import verify_query
from experiments.reid_adaptation import runtime as dispatch, protocol as dispatch_protocol
from experiments.reid_adaptation.schedule import ScheduledIHyRDP
from experiments.early_context import protocol, runtime, rank1


class EarlyContextTests(AdaptationRuntimeTests):
    def switch_fixture(self,sizes=(8,7,6)):
        g,a,_,_=self.fixture(sizes)
        random=np.random.default_rng(71).uniform(.02,1.7,g.shape)
        random=(random+random.T)/2
        context=sparse.csr_matrix(g.multiply(random));context.setdiag(g.diagonal());context.sort_indices()
        root=self.root/'original';root.mkdir(exist_ok=True)
        sparse.save_npz(root/'graph.npz',g)
        cfg=protocol.recipe('duke-osnet');cfg['original_cache']=str(root)
        return cfg,g,context,a

    def test_frozen_window_and_new_protocol_do_not_relabel_old_combination(self):
        for key in protocol.ORDER:
            cfg=protocol.recipe(key)
            self.assertEqual(dispatch_protocol.validate(cfg),protocol.validate(cfg))
            self.assertEqual(cfg['context_window_new_nodes'],5)
            for field,value in [('context_window_new_nodes',4),('promotion_schedule','fixed5'),
                                ('graph_schedule','context_forever'),('late_numerical_solver','stable_lu_certified')]:
                altered=copy.deepcopy(cfg);altered[field]=value
                with self.assertRaises(ValueError):protocol.validate(altered)

    def test_switch_scores_and_complete_ranking_match_dense_equations(self):
        cfg,g,c,a=self.switch_fixture()
        engine,*_=dispatch.initialize(cfg,c,a)
        def dense(w,e,alpha,beta,*,diagnostics,indices,targets,bridges):
            chosen=c if len(targets)-1<5 else g
            weights=chosen[indices][:,indices].toarray()
            boundary=np.zeros(len(indices))
            for target in targets:boundary+=chosen[indices,target].toarray().ravel()
            for outside,inside,value in bridges:boundary[np.asarray(indices)==outside]+=value
            h=weights.sum(axis=1);p=boundary/(h+boundary)
            matrix=np.diag(1+alpha*p)-weights/np.sqrt(h[:,None]*h[None,:])
            rhs=np.where(boundary>0,alpha,beta)
            values=np.linalg.solve(matrix,rhs)
            return values,float(np.linalg.norm(matrix@values-rhs)/np.linalg.norm(rhs))
        oracle=ScheduledIHyRDP(g,a,stage_solver=dense,promotion_schedule='early5')
        for query in (0,1):
            actual=self.capture(engine,a,query=query)
            expected,_=oracle.rank([query],nq=len(a),input_similarity=a[query])
            np.testing.assert_array_equal(actual[0],expected)
            verified=self.verify(cfg,c,a,actual,query=query)
            self.assertGreater(verified['bridge_events'],0)
            for row in actual[2]:
                if row['event'] is not None:
                    self.assertEqual(row['event']['scoring_graph'],'context' if len(row['labeled'])<=5 else 'original')
            # Five newly admitted nodes plus the original query => switch at |L|=6.
            self.assertEqual([len(t['selected']) for t in actual[2][:5]],[1]*5)
            self.assertEqual(actual[2][5]['labeled'],actual[0][:6].tolist())
            self.assertEqual(actual[2][5]['promotion_step'],5)

    def test_identical_graphs_reduce_to_existing_early5(self):
        cfg,g,_,a=self.switch_fixture()
        new=self.capture(dispatch.initialize(cfg,g,a)[0],a)
        ordinary=self.config('early5',self.root/'unused.npy')
        old=self.capture(dispatch.initialize(ordinary,g,a)[0],a)
        np.testing.assert_array_equal(new[0],old[0]);self.assertEqual(new[1],old[1])
        self.verify(cfg,g,a,new)

    def test_reader_rejects_wrong_phase_and_context_boundary(self):
        cfg,g,c,a=self.switch_fixture()
        out=self.capture(dispatch.initialize(cfg,c,a)[0],a)
        for field,value in [('scoring_graph','original'),('stage_boundary_sha256','0'*64),('context_window_new_nodes',6)]:
            altered=copy.deepcopy(out);altered[2][0]['event'][field]=value
            with self.assertRaises(AssertionError):self.verify(cfg,c,a,altered)
        changed=c.copy();changed[0,1]=changed[1,0]=0.;changed.eliminate_zeros()
        with self.assertRaises(ValueError):dispatch.initialize(cfg,changed,a)

    def test_exact_first_valid_gallery_can_occur_after_switch(self):
        cfg,g,c,a=self.switch_fixture((18,))
        full_output=self.capture(dispatch.initialize(cfg,c,a)[0],a)
        labels=dict(query_label=np.array([1,2]),query_cam=np.array([0,0]),
                    gallery_label=np.full(16,3),gallery_cam=np.ones(16,dtype=int))
        # Make the first several admitted gallery images evaluation-excluded.
        for node in full_output[0][1:8]:
            if node>=2:labels['gallery_label'][node-2]=1;labels['gallery_cam'][node-2]=0
        expected=rank1.first_valid(full_output[1],0,2,labels)
        engine,rg,_,sums,cert=dispatch.initialize(cfg,c,a)
        certified.EVENTS.clear();rows=[];prefix=[0]
        def capture(t):
            row=copy.deepcopy(t);values=row.pop('scores')
            row['returned_scores']=None if values is None else [certified.packed(v) for v in values]
            row['event']=None if values is None else copy.deepcopy(certified.EVENTS[-1])
            rows.append(row);prefix.extend(row['selected'])
            if rank1.first_valid(prefix,0,2,labels) is not None:raise rank1.FirstGalleryCommitted()
        with self.assertRaises(rank1.FirstGalleryCommitted):engine.rank([0],nq=len(a),trace=capture)
        self.assertEqual(rank1.first_valid(prefix,0,2,labels),expected)
        self.assertGreater(len(prefix),6)
        verify_query(rg,a,None,0,prefix,None,rows,'original',2,len(prefix),
                    full_sums=sums,stage_certifier=cert,promotion_schedule='early5',prefix_only=True)
        with self.assertRaises(AssertionError):
            verify_query(rg,a,np.ones(len(a)),0,prefix,None,rows,'original',2,len(prefix),
                         full_sums=sums,stage_certifier=cert,promotion_schedule='early5',prefix_only=True)


if __name__=='__main__':
    import unittest
    unittest.main()
