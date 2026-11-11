"""Numerical equivalence and state-lifetime checks, not accuracy exploration."""
import unittest
from unittest.mock import patch
import mpmath as mp
import numpy as np
from scipy import sparse
from experiments.continuous_fitting import certified, original_solver as opt


class OriginalSolverTests(unittest.TestCase):
    def graph(self, n=150):
        off = np.linspace(.02, .17, n-1)
        w = sparse.diags([off, np.linspace(.8, 1.4, n), off], [-1, 0, 1], format='csr')
        e = np.linspace(.013, .31, n)
        return w, e

    def test_exact_tie_and_one_ulp_boundary_keep_original_order(self):
        w = sparse.eye(7, format='csr')
        for last in (1., np.nextafter(1., 2.)):
            e = np.ones(7); e[-1] = last
            before, _, old = certified.solve(w, e, source='original')
            after, residual, new = opt.solve(w, e)
            self.assertEqual(old['returned_prefix_local'], new['returned_prefix_local'])
            self.assertEqual(certified.certify_saved(w, e, after, source='original')['prefix'], new['returned_prefix_local'])
            self.assertLessEqual(residual, certified.TOL)
            if last != 1.:
                self.assertGreater(new['dps'], 0)
                self.assertEqual(new['returned_prefix_local'][0], 6)

    def test_early_exit_uses_original_certificate_and_does_not_mutate_cg_iterate(self):
        w, e = self.graph()
        exact, _, old = certified.solve(w, e, source='original')
        x = np.asarray(exact, dtype=float)
        before = x.copy()
        def fake_cg(a, b, *, callback, **kw):
            for _ in range(32): callback(x)
            self.fail('Certified callback did not stop CG')
        with patch.object(opt, 'cg', fake_cg):
            values, residual, event = opt.solve(w, e)
        np.testing.assert_array_equal(x, before)
        self.assertTrue(event['stopped_early'])
        self.assertEqual(event['cg_iterations'], 32)
        self.assertEqual(event['returned_prefix_local'], old['returned_prefix_local'])
        self.assertEqual(certified.certify_saved(w, e, values, source='original')['prefix'], old['returned_prefix_local'])
        self.assertLessEqual(residual, certified.TOL)

    def test_rejected_iterates_do_not_stop_or_populate_success(self):
        w, e = self.graph()
        calls = []
        def fake_cg(a, b, *, callback, **kw):
            for _ in range(64): callback(np.ones(len(e)))
            calls.append('finished')
            return np.asarray(np.linalg.solve(a.toarray(), b)), 0
        with patch.object(opt, 'cg', fake_cg):
            values, _, event = opt.solve(w, e)
        self.assertEqual(calls, ['finished'])
        self.assertFalse(event['stopped_early'])
        certified.certify_saved(w, e, values, source='original')

    def test_warm_mapping_uses_returned_scores_and_reset(self):
        solver = opt.WarmOriginalSolver(events=[])
        guesses = []
        def fake_solve(w, e, alpha, beta, *, x0):
            guesses.append(None if x0 is None else x0.copy())
            return np.array([mp.mpf(i+2) for i in range(len(e))], dtype=object), 0., dict(cg_calls=1,cg_iterations=10,direct_calls=0)
        with patch.object(opt, 'solve', fake_solve):
            for ids in ([2,5,9], [1,5,9,10], [20,21]):
                solver(sparse.eye(len(ids)), np.ones(len(ids)), 5., 25., diagnostics={}, indices=ids, targets=(0,), bridges=())
            solver.reset()
            solver(sparse.eye(2), np.ones(2), 5., 25., diagnostics={}, indices=[20,21], targets=(0,), bridges=())
        self.assertIsNone(guesses[0])
        np.testing.assert_array_equal(guesses[1], [4.,3.,4.,4.])
        self.assertIsNone(guesses[2]); self.assertIsNone(guesses[3])
        self.assertEqual(solver.events[1]['warm_reused_nodes'], 2)

    def test_failed_stage_does_not_replace_cache(self):
        solver = opt.WarmOriginalSolver(events=[])
        w, e = self.graph(7)
        solver(w,e,5.,25.,diagnostics={},indices=np.arange(7),targets=(20,),bridges=())
        before = solver._scores.copy()
        with patch.object(opt,'solve',side_effect=certified.original.ArithmeticFailure('test')):
            with self.assertRaises(np.linalg.LinAlgError):
                solver(w,e,5.,25.,diagnostics={},indices=np.arange(7),targets=(20,21),bridges=())
        np.testing.assert_array_equal(solver._scores,before)
        self.assertEqual(len(solver.events),1)

    def test_unrepresentable_certified_values_disable_warm_cache_only(self):
        solver=opt.WarmOriginalSolver(events=[])
        event=dict(cg_calls=0,cg_iterations=0,direct_calls=1)
        for point in ('1e400', '1e-400'):
            values=np.array([mp.mpf(point)],dtype=object)
            with patch.object(opt,'solve',return_value=(values,0.,event.copy())):
                actual,_=solver(sparse.eye(1),np.ones(1),5.,25.,diagnostics={},indices=[1],targets=(0,),bridges=())
            self.assertEqual(actual[0],values[0])
            self.assertEqual(len(solver._ids),0)

    def test_failed_refinement_recovers_through_unchanged_legacy(self):
        w,e=self.graph(7)
        values,residual,event=certified.solve(w,e,source='original')
        with patch.object(opt,'_accept',return_value=None), patch.object(certified.Block,'solve',side_effect=RuntimeError('refinement test')), patch.object(certified,'solve',return_value=(values,residual,event)) as legacy:
            actual,_,report=opt.solve(w,e)
        legacy.assert_called_once()
        np.testing.assert_array_equal(actual,values)
        self.assertEqual(report['candidate_method'],'legacy-recovery')
        self.assertEqual(report['solver_id'],opt.SOLVER_ID)

    def test_complete_bridged_trajectory_and_query_reset(self):
        from experiments.continuous_fitting.solver_timing import execute_query,compare_saved
        from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
        from pathlib import Path
        import tempfile
        rng=np.random.default_rng(2041);n=17
        w=np.zeros((n,n))
        for ids in (range(8),range(8,n)):
            for i in ids:
                w[i,i]=1.
                for j in ids:
                    if i<j:w[i,j]=w[j,i]=rng.uniform(.05,.8)
        full=rng.uniform(.05,.95,(n,n));full=(full+full.T)/2;np.fill_diagonal(full,1.)
        graph=sparse.csr_matrix(w)
        with tempfile.TemporaryDirectory() as td:
            for route in ('original','optimized'):
                out=execute_query(route,0,graph,full,np.eye(n),1,Path(td)/route,depth=16)
                self.assertGreater(out['acceptance']['bridge_events'],0)
                self.assertLessEqual(out['acceptance']['max_reported_residual'],certified.TOL)
            self.assertTrue(compare_saved(Path(td)/'original',Path(td)/'optimized')['complete_gallery_identical'])
        solver=opt.WarmOriginalSolver(events=[]);core=FullIHyRDP(graph,full,stage_solver=solver)
        for query in (0,1):
            start=len(solver.events)
            core.rank([query],nq=16,input_similarity=full[query])
            self.assertEqual(solver.events[start]['warm_reused_nodes'],0)

    def test_opt_in_binding_and_manifest(self):
        from experiments.continuous_fitting import full
        cfg=dict(source='original',original_solver=opt.MODE)
        self.assertEqual(full.configured_solver_id(cfg),opt.SOLVER_ID)
        self.assertEqual(full.configured_solver_id(dict(source='original')),certified.SOLVER_ID)
        for change in (dict(source='continuous'),dict(relation_model='csw'),dict(relation_schema=1),dict(admission_policy='camera_gate'),dict(original_solver='typo')):
            with self.assertRaises(ValueError):opt.binding(dict(cfg,**change))
        manifest=dict(config=cfg,solver_id=opt.SOLVER_ID)
        full.check_admission_manifest(cfg,manifest)
        with self.assertRaises(AssertionError):full.check_admission_manifest(cfg,dict(config={'source':'original'},solver_id=certified.SOLVER_ID))

    def test_production_compute_and_readback_record_separate_costs(self):
        from experiments.continuous_fitting import full
        from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
        from pathlib import Path
        import tempfile,json
        w,e=self.graph(17);affinity=np.eye(17)+np.ones((17,17))*.01
        engine=FullIHyRDP(w,affinity,stage_solver=opt.WarmOriginalSolver())
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'queries').mkdir();(out/'accepted').mkdir()
            cfg=dict(source='original',original_solver=opt.MODE,depth=16,alpha=5.,beta=25.)
            context=dict(OUT=out,QN=1,GN=16,CFG=cfg,SIG='synthetic-warm',GRAPH=w,FULL=affinity,
                         X=np.eye(17),INIT_ERROR=None,ENGINE=engine,STAGE_CERTIFIER=None,
                         READBACK_KERNEL=None,READBACK_SUMS=None,EXECUTION_COMMIT=None)
            with patch.dict(full.__dict__,context):
                result=full.compute_query(0)
                self.assertNotIn('error',result,result.get('error'))
                accepted=full.accept_query(0)
                self.assertNotIn('error',accepted,accepted.get('error'))
            record=json.loads((out/'queries/0000.json').read_text())
            self.assertGreater(record['compute_seconds'],0.)
            self.assertGreater(record['saving_seconds'],0.)
            self.assertAlmostEqual(record['seconds'],record['compute_seconds']+record['saving_seconds'])


if __name__=='__main__':unittest.main()
