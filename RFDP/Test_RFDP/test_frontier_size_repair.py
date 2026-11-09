"""Functional regressions for complete sparse frontiers, not retrieval probes."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse

from experiments.continuous_fitting import certified, full
from experiments.continuous_fitting.verify import verify_query
from experiments.relation_routes.smoothing import System


class FullFrontierTests(unittest.TestCase):
    def test_original_solver_and_saved_points_above_old_cap(self):
        # Independent anchored singleton components have the exact solution
        # psi_i=(1+e_i)/e_i. Test just across the old rejected size boundary.
        n=8193
        w=sparse.eye(n,format='csr');e=np.linspace(.25,1.,n)
        values,residual,event=certified.solve(w,e,source='original')
        np.testing.assert_allclose(np.asarray(values,dtype=float),(1.+e)/e,rtol=1e-10)
        self.assertEqual(len(values),n)
        self.assertEqual(event['returned_prefix_local'],list(range(n-1,n-6,-1)))
        self.assertLessEqual(residual,1e-8)
        saved=certified.certify_saved(w,e,values,source='original')
        self.assertEqual(saved['prefix'],event['returned_prefix_local'])
        # A query connected to all singleton frontier nodes gives this exact
        # stage in a real sparse graph. Verify reconstruction and the complete
        # gallery, not just the standalone equation certificate.
        ids=np.arange(1,n+1);zeros=np.zeros(n,dtype=int)
        graph=sparse.eye(n+1,format='csr')+sparse.csr_matrix(
            (np.r_[e,e],(np.r_[zeros,ids],np.r_[ids,zeros])),shape=(n+1,n+1))
        chosen=[i+1 for i in event['returned_prefix_local']];prefix=[0]+chosen
        rank=chosen+[i for i in range(1,n+1) if i not in chosen]
        trace=dict(iteration=0,labeled=[0],direct=ids.tolist(),indirect=[],
            unlabeled=ids.tolist(),selected=chosen,bridges=[],residual=residual,
            returned_scores=[certified.packed(v) for v in values],event=event)
        checked=verify_query(graph,None,np.zeros(n+1),0,prefix,rank,[trace],
            'original',1,6,full_sums=np.ones(n+1))
        self.assertEqual(checked['max_unlabeled'],n)
        self.assertLessEqual(checked['max_recomputed_residual'],1e-8)

    def test_internal_smoothing_preserves_large_frontier(self):
        n=8193;w=sparse.eye(n,format='csr');e=np.linspace(.25,1.,n)
        for source,cameras in [('camera_channel',np.arange(n)%2),('th_target',None)]:
            system=System(w,e,5.,25.,source,cameras,1,0.)
            self.assertEqual(system.n,n)
            self.assertEqual(system.w.shape,(n,n))


class ExecutionRepairTests(unittest.TestCase):
    def test_repair_requires_exact_config_commits_diff_and_clean_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'csw_grouped.json'
            cfg={'commit':'a'*40};config.write_text(json.dumps(cfg))
            head='b'*40;diff=b'reviewed complete diff';dirty=''
            def git(args,**kwargs):
                if args[1]=='rev-parse':return head+'\n'
                if args[1]=='status':return dirty
                if args[1]=='diff':return diff
                raise AssertionError(args)
            with mock.patch.object(full,'CFG',cfg,create=True),mock.patch.object(full,'CONFIG_PATH',str(config),create=True),mock.patch.object(full.subprocess,'check_output',side_effect=git):
                with self.assertRaises(ValueError):full.execution_revision()
                sidecar=config.with_suffix('.repair.json')
                record=dict(base_commit=cfg['commit'],execution_commit=head,config_sha256=full.sha(config),
                    code_diff_sha256=hashlib.sha256(diff).hexdigest(),scientific_config_unchanged=True,
                    rationale='Remove fixed frontier node cap; preserve all equations and certificates.')
                sidecar.write_text(json.dumps(record))
                self.assertEqual(full.execution_revision(),head)
                for key in ('base_commit','execution_commit','config_sha256','code_diff_sha256'):
                    wrong=dict(record);wrong[key]='bad';sidecar.write_text(json.dumps(wrong))
                    with self.assertRaises(ValueError):full.execution_revision()
                sidecar.write_text(json.dumps(record));dirty=' M solver.py'
                with self.assertRaises(AssertionError):full.execution_revision()


if __name__=='__main__':
    unittest.main()
