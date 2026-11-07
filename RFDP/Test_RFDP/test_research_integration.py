"""Runner/readback contracts for the fixed research methods; no data experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from MODEL.Paper_RFDP import PaperFullIterative as model
from experiments.continuous_fitting import certified, full, research_config as cfg
from Test_RFDP import test_f1_integration as previous
from Test_RFDP.test_f1_config import checked_base


class ResearchIntegrationTests(unittest.TestCase):
    def test_new_sources_rebuild_complete_synthetic_trajectories(self):
        for source in cfg.SOURCES:
            with self.subTest(source=source):
                args,diag,accepted=previous.F1IntegrationTests.trajectory(self,source)
                self.assertGreater(accepted['bridge_events'],0)
                self.assertEqual(accepted['linear_systems'],diag['linear_systems'])
                self.assertEqual(accepted['mm_iterations'],diag['mm_iterations'])
                self.assertGreater(accepted['solver_calls'],0)
                self.assertEqual(sorted(args[5]),list(range(1,17)))
                if source in cfg.MAIN:self.assertGreater(diag['mm_iterations'],0)

    def test_initial_state_survives_successful_inline_trace_pruning(self):
        n=8
        affinity=np.full((n,n),.25);np.fill_diagonal(affinity,1.)
        graph=sparse.csr_matrix(affinity)
        config=dict(source=cfg.MAIN[0],alpha=5.,beta=25.,depth=7,
                    trace_retention='sampled_after_readback',retained_queries=[],retain_initial_state=True)
        old=full.install_solver(config)
        try:
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory);(out/'queries').mkdir();(out/'accepted').mkdir()
                with patch.multiple(full,create=True,CFG=config,OUT=out,QN=1,GN=n-1,
                                    SIG='integration',GRAPH=graph,FULL=affinity,X=np.eye(n),
                                    INIT_ERROR=None,READBACK_KERNEL=None,READBACK_SUMS=None,
                                    ADMISSION_CAMERAS=None,ENGINE=model.FullIHyRDP(graph,affinity)):
                    row=full.compute_query(0)
                    self.assertNotIn('error',row,row.get('error'))
                    self.assertFalse((out/'queries/0000.trace.jsonl.gz').exists())
                    first=out/'queries/0000.initial.jsonl.gz'
                    self.assertTrue(first.exists())
                    accepted=json.loads((out/'accepted/0000.json').read_text())
                    self.assertEqual(accepted['initial_state_sha256'],full.sha(first))
                    self.assertTrue((out/'queries/0000.rankings.npy').exists())
        finally:
            model.solve_hybrid_sparse=old;certified.EVENTS.clear()

    def test_unbound_and_disabled_recipes_stop_before_sources(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(full,'sources',side_effect=AssertionError('data access')):
            for source in cfg.SOURCES:
                record=cfg.recipe('market-osnet',source)
                path=Path(directory)/'recipe.json';path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):full.main('compute',path)
            bound=cfg.bind(cfg.recipe('market-osnet',cfg.BACKUP),checked_base('market-osnet'),output_root='/new/runs',commit='d'*40)
            path.write_text(json.dumps(bound))
            with self.assertRaisesRegex(ValueError,'disabled'):full.main('compute',path)

    def test_algorithm_ids_bind_manifest_and_sources(self):
        configs=[cfg.bind(cfg.recipe('market-osnet',s),checked_base('market-osnet'),output_root='/new/runs',commit='d'*40) for s in cfg.SOURCES]
        signatures=[full.fitting_binding(c) for c in configs]
        self.assertEqual(len({json.dumps(x,sort_keys=True) for x in signatures}),3)
        for c in configs:
            manifest={'config':copy.deepcopy(c),'solver_id':full.solver_id(c['source'])}
            full.check_admission_manifest(c,manifest)
            damaged=copy.deepcopy(manifest);damaged['solver_id']='old'
            with self.assertRaises(AssertionError):full.check_admission_manifest(c,damaged)
            damaged=copy.deepcopy(manifest);damaged['config']['commit']='e'*40
            with self.assertRaises(AssertionError):full.check_admission_manifest(c,damaged)
        with self.assertRaises(AssertionError):full.check_admission_manifest(configs[0],{'config':configs[1]})


if __name__=='__main__':unittest.main()
