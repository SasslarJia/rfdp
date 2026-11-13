"""Synthetic-only protocol and saved-score pipeline checks; no ReID runs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy import sparse
from experiments.continuous_fitting import full, certified
from experiments.reid_adaptation import protocol, runtime, resources
from experiments.reid_adaptation import full_run
from experiments.reid_adaptation.io import labels_from_manifest


class AdaptationProtocolTests(unittest.TestCase):
    def test_exact_matrix_and_fixed_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan=protocol.write_recipes(p)
            self.assertEqual((plan['first_stage_count'],plan['conditional_combination_max'],plan['conditional_late5_max']),(8,4,1))
            self.assertEqual(len(list(p.glob('*.json'))),13)
            for entry in plan['runs']:
                cfg=json.loads((p/entry['recipe']).read_text());protocol.validate(cfg)
                with self.assertRaises(ValueError):protocol.validate(cfg,require_bound=True)
                self.assertEqual((cfg['alpha'],cfg['beta'],cfg['depth'],cfg['bridge_budget']),(5,25,500,5))
                for key,value in [('alpha',10),('gamma',.3),('support_lambda',.25),('workers',4)]:
                    changed=dict(cfg,**{key:value})
                    with self.assertRaises(ValueError):protocol.validate(changed)
            with self.assertRaises(ValueError):protocol.recipe('early5','market-osnet')
            with self.assertRaises(ValueError):protocol.recipe('camera_boundary','duke-transreid')

    def test_metadata_order_and_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'labels.json'
            rows=[dict(path=f'/images/{i}.jpg',split='query' if i<2 else 'gallery',pid=i//2,camid=i%2) for i in range(6)]
            p.write_text(json.dumps(rows));got=labels_from_manifest(p,2,4)
            np.testing.assert_array_equal(got['gallery_cam'],[0,1,0,1])
            for mutation in ('split','duplicate','count'):
                changed=copy.deepcopy(rows)
                if mutation=='split':changed[2]['split']='query'
                elif mutation=='duplicate':changed[-1]['path']=changed[-2]['path']
                else:changed.pop()
                p.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):labels_from_manifest(p,2,4)

    def test_cpu_quota_and_native_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cpu.max').write_text('1400000 100000')
            with patch.object(resources.os,'sched_getaffinity',return_value=set(range(14)),create=True):
                self.assertEqual(resources.cpu_allocation(root)['workers'],12)
                (root/'cpu.max').write_text('800000 100000')
                self.assertEqual(resources.cpu_allocation(root)['workers'],8)
            with self.assertRaises(ValueError):resources.native_path('/mnt/d/active-run')
            info=root/'meminfo';info.write_text('MemTotal: 33554432 kB\nMemAvailable: 32505856 kB\n')
            (root/'memory.max').write_text(str(20*2**30));(root/'memory.current').write_text(str(2**30))
            self.assertEqual(resources.memory_allocation(root,info)['cap'],20*2**30)

    def test_conditional_decision_has_predecessors(self):
        cfg=protocol.recipe('p1_graph','market-osnet')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'decision.json'
            path.write_text(json.dumps(dict(verdict='continue',rationale='test',allowed_units=[full_run.unit_id(cfg)],evidence=[])))
            with self.assertRaises(ValueError):full_run.check_decision(cfg,None)
            with self.assertRaises(ValueError):full_run.check_decision(cfg,path)
            with patch.object(full_run,'decision_evidence',return_value={('p1_graph','market-transreid'),('p1_graph','duke-transreid')}):
                full_run.check_decision(cfg,path)
            changed=json.loads(path.read_text());changed['allowed_units']=[];path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):full_run.check_decision(cfg,path)

    def test_wsl_resources_do_not_call_gpu_or_cloud_guard(self):
        with patch.object(full,'CFG',dict(adaptation_schema=protocol.SCHEMA),create=True), \
             patch.object(resources,'snapshot',return_value={'cpu':'wsl'}) as snap, \
             patch.object(full.subprocess,'check_output',side_effect=AssertionError('GPU invoked')):
            self.assertEqual(full.resources(),{'cpu':'wsl'});snap.assert_called_once()

    def test_run_rechecks_metadata_and_graph_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'labels.json'
            rows=[dict(path=f'{i}.jpg',split='query' if i<2 else 'gallery',pid=i//2,camid=i%2) for i in range(6)]
            manifest.write_text(json.dumps(rows));expected=labels_from_manifest(manifest,2,4)
            np.savez(root/'labels.npz',**expected);np.save(root/'cameras.npy',np.arange(6)%2)
            cfg=dict(k=20,queries=2,gallery=4,label_manifest=str(manifest),labels_file=str(root/'labels.npz'),camera_ids_file=str(root/'cameras.npy'))
            full_run.check_bound_inputs(cfg)
            np.save(root/'cameras.npy',np.zeros(6,dtype=int))
            with self.assertRaises(ValueError):full_run.check_bound_inputs(cfg)
            meta=dict(status='complete',builder_id=full_run.graph.BUILDER_ID,parameters=dict(kd=30,ks=30,gamma=.4),
                      n=6,features_renormalized=False,files={n:('1'*64) for n in ('features.npy','affinity.npy','graph.npz')})
            gcfg=dict(kd=30,ks=30,gamma=.4,queries=2,gallery=4,expected_cache_input_hashes={'features.npy':'1'*64})
            (root/'manifest.json').write_text(json.dumps(meta));hashes={str(root/n):h for n,h in meta['files'].items()}
            full_run.check_graph(root,gcfg,hashes)
            del meta['files']['affinity.npy'];(root/'manifest.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):full_run.check_graph(root,gcfg,hashes)

    def test_graph_build_memory_line_is_checked(self):
        with patch.object(resources,'memory_allocation',return_value=dict(current=71,cap=100)):
            with self.assertRaises(RuntimeError):full_run.graph_memory_guard()


class AdaptationFullIOTests(unittest.TestCase):
    def test_real_capture_readback_and_evaluation_on_synthetic_graph(self):
        rng=np.random.default_rng(19);n=18;qn=2;gn=n-qn
        features=rng.normal(size=(n,7)).astype(np.float32)
        features/=np.linalg.norm(features,axis=1,keepdims=True)
        affinity=rng.uniform(.1,.8,size=(n,n));affinity=(affinity+affinity.T)/2;np.fill_diagonal(affinity,1.)
        # Several disconnected components exercise original full-affinity bridging.
        g=np.zeros_like(affinity)
        for start,stop in [(0,4),(4,10),(10,n)]:g[start:stop,start:stop]=affinity[start:stop,start:stop]
        graph=sparse.csr_matrix(g)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);camera=root/'camera.npy';np.save(camera,np.arange(n)%3)
            labels=root/'labels.npz';np.savez(labels,query_label=[0,1],query_cam=[0,1],
                                            gallery_label=np.arange(gn)%2,gallery_cam=np.arange(gn)%3)
            for route,schedule in [('p1_graph','fixed5'),('early5','early5'),('camera_boundary','fixed5')]:
                out=root/route;out.mkdir();(out/'queries').mkdir();(out/'accepted').mkdir()
                cfg=dict(output=str(out),adaptation_schema=protocol.SCHEMA,adaptation_route=route,source='original',
                         relation_model='original',admission_policy='original',promotion_schedule=schedule,
                         camera_ids_file=str(camera),boundary_rho=.5 if route=='camera_boundary' else 1.,
                         alpha=5,beta=25,step=5,depth=15,queries=qn,gallery=gn,labels_file=str(labels),
                         trace_retention='sampled_after_readback',retained_queries=[0],retain_initial_state=True)
                engine,readgraph,kernel,sums,certifier=runtime.initialize(cfg,graph,affinity)
                signature=protocol.digest(cfg)
                # Only the full-size recipe gate is bypassed for this synthetic IO fixture.
                globals_=dict(CFG=cfg,OUT=out,QN=qn,GN=gn,INIT_ERROR=None,ENGINE=engine,GRAPH=readgraph,
                              FULL=affinity,X=features,SIG=signature,READBACK_KERNEL=kernel,
                              READBACK_SUMS=sums,STAGE_CERTIFIER=certifier,ADMISSION_CAMERAS=None,EXECUTION_COMMIT=None)
                with patch.multiple(full,create=True,**globals_):
                    for q in range(qn):
                        result=full.compute_query(q)
                        self.assertNotIn('error',result,result.get('error'))
                        self.assertTrue(full_run.query_complete(cfg,signature,q))
                        row=json.loads((out/'queries'/f'{q:04d}.json').read_text())
                        for key in ('compute_seconds','saving_seconds','readback_seconds'):self.assertGreaterEqual(row[key],0)
                    metrics=full.evaluate();self.assertEqual(metrics['valid_queries'],qn)
                    full.dump(out/'COMPUTE_COMPLETE.json',dict(signature=signature,rankings_sha256=full.sha(out/'rankings.npy'),metrics_sha256=full.sha(out/'metrics.json')))
                    full.aggregate_inline_readback()
                    self.assertEqual(np.load(out/'rankings.npy').shape,(qn,gn))
                    self.assertFalse((out/'queries'/'0001.trace.jsonl.gz').exists())
                    self.assertTrue((out/'queries'/'0001.initial.jsonl.gz').exists())
                    trace=list(full.read_rows(out/'queries'/'0000.trace.jsonl.gz'))
                    self.assertEqual(trace[0]['promotion_schedule'],schedule)
                    if schedule=='early5':
                        self.assertEqual(len(trace[0]['selected']),1)
                        self.assertGreater(len(trace[0]['event']['returned_prefix_local']),1)
                    full.dump(out/'run-manifest.json',dict(config=cfg,signature=signature,solver_id=runtime.SOLVER_ID))
                    full.dump(out/'FULL_ACCEPTANCE.json',dict(status='accepted_full_population',queries=qn,gallery=gn,signature=signature,
                        max_recomputed_residual=1e-10,rankings_sha256=full.sha(out/'rankings.npy'),metrics_sha256=full.sha(out/'metrics.json')))
                    with patch.object(protocol,'validate',return_value={}):
                        self.assertIsNotNone(full_run.accepted_result(out,cfg))
                        saved=(out/'metrics.json').read_bytes();(out/'metrics.json').write_text('{}')
                        with self.assertRaises(ValueError):full_run.accepted_result(out,cfg)
                        (out/'metrics.json').write_bytes(saved)
                        (out/'queries'/'0000.initial.jsonl.gz').unlink()
                        with self.assertRaises(FileNotFoundError):full_run.accepted_result(out,cfg)
        certified.EVENTS.clear()

if __name__=='__main__':unittest.main()
