"""Numerical/IO fixtures only; no ReID experiment, server, or timing probe."""
import contextlib
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse
from experiments.relation_routes import protocol, runtime, full_run, report
from experiments.relation_routes.csw_grouped import build_grouped_csw_inputs
from experiments.continuous_fitting import full, certified
from MODEL.Paper_RFDP import PaperFullIterative as model


def bound(route,root):
    cfg=protocol.recipe(route)
    cfg.update(path_binding='bound',commit='a'*40,cache=str(root/'cache'),mat=str(root/'features.mat'),
               output=str(root/'runs'/route),camera_ids_file=str(root/'cameras.npy'),workers=16)
    cfg['input_sha256']={str(root/'cache'/name):value for name,value in protocol.FROZEN['cache_input_hashes'].items()}
    cfg['input_sha256'][cfg['mat']]=protocol.FROZEN['feature_sha256']
    cfg['input_sha256'][cfg['camera_ids_file']]='b'*64
    if route=='csw_grouped':
        for key,name in [('grouped_graph_file','grouped.npz'),('support_file','support.npz'),('grouped_metadata_file','metadata.json')]:
            cfg[key]=str(root/name);cfg['input_sha256'][cfg[key]]='b'*64
    protocol.validate(cfg,require_bound=True)
    return cfg


def references():
    return dict(records=[dict(method=name,input_key='market-osnet',queries=3368,gallery=19732,
        feature_sha256=protocol.FROZEN['feature_sha256'],mAP=.5,rank1=.6,mINP=.4,
        provenance={'verification':{'official_population_verified':True}}) for name in protocol.REFERENCE_METHODS])


@contextlib.contextmanager
def tiny_transaction(root,route):
    """Exercise real producer/readback on an algebraic 14-node test fixture.

    Only this in-process unit fixture overrides population/depth after runtime
    binding. The public entry point rejects these settings before opening data.
    """
    cfg=bound(route,root)
    rng=np.random.default_rng(84)
    affinity=rng.uniform(.1,.9,(14,14));affinity=(affinity+affinity.T)/2
    np.fill_diagonal(affinity,1.)
    graph=sparse.csr_matrix(affinity);cameras=np.arange(14)%3
    np.save(cfg['camera_ids_file'],cameras)
    if route=='csw_grouped':
        raw,support,metadata=build_grouped_csw_inputs(affinity,cameras)
        sparse.save_npz(cfg['grouped_graph_file'],raw);sparse.save_npz(cfg['support_file'],support)
        full.dump(cfg['grouped_metadata_file'],metadata)
    original_solver=model.solve_hybrid_sparse
    engine,graph,kernel,sums,certifier=runtime.initialize(cfg,graph,affinity)
    out=Path(cfg['output']);(out/'queries').mkdir(parents=True);(out/'accepted').mkdir()
    features=rng.normal(size=(14,6));features/=np.linalg.norm(features,axis=1)[:,None]
    unit_cfg=dict(cfg,depth=12,queries=2,gallery=12,retained_queries=[0])
    values=dict(CFG=unit_cfg,OUT=out,QN=2,GN=12,GRAPH=graph,FULL=affinity,X=features,
        SIG='fixture-signature',INIT_ERROR=None,ENGINE=engine,READBACK_KERNEL=kernel,
        READBACK_SUMS=sums,ADMISSION_CAMERAS=None,STAGE_CERTIFIER=certifier)
    with contextlib.ExitStack() as stack:
        for key,value in values.items():stack.enter_context(mock.patch.object(full,key,value,create=True))
        try:yield unit_cfg,out
        finally:model.solve_hybrid_sparse=original_solver;certified.EVENTS.clear()


class FullRelationProtocolTests(unittest.TestCase):
    def test_checked_in_configs_match_the_actual_prepare_recipes(self):
        folder=full.REPO/'configs/relation-routes/full-market-osnet'
        self.assertEqual(len(list(folder.glob('*.json'))),3)
        for route in protocol.ROUTES:
            self.assertEqual(json.loads((folder/f'market-osnet.{route}.json').read_text()),protocol.recipe(route))

    def test_prepare_binds_three_jobs_and_reuses_graph_without_query_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);root=directory/'new';old=directory/'old'
            cache,mat=old/'cache',old/'source.mat'
            base=directory/'base.json';full.dump(base,dict(cache=str(cache),mat=str(mat),output=str(old/'baseline')))
            ref=directory/'references.json';full.dump(ref,references())
            hashes={str(cache/name):value for name,value in protocol.FROZEN['cache_input_hashes'].items()}
            hashes[str(mat)]=protocol.FROZEN['feature_sha256']
            fake_affinity=mock.Mock(shape=(23100,23100))
            graph=sparse.eye(23100,format='csr')
            cams=dict(query_cam=np.ones(3368,dtype=int),gallery_cam=np.ones(19732,dtype=int))
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(full_run,'checkout',return_value='a'*40))
                stack.enter_context(mock.patch.object(full_run,'checked_inputs',return_value=(cache,mat,hashes)))
                stack.enter_context(mock.patch.object(full_run,'allocated_cpu_count',return_value=(16,{'workers':16})))
                stack.enter_context(mock.patch.object(full_run,'resource_snapshot',return_value={}))
                stack.enter_context(mock.patch.object(full_run,'loadmat',return_value=cams))
                stack.enter_context(mock.patch.object(full_run.np,'load',return_value=fake_affinity))
                builder=stack.enter_context(mock.patch('experiments.relation_routes.csw_grouped.build_grouped_csw_inputs',return_value=(graph,graph,{'unit_fixture':'builder independently tested'})))
                compute=stack.enter_context(mock.patch.object(full,'compute_query',side_effect=AssertionError('prepare must not execute queries')))
                first=full_run.prepare(base,root,ref)
                second=full_run.prepare(base,root,ref)
                self.assertEqual(first,second);self.assertEqual(builder.call_count,1)
                self.assertFalse(compute.called)
                for route in protocol.ROUTES:
                    cfg=json.loads((root/'configs'/f'{route}.json').read_text())
                    protocol.validate(cfg,True)
                    self.assertEqual(cfg['workers'],16)
                self.assertFalse((root/'runs').exists())

    def test_only_three_official_recipes_and_frozen_parameters(self):
        self.assertEqual(set(protocol.ROUTES),{'camera_channel','csw_grouped','th_target'})
        for route in protocol.ROUTES:
            cfg=protocol.recipe(route)
            self.assertEqual((cfg['queries'],cfg['gallery'],cfg['depth'],cfg['step']),(3368,19732,500,5))
            self.assertEqual(cfg['reference_methods_reuse'],['I-HyRDP','CSW+I-HyRDP'])
            with self.assertRaises(ValueError):protocol.validate(cfg,require_bound=True)
            for key,value in [('queries',12),('gallery',84),('depth',100),('alpha',10),('step',1),('auto_start',True)]:
                bad=dict(cfg);bad[key]=value
                with self.assertRaises(ValueError):protocol.validate(bad)
        for stopped in ('original','csw','quaternary','f1_robust_converged'):
            with self.assertRaises(ValueError):protocol.recipe(stopped)

    def test_bound_paths_hashes_and_uncombined_routes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for route in protocol.ROUTES:
                cfg=bound(route,root)
                bad=copy.deepcopy(cfg);bad['input_sha256'][str(root/'cache/graph.npz')]='c'*64
                with self.assertRaises(ValueError):protocol.validate(bad,True)
                bad=dict(cfg,output=cfg['cache'])
                with self.assertRaises(ValueError):protocol.validate(bad,True)
            for cfg in [dict(bound('camera_channel',root),rho=.5),dict(bound('csw_grouped',root),theta=.5)]:
                with self.assertRaises(ValueError):protocol.validate(cfg,True)

    def test_cpu_uses_quota_and_affinity_not_host_count(self):
        with tempfile.TemporaryDirectory() as temp:
            cg=Path(temp);(cg/'cpu.max').write_text('1600000 100000\n')
            with mock.patch.object(full_run.os,'sched_getaffinity',return_value=set(range(128)),create=True):
                count,meta=full_run.allocated_cpu_count(cg)
                self.assertEqual(count,16);self.assertEqual(meta['visible_affinity_cpus'],128)
            (cg/'cpu.max').write_text('max 100000\n')
            with mock.patch.object(full_run.os,'sched_getaffinity',return_value={0,1},create=True):
                self.assertEqual(full_run.allocated_cpu_count(cg)[0],2)

    def test_cgroup_v1_and_v2_preserve_the_same_cpu_and_memory_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            cg=Path(temp);(cg/'cpu').mkdir();(cg/'memory').mkdir()
            (cg/'cpu/cpu.cfs_quota_us').write_text('1600000')
            (cg/'cpu/cpu.cfs_period_us').write_text('100000')
            (cg/'memory/memory.limit_in_bytes').write_text('66571993088')
            (cg/'memory/memory.usage_in_bytes').write_text('873558016')
            with mock.patch.object(full_run.os,'sched_getaffinity',return_value=set(range(128)),create=True):
                self.assertEqual(full_run.allocated_cpu_count(cg)[0],16)
            self.assertEqual(full_run.allocated_memory(cg),(66571993088,873558016))
            (cg/'memory.max').write_text('66571993088');(cg/'memory.current').write_text('873558016')
            self.assertEqual(full_run.allocated_memory(cg),(66571993088,873558016))

    def test_real_worker_equations_and_actual_saved_score_readback_all_routes(self):
        for route in protocol.ROUTES:
            with self.subTest(route=route),tempfile.TemporaryDirectory() as temp:
                with tiny_transaction(Path(temp),route) as (cfg,out):
                    with self.assertRaises(ValueError):protocol.validate(cfg,True)
                    result=full.compute_query(0)
                    self.assertNotIn('error',result,result.get('error'))
                    self.assertTrue(full_run.query_complete(cfg,full.SIG,0))
                    accepted=json.loads((out/'accepted/0000.json').read_text())
                    self.assertLessEqual(accepted['max_recomputed_residual'],1e-8)
                    np.testing.assert_array_equal(np.sort(np.load(out/'queries/0000.rankings.npy')),np.arange(2,14))
                    with gzip.open(out/'queries/0000.trace.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
                    if route!='csw_grouped':
                        self.assertTrue(all(r['event']['source']==cfg['source'] for r in rows if r['event']))
                        if route=='th_target':
                            self.assertTrue(all(r['event']['target_count']==len(r['labeled']) for r in rows if r['event']))

    def test_resume_skips_intact_query_but_rejects_damage_and_wrong_binding(self):
        with tempfile.TemporaryDirectory() as temp,tiny_transaction(Path(temp),'camera_channel') as (cfg,out):
            self.assertFalse(full_run.query_complete(cfg,full.SIG,0))
            self.assertNotIn('error',full.compute_query(0))
            self.assertTrue(full_run.query_complete(cfg,full.SIG,0))
            with self.assertRaises(ValueError):full_run.query_complete(cfg,'different-signature',0)
            path=out/'queries/0000.rankings.npy'
            rank=np.load(path);rank[0]=rank[1];np.save(path,rank)
            with self.assertRaises(ValueError):full_run.query_complete(cfg,full.SIG,0)

    def test_interrupted_query_without_completed_readback_is_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=bound('th_target',Path(temp));out=Path(cfg['output']);(out/'queries').mkdir(parents=True)
            full.dump(out/'queries/0000.json',dict(query=0,signature='s',source=cfg['source']))
            self.assertFalse(full_run.query_complete(cfg,'s',0))

    def test_pending_or_partial_metrics_never_enter_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref=root/'references.json';full.dump(ref,references())
            run=root/'runs/camera_channel';run.mkdir(parents=True)
            full.dump(run/'metrics.json',dict(mAP=1.,valid_queries=1))
            result=report.collect(root,ref)
            self.assertEqual(len(result['records']),2)
            self.assertEqual(len(result['pending']),3)
            self.assertNotIn('100.0000',(root/'结果汇总.md').read_text())
            wrong=references();wrong['records'][0]['queries']=12;full.dump(ref,wrong)
            with self.assertRaises(ValueError):report.collect(root,ref)

    def test_queue_only_calls_three_candidates_and_stops_on_technical_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);full.dump(root/'PREPARED.json',dict(routes=list(protocol.ROUTES),commit='a'*40,files_sha256={}))
            with mock.patch.object(full_run,'checkout',return_value='a'*40),mock.patch.object(report,'collect'),mock.patch.object(full_run,'run_one',return_value='completed') as run:
                full_run.run_queue(root,resume=True)
                self.assertEqual([c.args[0].stem for c in run.call_args_list],list(protocol.ROUTES))
                self.assertTrue(all(c.args[1] is True for c in run.call_args_list))
            with mock.patch.object(full_run,'checkout',return_value='a'*40),mock.patch.object(report,'collect'),mock.patch.object(full_run,'run_one',side_effect=[None,RuntimeError('numerical failure')]) as run:
                with self.assertRaises(RuntimeError):full_run.run_queue(root,resume=True)
                self.assertEqual(run.call_count,2)
                self.assertEqual(json.loads((root/'QUEUE_STATUS.json').read_text())['status'],'failed')

    def test_manifest_checks_route_context_before_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=bound('th_target',Path(temp));manifest=dict(config=copy.deepcopy(cfg),solver_id=full.solver_id(cfg['source']))
            full.check_admission_manifest(cfg,manifest)
            manifest['config']['camera_ids_file']='/other/cameras.npy'
            with self.assertRaises((AssertionError,ValueError)):full.check_admission_manifest(cfg,manifest)

    def test_final_aggregation_can_resume_after_duplicate_rank_files_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'queries').mkdir();(out/'accepted').mkdir()
            ranks=np.array([[2,3,4],[4,3,2]],dtype=np.int32);np.save(out/'rankings.npy',ranks)
            full.dump(out/'COMPUTE_COMPLETE.json',dict(signature='s',rankings_sha256=full.sha(out/'rankings.npy')))
            for q in range(2):
                dest=out/'queries';stem=f'{q:04d}'
                np.save(dest/f'{stem}.prefix.npy',np.array([q,2,3]))
                np.save(dest/f'{stem}.rankings.npy',ranks[q])
                trace=dest/f'{stem}.trace.jsonl.gz'
                trace.write_bytes(b'fixture trace');trace_sha=full.sha(trace)
                if q:trace.unlink()
                acceptance=dict(query=q,signature='s',trace_sha256=trace_sha)
                full.dump(out/'accepted'/f'{stem}.json',acceptance)
                row=dict(query=q,signature='s',trace_sha256=trace_sha,
                    readback_sha256=full.sha(out/'accepted'/f'{stem}.json'),
                    prefix_sha256=full.sha(dest/f'{stem}.prefix.npy'),
                    rankings_sha256=full.sha(dest/f'{stem}.rankings.npy'),trace_retained=q==0)
                full.dump(dest/f'{stem}.json',row)
            with mock.patch.object(full,'OUT',out,create=True),mock.patch.object(full,'QN',2,create=True),mock.patch.object(full,'GN',3,create=True),mock.patch.object(full,'CFG',dict(retained_queries=[0]),create=True):
                full.aggregate_inline_readback()
                self.assertFalse(list((out/'queries').glob('*.rankings.npy')))
                full.aggregate_inline_readback()


if __name__=='__main__':unittest.main()
