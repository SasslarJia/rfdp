"""Small-graph admission semantics and certified saved-trajectory read-back."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from MODEL.Paper_RFDP import PaperFullIterative as model
from experiments.continuous_fitting import certified, full
from experiments.continuous_fitting.verify import verify_query, read_rows


def id_scores(w, e, *args, **kwargs):
    """A fixed nomination order for tests of admission, never a numeric oracle."""
    return np.arange(len(e), dtype=float), 0.


class AdmissionLogicTests(unittest.TestCase):
    def fixture(self, policy, cameras=None):
        n=9;graph=np.eye(n)
        graph[0,2:]=graph[2:,0]=.4
        for node in (4,6,7):graph[1,node]=graph[node,1]=.3
        if cameras is None:cameras=np.array([10,20,30,10,20,30,10,30,20])
        return model.FullIHyRDP(graph,np.ones((n,n)),admission_policy=policy,camera_ids=cameras)

    def first_stage(self, policy, depth=8, cameras=None):
        trace=[]
        with patch.object(model,'solve_hybrid_sparse',side_effect=id_scores):
            self.fixture(policy,cameras).rank([0,1],nq=depth,trace=trace.append)
        return trace[0]

    def test_gate_and_same_state_prefix_control(self):
        gate=self.first_stage('camera_gate')
        prefix=self.first_stage('camera_gate_prefix')
        for t in (gate,prefix):
            self.assertEqual(t['nominated'],[2,3,4,5,6])
            self.assertEqual(t['support_camera_counts'],[1,1,2,1,2])
            self.assertNotIn(7,t['selected'])  # Qualified, but outside original nominations.
        self.assertEqual(gate['selected'],[2,4,6])
        self.assertEqual(prefix['selected'],[2,3,4])

    def test_remaining_depth_truncates_before_gate(self):
        for policy in ('camera_gate','camera_gate_prefix'):
            t=self.first_stage(policy,depth=4)
            self.assertEqual(t['nominated'],[2,3])
            self.assertEqual(t['selected'],[2])  # Do not refill with qualified node 4.

    def test_duplicate_camera_sources_and_candidate_camera_do_not_count(self):
        cameras=np.array([10,10,20,20,20,20,20,20,20])
        t=self.first_stage('camera_gate',cameras=cameras)
        self.assertEqual(t['support_camera_counts'],[1,1,1,1,1])
        self.assertEqual(t['selected'],[2])

    def test_small_frontier_preserves_id_nominations_and_old_batch_snapshot(self):
        graph=np.ones((6,6));cameras=np.array([0,1,1,1,1,1]);trace=[]
        with patch.object(model,'solve_hybrid_sparse',side_effect=AssertionError('small frontier must not solve')):
            model.FullIHyRDP(graph,graph,admission_policy='camera_gate',camera_ids=cameras).rank([0],trace=trace.append)
        self.assertEqual(trace[0]['nominated'],[1,2,3,4,5])
        self.assertEqual(trace[0]['support_camera_counts'],[1,1,1,1,1])
        self.assertEqual(trace[0]['selected'],[1])
        self.assertEqual(trace[1]['support_camera_counts'],[2,2,2,2])
        self.assertEqual(trace[1]['selected'],[2,3,4,5])

    def test_default_and_explicit_original_keep_legacy_trace(self):
        w=np.eye(9)+.4*np.eye(9,k=1)+.4*np.eye(9,k=-1)
        records=[]
        for kwargs in ({},{'admission_policy':'original'}):
            trace=[];order,_=model.FullIHyRDP(w,np.ones((9,9)),**kwargs).rank([0],trace=trace.append)
            np.testing.assert_array_equal(order,np.arange(9))
            self.assertTrue(all('nominated' not in t and 'admission_policy' not in t for t in trace))
            records.append(trace)
        self.assertEqual(records[0],records[1])

    def test_bridges_are_transient_and_unselected_components_leave_frontier(self):
        n=9;w=np.eye(n)
        for ids in ([1,2,3],[4,5,6],[7,8]):
            for i,j in zip(ids,ids[1:]):w[i,j]=w[j,i]=.4
        affinity=np.ones((n,n));affinity[4,1]=affinity[1,4]=3.
        affinity[4,0]=affinity[0,4]=2.
        cameras=np.array([10,20,20,20,30,30,30,40,40])
        trace=[]
        with patch.object(model,'solve_hybrid_sparse',side_effect=id_scores):
            model.FullIHyRDP(w,affinity,admission_policy='camera_gate',camera_ids=cameras).rank([0],trace=trace.append)
        self.assertEqual(trace[0]['selected'],[1])
        self.assertTrue(any(i==4 for i,j,v in trace[0]['bridges']))
        self.assertNotIn(4,trace[1]['unlabeled'])
        self.assertFalse(trace[1]['bridges'])
        bridge_stages=[t for t in trace if t['bridges']]
        self.assertGreaterEqual(len(bridge_stages),2)
        for t in bridge_stages:
            self.assertTrue(all(c<=1 for c in t['support_camera_counts']))
            self.assertEqual(len(t['selected']),1)

    def test_policy_inputs_and_numerical_failure(self):
        w=np.ones((8,8))
        for kwargs in ({'admission_policy':'unknown'},
                       {'admission_policy':'camera_gate'},
                       {'admission_policy':'camera_gate','camera_ids':np.zeros(7,dtype=int)},
                       {'admission_policy':'camera_gate','camera_ids':np.zeros(8)},
                       {'admission_policy':'camera_gate','camera_ids':np.zeros(8,dtype=int),'step':2}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):model.FullIHyRDP(w,w,**kwargs)
        with patch.object(model,'solve_hybrid_sparse',side_effect=np.linalg.LinAlgError('numeric failure')):
            with self.assertRaises(np.linalg.LinAlgError):
                model.FullIHyRDP(w,w,admission_policy='camera_gate',camera_ids=np.arange(8)).rank([0],nq=7)


class AdmissionReadbackTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2041);self.n=17
        affinity=rng.uniform(.1,1.,(self.n,self.n));self.affinity=(affinity+affinity.T)/2
        np.fill_diagonal(self.affinity,1.)
        graph=np.zeros_like(self.affinity)
        for ids in (list(range(8)),list(range(8,self.n))):graph[np.ix_(ids,ids)]=self.affinity[np.ix_(ids,ids)]
        self.graph=sparse.csr_matrix(graph);self.cameras=np.arange(self.n)%3
        self.old_solver=model.solve_hybrid_sparse

    def tearDown(self):model.solve_hybrid_sparse=self.old_solver

    def config(self,out,policy='camera_gate'):
        camera_file=out/'cameras.npy';np.save(camera_file,self.cameras)
        return dict(source='original',relation_model='original',alpha=5.,beta=25.,step=5,depth=16,
                    admission_policy=policy,admission_version=model.ADMISSION_VERSION,
                    camera_ids_file=str(camera_file),input_sha256={str(camera_file):full.sha(camera_file)})

    def test_camera_file_and_policy_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.config(Path(directory))
            np.testing.assert_array_equal(full.load_admission_cameras(cfg,self.n),self.cameras)
            a=full.admission_binding(cfg);other=dict(cfg,admission_policy='camera_gate_prefix')
            self.assertNotEqual(a,full.admission_binding(other))
            manifest={'config':copy.deepcopy(cfg)}
            full.check_admission_manifest(cfg,manifest)
            for changed in (other,dict(cfg,admission_policy='original'),
                            dict(cfg,input_sha256={cfg['camera_ids_file']:'0'*64})):
                with self.assertRaises(AssertionError):full.check_admission_manifest(changed,manifest)
            for update in ({'source':'continuous'},{'relation_model':'csw'},
                           {'admission_version':'wrong'},{'input_sha256':{}},{'alpha':10.}):
                with self.subTest(update=update),self.assertRaises(AssertionError):full.admission_binding(dict(cfg,**update))
            np.save(cfg['camera_ids_file'],np.zeros(self.n,dtype=int))
            with self.assertRaises(AssertionError):full.load_admission_cameras(cfg,self.n)
        self.assertEqual(full.admission_binding({'source':'original'}),{})

    def test_both_policies_certify_nominations_and_reject_tampering(self):
        for policy in ('camera_gate','camera_gate_prefix'):
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as directory:
                out=Path(directory);(out/'queries').mkdir();(out/'accepted').mkdir()
                cfg=self.config(out,policy);certified.install('original')
                engine=model.FullIHyRDP(self.graph,self.affinity,admission_policy=policy,camera_ids=self.cameras)
                with patch.multiple(full,create=True,OUT=out,CFG=cfg,QN=1,GN=self.n-1,ENGINE=engine,
                        GRAPH=self.graph,FULL=self.affinity,X=np.eye(self.n),SIG='synthetic',INIT_ERROR=None,
                        READBACK_KERNEL=None,READBACK_SUMS=None,ADMISSION_CAMERAS=self.cameras):
                    result=full.compute_query(0);self.assertNotIn('error',result,result.get('error'))
                    accepted=full.accept_query(0);self.assertNotIn('error',accepted,accepted.get('error'))
                    self.assertGreater(accepted['bridge_events'],0)
                    row=json.loads((out/'queries/0000.json').read_text())
                    self.assertEqual(row['camera_ids_sha256'],cfg['input_sha256'][cfg['camera_ids_file']])
                prefix=np.load(out/'queries/0000.prefix.npy');rank=np.load(out/'queries/0000.rankings.npy')
                rows=list(read_rows(out/'queries/0000.trace.jsonl.gz'))
                self.assertEqual(rows[0]['selected'],rows[0]['nominated'][:1])
                self.assertTrue(any(t['returned_scores'] is not None for t in rows))
                def verify(records,**kwargs):
                    options=dict(admission_policy=policy,camera_ids=self.cameras,admission_version=model.ADMISSION_VERSION)
                    options.update(kwargs)
                    return verify_query(self.graph,self.affinity,np.eye(self.n)[0],0,prefix,rank,records,'original',1,16,**options)
                verify(rows)
                for field,value in [('nominated',list(reversed(rows[0]['nominated']))),
                                    ('support_camera_counts',[2]*len(rows[0]['nominated'])),
                                    ('selected',rows[0]['nominated'][:2]),
                                    ('admission_policy','original'),('admission_version','wrong')]:
                    damaged=copy.deepcopy(rows);damaged[0][field]=value
                    with self.subTest(field=field),self.assertRaises(AssertionError):verify(damaged)
                with self.assertRaises(AssertionError):verify(rows,camera_ids=np.zeros(self.n,dtype=int))
                with self.assertRaises(AssertionError):verify(rows,admission_version='wrong')
                damaged=copy.deepcopy(rows);t=next(t for t in damaged if t['returned_scores'] is not None)
                t['returned_scores']=[certified.packed(1.) for _ in t['returned_scores']]
                t['event']['returned_scores_sha256']=hashlib.sha256(json.dumps(t['returned_scores'],separators=(',',':')).encode()).hexdigest()
                with self.assertRaises(np.linalg.LinAlgError):verify(damaged)


if __name__=='__main__':unittest.main()
