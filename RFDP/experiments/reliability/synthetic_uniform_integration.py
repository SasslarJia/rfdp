#!/usr/bin/env python3
"""Local-only synthetic integration; no real retrieval data or outcomes are read.

Writes only a new directory beneath this draft. Uses a copied repository to keep
legacy import-time log creation away from the immutable experiment snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy import sparse
from scipy.io import savemat
from synthetic_v2_helpers import generate_fixture,QUERY_COUNT,GALLERY_COUNT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repo',type=Path)
    args=parser.parse_args();source=Path(__file__).resolve().parent;draft=source.parent
    root=args.output_dir.resolve();require=lambda ok,msg: None if ok else (_ for _ in ()).throw(AssertionError(msg))
    require(root.is_relative_to(draft),'Synthetic output must remain inside draft')
    repo=(args.repo or draft/'repo').resolve();require(repo.is_relative_to(draft),'Synthetic repository must be the draft copy')
    root.mkdir(parents=True,exist_ok=False)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RFDP_PAPER_DIR=str(repo/'RFDP/MODEL/Paper_RFDP'))
    environment['PYTHONPATH']=str(source)+os.pathsep+environment.get('PYTHONPATH','')
    os.environ['RFDP_PAPER_DIR']=environment['RFDP_PAPER_DIR']
    fingerprint={str(path):sha(path)for path in source.glob('*.py')}
    stages=[];checks={};summary=dict(status='running',scope='96 artificial nodes only; not Market1501 or model-performance evidence',query_count=QUERY_COUNT,gallery_count=GALLERY_COUNT,depth=70,sources=fingerprint,stages=stages,checks=checks)
    def stage(name,command,success=True):
        log=root/(name+'.log');started=time.monotonic()
        print(json.dumps(dict(stage=name,expected_success=success)),flush=True)
        with log.open('w')as stream:result=subprocess.run([str(v)for v in command],cwd=repo,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        stages.append(dict(name=name,command=[str(v)for v in command],exit_code=result.returncode,seconds=time.monotonic()-started,log=str(log),log_sha256=sha(log)))
        require((result.returncode==0)==success,name+' unexpected exit; '+log.read_text()[-6000:])
    try:
        mat=root/'synthetic.mat';data,counts=generate_fixture(mat)
        # Two separated feature clusters, each 48 nodes, force cross-component
        # bridging before the predeclared depth 70. Labels remain evaluation-only.
        rng=np.random.default_rng(20261016)
        positions=np.arange(96);query_ids=np.arange(0,96,8);gallery_ids=np.setdiff1d(positions,query_ids)
        cloud=rng.normal(scale=.02,size=(96,64))
        features=np.column_stack((np.ones(96),cloud,np.where(positions<48,-.12,.12))).astype(np.float32)
        data['query_f']=features[query_ids];data['gallery_f']=features[gallery_ids]
        savemat(mat,data,do_compression=False,oned_as='row')
        with mat.open('r+b')as stream:stream.write(b'Synthetic uniform integration 20261015; no real ReID data'.ljust(116,b' '))
        summary['cross_camera_positives']=counts
        protocol=root/'SYNTHETIC-PROTOCOL.md';protocol.write_text('Synthetic software integration only. All 12 queries, 84 gallery, 96 artificial nodes. k20 gamma.4, rho.5, depth70, alpha5 beta25 step5. Ordinary and CB source lambda.5 controls; both source lambda0 controls against unchanged CSW. No real data, no numerical tail, tolerance1e-8. One separate test injects SYNTHETIC_FORCE_ONE_MP_FALLBACK to exercise same-equation MP evidence; it is not a real numerical failure.\n')
        cache=root/'cache'
        stage('build-cache',[sys.executable,'-B',repo/'RFDP/experiments/full_reid/build_feature_cache.py','--mat',mat,'--output',cache])
        cached=json.loads((cache/'manifest.json').read_text());require(cached['components']>1 and max(cached['component_sizes'])<70,'Fixture must exercise bridges')
        from reliability_model import build_supports,save_support
        cameras=np.r_[data['query_cam'],data['gallery_cam']]
        supports,metadata=build_supports(np.load(cache/'affinity.npy',mmap_mode='r'),cameras,k=20)
        support_paths={}
        for mode in ['ordinary','cb']:
            support_paths[mode]=root/f'support-{mode}.npz';save_support(support_paths[mode],supports[mode],metadata[mode])
        selections={}
        for lam in [0.,.5]:
            selection={'scope':'synthetic fixture only','models':{'resnet50':{mode:dict(model='resnet50',mode=mode,rho=.5,lambda_=lam,split='val',method='synthetic-source-'+mode,verified_sha256='0'*64)for mode in ['ordinary','cb']}}}
            path=root/f'synthetic-selection-{lam}.json';dump(path,selection);selections[lam]=path
        common=['--repo',repo,'--mat-path',mat,'--cache-dir',cache,'--protocol',protocol,'--rho','.5','--depth','70','--workers','1','--expected-queries',str(QUERY_COUNT),'--expected-gallery',str(GALLERY_COUNT)]
        outcomes=[]
        for mode,lam,forced in [('ordinary',.5,False),('cb',.5,False),('ordinary',0.,False),('cb',0.,False),('ordinary',.5,True)]:
            method='uniform-mass-'+mode;suffix='-forced-mp' if forced else '';out=root/f'{method}-{lam}{suffix}'
            binding=dict(schema='uniform-frozen-input-binding-v1',scope='synthetic-only',description='artificial software fixture; no real ReID results',
                model='resnet50',source_support_mode=mode,source_lambda=lam,rho=.5,selection_sha256=sha(selections[lam]),
                feature_sha256=sha(mat),support_sha256=sha(support_paths[mode]),
                cache_files={name:sha(cache/name)for name in ['manifest.json','features.npy','affinity.npy','graph.npz']})
            binding_path=root/f'input-binding-{mode}-{lam}{suffix}.json';dump(binding_path,binding)
            command=[sys.executable,'-B',source/('synthetic_force_mp.py' if forced else 'run_uniform_candidate.py'),*common,'--output-dir',out,'--method',method,'--support-mode',mode,'--support-file',support_paths[mode],'--source-lambda',str(lam),'--selection-file',selections[lam],'--selection-sha256',sha(selections[lam]),'--model','resnet50','--input-binding-file',binding_path,'--input-binding-sha256',sha(binding_path)]
            stem=f'{mode}-{lam}{suffix}'
            stage(stem+'-run',command)
            checkpoint=out/method/'0000.json';checkpoint_sha=sha(checkpoint)
            before=np.load(out/f'{method}.rankings.npy').copy();signature=json.loads((out/'run-manifest.json').read_text())['signature']
            if lam==.5 and not forced:
                stage(stem+'-resume',command)
                require(sha(checkpoint)==checkpoint_sha and np.array_equal(before,np.load(out/f'{method}.rankings.npy')),'Complete resume changed checkpoint/rankings')
                # Recreate a missing query checkpoint while preserving already completed ones.
                (out/method/'0001.json').unlink();stage(stem+'-resume-missing-query',command)
                require(sha(checkpoint)==checkpoint_sha and np.array_equal(before,np.load(out/f'{method}.rankings.npy')),'Interrupted resume changed intact checkpoint/rankings')
                require(json.loads((out/'run-manifest.json').read_text())['signature']==signature,'Resume changed signature')
                checks[stem+'-resume-signature-and-rankings']=True
            stage(stem+'-standard',[sys.executable,'-B',source/'verify_experiment.py','--mat-path',mat,'--output-dir',out,'--method',method,'--protocol',protocol,'--support-file',support_paths[mode],'--expected-queries',str(QUERY_COUNT),'--expected-gallery',str(GALLERY_COUNT),'--depth','70','--progress-every','0'])
            stage(stem+'-numerical',[sys.executable,'-B',source/'verify_numerics.py','--output-dir',out,'--method',method])
            verifier=[sys.executable,'-B',source/'verify_uniform_control.py','--output-dir',out,'--method',method]
            stage(stem+'-uniform',verifier)
            report=json.loads((out/'verified-uniform-control.json').read_text())
            require(report['independently_checked_bridge_events']>0,'Bridge verification unexercised')
            if lam==.5 and not forced:
                original=(out/'control-contract.json').read_bytes()
                changed=json.loads(original);changed['statistics']['qbar']+=.01;dump(out/'control-contract.json',changed)
                stage(stem+'-contract-tamper-verifier',verifier,False)
                stage(stem+'-contract-tamper-resume',command,False)
                (out/'control-contract.json').write_bytes(original)
                require(sha(checkpoint)==checkpoint_sha,'Rejected contract altered completed checkpoint')
                stage(stem+'-restored-uniform',verifier)
                bad=list(command);bad[bad.index('--selection-sha256')+1]='1'*64
                stage(stem+'-selection-sha-rejected',bad,False)
                checks[stem+'-tamper-rejection']=True
                wrong=dict(binding,model='osnet');wrong_path=root/(stem+'-wrong-model-binding.json');dump(wrong_path,wrong)
                bad=list(command);bad[bad.index('--input-binding-file')+1]=wrong_path;bad[bad.index('--input-binding-sha256')+1]=sha(wrong_path)
                stage(stem+'-wrong-model-binding-rejected',bad,False)
                checks[stem+'-model-input-binding-rejection']=True
            if forced:
                require(report['mp_systems_rebound_to_controlled_graph']==1,'Forced fallback did not produce one graph-bound MP event')
                checks['forced-MP-exact-certificate-and-controlled-equation']=True
            outcomes.append(dict(mode=mode,source_lambda=lam,forced_test_fallback=forced,output=str(out),control_report_sha256=sha(out/'verified-uniform-control.json'),qbar=report['qbar'],effective_lambda=report['effective_lambda'],bridge_events=report['independently_checked_bridge_events'],mp_events=report['mp_systems_rebound_to_controlled_graph']))
        baseline=root/'csw-baseline';method='synthetic-csw-lam0'
        stage('csw-lam0-run',[sys.executable,'-B',source/'run_candidate.py',*common,'--output-dir',baseline,'--method',method,'--lam','0'])
        stage('csw-lam0-standard',[sys.executable,'-B',source/'verify_experiment.py','--mat-path',mat,'--output-dir',baseline,'--method',method,'--protocol',protocol,'--expected-queries',str(QUERY_COUNT),'--expected-gallery',str(GALLERY_COUNT),'--depth','70','--progress-every','0'])
        stage('csw-lam0-numerical',[sys.executable,'-B',source/'verify_numerics.py','--output-dir',baseline,'--method',method])
        reference=np.load(baseline/f'{method}.rankings.npy')
        for mode in ['ordinary','cb']:
            control=root/f'uniform-mass-{mode}-0.0';control_method='uniform-mass-'+mode
            require(np.array_equal(reference,np.load(control/f'{control_method}.rankings.npy')),'lambda0 full rankings differ from CSW')
            for q in range(QUERY_COUNT):require(np.array_equal(np.load(baseline/method/f'{q:04d}.prefix.npy'),np.load(control/control_method/f'{q:04d}.prefix.npy')),'lambda0 diffusion prefixes differ')
            checks[mode+'-lambda0-bitwise-all-prefixes-and-rankings']=True
        for path,digest in fingerprint.items():require(sha(path)==digest,'Source changed during integration')
        checks['all-queries-full-gallery']=True;checks['source-files-stable']=True
        summary.update(status='passed',outcomes=outcomes,components=cached['component_sizes'],not_exercised=['No natural fallback was needed in the successful fixture. One explicitly injected synthetic failure exercised MP certificates, score ordering and independent controlled-equation binding.'])
    except Exception as exc:
        summary.update(status='failed',error=repr(exc));raise
    finally:dump(root/'integration-summary.json',summary)
    print(json.dumps(dict(status=summary['status'],summary=str(root/'integration-summary.json'),checks=checks)),flush=True)

if __name__=='__main__':main()
