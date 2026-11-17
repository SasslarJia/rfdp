"""Exact first evaluated gallery item, using the production scorer and reader.

The external evaluation callback stops after a committed batch contains the
first evaluation-valid gallery item. Identity labels never enter the engine.
This is a full-query-population Rank-1 result, not a full mAP/mINP experiment.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import copy
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from scipy import sparse

from experiments.continuous_fitting import full, certified
from experiments.continuous_fitting.verify import verify_query, read_rows
from experiments.full_reid.common import complete_gallery
from experiments.reid_adaptation import resources
from experiments.reid_adaptation.io import lock, load_labels
from experiments.reid_adaptation.protocol import digest
from . import protocol, runtime


class FirstGalleryCommitted(Exception):
    pass


def first_valid(nodes, query, qn, labels):
    for node in nodes:
        if node < qn: continue
        index = int(node)-qn
        pid, cam = labels['gallery_label'][index], labels['gallery_cam'][index]
        if pid == -1 or (pid == labels['query_label'][query] and cam == labels['query_cam'][query]):
            continue
        return int(node)
    return None


def initialize(config_path, destination):
    global CFG, DEST, ENGINE, GRAPH, AFFINITY, FEATURES, CERTIFIER, SUMS, LABELS, ERROR
    ERROR = None
    try:
        full.configure(config_path); full.execution_revision()
        CFG = full.CFG; DEST = Path(destination)
        resource.setrlimit(resource.RLIMIT_AS, (full.WORKER_AS, full.WORKER_AS))
        cache = Path(CFG['cache'])
        graph = sparse.load_npz(cache/'graph.npz')
        AFFINITY = np.load(cache/'affinity.npy', mmap_mode='r')
        FEATURES = np.load(cache/'features.npy', mmap_mode='r')
        ENGINE, GRAPH, _, SUMS, CERTIFIER = runtime.initialize(CFG, graph, AFFINITY)
        LABELS = load_labels(CFG['labels_file'], CFG['queries'], CFG['gallery'])
    except BaseException:
        ERROR = traceback.format_exc()


def compute(query):
    try:
        if ERROR: raise RuntimeError(ERROR)
        certified.EVENTS.clear(); seen=0; prefix=[query]; saving=0.
        start=time.monotonic(); path=DEST/'queries'/f'{query:04d}.trace.jsonl.gz'
        save_start=time.monotonic()
        with gzip.open(path, 'wt', compresslevel=6) as stream:
            saving += time.monotonic()-save_start
            def capture(row):
                nonlocal seen, saving
                assert row['labeled'] == prefix
                event = None
                if row['scores'] is not None:
                    assert len(certified.EVENTS) == seen+1
                    event=copy.deepcopy(certified.EVENTS[seen]); seen+=1
                    expected=np.asarray(row['unlabeled'])[event['returned_prefix_local']]
                    assert row['selected'] == expected[:row['promotion_step']][:CFG['depth']-len(prefix)].tolist()
                else:
                    assert len(certified.EVENTS) == seen
                before=time.monotonic()
                saved={k:v for k,v in row.items() if k!='scores'}
                saved.update(event=event,returned_scores=None if row['scores'] is None else [certified.packed(v) for v in row['scores']])
                stream.write(json.dumps(saved,separators=(',',':'),allow_nan=False)+'\n')
                saving+=time.monotonic()-before
                prefix.extend(row['selected'])
                if first_valid(prefix,query,CFG['queries'],LABELS) is not None:
                    raise FirstGalleryCommitted()
            try:
                ENGINE.rank([query],nq=CFG['depth'],numerical_tail=False,trace=capture)
            except FirstGalleryCommitted:
                pass
            save_start=time.monotonic()
        saving += time.monotonic()-save_start
        computation=time.monotonic()-start-saving
        before=time.monotonic()
        checked=verify_query(GRAPH,AFFINITY,None,query,prefix,None,read_rows(path),
              'original',CFG['queries'],len(prefix),full_sums=SUMS,alpha=CFG['alpha'],beta=CFG['beta'],
              stage_certifier=CERTIFIER,promotion_schedule='early5',prefix_only=True)
        node=first_valid(prefix,query,CFG['queries'],LABELS); from_tail=node is None
        if from_tail:
            assert len(prefix)==CFG['depth']
            node=first_valid(complete_gallery(np.asarray(prefix),FEATURES@FEATURES[query],CFG['queries']),
                             query,CFG['queries'],LABELS)
        if node is None: raise ValueError('No evaluation-valid gallery item')
        result=dict(query=query,first_gallery=node,
                    rank1=int(LABELS['gallery_label'][node-CFG['queries']]==LABELS['query_label'][query]),
                    prefix=prefix,from_original_tail=from_tail,
                    compute_seconds=computation,saving_seconds=saving,
                    readback_seconds=time.monotonic()-before,
                    max_recomputed_residual=checked['max_recomputed_residual'],
                    certified_batches=checked['iterations'],solver_calls=checked['solver_calls'],
                    trace_sha256=full.sha(path),signature=digest(CFG))
        full.dump(DEST/'queries'/f'{query:04d}.json',result)
        return dict(query=query)
    except BaseException:
        return dict(query=query,error=traceback.format_exc())


def run(config_path, destination):
    full.configure(config_path); cfg=full.CFG
    protocol.validate(cfg,require_bound=True); full.sources()
    from experiments.reid_adaptation.full_run import check_bound_inputs
    check_bound_inputs(cfg)
    dest=resources.native_path(destination)
    if dest.exists(): raise ValueError('Do not overwrite a previous Rank-1 screen')
    if dest==Path(cfg['output']) or dest in Path(cfg['output']).parents or Path(cfg['output']) in dest.parents:
        raise ValueError('Rank-1 evidence must be separate from the full experiment')
    with lock(Path(cfg['workspace'])/'.execution.lock'):
        resources.snapshot(dict(cfg,output=str(dest)),reserve=True)
        dest.mkdir(parents=True);(dest/'queries').mkdir()
        full.dump(dest/'run-manifest.json',dict(scope='full_population_exact_rank1_only',config=cfg,
                  signature=digest(cfg),time=full.now(),no_map_or_minp_claim=True))
        start=time.monotonic()
        try:
            with mp.get_context('spawn').Pool(cfg['workers'],initializer=initialize,
                      initargs=(str(config_path),str(dest))) as pool:
                result=pool.imap_unordered(compute,range(cfg['queries']),chunksize=1)
                done=0
                while done<cfg['queries']:
                    try:row=result.next(timeout=20)
                    except mp.TimeoutError:
                        resources.snapshot(dict(cfg,output=str(dest)));continue
                    if 'error' in row:raise RuntimeError(row['error'])
                    done+=1
                    if done%64==0 or done==cfg['queries']:
                        state=dict(stage='rank1_only',completed=done,total=cfg['queries'],elapsed_seconds=time.monotonic()-start)
                        full.dump(dest/'progress.json',state);print(json.dumps(state),flush=True)
                        resources.snapshot(dict(cfg,output=str(dest)))
            records=[json.loads((dest/'queries'/f'{q:04d}.json').read_text()) for q in range(cfg['queries'])]
            for q,row in enumerate(records):
                assert row['query']==q and row['signature']==digest(cfg)
                assert full.sha(dest/'queries'/f'{q:04d}.trace.jsonl.gz')==row['trace_sha256']
                assert row['max_recomputed_residual']<=1e-8
            full.sources()
            full.dump(dest/'per-query.json',records)
            accepted=dict(status='accepted_full_population_rank1_only',signature=digest(cfg),
                  queries=cfg['queries'],gallery=cfg['gallery'],time=full.now(),
                  rank1=float(np.mean([r['rank1'] for r in records])),
                  correct=int(sum(r['rank1'] for r in records)),
                  elapsed_seconds=time.monotonic()-start,
                  timing={k:sum(r[k] for r in records) for k in ('compute_seconds','saving_seconds','readback_seconds')},
                  max_recomputed_residual=max(r['max_recomputed_residual'] for r in records),
                  per_query_sha256=full.sha(dest/'per-query.json'),source_commit=cfg['commit'],
                  no_map_or_minp_claim=True,all_saved_score_prefixes_verified=True)
            full.dump(dest/'RANK1_ACCEPTANCE.json',accepted)
            return accepted
        except BaseException:
            full.dump(dest/'FAILED.json',dict(time=full.now(),error=traceback.format_exc()))
            raise


def check(config, destination):
    dest=Path(destination);a=json.loads((dest/'RANK1_ACCEPTANCE.json').read_text())
    assert a['status']=='accepted_full_population_rank1_only' and a['signature']==digest(config)
    assert (a['queries'],a['gallery'])==(config['queries'],config['gallery'])
    assert a['max_recomputed_residual']<=1e-8 and a['all_saved_score_prefixes_verified']
    assert full.sha(dest/'per-query.json')==a['per_query_sha256']
    rows=json.loads((dest/'per-query.json').read_text())
    assert len(rows)==config['queries'] and [r['query'] for r in rows]==list(range(config['queries']))
    return a,rows
