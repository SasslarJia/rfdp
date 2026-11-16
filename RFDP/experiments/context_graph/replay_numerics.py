"""Paired numerical replay of two predesignated retained traces; no metrics.

The saved L/U and bridges are fixed. Every returned ordered first five must
match the saved trace; the 500-node prefix and full gallery rank must match as
well. This validates a numerical solver, not a new method or parameter point.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy import sparse
import scipy

from experiments.continuous_fitting import certified
from experiments.continuous_fitting.original_solver import WarmOriginalSolver
from experiments.full_reid.common import complete_gallery


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def stages(graph, trace, query):
    boundary = np.asarray(graph[[query]].sum(axis=0)).ravel()
    labeled = [query]
    for row in trace:
        assert row['labeled'] == labeled
        e = boundary.copy()
        for outside, inside, weight in row['bridges']:
            assert inside in labeled
            e[outside] += weight
        u = np.asarray(row['unlabeled'])
        yield row, u, graph[u][:, u], e[u]
        for node in row['selected']:
            begin, end = graph.indptr[node:node+2]
            boundary[graph.indices[begin:end]] += graph.data[begin:end]
        labeled += row['selected']


def run(args):
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    source = Path(cfg['output'])
    assert (source/'STOPPED_BY_DECISION.json').exists()
    # Two earliest traces selected by the existing uniform retention schedule.
    queries = cfg['retained_queries'][:2]
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    graph_path = Path(cfg['cache'])/'graph.npz'
    assert sha(graph_path) == cfg['graph_sha256']
    graph = sparse.load_npz(graph_path).astype(np.float64)
    features = np.load(Path(cfg['cache'])/'features.npy', mmap_mode='r')
    if args.solver == 'stable':
        from experiments.continuous_fitting.stable_original_solver import StableOriginalSolver
        constructor = StableOriginalSolver
    else:
        constructor = WarmOriginalSolver
    report = dict(scope='saved-equation numerical replay only; no retrieval metrics', solver=args.solver,
                  queries=queries, config_sha256=sha(config_path), graph_sha256=sha(graph_path),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  longdouble_mantissa_bits=np.finfo(np.longdouble).nmant+1,
                  blas_threads=1, worker_processes=1, original_output=str(source),
                  replay_script_sha256=sha(__file__),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  rows=[], timing_scope='solver includes certification; preparation separate; saving packs/compresses traces and writes arrays; readback certifies saved points on rebuilt coefficients')
    for query in queries:
        metadata = json.loads((source/'queries'/f'{query:04d}.json').read_text())
        trace_path = source/'queries'/f'{query:04d}.trace.jsonl.gz'
        assert sha(trace_path) == metadata['trace_sha256'] and 'readback_seconds' in metadata
        with gzip.open(trace_path,'rt') as f:
            trace = [json.loads(line) for line in f]
        events = []
        solver = constructor(events=events)
        output_trace = destination/f'{query:04d}.trace.jsonl.gz'
        solve_seconds = saving_seconds = prepare_seconds = 0.
        prefix = [query]
        start = time.monotonic()
        stream = gzip.open(output_trace,'wt',encoding='utf-8',compresslevel=6)
        saving_seconds += time.monotonic()-start
        iterator = iter(stages(graph,trace,query))
        while True:
            start = time.monotonic()
            try:
                row, u, w, e = next(iterator)
            except StopIteration:
                break
            prepare_seconds += time.monotonic()-start
            assert row['returned_scores'] is not None
            start = time.monotonic()
            values, residual = solver(w,e,cfg['alpha'],cfg['beta'],diagnostics={},indices=u,
                                      targets=tuple(row['labeled']),bridges=tuple(row['bridges']))
            solve_seconds += time.monotonic()-start
            event = events[-1]
            assert event['returned_prefix_local'] == row['event']['returned_prefix_local']
            selected = u[event['returned_prefix_local']][:cfg['step']][:cfg['depth']-len(prefix)].tolist()
            assert selected == row['selected']
            prefix += selected
            start = time.monotonic()
            saved = dict(row,returned_scores=[certified.packed(v) for v in values],residual=residual,event=event)
            stream.write(json.dumps(saved,separators=(',',':'),allow_nan=False)+'\n')
            stream.flush()
            saving_seconds += time.monotonic()-start
        start = time.monotonic()
        stream.close()
        saving_seconds += time.monotonic()-start
        prefix = np.asarray(prefix,dtype=np.int32)
        np.testing.assert_array_equal(prefix,np.load(source/'queries'/f'{query:04d}.prefix.npy'))
        start = time.monotonic()
        rank = complete_gallery(prefix,features@features[query],cfg['queries'])
        prepare_seconds += time.monotonic()-start
        np.testing.assert_array_equal(rank,np.load(source/'queries'/f'{query:04d}.rankings.npy'))
        start = time.monotonic()
        np.save(destination/f'{query:04d}.prefix.npy',prefix)
        np.save(destination/f'{query:04d}.rankings.npy',rank)
        saving_seconds += time.monotonic()-start
        start = time.monotonic()
        with gzip.open(output_trace,'rt') as f:
            saved_trace = [json.loads(line) for line in f]
        for row, u, w, e in stages(graph,saved_trace,query):
            cert = certified.certify_saved(w,e,certified.unpack(row['returned_scores']),cfg['alpha'],cfg['beta'],'original')
            assert cert['prefix'] == row['event']['returned_prefix_local']
            assert cert['relative'] <= certified.TOL
        readback_seconds = time.monotonic()-start
        item = dict(query=query,states=len(events),prepare_seconds=prepare_seconds,solve_seconds=solve_seconds,
                    saving_seconds=saving_seconds,readback_seconds=readback_seconds,
                    cg_iterations=sum(x.get('cg_iterations',0) for x in events),
                    direct_calls=sum(x.get('direct_calls',0) for x in events),
                    mp_stages=sum(x['dps']>0 for x in events),
                    wide_stages=sum(x['certificate_backend']=='outward-longdouble' for x in events),
                    source_trace_sha256=sha(trace_path),output_trace_sha256=sha(output_trace),
                    ordered_top5_all_identical=True,prefix500_identical=True,full_gallery_ranking_identical=True,
                    gallery_size=len(rank),saved_points_independently_certified=True)
        report['rows'].append(item)
        (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    report['status'] = 'numerical_replay_passed'
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--solver',choices=('warm','stable'),required=True)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
