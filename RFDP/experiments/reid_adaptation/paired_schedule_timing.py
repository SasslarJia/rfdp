"""Measure fixed-batch versus early sequential feedback on paired full queries."""
import os

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.continuous_fitting.full import REPO, sha
from experiments.continuous_fitting.original_solver import WarmOriginalSolver
from experiments.full_reid.common import complete_gallery
from experiments.reid_adaptation.io import lock
from experiments.reid_adaptation.schedule import ScheduledIHyRDP


def clock_stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<i4').tobytes()).hexdigest()


def time_input(group, plan_index, cache, stream):
    for filename, expected in group['sha256'].items():
        if sha(cache / filename) != expected:
            raise ValueError(f'Frozen input mismatch: {cache / filename}')
    graph = sparse.load_npz(cache / 'graph.npz').astype(np.float64)
    affinity = np.load(cache / 'affinity.npy', mmap_mode='r')
    features = np.load(cache / 'features.npy', mmap_mode='r')
    n = group['queries'] + group['gallery']
    assert graph.shape == affinity.shape == (n, n) and len(features) == n
    events = {method: [] for method in ('original', 'early5')}
    engines = {
        method: ScheduledIHyRDP(
            graph, affinity, alpha=5., beta=25., step=5, promotion_schedule=schedule,
            stage_solver=WarmOriginalSolver(events=events[method]))
        for method, schedule in [('original', 'fixed5'), ('early5', 'early5')]
    }
    for pair_index, pair in enumerate(group['pairs']):
        query = pair['query']
        assert 0 <= query < group['queries'] and set(pair['order']) == set(engines)
        for method in pair['order']:
            engine = engines[method]
            events[method].clear()
            engine.bridge_cache.clear()
            cpu = time.process_time()
            start = time.perf_counter()
            similarity = features @ features[query]
            prefix, diagnostics = engine.rank([query], nq=500, input_similarity=similarity,
                                               numerical_tail=False)
            rank = complete_gallery(prefix, similarity, group['queries'])
            wall_seconds = time.perf_counter() - start
            cpu_seconds = time.process_time() - cpu
            # Evidence is written after the timed interval. No test label is loaded.
            assert len(prefix) == len(set(prefix)) == 500 and len(rank) == group['gallery']
            assert np.array_equal(np.sort(rank), np.arange(group['queries'], n))
            assert diagnostics['status'] == 'complete_diffusion'
            assert diagnostics['numerical_tail_count'] == 0
            assert diagnostics['max_relative_residual'] <= 1e-8
            assert len(events[method]) == diagnostics['solver_calls']
            row = dict(input=group['input'], query=query, method=method, order=pair['order'],
                       block=plan_index, pair=pair_index, wall_seconds=wall_seconds,
                       cpu_seconds=cpu_seconds, stages=diagnostics['iterations'],
                       solves=diagnostics['solver_calls'], max_residual=diagnostics['max_relative_residual'],
                       prefix_sha256=array_sha(prefix), gallery_sha256=array_sha(rank))
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        print(group['input'], pair_index+1, 'complete pairs', flush=True)


def run(plan_path, workspace, destination):
    plan = json.loads(plan_path.read_text())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if source != plan['scoring_commit']:
        raise ValueError('Execution checkout differs from the frozen plan')
    with lock(workspace / '.execution.lock'), threadpool_limits(limits=1):
        destination.mkdir(parents=True, exist_ok=False)
        affinity = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {affinity[0]})
        pools = threadpool_info()
        if not pools or any(p['num_threads'] != 1 for p in pools):
            raise RuntimeError('Single-thread math backend could not be confirmed')
        meta = dict(start_utc=clock_stamp(), plan_sha256=sha(plan_path),
                    scorer=source, driver_sha256=sha(__file__), worker_count=1,
                    cpu_affinity=sorted(os.sched_getaffinity(0)), available_cpus=affinity,
                    math_backends=pools, uname=platform.uname()._asdict(),
                    platform=platform.platform(), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    cpu_model=next(line.split(':', 1)[1].strip() for line in
                                   Path('/proc/cpuinfo').read_text().splitlines()
                                   if line.startswith('model name')))
        (destination / 'environment.json').write_text(json.dumps(meta, indent=2)+'\n')
        output = destination / 'paired-timing.jsonl'
        with output.open('x') as stream:
            for i, group in enumerate(plan['groups']):
                time_input(group, i, workspace / 'inputs' / group['input'], stream)
        (destination / 'complete.json').write_text(json.dumps(
            dict(end_utc=clock_stamp(), data_sha256=sha(output), plan_sha256=sha(plan_path),
                 measurements=sum(1 for line in output.read_text().splitlines() if line),
                 environment_sha256=sha(destination/'environment.json')), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'workspace', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.workspace, args.output)
