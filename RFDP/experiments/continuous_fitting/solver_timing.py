"""Pair the unchanged certified original solver with its warm-start implementation.

Run as ``python -m experiments.continuous_fitting.solver_timing --config FILE``.
The JSON config requires cache, cache_sha256 (or sha), query_count,
gallery_count, query_ids and a new output directory. Hash all three files:
graph.npz, affinity.npy and features.npy. Optional archive_run names an extracted
run directory containing queries/NNNN.prefix.npy and rankings.npy (or individual
queries/NNNN.rankings.npy). No population accuracy is evaluated.

Preparation, complete-query computation (including online certificates and trace
construction), artifact serialization, and independent saved-score readback use
separate wall/CPU clocks. Both methods run their own complete dynamic trajectory.
Thread environment variables and threadpoolctl request one thread; an empty
backend inventory is recorded as unverified, not as proof of a native limit.
Historical archive comparison is a separate check: its failure preserves the
accepted local runs and their pair comparison, but still makes the CLI fail.
"""
import os

THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
for _key in THREAD_ENV:
    os.environ[_key] = '1'

import argparse
import gzip
import hashlib
import itertools
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP
from experiments.full_reid.common import complete_gallery
from . import certified
from .verify import read_rows, verify_query

REPO = Path(__file__).resolve().parents[3]
DEPTH = 500
ROUTES = ('original', 'optimized')
INPUTS = {'graph.npz', 'affinity.npy', 'features.npy'}
TRACE_FIELDS = ('iteration', 'labeled', 'direct', 'indirect', 'unlabeled',
                'selected', 'bridges', 'residual')
COMPARE_FIELDS = TRACE_FIELDS[:-1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def clock():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    return dict(wall_seconds=time.perf_counter() - start[0],
                cpu_seconds=time.process_time() - start[1])


def validate_config(cfg):
    cfg = dict(cfg)
    hashes = cfg.get('cache_sha256', cfg.get('sha'))
    if 'sha' in cfg and 'cache_sha256' in cfg and cfg['sha'] != cfg['cache_sha256']:
        raise ValueError('sha and cache_sha256 disagree')
    if not isinstance(hashes, dict) or set(hashes) != INPUTS:
        raise ValueError('Hashes for all three frozen input files are required')
    for value in hashes.values():
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Expected a lowercase SHA-256 digest')
    cfg['cache_sha256'] = hashes
    for name in ('query_count', 'gallery_count'):
        if type(cfg.get(name)) is not int or cfg[name] <= 0:
            raise ValueError('Positive integer ' + name + ' required')
    if cfg['query_count'] + cfg['gallery_count'] < DEPTH:
        raise ValueError('This protocol requires the complete depth-500 trajectory')
    queries = cfg.get('query_ids')
    if (not isinstance(queries, list) or not queries
            or any(type(q) is not int or not 0 <= q < cfg['query_count'] for q in queries)
            or len(set(queries)) != len(queries)):
        raise ValueError('Explicit unique query_ids are required')
    for key in ('cache', 'output'):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError(key + ' path required')
    if cfg.get('parameters', dict(alpha=5, beta=25, step=5, depth=500)) != dict(
            alpha=5, beta=25, step=5, depth=500):
        raise ValueError('Original alpha5/beta25/step5/depth500 are fixed')
    if cfg.get('workers', 1) != 1 or cfg.get('blas_threads', 1) != 1:
        raise ValueError('Only one worker and one BLAS thread are allowed')
    if cfg.get('archive_run') and not Path(cfg['archive_run']).is_dir():
        raise ValueError('archive_run must be an already extracted run directory')
    return cfg


def route_order(query_position):
    """Alternation uses the position in the declared query list, not query ID."""
    return ROUTES if query_position % 2 == 0 else ROUTES[::-1]


def make_stage_solver(route, events):
    if route == 'optimized':
        from .original_solver import WarmOriginalSolver
        solver = WarmOriginalSolver(events=events)
        solver.reset()
        return solver
    if route != 'original':
        raise ValueError('Unknown solver route: ' + str(route))

    def solve(w, e, alpha, beta, *, diagnostics, indices, targets, bridges):
        values, residual, event = certified.solve(w, e, alpha, beta, source='original')
        events.append(event)
        diagnostics.update(cg_calls=int(event.get('cg_iterations', 0) > 0),
                           cg_iterations=event.get('cg_iterations', 0),
                           direct_calls=int(event['candidate_method'] == 'sparse-lu'),
                           uncertified_cg=0)
        return values, residual
    return solve


class TraceCapture:
    """Keep producer-created rows and event references; never serialize in rank."""
    def __init__(self, events, depth):
        self.events, self.depth = events, depth
        self.rows = []
        self.seen = 0

    def __call__(self, row):
        event = None
        if row['scores'] is None:
            if len(self.events) != self.seen:
                raise AssertionError('An event was emitted for a no-solve batch')
            expected = row['unlabeled'][:self.depth - len(row['labeled'])]
        else:
            if len(self.events) != self.seen + 1:
                raise AssertionError('Each solved stage must emit exactly one event')
            event = self.events[self.seen]
            self.seen += 1
            expected = [row['unlabeled'][i] for i in event['returned_prefix_local']]
            expected = expected[:self.depth - len(row['labeled'])]
        self.rows.append((row, event))  # Keep completed evidence before checking order.
        if expected != row['selected']:
            raise AssertionError('Ordered selection differs from the certified prefix')


def save_artifacts(dest, captures, events, prefix=None, rank=None):
    """Packing MP values, JSON/gzip and compressed NumPy are all saving work."""
    with gzip.open(dest / 'trace.jsonl.gz', 'wt', encoding='utf-8', compresslevel=6) as stream:
        for row, event in captures:
            saved = {key: row[key] for key in TRACE_FIELDS}
            saved['returned_scores'] = (None if row['scores'] is None else
                                        [certified.packed(v) for v in row['scores']])
            saved['event'] = event
            stream.write(json.dumps(saved, separators=(',', ':'), allow_nan=False) + '\n')
    dump(dest / 'events.json', events)
    paths = [dest / 'trace.jsonl.gz', dest / 'events.json']
    if prefix is not None and rank is not None:
        np.savez_compressed(dest / 'arrays.npz', prefix=np.asarray(prefix, dtype=np.int32),
                            rankings=np.asarray(rank, dtype=np.int32))
        paths.append(dest / 'arrays.npz')
    return {path.name: sha(path) for path in paths}


def verify_artifacts(dest, graph, affinity, features, query, qn, depth, sums, hashes):
    """Use the existing reader and its own unchanged original certificate."""
    for name, expected in hashes.items():
        if sha(dest / name) != expected:
            raise AssertionError('Saved artifact hash mismatch: ' + name)
    with np.load(dest / 'arrays.npz', allow_pickle=False) as saved:
        prefix, rank = saved['prefix'], saved['rankings']
    return verify_query(graph, affinity, features @ features[query], query,
                        prefix, rank, read_rows(dest / 'trace.jsonl.gz'),
                        'original', qn, depth, full_sums=sums, alpha=5., beta=25.)


def execute_query(route, query, graph, affinity, features, qn, dest, *, depth=DEPTH):
    """The public CLI always uses DEPTH; smaller depth is for synthetic tests."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    events = []
    capture = TraceCapture(events, depth)
    timings = {}
    phase = 'preparation'
    start = clock()
    total = start
    try:
        solver = make_stage_solver(route, events)
        core = FullIHyRDP(graph, affinity, alpha=5., beta=25., step=5, stage_solver=solver)
        timings[phase] = elapsed(start)
        phase, start = 'compute', clock()
        similarity = features @ features[query]
        prefix, diagnostics = core.rank([query], nq=depth, input_similarity=similarity,
                                        numerical_tail=False, trace=capture)
        if (diagnostics['status'] != 'complete_diffusion'
                or diagnostics['numerical_tail_count'] != 0
                or capture.seen != len(events) or len(prefix) != depth):
            raise AssertionError('Incomplete or untracked dynamic ranking')
        rank = complete_gallery(prefix, similarity, qn)
        timings[phase] = elapsed(start)
        phase, start = 'save', clock()
        hashes = save_artifacts(dest, capture.rows, events, prefix, rank)
        dump(dest / 'compute.json', dict(query=query, route=route, diagnostics=diagnostics,
                                        timings=timings, files_sha256=hashes))
        timings[phase] = elapsed(start)
        phase, start = 'verify', clock()
        acceptance = verify_artifacts(dest, graph, affinity, features, query, qn,
                                      depth, core.full_sums, hashes)
        for key in ('solver_calls', 'iterations', 'bridge_events', 'bridge_edges', 'max_unlabeled'):
            if acceptance[key] != diagnostics[key]:
                raise AssertionError('Saved trajectory diagnostics mismatch: ' + key)
        if acceptance['max_reported_residual'] != diagnostics['max_relative_residual']:
            raise AssertionError('Saved reported residual differs from producer')
        timings[phase] = elapsed(start)
        before = clock()
        dump(dest / 'acceptance.json', acceptance)
        acceptance_save = elapsed(before)
        for key in ('wall_seconds', 'cpu_seconds'):
            timings['save'][key] += acceptance_save[key]
        item = dict(query=query, route=route, status='accepted', timings=timings,
                    total=elapsed(total), full_query_accepted=True,
                    diagnostics=diagnostics, acceptance=acceptance, files_sha256=hashes,
                    gallery_count=len(rank), trace_rows=len(capture.rows), solver_events=len(events),
                    certificate_precisions=sorted({e.get('dps', 0) for e in events}),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                    (1 if sys.platform == 'darwin' else 1024),
                    peak_rss_scope='process lifetime high-water mark, not per-method memory')
        # Only the final stopwatch report itself is outside the four phases.
        dump(dest / 'timing.json', item)
        return item
    except Exception:
        timings.setdefault(phase, elapsed(start))
        failure = dict(query=query, route=route, status='failed', phase=phase,
                       error=traceback.format_exc(), timings=timings, total=elapsed(total))
        if phase in ('preparation', 'compute'):
            try:
                failure['partial_files_sha256'] = save_artifacts(dest, capture.rows, events)
            except Exception:
                failure['evidence_save_error'] = traceback.format_exc()
        dump(dest / 'FAILED.json', failure)
        raise


def compare_saved(left, right):
    """Compare order, not membership; scores may differ but each is certified."""
    sentinel = object()
    count = 0
    for count, (a, b) in enumerate(itertools.zip_longest(
            read_rows(Path(left) / 'trace.jsonl.gz'), read_rows(Path(right) / 'trace.jsonl.gz'),
            fillvalue=sentinel), start=1):
        if a is sentinel or b is sentinel:
            raise AssertionError('Trace length differs at stage ' + str(count - 1))
        for key in COMPARE_FIELDS:
            if a[key] != b[key]:
                raise AssertionError('Ordered trajectory mismatch at stage %d: %s' % (count - 1, key))
        ea, eb = a['event'], b['event']
        if (ea is None) != (eb is None):
            raise AssertionError('Solve/no-solve events differ')
        if ea is not None:
            for key in ('source', 'size', 'returned_prefix_local'):
                if ea[key] != eb[key]:
                    raise AssertionError('Stage %d event mismatch: %s' % (count - 1, key))
    with np.load(Path(left) / 'arrays.npz', allow_pickle=False) as a, np.load(
            Path(right) / 'arrays.npz', allow_pickle=False) as b:
        for key in ('prefix', 'rankings'):
            if not np.array_equal(a[key], b[key]):
                raise AssertionError('Complete ordered ' + key + ' differs')
    return dict(ordered_trajectories_identical=True, ordered_first_five_identical=True,
                complete_prefix_identical=True, complete_gallery_identical=True, stages=count)


def compare_archive(dest, archive, query):
    archive = Path(archive)
    pp = archive / 'queries' / f'{query:04d}.prefix.npy'
    rp = archive / 'queries' / f'{query:04d}.rankings.npy'
    if not rp.exists():
        rp = archive / 'rankings.npy'
        archived_rank = np.load(rp, mmap_mode='r', allow_pickle=False)[query]
    else:
        archived_rank = np.load(rp, allow_pickle=False)
    with np.load(Path(dest) / 'arrays.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['prefix'], np.load(pp, allow_pickle=False)):
            raise AssertionError('Baseline prefix differs from archive')
        if not np.array_equal(saved['rankings'], archived_rank):
            raise AssertionError('Baseline complete gallery differs from archive')
    return dict(prefix_identical=True, complete_gallery_identical=True,
                prefix_path=str(pp), rankings_path=str(rp), prefix_sha256=sha(pp))


def run(config_path):
    cfg = validate_config(json.loads(Path(config_path).read_text()))
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='single-worker paired listed-query timing; no population metrics',
                  config=cfg, workers=1, blas_threads_requested=1, runs=[], pairs=[], failures=[],
                  platform=platform.platform(), python=sys.version, numpy=np.__version__,
                  scipy=scipy.__version__, thread_environment={k: os.environ[k] for k in THREAD_ENV},
                  timing_contract=dict(compute='similarity, real rank, online certificate, trace construction, complete gallery',
                                       save='MP packing, JSON/gzip, compressed NumPy, artifact hashing, compute and acceptance records',
                                       verify='disk readback, recomputed similarity, verify_query, diagnostic checks',
                                       reports='final stopwatch/controller report writes excluded; total includes phase transitions'))
    dump(output / 'timing.json', report)
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            matched_pools = [p for p in pools if p.get('user_api') in ('blas', 'openmp')]
            if any(p.get('num_threads') != 1 for p in matched_pools):
                raise RuntimeError('Native thread-pool limit was not enforced')
            report['native_thread_pools'] = pools
            report['native_thread_limits_verified'] = bool(matched_pools)
            report['native_backend_detection'] = ('supported_backends_detected' if matched_pools
                                                  else 'no_supported_backend_detected')
            report['thread_limit_request'] = 'environment variables and threadpoolctl(limits=1)'
            report['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
            report['worktree_changes'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
            if cfg.get('expected_commit') and cfg['expected_commit'] != report['commit']:
                raise ValueError('Checkout differs from expected_commit')
            names = ('solver_timing.py', 'original_solver.py', 'certified.py', 'model.py', 'verify.py')
            files = [Path(__file__).with_name(name) for name in names]
            files += [REPO / 'RFDP/MODEL/Paper_RFDP/PaperFullIterative.py',
                      REPO / 'RFDP/experiments/full_reid/common.py']
            report['source_sha256'] = {str(p.relative_to(REPO)): sha(p) for p in files}
            start = clock()
            cache = Path(cfg['cache'])
            for name, expected in cfg['cache_sha256'].items():
                if sha(cache / name) != expected:
                    raise ValueError('Frozen input SHA mismatch: ' + name)
            graph = sparse.load_npz(cache / 'graph.npz').astype(np.float64).tocsr()
            graph.eliminate_zeros()
            graph.sort_indices()
            affinity = np.load(cache / 'affinity.npy', mmap_mode='r', allow_pickle=False)
            features = np.load(cache / 'features.npy', mmap_mode='r', allow_pickle=False)
            n = cfg['query_count'] + cfg['gallery_count']
            if graph.shape != (n, n) or affinity.shape != (n, n) or features.shape[0] != n:
                raise ValueError('Frozen input dimensions disagree')
            report['common_preparation'] = elapsed(start)
            dump(output / 'timing.json', report)
            for position, query in enumerate(cfg['query_ids']):
                pair = dict(query=query, order=list(route_order(position)), status='incomplete')
                items = {}
                for route in route_order(position):
                    dest = output / f'{query:04d}' / route
                    try:
                        item = execute_query(route, query, graph, affinity, features,
                                             cfg['query_count'], dest)
                        # Local mathematical acceptance is independent of an old
                        # platform's complete-tail ordering. Preserve both verdicts.
                        items[route] = item
                        report['runs'].append(item)
                        if route == 'original' and cfg.get('archive_run'):
                            before = clock()
                            try:
                                item['archive'] = compare_archive(dest, cfg['archive_run'], query)
                                item['archive_comparison_status'] = 'passed'
                            except Exception:
                                failure = dict(query=query, route=route, phase='archive_comparison',
                                               error=traceback.format_exc())
                                item['archive_comparison_status'] = 'failed'
                                item['archive_error'] = failure['error']
                                report['failures'].append(failure)
                                dump(dest / 'ARCHIVE_FAILED.json', failure)
                            item['archive_comparison'] = elapsed(before)
                            dump(dest / 'timing.json', item)
                    except Exception:
                        failure = dict(query=query, route=route, error=traceback.format_exc())
                        report['failures'].append(failure)
                        dump(dest / 'RUN_FAILED.json', failure)
                    dump(output / 'timing.json', report)
                if len(items) == 2:
                    try:
                        before = clock()
                        pair.update(compare_saved(output / f'{query:04d}' / 'original',
                                                  output / f'{query:04d}' / 'optimized'))
                        pair['comparison'] = elapsed(before)
                        pair['ratios_optimized_over_original'] = {
                            phase: items['optimized']['timings'][phase]['wall_seconds'] /
                            items['original']['timings'][phase]['wall_seconds']
                            for phase in ('preparation', 'compute', 'save', 'verify')}
                        pair['status'] = 'accepted'
                    except Exception:
                        pair['status'], pair['error'] = 'failed', traceback.format_exc()
                        report['failures'].append(dict(query=query, phase='ordered_pair_comparison', error=pair['error']))
                report['pairs'].append(pair)
                dump(output / f'{query:04d}' / 'pair.json', pair)
                dump(output / 'timing.json', report)
                print(json.dumps(pair, ensure_ascii=False), flush=True)
            report['status'] = 'completed' if not report['failures'] else 'completed_with_failures'
    except Exception:
        report['status'] = 'failed_preparation'
        report['failures'].append(dict(phase='common_preparation', error=traceback.format_exc()))
        dump(output / 'FAILED.json', report['failures'][-1])
        dump(output / 'timing.json', report)
        raise
    dump(output / 'timing.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    result = run(parser.parse_args().config)
    raise SystemExit(0 if result['status'] == 'completed' else 1)
