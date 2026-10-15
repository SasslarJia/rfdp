"""Full-query I-HyRDP/CSW experiment on a frozen Market1501-style feature MAT.

The graph covers every query/gallery image. Depth counts dynamic seeds (including
query), not a fixed candidate pool. Remaining gallery items retain cosine order.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '1'

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
from scipy import sparse

from common import REPO, complete_gallery, dump, load_features, save, sha, validate_ranking
sys.path.insert(0, str(REPO / 'RFDP'))
from MODEL.Paper_RFDP import PaperPrecision
from MODEL.Paper_RFDP.PaperCamera import CameraIHyRDP
from EVAL.ReIDEval import evaluate_reid_rankings

HERE = Path(__file__).resolve().parent
METHODS = ['original', 'ihyrdp', 'csw', 'k-reciprocal', 'ecn-rankdist']


def metrics(rankings):
    return evaluate_reid_rankings(rankings, range(QN),
                                  *[D[k] for k in ['query_label', 'query_cam', 'gallery_label', 'gallery_cam']],
                                  QN, max_rank=min(500, GN), return_per_query=True)


def work(query):
    try:
        out = OUT / MODE
        record = out / f'{query:04d}.json'
        ranking_path = out / f'{query:04d}.rankings.npy'
        prefix_path = out / f'{query:04d}.prefix.npy'
        if record.exists():
            previous = json.loads(record.read_text())
            if previous['signature'] != SIG:
                raise ValueError('Checkpoint configuration mismatch')
            if (ranking_path.exists() and prefix_path.exists()
                    and sha(ranking_path) == previous['rankings_sha256']
                    and sha(prefix_path) == previous['prefix_sha256']):
                validate_ranking(np.load(ranking_path), QN, GN)
                return dict(query=query, resumed=True)
        start = time.monotonic()
        PaperPrecision.EVENTS.clear()
        similarity = X @ X[query]
        trace = []
        prefix, diagnostics = ENGINE.rank(
            [query], nq=DEPTH, input_similarity=similarity, numerical_tail=False,
            trace=lambda r: trace.append(dict(iteration=r['iteration'], selected=r['selected'],
                                             residual=r['residual'], size=len(r['unlabeled']))))
        if (len(prefix) != DEPTH or prefix[0] != query or len(set(prefix)) != DEPTH
                or diagnostics['status'] != 'complete_diffusion'
                or diagnostics['numerical_tail_count'] != 0
                or diagnostics['max_relative_residual'] > 1e-8):
            raise RuntimeError('Incomplete or numerically invalid diffusion')
        rank = complete_gallery(prefix, similarity, QN)
        save(ranking_path, rank)
        save(prefix_path, prefix.astype(np.int32))
        dump(record, dict(signature=SIG, query=query, mode=MODE,
                          rankings_sha256=sha(ranking_path), prefix_sha256=sha(prefix_path),
                          diagnostics=diagnostics, trace=trace,
                          refinement_events=list(PaperPrecision.EVENTS),
                          seconds=time.monotonic() - start))
        return dict(query=query, seconds=time.monotonic() - start)
    except Exception:
        return dict(query=query, error=traceback.format_exc())


def command(arguments, log):
    with log.open('w') as stream:
        subprocess.run([sys.executable, *map(str, arguments)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT)


def main():
    global X, D, ENGINE, MODE, OUT, SIG, QN, GN, DEPTH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat-path', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=500)
    parser.add_argument('--rho', type=float, default=.5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    args = parser.parse_args()
    if not 0 < args.rho <= 1 or args.workers < 1:
        parser.error('rho must be in (0,1], workers must be positive')
    if args.workers > 1 and 'fork' not in mp.get_all_start_methods():
        parser.error('Multiple workers require POSIX fork; use --workers 1 otherwise')
    D, QN, GN = load_features(args.mat_path)
    DEPTH = args.depth
    if not 1 <= DEPTH <= QN + GN:
        parser.error('depth must be between 1 and query_count + gallery_count')
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Fail before expensive computation if pinned comparator sources are missing.
    upstream = json.loads((HERE / 'upstream_sources.json').read_text())
    for method, family in [('k-reciprocal', 'kr'), ('ecn-rankdist', 'ecn')]:
        if method in args.methods:
            for record in upstream[family]['files'].values():
                path = HERE / record['destination']
                if not path.exists() or sha(path) != record['sha256']:
                    raise ValueError('Run fetch_controls.py first; missing/changed source: ' + str(path))
    # Hash the whole tracked computation path, including evaluator and graph builder.
    source_paths = set((REPO / 'RFDP/MODEL').rglob('*.py'))
    source_paths.update((REPO / 'RFDP/EVAL').rglob('*.py'))
    source_paths.update(HERE.glob('*.py'))
    source_paths.add(REPO / 'RFDP/Test_RFDP/reid_paper_experiment.py')
    sources = {str(p.relative_to(REPO)): sha(p) for p in sorted(source_paths)}
    config = dict(query_count=QN, gallery_count=GN, depth=DEPTH, rho=args.rho,
                  alpha=5, beta=25, step=5, k=20, gamma=.4, workers=args.workers,
                  methods=args.methods, kr=[20, 6, .3], ecn=[25, 3, 8],
                  feature_sha256=sha(args.mat_path), sources=sources,
                  upstream=upstream, numpy=np.__version__, scipy=scipy.__version__)
    SIG = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = OUT / 'run-manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['signature'] != SIG:
            raise ValueError('Output directory belongs to a different configuration/source/input')
    else:
        dump(manifest_path, dict(signature=SIG, config=config))
    # A previous COMPLETE marker is reissued only after this invocation verifies all results.
    (OUT / 'COMPLETE.json').unlink(missing_ok=True)
    cache = args.cache_dir
    if not (cache / 'manifest.json').exists():
        command([HERE / 'build_feature_cache.py', '--mat', args.mat_path, '--output', cache], OUT / 'cache.log')
    cached = json.loads((cache / 'manifest.json').read_text())
    if cached['feature_sha256'] != config['feature_sha256'] or cached['parameters'] != dict(k=20, gamma=.4, alpha=5, beta=25, step=5):
        raise ValueError('Cache input/protocol mismatch')
    if cached['builder_sha256'] != sha(HERE / 'build_feature_cache.py'):
        raise ValueError('Cache was built by a different builder; rebuild in a new directory')
    for relative, digest in cached['sources'].items():
        if sha(REPO / relative) != digest:
            raise ValueError('Cache computation source changed: ' + relative)
    for name in ['features.npy', 'affinity.npy', 'graph.npz']:
        if sha(cache / name) != cached['files'][name]:
            raise ValueError('Cache checksum mismatch: ' + name)
    X = np.load(cache / 'features.npy', mmap_mode='r')
    full = np.load(cache / 'affinity.npy', mmap_mode='r')
    graph = sparse.load_npz(cache / 'graph.npz')
    if X.shape[0] != QN + GN or full.shape != (QN + GN, QN + GN) or full.dtype != np.float32:
        raise ValueError('Cache shape/dtype mismatch')
    PaperPrecision.install()
    started = time.monotonic()
    if 'original' in args.methods:
        ranks = np.empty((QN, GN), np.int32)
        for query in range(QN):
            ranks[query] = np.lexsort((np.arange(QN, QN + GN), -(X[QN:] @ X[query]))).astype(np.int32) + QN
        save(OUT / 'original.rankings.npy', ranks)
        dump(OUT / 'original.json', metrics(ranks))
        del ranks
    cameras = np.r_[D['query_cam'].ravel(), D['gallery_cam'].ravel()]
    for MODE, eta in [('ihyrdp', 1.), ('csw', args.rho)]:
        if MODE not in args.methods:
            continue
        (OUT / MODE).mkdir(exist_ok=True)
        ENGINE = CameraIHyRDP(graph, full, cameras, eta)
        first = work(0)
        if 'error' in first:
            raise RuntimeError(first['error'])
        pool = mp.get_context('fork').Pool(args.workers) if args.workers > 1 else None
        try:
            pending = pool.imap_unordered(work, range(1, QN), chunksize=1) if pool else map(work, range(1, QN))
            for count, result in enumerate(pending, 2):
                if 'error' in result:
                    raise RuntimeError(result['error'])
                if count % 25 == 0 or count == QN:
                    progress = dict(stage='diffusion', mode=MODE, completed=count, total=QN,
                                    elapsed=time.monotonic() - started)
                    dump(OUT / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        finally:
            if pool:
                pool.terminate()
                pool.join()
        ranks = np.stack([np.load(OUT / MODE / f'{q:04d}.rankings.npy') for q in range(QN)])
        save(OUT / f'{MODE}.rankings.npy', ranks)
        dump(OUT / f'{MODE}.json', metrics(ranks))
        del ranks, ENGINE
    del X, graph, full
    for method, script, extra in [('k-reciprocal', 'run_k_reciprocal.py', []),
                                  ('ecn-rankdist', 'run_classic_comparators.py', ['--method', 'ecn-rankdist'])]:
        if method not in args.methods:
            continue
        marker = OUT / f'{method}.checkpoint.json'
        rankfile = OUT / f'{method}.rankings.npy'
        metricfile = OUT / f'{method}.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            if (previous['signature'] == SIG and rankfile.exists() and metricfile.exists()
                    and previous['rankings_sha256'] == sha(rankfile)
                    and previous['metrics_sha256'] == sha(metricfile)):
                continue
        command([HERE / script, '--repo-root', REPO, '--mat-path', args.mat_path,
                 '--output-dir', OUT, '--eval-max-rank', 500, *extra], OUT / f'{method}.log')
        dump(marker, dict(signature=SIG, rankings_sha256=sha(rankfile), metrics_sha256=sha(metricfile)))
    dump(OUT / 'COMPUTE_COMPLETE.json', dict(signature=SIG, status='computed', seconds=time.monotonic() - started))
    command([HERE / 'verify_full.py', '--mat-path', args.mat_path, '--output-dir', OUT,
             '--methods', *args.methods], OUT / 'verify.log')
    dump(OUT / 'progress.json', dict(stage='verified', total=QN, methods=args.methods))
    print('Verified all queries:', OUT / 'verified-results.json', flush=True)
    lock.close()


if __name__ == '__main__':
    main()
