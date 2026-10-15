#!/usr/bin/env python3
"""Frozen-feature k-reciprocal comparison using unchanged author distance API."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from scipy.io import loadmat

UPSTREAM_COMMIT = '278f2c704e7f033b767f4158dff4febc03a0b78a'
UPSTREAM_URL = ('https://github.com/zhunzhong07/person-re-ranking/blob/'
                + UPSTREAM_COMMIT + '/python-version/re_ranking_ranklist.py')


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def normalize_float32(features):
    # Match the frozen-feature baseline's float64 normalization then float32 dot.
    features = np.asarray(features, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return (features / norms).astype(np.float32)


def euclidean_matrix(features, batch_size=128):
    """Unsquared Euclidean distances; upstream squares them once internally."""
    count = features.shape[0]
    distances = np.empty((count, count), dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        similarity = features[start:stop].dot(features.T)
        block = np.maximum(2.0 - 2.0 * similarity, 0.0)
        np.sqrt(block, out=block)
        distances[start:stop] = block
        print('Euclidean rows', stop, '/', count, flush=True)
    np.fill_diagonal(distances, 0.0)
    return distances


def load_author_implementation(path):
    specification = importlib.util.spec_from_file_location('official_k_reciprocal', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.re_ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--mat-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--k1', type=int, default=20)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda-value', type=float, default=0.3)
    parser.add_argument('--eval-max-rank', type=int, default=50)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / 'RFDP'))
    from EVAL.ReIDEval import evaluate_reid_rankings

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(__file__).with_name('re_ranking_ranklist.py')
    expected = json.loads(Path(__file__).with_name('upstream_sources.json').read_text())['kr']['files']['re_ranking_ranklist.py']['sha256']
    if sha256(source_file) != expected:
        raise RuntimeError('Official k-reciprocal source hash mismatch')
    re_ranking = load_author_implementation(source_file)
    started = time.perf_counter()
    data = loadmat(args.mat_path)
    query_count = data['query_f'].shape[0]
    gallery_count = data['gallery_f'].shape[0]
    features = normalize_float32(np.vstack([data['query_f'], data['gallery_f']]))
    if args.batch_size <= 0 or not (1 <= args.k1 < len(features)) or not (1 <= args.k2 <= len(features)):
        raise ValueError('Invalid batch_size or neighborhood parameters for feature count.')
    if not 0.0 <= args.lambda_value <= 1.0:
        raise ValueError('lambda-value must be in [0, 1].')
    distance_started = time.perf_counter()
    distances = euclidean_matrix(features, args.batch_size)
    distance_seconds = time.perf_counter() - distance_started
    rerank_started = time.perf_counter()
    final_distances = re_ranking(
        distances[:query_count, query_count:], distances[:query_count, :query_count],
        distances[query_count:, query_count:], k1=args.k1, k2=args.k2,
        lambda_value=args.lambda_value)
    rerank_seconds = time.perf_counter() - rerank_started
    del distances, features
    if not np.isfinite(final_distances).all():
        raise RuntimeError('Official reranker returned nonfinite distances; do not evaluate.')
    ranking_started = time.perf_counter()
    rankings = np.argsort(final_distances, axis=1, kind='stable').astype(np.int32)
    rankings += query_count  # Same global-ID convention as the diffusion suite.
    del final_distances
    ranking_seconds = time.perf_counter() - ranking_started
    evaluate_started = time.perf_counter()
    metrics = evaluate_reid_rankings(
        rankings, list(range(query_count)), data['query_label'], data['query_cam'],
        data['gallery_label'], data['gallery_cam'], query_count,
        max_rank=args.eval_max_rank, return_per_query=True)
    evaluate_seconds = time.perf_counter() - evaluate_started
    per_query = metrics.pop('per_query')
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo_root,
                                           text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        git_head = None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    max_rss_bytes = int(max_rss if sys.platform == 'darwin' else max_rss * 1024)
    output = {
        'method': 'k-reciprocal', 'query_num': query_count, 'gallery_num': gallery_count,
        'metrics': metrics, 'k1': args.k1, 'k2': args.k2, 'lambda_value': args.lambda_value,
        'distance_input': 'unsquared Euclidean on L2-normalized float32 features',
        'input_transform': 'upstream squares once, column-max normalizes, then transposes',
        'graph_population': 'all query + all gallery; pid=-1 retained for graph, removed in evaluation',
        'feature_sha256': sha256(args.mat_path), 'code_git_head': git_head,
        'upstream_commit': UPSTREAM_COMMIT, 'upstream_url': UPSTREAM_URL,
        'upstream_source_sha256': sha256(source_file), 'runner_sha256': sha256(__file__),
        'timings_seconds': {'distance': distance_seconds, 'rerank': rerank_seconds,
                            'ranking': ranking_seconds, 'evaluation': evaluate_seconds,
                            'total_before_save': time.perf_counter() - started},
        'process_max_rss_bytes': max_rss_bytes,
        'numpy_version': np.__version__,
    }
    (args.output_dir / 'k-reciprocal.json').write_text(json.dumps(output, indent=2))
    (args.output_dir / 'k-reciprocal.per-query.json').write_text(
        json.dumps({'method': 'k-reciprocal', 'queries': per_query}, indent=2))
    np.save(args.output_dir / 'k-reciprocal.rankings.npy', rankings, allow_pickle=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
