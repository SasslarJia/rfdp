#!/usr/bin/env python3
"""End-to-end synthetic integration of cache -> candidate -> independent audit.

This generates 96 artificial samples (12 queries, 84 gallery images), with an
eligible cross-camera positive for every query. It exercises the actual frozen
feature-cache builder, the real production diffusion runner and the independent
verifier for lambda=0 and camera-balanced reliability lambda=.5. It is a software
integration check, not evidence about any real ReID dataset or model quality.

Run only after all source/*.py files are frozen, because run_candidate binds
their content hashes. This script never edits the runner, verifier or model.
Default outputs are a new uniquely named directory inside this experiment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import sparse
from scipy.io import savemat


QUERY_COUNT = 12
GALLERY_COUNT = 84
SEED = 20261015


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_fingerprints(source_dir):
    return {str(path.resolve()): file_sha256(path) for path in sorted(source_dir.glob('*.py'))}


def generate_fixture(path):
    """A noisy circular feature manifold with integer, evaluation-only labels."""
    rng = np.random.default_rng(SEED)
    count = QUERY_COUNT + GALLERY_COUNT
    positions = np.arange(count)
    theta = 2 * np.pi * positions / count
    features = np.column_stack((np.cos(theta), np.sin(theta),
                                .35 * np.cos(2 * theta), .35 * np.sin(2 * theta),
                                .12 * np.cos(3 * theta), .12 * np.sin(3 * theta),
                                .06 * np.cos(5 * theta), .06 * np.sin(5 * theta)))
    features += rng.normal(scale=.008, size=features.shape)
    features = features.astype(np.float32)
    pids = positions // 8 + 1
    cameras = 1 + ((positions % 8 + positions // 8) % 3)
    query_ids = np.arange(0, count, 8)
    gallery_ids = np.setdiff1d(positions, query_ids)
    query_pid, query_cam = pids[query_ids], cameras[query_ids]
    gallery_pid, gallery_cam = pids[gallery_ids].copy(), cameras[gallery_ids]
    # Exercise pid=-1 junk removal, while keeping eligible positives for all queries.
    gallery_pid[-2:] = -1
    counts = [int(np.count_nonzero((gallery_pid == pid) & (gallery_cam != cam)))
              for pid, cam in zip(query_pid, query_cam)]
    if len(query_ids) != QUERY_COUNT or len(gallery_ids) != GALLERY_COUNT or min(counts) <= 0:
        raise AssertionError('Synthetic query/gallery coverage contract failed')
    data = dict(query_f=features[query_ids], gallery_f=features[gallery_ids],
                query_label=query_pid, query_cam=query_cam,
                gallery_label=gallery_pid, gallery_cam=gallery_cam)
    savemat(path, data, do_compression=False, oned_as='row')
    # The MAT description field otherwise embeds wall-clock time. Fix only that
    # free-text field; preserve the format version/endian bytes at offsets 124+.
    with Path(path).open('r+b') as stream:
        description = b'MATLAB 5.0 MAT-file, synthetic integration fixture, seed 20261015; not real ReID data'
        stream.write(description.ljust(116, b' '))
    return data, counts


def run_stage(command, name, root, repo, environment, stages):
    log = root / (name + '.log')
    started = time.monotonic()
    print(json.dumps({'stage': name, 'status': 'running', 'log': str(log)}), flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=repo, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    stage = dict(name=name, command=[str(part) for part in command],
                 exit_code=result.returncode, seconds=time.monotonic() - started,
                 log=str(log), log_sha256=file_sha256(log))
    stages.append(stage)
    if result.returncode != 0:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-35:])
        raise RuntimeError(f'{name} exited {result.returncode}; log: {log}\n{tail}')


def main():
    source_dir = Path(__file__).resolve().parent
    experiment = source_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=source_dir.parents[2])
    parser.add_argument('--output-dir', type=Path, help='New directory; defaults to a unique directory inside the experiment')
    parser.add_argument('--depth', type=int, default=25)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not 2 <= args.depth <= QUERY_COUNT + GALLERY_COUNT or args.workers < 1:
        raise ValueError('Invalid integration depth or worker count')
    builder = repo / 'RFDP/experiments/full_reid/build_feature_cache.py'
    runner = source_dir / 'run_candidate.py'
    verifier = source_dir / 'verify_experiment.py'
    if not all(path.is_file() for path in (builder, runner, verifier)):
        raise ValueError('The experiment repository snapshot, runner and verifier are required')
    if args.output_dir is None:
        root = Path(tempfile.mkdtemp(prefix='synthetic-integration-')).resolve()
    else:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       RFDP_PAPER_DIR=str(repo / 'RFDP/MODEL/Paper_RFDP'))
    source_before = source_fingerprints(source_dir)
    stages = []
    summary = dict(scope='Synthetic software integration only; not real ReID evidence',
                   status='running', seed=SEED, query_count=QUERY_COUNT,
                   gallery_count=GALLERY_COUNT, total_nodes=QUERY_COUNT + GALLERY_COUNT,
                   depth=args.depth, workers=args.workers, output_dir=str(root),
                   repo=str(repo), source_sha256=source_before, stages=stages)
    write_json(root / 'integration-summary.json', summary)
    try:
        mat = root / 'synthetic-features.mat'
        data, positive_counts = generate_fixture(mat)
        protocol = root / 'SYNTHETIC-PROTOCOL.md'
        protocol.write_text(
            '# Synthetic integration protocol\n\n'
            'Artificial data only. No Market1501/Duke image, feature, identity or outcome is used.\n'
            f'Seed {SEED}; {QUERY_COUNT} queries; {GALLERY_COUNT} gallery samples; '
            f'{QUERY_COUNT + GALLERY_COUNT} total nodes. Every query has cross-camera positives.\n'
            'Evaluate every generated query against the entire generated gallery; retain junk filtering.\n'
            'Use the existing full feature-cache builder: k=20, gamma=0.4.\n'
            f'Use rho=0.5, alpha=5, beta=25, step=5, diffusion depth={args.depth}.\n'
            'Run CSW lambda=0 and camera-balanced fixed reliability lambda=0.5.\n'
            'Reliability is derived only from the full affinity and cameras. Identity labels are evaluation-only.\n'
            'Numerical tails are disabled. Independently verify rankings, checkpoints, metrics and provenance.\n'
            'Passing this protocol establishes software integration, not real-world retrieval improvement.\n'
        )
        summary.update(feature_sha256=file_sha256(mat), protocol_sha256=file_sha256(protocol),
                       eligible_positives_per_query=positive_counts)
        cache = root / 'cache'
        run_stage([sys.executable, '-B', str(builder), '--mat', str(mat), '--output', str(cache)],
                  'build_feature_cache', root, repo, environment, stages)
        # Importing the support module uses the snapshot explicitly; it cannot
        # silently resolve a different local production checkout.
        os.environ['RFDP_PAPER_DIR'] = str(repo / 'RFDP/MODEL/Paper_RFDP')
        from reliability_model import build_support, save_support, support_stats
        full = np.load(cache / 'affinity.npy', mmap_mode='r')
        graph = sparse.load_npz(cache / 'graph.npz')
        cameras = np.r_[data['query_cam'], data['gallery_cam']]
        support, metadata = build_support(full, cameras, k=20, mode='cb')
        support_file = root / 'support-cb-k20.npz'
        save_support(support_file, support, metadata)
        stats = support_stats(support, graph, cameras)
        if not 0 < stats['graph_q_mean'] < 1:
            raise AssertionError('Synthetic fixture must exercise a nonidentity reliability gate')
        summary['support_statistics'] = stats
        summary['support_sha256'] = file_sha256(support_file)
        summaries = []
        rankings = []
        for method, lam in (('synthetic_csw_lam0', 0.0), ('synthetic_cb_lam05', 0.5)):
            output = root / method
            command = [sys.executable, '-B', str(runner), '--repo', str(repo),
                       '--mat-path', str(mat), '--cache-dir', str(cache), '--output-dir', str(output),
                       '--protocol', str(protocol), '--method', method, '--rho', '.5', '--lam', str(lam),
                       '--depth', str(args.depth), '--workers', str(args.workers),
                       '--expected-queries', str(QUERY_COUNT), '--expected-gallery', str(GALLERY_COUNT)]
            if lam > 0:
                command += ['--support-file', str(support_file), '--support-mode', 'cb']
            run_stage(command, method + '-run', root, repo, environment, stages)
            verify_command = [sys.executable, '-B', str(verifier), '--mat-path', str(mat),
                              '--output-dir', str(output), '--method', method,
                              '--protocol', str(protocol), '--depth', str(args.depth),
                              '--expected-queries', str(QUERY_COUNT), '--expected-gallery', str(GALLERY_COUNT),
                              '--progress-every', '0']
            if lam > 0:
                verify_command += ['--support-file', str(support_file)]
            run_stage(verify_command, method + '-verify', root, repo, environment, stages)
            report_path = output / ('verified-' + method + '.json')
            report = json.loads(report_path.read_text())
            checks = report['checks']
            if (report['status'] != 'verified'
                    or checks['full_permutation_queries'] != QUERY_COUNT
                    or checks['independent_metric_queries'] != QUERY_COUNT
                    or checks['diffusion_checkpoints'] != QUERY_COUNT
                    or checks['diffusion_depth'] != args.depth):
                raise AssertionError('Independent verifier did not confirm complete coverage')
            rank = np.load(output / (method + '.rankings.npy'), allow_pickle=False)
            if rank.shape != (QUERY_COUNT, GALLERY_COUNT):
                raise AssertionError('Wrong aggregate ranking dimensions')
            rankings.append(rank)
            summaries.append(dict(method=method, lam=lam, status='verified',
                                  ranking_shape=list(rank.shape), checks=checks,
                                  report=str(report_path), report_sha256=file_sha256(report_path),
                                  manifest_signature=json.loads((output / 'run-manifest.json').read_text())['signature']))
        if source_fingerprints(source_dir) != source_before:
            raise RuntimeError('source/*.py changed during synthetic integration; rerun after source freeze')
        summary.update(status='passed', methods=summaries,
                       changed_query_rankings=int(np.count_nonzero(np.any(rankings[0] != rankings[1], axis=1))),
                       source_freeze_verified=True,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / 'integration-summary.json', summary)
        print(json.dumps({'status': 'passed', 'scope': summary['scope'],
                          'summary': str(root / 'integration-summary.json'),
                          'queries_per_method': QUERY_COUNT, 'gallery_per_query': GALLERY_COUNT,
                          'methods': [item['method'] for item in summaries]}, indent=2), flush=True)
    except Exception as error:
        summary.update(status='failed', error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / 'integration-summary.json', summary)
        raise


if __name__ == '__main__':
    main()
