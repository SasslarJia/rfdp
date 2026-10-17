"""Build a hash-bound fixed neighbourhood support matrix from a feature cache."""
import argparse
from pathlib import Path
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--mat-path', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['ordinary', 'cb'], default='ordinary')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'RFDP'))
    sys.path.insert(0, str(args.repo / 'RFDP/experiments/full_reid'))
    from common import load_features, sha
    from reliability_model import build_support, save_support
    if args.output.exists():
        raise FileExistsError('Support output already exists; preserve the frozen input')
    metadata = json.loads((args.cache_dir / 'manifest.json').read_text())
    if metadata['feature_sha256'] != sha(args.mat_path):
        raise ValueError('Feature/cache fingerprint mismatch')
    full_path = args.cache_dir / 'affinity.npy'
    if metadata['files']['affinity.npy'] != sha(full_path):
        raise ValueError('Affinity fingerprint mismatch')
    data, queries, gallery = load_features(args.mat_path)
    full = np.load(full_path, mmap_mode='r')
    if full.shape != (queries + gallery, queries + gallery):
        raise ValueError('Affinity population differs from feature file')
    cameras = np.r_[data['query_cam'].ravel(), data['gallery_cam'].ravel()]
    support, info = build_support(full, cameras, k=20, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_support(args.output, support, info)
    print(json.dumps(dict(status='ready', mode=args.mode, queries=queries,
                          gallery=gallery, sha256=sha(args.output))))


if __name__ == '__main__':
    main()
