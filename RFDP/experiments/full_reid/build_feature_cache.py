"""Build the historical full affinity and mutual graph, without identity labels."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from common import REPO, dump, load_features, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-manifest', type=Path)
    args = parser.parse_args()
    data, nq, ng = load_features(args.mat)
    feature_hash = sha(args.mat)
    if args.feature_manifest and json.loads(args.feature_manifest.read_text())['feature_sha256'] != feature_hash:
        raise ValueError('Feature manifest hash mismatch')
    features = np.vstack([data['query_f'], data['gallery_f']]).astype(float)
    if len(features) <= 20:
        raise ValueError('The frozen k=20 graph requires at least 21 images')
    spec = importlib.util.spec_from_file_location('entry', REPO / 'RFDP/Test_RFDP/reid_paper_experiment.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    opts = SimpleNamespace(cache_path=None, graph_mode='dense', k_num=20, gamma=.4,
                           dtype='float32', batch_size=128, save_cache=None)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'features.npy', entry.l2_normalize(features).astype(np.float32))
    start = time.monotonic()
    affinity, dense_graph = entry.build_or_load_graph(opts, features)
    np.save(args.output / 'affinity.npy', affinity)
    graph = sparse.csr_matrix(dense_graph)
    sparse.save_npz(args.output / 'graph.npz', graph)
    delta = graph - graph.T
    nc, components = connected_components(graph, directed=False)
    source_paths = set((REPO / 'RFDP/MODEL').rglob('*.py'))
    source_paths.add(REPO / 'RFDP/Test_RFDP/reid_paper_experiment.py')
    manifest = dict(sources={str(p.relative_to(REPO)): sha(p) for p in sorted(source_paths)},
                    graph_seconds=time.monotonic() - start, n=nq + ng, query_count=nq,
                    gallery_count=ng, nnz=graph.nnz, components=nc,
                    component_sizes=np.bincount(components).tolist(),
                    max_asymmetry=float(abs(delta.data).max()) if delta.nnz else 0.,
                    feature_sha256=feature_hash,
                    parameters=dict(k=20, gamma=.4, alpha=5, beta=25, step=5),
                    graph_mode='full affinity; historical mutual-kernel-kNN; CSR is storage only',
                    builder_sha256=sha(__file__),
                    files={name: sha(args.output / name) for name in
                           ['features.npy', 'affinity.npy', 'graph.npz']})
    dump(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
