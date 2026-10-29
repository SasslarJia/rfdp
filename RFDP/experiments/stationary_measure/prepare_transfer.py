"""Restore derived Market TransReID arrays, requiring the accepted byte hashes.

No feature extraction, graph change, retrieval, or accuracy selection occurs here.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from experiments.full_reid.common import REPO, dump, load_features, sha


def main(root):
    start = time.monotonic()
    cap = int(Path('/sys/fs/cgroup/memory.max').read_text())
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    assert current + 20 * 2**30 <= .7 * cap, 'Cache reconstruction memory reservation'
    disk = shutil.disk_usage(root)
    assert disk.free - 3 * 2**30 >= max(disk.total // 4, 10 * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (20 * 2**30, 20 * 2**30))
    for entry in json.loads((root / 'restore-receipt.json').read_text()):
        for item in entry['selected_members'].values():
            assert sha(root / item['local_name']) == item['sha256'], item['local_name']
    mat = root / 'market-transreid.mat'
    assert sha(mat) == 'ed8925dc55a02818304dcd89b40ec3a21ed7d23a5d56c62a9279b826c4a25967'
    old = json.loads((root / 'manifest.json').read_text())
    assert old['feature_sha256'] == sha(mat)
    assert sha(root / 'graph.npz') == old['files']['graph.npz']
    # The baseline JSON contains both its summary and all accepted per-query values.
    baseline = json.loads((root / 'baseline-metrics.json').read_text())
    assert len(baseline['per_query']) == 3368
    assert abs(baseline['mAP'] - .9453089037436823) < 1e-14
    assert abs(baseline['rank-1'] - .952790973871734) < 1e-14
    data, nq, ng = load_features(mat)
    assert (nq, ng, data['query_f'].shape[1]) == (3368, 19732, 3840)
    features = np.vstack([data['query_f'], data['gallery_f']]).astype(float)
    del data
    spec = importlib.util.spec_from_file_location('historical_entry', REPO / 'RFDP/Test_RFDP/reid_paper_experiment.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    cache = root / 'cache'
    cache.mkdir(exist_ok=False)
    np.save(cache / 'features.npy', entry.l2_normalize(features).astype(np.float32))
    assert sha(cache / 'features.npy') == old['files']['features.npy'], 'Normalized input differs from frozen baseline'
    distances = entry.pairwise_l2_distance(features, 128, np.float32)
    del features
    affinity = entry.Dist2Aff(20, gamma=.4, asst_rela=entry.OperAsstNumpy())(distances)
    del distances
    np.save(cache / 'affinity.npy', np.asarray(affinity, dtype=np.float32))
    del affinity
    assert sha(cache / 'affinity.npy') == old['files']['affinity.npy'], 'Affinity differs from frozen baseline; do not launch'
    shutil.copyfile(root / 'graph.npz', cache / 'graph.npz')
    shutil.copyfile(root / 'manifest.json', cache / 'manifest.json')
    hashes = {name: sha(cache / name) for name in ('features.npy', 'affinity.npy', 'graph.npz', 'manifest.json')}
    dump(root / 'CACHE_RESTORED.json', dict(status='all_frozen_cache_hashes_match', files=hashes,
         git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
         seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         feature_extraction=False, graph_rebuilt=False, retrieval_rerun=False))
    print(json.dumps(hashes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    main(parser.parse_args().input.resolve())
