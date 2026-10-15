"""Portable experiment IO; saved rankings use global query+gallery indices."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO = Path(__file__).resolve().parents[3]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save(path, array):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def load_features(path):
    data = loadmat(path)
    q, g = (np.asarray(data[k]) for k in ('query_f', 'gallery_f'))
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or min(len(q), len(g)) == 0:
        raise ValueError('Expected nonempty query/gallery row features of equal width')
    for values in (q, g):
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError('Feature vectors must be finite and nonzero')
    for split, count in [('query', len(q)), ('gallery', len(g))]:
        for field in ('label', 'cam'):
            values = np.asarray(data[split + '_' + field]).ravel()
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError('Invalid label/camera metadata')
    return data, len(q), len(g)


def validate_ranking(rank, query_count, gallery_count):
    rank = np.asarray(rank)
    if rank.dtype.kind not in 'iu' or rank.shape != (gallery_count,) or not np.array_equal(
            np.sort(rank), np.arange(query_count, query_count + gallery_count)):
        raise ValueError('Ranking must be a complete permutation of global gallery IDs')


def complete_gallery(prefix, similarity, query_count):
    """Keep diffusion prefix, then original distance order, removing query nodes."""
    prefix = np.asarray(prefix, dtype=np.int32)
    similarity = np.asarray(similarity)
    n = len(similarity)
    if len(np.unique(prefix)) != len(prefix) or np.any(prefix < 0) or np.any(prefix >= n):
        raise ValueError('Invalid diffusion prefix')
    original = np.lexsort((np.arange(n), -similarity))
    used = np.zeros(n, dtype=bool)
    used[prefix] = True
    rank = np.r_[prefix, original[~used[original]]]
    rank = rank[rank >= query_count].astype(np.int32)
    validate_ranking(rank, query_count, n - query_count)
    return rank
