"""Blocked historical Dist2Aff / mutual-affinity-kNN on frozen float32 Z.

This module deliberately has no parameter-search CLI. P1 uses Kd=Ks=30 and
gamma=.4; explicit smaller arguments exist for synthetic numerical oracles.
The complete affinity and the sparse graph are always built together. Distances
and affinity use disk-backed arrays; no additional in-memory N-by-N array is
created. Frozen features are never normalized again or cast to another dtype.
"""
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import scipy

BUILDER_ID = 'frozen-z-historical-affinity-blocked-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def _blocks(n, size):
    for start in range(0, n, size):
        yield start, min(n, start + size)


def _notice(progress, phase, completed, total):
    if progress is not None:
        progress(dict(phase=phase, completed=completed, total=total))


def _close(array):
    if isinstance(array, np.memmap):
        array.flush()
        array._mmap.close()


def _symmetrize_distance(distance, block_rows):
    """AffXfrm's float32 (D+D.T)*.5, with bounded tile temporaries.

Applying it to already symmetric finite distances leaves their bits unchanged.
Off-diagonal tiles are read before either half is overwritten.
    """
    n = len(distance)
    for start, end in _blocks(n, block_rows):
        for col, stop in _blocks(n, block_rows):
            if col < start:
                continue
            tile = np.array(distance[start:end, col:stop], copy=True)
            tile += distance[col:stop, start:end].T
            tile *= .5
            distance[start:end, col:stop] = tile
            if col != start:
                distance[col:stop, start:end] = tile.T


def _write_distances(features, path, block_rows, progress=None):
    n = len(features)
    distance = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, n))
    try:
        for start, end in _blocks(n, block_rows):
            similarity = features[start:end].dot(features.T)
            # Preserve pairwise_l2_distance's operation order and float32 dtype.
            block = np.maximum(2.0 - 2.0 * similarity, 0.0)
            np.sqrt(block, out=block)
            distance[start:end] = block
            _notice(progress, 'distance', end, n)
        np.fill_diagonal(distance, 0.)
        _symmetrize_distance(distance, block_rows)
        distance.flush()
    except BaseException:
        _close(distance)
        raise
    return distance


def _distance_sums(distance, kd, block_rows, progress=None):
    """Same sorted column top-(Kd+1), omit first, float32 axis-0 sum."""
    n = len(distance)
    sums = np.empty(n, dtype=np.float32)
    for start, end in _blocks(n, block_rows):
        column_block = np.array(distance[:, start:end], copy=True, order='C')
        column_block.partition(kd, axis=0)
        top = np.array(column_block[:kd + 1], copy=True, order='C')
        top.sort(axis=0)
        # The dense original sums a non-fast axis in a matrix with N>1
        # columns. Keep that reduction layout for a one-column final block.
        if end - start == 1:
            padded = np.repeat(top, 2, axis=1)
            values = padded[1:].sum(axis=0, dtype=np.float32)[:1]
        else:
            values = top[1:].sum(axis=0, dtype=np.float32)
        sums[start:end] = values
        _notice(progress, 'distance_scales', end, n)
    if not np.isfinite(sums).all() or np.any(sums <= 0):
        raise ValueError('Zero/nonfinite local distance scale; no epsilon floor is added to Dist2Aff')
    return sums


def _write_affinity(distance, sums, path, kd, gamma, block_rows, progress=None):
    n = len(distance)
    affinity = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, n))
    weight = gamma / (2 * kd)
    try:
        for start, end in _blocks(n, block_rows):
            # Do not rewrite as (D / denominator)**2: rounding differs.
            delta = weight * (sums[start:end, None] + sums[None, :])
            np.divide(1, delta, out=delta)
            np.power(delta, 2.0, out=delta)
            values = np.array(distance[start:end], copy=True, order='C')
            np.power(values, 2.0, out=values)
            np.negative(values, out=values)
            values *= delta
            np.exp(values, out=values)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite Dist2Aff result')
            affinity[start:end] = values
            _notice(progress, 'affinity', end, n)
        if not np.array_equal(affinity.diagonal(), np.ones(n, dtype=np.float32)):
            raise ValueError('Historical affinity self-loops must equal one')
        affinity.flush()
    except BaseException:
        _close(affinity)
        raise
    return affinity


def _symmetric_rows(affinity, start, end):
    """KnnRstct invokes AffXfrm symmetry again; keep A itself unchanged."""
    values = np.array(affinity[start:end], copy=True, order='C')
    values += affinity[:, start:end].T
    values *= .5
    return values


def _affinity_thresholds(affinity, ks, block_rows):
    n = len(affinity)
    thresholds = np.empty(n, dtype=np.float32)
    for start, end in _blocks(n, block_rows):
        # Symmetry means the sorted row and column thresholds are identical.
        values = _symmetric_rows(affinity, start, end)
        np.negative(values, out=values)
        values.partition(ks, axis=1)
        thresholds[start:end] = -values[:, ks]
    return thresholds


def _mutual_block(affinity, thresholds, start, end):
    values = _symmetric_rows(affinity, start, end)
    keep = values >= thresholds[start:end, None]
    keep &= values >= thresholds[None, :]
    # Dense KnnRstct may retain zero-valued ties; CSR omits exact zero only.
    keep &= values > 0
    return values, keep


def _write_mutual_graph(affinity, path, ks, block_rows, progress=None):
    """Two passes size CSR exactly, including all positive threshold ties."""
    n = len(affinity)
    thresholds = _affinity_thresholds(affinity, ks, block_rows)
    counts = np.empty(n, dtype=np.int64)
    for start, end in _blocks(n, block_rows):
        _, keep = _mutual_block(affinity, thresholds, start, end)
        counts[start:end] = keep.sum(axis=1, dtype=np.int64)
        _notice(progress, 'graph_count', end, n)
    indptr64 = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    nnz = int(indptr64[-1])
    index_dtype = np.int32 if max(n, nnz) <= np.iinfo(np.int32).max else np.int64
    indptr = indptr64.astype(index_dtype)
    data_path, indices_path = path.parent / '.graph-data.npy', path.parent / '.graph-indices.npy'
    data = np.lib.format.open_memmap(data_path, mode='w+', dtype=np.float32, shape=(nnz,))
    indices = np.lib.format.open_memmap(indices_path, mode='w+', dtype=index_dtype, shape=(nnz,))
    try:
        for start, end in _blocks(n, block_rows):
            values, keep = _mutual_block(affinity, thresholds, start, end)
            for offset, row in enumerate(range(start, end)):
                ids = np.flatnonzero(keep[offset])
                lo, hi = indptr[row:row + 2]
                indices[lo:hi] = ids
                data[lo:hi] = values[offset, ids]
                if not keep[offset, row] or values[offset, row] != 1:
                    raise ValueError('Mutual graph lost a unit self-loop')
            _notice(progress, 'graph_write', end, n)
        # Same CSR archive fields as scipy.sparse.save_npz. Writing memmaps
        # directly avoids materializing another CSR-sized copy under many ties.
        np.savez_compressed(path, indices=indices, indptr=indptr, format=np.array(b'csr'),
                            shape=np.asarray([n, n], dtype=np.int64), data=data)
    finally:
        _close(data)
        _close(indices)
    data_path.unlink()
    indices_path.unlink()
    return dict(nnz=nnz, dtype='float32', index_dtype=np.dtype(index_dtype).name,
                symmetric=True, symmetry_basis='symmetric affinity and mutual >= row/column thresholds',
                self_loop=1., max_nonself_neighbors=int(counts.max() - 1),
                rows_exceeding_ks_due_to_ties=int(np.count_nonzero(counts > ks + 1)))


def build_graph_cache(features_path, output, *, kd=30, ks=30, gamma=.4,
                      block_rows=128, expected_features_sha256=None, progress=None):
    """Build one frozen graph cache; no extraction, re-normalization or search.

    A new directory is required. A failed construction retains its evidence
    with FAILED.json and never publishes a complete manifest.
    """
    features_path, output = Path(features_path), Path(output)
    features = np.load(features_path, mmap_mode='r', allow_pickle=False)
    if features.ndim != 2 or features.dtype != np.dtype('float32'):
        raise ValueError('Expected a frozen normalized two-dimensional float32 features.npy')
    n = len(features)
    if any(type(k) is not int or not 1 <= k < n for k in (kd, ks)):
        raise ValueError('Kd and Ks must be integers in [1, N-1]')
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError('block_rows must be a positive integer')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    for start, end in _blocks(n, block_rows):
        if not np.isfinite(features[start:end]).all():
            raise ValueError('Frozen features must be finite')
    feature_hash = sha256(features_path)
    if expected_features_sha256 is not None and feature_hash != expected_features_sha256:
        raise ValueError('Frozen feature SHA-256 mismatch')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    distance = affinity = None
    try:
        try:
            os.link(features_path, output / 'features.npy')
            feature_storage = 'hardlink'
        except OSError:
            shutil.copyfile(features_path, output / 'features.npy')
            feature_storage = 'byte_copy'
        distance_path = output / '.distance.npy'
        distance = _write_distances(features, distance_path, block_rows, progress)
        sums = _distance_sums(distance, kd, block_rows, progress)
        affinity = _write_affinity(distance, sums, output / 'affinity.npy', kd,
                                   gamma, block_rows, progress)
        _close(distance)
        distance = None
        distance_path.unlink()
        graph_info = _write_mutual_graph(affinity, output / 'graph.npz', ks, block_rows, progress)
        _close(affinity)
        affinity = None
        files = {name: sha256(output / name) for name in ('features.npy', 'affinity.npy', 'graph.npz')}
        if files['features.npy'] != feature_hash:
            raise ValueError('Frozen feature bytes changed while building the cache')
        source_root = Path(__file__).resolve().parents[2]
        references = ('MODEL/Aff_RFDP/Dist2Aff.py', 'MODEL/Aff_RFDP/KnnRstct.py',
                      'MODEL/Aff_RFDP/AffXfrm.py', 'MODEL/Asst_RFDP/OperAsstNumpy.py')
        manifest = dict(status='complete', builder_id=BUILDER_ID,
                        builder_sha256=sha256(__file__), n=n,
                        features_source=str(features_path.resolve()), features_sha256=feature_hash,
                        features_dtype='float32', features_renormalized=False,
                        features_storage=feature_storage, affinity_dtype='float32',
                        parameters=dict(kd=kd, ks=ks, gamma=float(gamma)), block_rows=block_rows,
                        graph_mode='full affinity and historical mutual affinity kNN; >= threshold ties',
                        graph=graph_info, nnz=graph_info['nnz'], files=files,
                        local_distance_sum_range=[float(sums.min()), float(sums.max())],
                        temporary_storage='one distance memmap; disk-backed CSR buffers; bounded row/tile work',
                        numpy=np.__version__, scipy=scipy.__version__,
                        historical_source_sha256={name: sha256(source_root / name) for name in references},
                        seconds=time.perf_counter() - started)
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception:
        _json(output / 'FAILED.json', dict(error=traceback.format_exc(), features_sha256=feature_hash,
                                          parameters=dict(kd=kd, ks=ks, gamma=float(gamma))))
        raise
    finally:
        if distance is not None:
            _close(distance)
        if affinity is not None:
            _close(affinity)
        _close(features)
