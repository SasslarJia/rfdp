"""Camera-grouped reciprocal input graph for the existing CSW solver.

This borrows the separate within/across-camera neighbor selection in CA-Jaccard;
it is not a reproduction of the complete CA-Jaccard distance.  The returned
graph has raw affinity weights and the returned ordinary Jaccard support uses
the new reciprocal neighborhoods.  Pass both to ReliabilityIHyRDP with
eta=.5, lam=.25.  That existing class applies CSW to the graph AND full kernel.

Nothing is written by this module.  The caller owns cache persistence and the
original full affinity, which may be a read-only memory map.  Auxiliary storage
is O(n*k + block_rows*n); returned support storage is additionally O(nnz(Q)).
Q includes all nonzero two-hop pairs, not just graph edges, so its necessary
output size can reach O(n*k**2).
"""

import hashlib
import json

import numpy as np
from scipy import sparse

from experiments.reliability.reliability_model import (
    DEFINITION_VERSION as SUPPORT_DEFINITION_VERSION,
    _dense_hasher,
    _validate_cameras,
    array_sha256,
)


GROUPED_DEFINITION_VERSION = 'camera-grouped-reciprocal-raw-v1'
_TIE_RULE = 'descending affinity, ascending sample ID'


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a positive integer')
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be a positive integer') from error
    if integer != value or integer < 1:
        raise ValueError(name + ' must be a positive integer')
    return integer


def _topk_ids(row, eligible, k):
    """eligible is ID-sorted; select only positive entries, resolving cutoff ties."""
    if len(eligible) > k:
        values = row[eligible]
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        above = eligible[values > threshold]
        tied = eligible[values == threshold]
        return np.concatenate((above, tied[:k - len(above)]))
    return eligible


def _csr_hash(matrix, chunk_entries=65536):
    """Match reliability_model.csr_sha256 without copying a whole large Q.

    Matrices produced here are canonical CSR with no stored zero entries.  A
    cached artifact must have that same representation; validation rejects a
    noncanonical one rather than silently rewriting its contents.
    """
    if not sparse.isspmatrix_csr(matrix) or not matrix.has_canonical_format:
        raise ValueError('Expected canonical CSR cache artifact')
    for start in range(0, len(matrix.data), chunk_entries):
        if np.any(matrix.data[start:start + chunk_entries] == 0):
            raise ValueError('CSR cache artifact contains stored zeros')
    digest = hashlib.sha256(_json_bytes(
        {'shape': matrix.shape, 'format': 'csr-f64-i64-v1'}))
    for array, dtype in ((matrix.indptr, '<i8'), (matrix.indices, '<i8'),
                         (matrix.data, '<f8')):
        for start in range(0, len(array), chunk_entries):
            block = np.ascontiguousarray(array[start:start + chunk_entries], dtype=dtype)
            digest.update(memoryview(block).cast('B'))
    return digest.hexdigest()


def _ordinary_support_block(reciprocal, degrees, start, stop):
    common = (reciprocal[start:stop] @ reciprocal).tocoo()
    rows = common.row + start
    keep = rows != common.col
    rows, cols, overlap = rows[keep], common.col[keep], common.data[keep]
    adjacent = (np.asarray(reciprocal[rows, cols]).ravel() if len(rows)
                else np.empty(0, dtype=np.float64))
    union = degrees[rows] + degrees[cols] - 2 * adjacent - overlap
    if np.any(union <= 0):
        raise ValueError('Invalid endpoint-removed Jaccard union')
    diagonal = np.arange(start, stop, dtype=np.int64)
    result = sparse.csr_matrix((np.r_[overlap / union, np.ones(stop - start)],
                               (np.r_[rows - start, diagonal - start],
                                np.r_[cols, diagonal])),
                              shape=(stop - start, reciprocal.shape[0]))
    result.sum_duplicates()
    result.sort_indices()
    return result


def _ordinary_support(reciprocal, block_rows, progress):
    """Two passes avoid retaining both a full R@R and a second full Q."""
    n = reciprocal.shape[0]
    degrees = np.asarray(reciprocal.sum(axis=1)).ravel()
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        block = _ordinary_support_block(reciprocal, degrees, start, stop)
        counts[start:stop] = np.diff(block.indptr)
        if progress is not None:
            progress(dict(stage='grouped_support_count', rows_done=stop, rows_total=n))
    total = int(counts.sum())
    index_dtype = np.int32 if max(n, total) <= np.iinfo(np.int32).max else np.int64
    indptr = np.empty(n + 1, dtype=index_dtype)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(total, dtype=index_dtype)
    data = np.empty(total, dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        block = _ordinary_support_block(reciprocal, degrees, start, stop)
        begin, end = int(indptr[start]), int(indptr[stop])
        if end - begin != block.nnz:
            raise ValueError('Reciprocal input changed while constructing support')
        indices[begin:end], data[begin:end] = block.indices, block.data
        if progress is not None:
            progress(dict(stage='grouped_support_values', rows_done=stop, rows_total=n))
    return sparse.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)


def _identity(metadata):
    return {key: metadata[key] for key in (
        'neighbor_definition_version', 'definition_version', 'mode',
        'full_sha256', 'cameras_sha256', 'k_intra', 'k_inter',
        'tie_rule', 'positive_neighbors_only', 'exclude_self', 'graph_diagonal')}


def build_grouped_csw_inputs(full, cameras, *, k_intra=5, k_inter=20,
                             block_rows=64, progress=None):
    """Return ``(raw_graph, ordinary_support, metadata)`` without writing files.

    Each row selects up to k_intra same-camera and k_inter other-camera
    positive, non-self affinities.  Both directed selections must contain an
    edge for it to survive.  Because camera equality is symmetric, taking
    reciprocity after the disjoint union is identical to taking it separately
    in each group.  Insufficient positive neighbors are all retained; groups
    do not borrow each other's unused budgets.

    The raw positive diagonal is retained.  Selected reciprocal weights must
    be symmetric exactly; asymmetric input weights are never averaged.  All
    full-affinity entries are checked for finiteness/nonnegativity and hashed
    in row blocks.  The input arrays must remain frozen throughout the call.

    Metadata is compatible with existing save_support/load_support and the
    ReliabilityIHyRDP ``support_metadata`` argument.  Its legacy ``k`` field
    is the total budget bound, not a claim of global top-k selection.
    """
    full = np.asarray(full)
    if full.ndim != 2 or full.shape[0] != full.shape[1] or not len(full):
        raise ValueError('full must be a nonempty square affinity matrix')
    if not np.issubdtype(full.dtype, np.number) or np.iscomplexobj(full):
        raise ValueError('full must contain real numeric affinities')
    k_intra = _positive_integer(k_intra, 'k_intra')
    k_inter = _positive_integer(k_inter, 'k_inter')
    block_rows = _positive_integer(block_rows, 'block_rows')
    n = len(full)
    cameras = _validate_cameras(cameras, n).copy()
    diagonal = np.asarray(full.diagonal(), dtype=np.float64).copy()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError('Raw affinity diagonal must be finite and positive')

    digest, hash_dtype = _dense_hasher(full)
    upper = min(n - 1, k_intra + k_inter)
    indices = np.empty(n * upper, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    cursor = 0
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        raw = np.asarray(full[start:stop])
        hash_block = np.ascontiguousarray(raw, dtype=hash_dtype)
        digest.update(memoryview(hash_block).cast('B'))
        block = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError('Full affinity must be finite and nonnegative')
        for offset, row in enumerate(block):
            i = start + offset
            eligible = np.flatnonzero(row > 0)
            eligible = eligible[eligible != i]
            same = cameras[eligible] == cameras[i]
            chosen = np.sort(np.r_[
                _topk_ids(row, eligible[same], k_intra),
                _topk_ids(row, eligible[~same], k_inter)])
            indices[cursor:cursor + len(chosen)] = chosen
            cursor += len(chosen)
            indptr[i + 1] = cursor
        if progress is not None:
            progress(dict(stage='grouped_reciprocal_topk', rows_done=stop, rows_total=n))
    directed = sparse.csr_matrix((np.ones(cursor), indices[:cursor], indptr), shape=(n, n))
    reciprocal = directed.multiply(directed.T).tocsr()
    reciprocal.sum_duplicates()
    reciprocal.sort_indices()
    del directed, indices, indptr

    rows = np.repeat(np.arange(n), np.diff(reciprocal.indptr))
    cols = reciprocal.indices
    values = np.asarray(full[rows, cols], dtype=np.float64).ravel()
    graph = sparse.csr_matrix((values, cols.copy(), reciprocal.indptr.copy()), shape=(n, n))
    graph.setdiag(diagonal)
    graph.sort_indices()
    if (graph != graph.T).nnz:
        raise ValueError('Reciprocal raw affinity weights must be symmetric')
    support = _ordinary_support(reciprocal, block_rows, progress)
    metadata = dict(
        n=n, full_shape=list(full.shape), full_dtype=full.dtype.str,
        full_sha256=digest.hexdigest(), cameras_sha256=array_sha256(cameras),
        neighbor_definition_version=GROUPED_DEFINITION_VERSION,
        definition_version=SUPPORT_DEFINITION_VERSION, mode='ordinary',
        k=k_intra + k_inter, k_intra=k_intra, k_inter=k_inter,
        k_semantics='sum of separate group budgets, not global top-k',
        tie_rule=_TIE_RULE, positive_neighbors_only=True, exclude_self=True,
        graph_diagonal='raw positive affinity', diagonal_support=1., endpoints_excluded=True,
        raw_graph_sha256=_csr_hash(graph), reciprocal_sha256=_csr_hash(reciprocal),
        support_sha256=_csr_hash(support), raw_graph_nnz=int(graph.nnz),
        reciprocal_nnz=int(reciprocal.nnz), support_nnz=int(support.nnz),
        block_rows=block_rows,
        csw_postprocess=dict(rho=.5, lam=.25, applied=False,
                             implementation='ReliabilityIHyRDP'),
        scope='grouped input graph with ordinary support; not full CA-Jaccard',
    )
    metadata['cache_key_sha256'] = hashlib.sha256(_json_bytes(_identity(metadata))).hexdigest()
    metadata['manifest_sha256'] = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    return graph, support, metadata


def validate_grouped_inputs(graph, support, metadata, *, full_sha256=None,
                            cameras=None, k_intra=None, k_inter=None):
    """Check cached content/signatures and caller-specified input expectations.

    This does not read or write files.  Pass the already-verified dense input
    hash to avoid rescanning a memory map just to check cache identity.
    """
    info = dict(metadata)
    # Existing save_support/load_support adds these archive-envelope fields.
    # The grouped manifest binds the semantic metadata before that envelope.
    info.pop('archive_version', None)
    info.pop('metadata_sha256', None)
    signature = info.pop('manifest_sha256', None)
    if signature != hashlib.sha256(_json_bytes(info)).hexdigest():
        raise ValueError('Grouped manifest hash mismatch')
    if (info.get('neighbor_definition_version') != GROUPED_DEFINITION_VERSION
            or info.get('definition_version') != SUPPORT_DEFINITION_VERSION
            or info.get('mode') != 'ordinary'):
        raise ValueError('Unsupported grouped graph/support definition')
    if info['cache_key_sha256'] != hashlib.sha256(_json_bytes(_identity(info))).hexdigest():
        raise ValueError('Grouped cache identity mismatch')
    shape = (info['n'], info['n'])
    if graph.shape != shape or support.shape != shape:
        raise ValueError('Grouped graph/support shape mismatch')
    if _csr_hash(graph) != info['raw_graph_sha256']:
        raise ValueError('Grouped raw graph hash mismatch')
    if _csr_hash(support) != info['support_sha256']:
        raise ValueError('Grouped support hash mismatch')
    for key, expected in (('full_sha256', full_sha256), ('k_intra', k_intra),
                          ('k_inter', k_inter)):
        if expected is not None and info[key] != expected:
            raise ValueError('Grouped cache expectation mismatch: ' + key)
    if cameras is not None and array_sha256(np.asarray(cameras)) != info['cameras_sha256']:
        raise ValueError('Grouped cache expectation mismatch: cameras_sha256')
    return True
