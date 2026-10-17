"""Frozen reciprocal-witness reliability for the production I-HyRDP solver.

No identity labels are accepted. The full affinity and the camera vector must
have the same sample order as the graph. Support is computed once from the
original full affinity, never from diffusion scores or selected seeds.

Public API:
    build_support(full, cameras, k=20, mode='cb') -> (CSR, metadata)
    build_supports(full, cameras, ...) -> ({mode: CSR}, {mode: metadata})
    save_support(path, support, metadata)
    load_support(path, expected=None) -> (CSR, metadata)
    ReliabilityIHyRDP(graph, full, cameras, eta=.5, support=Q, lam=.5)

Run with the repository's RFDP directory on PYTHONPATH. Ranking and linear solves are
inherited unchanged from that repository. Missing sparse support entries mean
q=0; diagonal support is always q_ii=1. No dense n-by-n support cache is built.

Hash contract: dense hashes include shape, little-endian dtype and C-order
values; CSR hashes include shape and canonical sorted CSR buffers, with indices
as little-endian int64 and values as little-endian float64. Dense dtype changes
therefore change a dense hash, while equivalent CSR index widths do not.
Support archives bind the exact full-affinity content, camera order, k, mode,
definition version and support values. Passing archive metadata to the model
checks those input hashes, including a streamed full-affinity hash.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
from scipy import sparse


def _load_production_classes():
    """Load the two actual source modules without package-level logging writes.

    RFDP's legacy package __init__ imports unrelated modules with file-writing
    side effects. A private package path preserves the source files' relative
    imports while leaving those initializers untouched and unexecuted.
    """
    # Reuse the exact module patched by the runner's PaperPrecision.install().
    existing_full = sys.modules.get('MODEL.Paper_RFDP.PaperFullIterative')
    if existing_full is not None:
        camera_module = importlib.import_module('MODEL.Paper_RFDP.PaperCamera')
        return camera_module.CameraIHyRDP, existing_full.FullIHyRDP, Path(existing_full.__file__).resolve().parent
    candidates = []
    if os.environ.get('RFDP_PAPER_DIR'):
        candidates.append(Path(os.environ['RFDP_PAPER_DIR']))
    else:
        for entry in sys.path:
            root = Path(entry or '.')
            candidates.extend((root / 'MODEL' / 'Paper_RFDP',
                               root / 'RFDP' / 'MODEL' / 'Paper_RFDP', root))
    folder = next((path.resolve() for path in candidates
                   if (path / 'PaperCamera.py').is_file()
                   and (path / 'PaperFullIterative.py').is_file()), None)
    if folder is None:
        raise ImportError('Set RFDP_PAPER_DIR to the production Paper_RFDP directory, '
                          'or add the repository/RFDP directory to PYTHONPATH')
    name = '_reliability_production_' + hashlib.sha256(str(folder).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        package.__package__ = name
        sys.modules[name] = package
    camera_module = importlib.import_module(name + '.PaperCamera')
    full_module = importlib.import_module(name + '.PaperFullIterative')
    if (folder / 'PaperPrecision.py').is_file():
        precision_module = importlib.import_module(name + '.PaperPrecision')
        precision_module.install()
    return camera_module.CameraIHyRDP, full_module.FullIHyRDP, folder


CameraIHyRDP, FullIHyRDP, PRODUCTION_DIR = _load_production_classes()
PRODUCTION_SOURCE_HASHES = {
    name: hashlib.sha256((PRODUCTION_DIR / name).read_bytes()).hexdigest()
    for name in ('PaperCamera.py', 'PaperFullIterative.py', 'PaperPrecision.py')
    if (PRODUCTION_DIR / name).is_file()
}


DEFINITION_VERSION = 'reciprocal-witness-v1-endpoints-excluded-diag-one'
ARCHIVE_VERSION = 1
_MODES = {'ordinary': 'ordinary', 'cb': 'cb', 'camera_balanced': 'cb'}


def _canonical_csr(matrix):
    result = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _pair_values(matrix, rows, cols):
    if len(rows) == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(matrix[rows, cols], dtype=np.float64).ravel()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _dense_hasher(array):
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise TypeError('Object arrays cannot be content hashed')
    dtype = array.dtype.newbyteorder('<')
    digest = hashlib.sha256()
    digest.update(_json_bytes({'shape': list(array.shape), 'dtype': dtype.str}))
    digest.update(b'\0')
    return digest, dtype


def array_sha256(array, block_rows=64):
    """Hash an ndarray/memmap without making a full copy."""
    array = np.asarray(array)
    if block_rows < 1:
        raise ValueError('block_rows must be positive')
    digest, dtype = _dense_hasher(array)
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    else:
        for start in range(0, len(array), block_rows):
            block = np.ascontiguousarray(array[start:start + block_rows], dtype=dtype)
            digest.update(memoryview(block).cast('B'))
    return digest.hexdigest()


def csr_sha256(matrix):
    matrix = _canonical_csr(matrix)
    digest = hashlib.sha256()
    digest.update(_json_bytes({'shape': list(matrix.shape), 'format': 'csr-f64-i64-v1'}))
    for array, dtype in ((matrix.indptr, '<i8'), (matrix.indices, '<i8'),
                         (matrix.data, '<f8')):
        digest.update(np.asarray(array, dtype=dtype).tobytes(order='C'))
    return digest.hexdigest()


def _validate_cameras(cameras, n):
    cameras = np.asarray(cameras)
    if cameras.ndim != 1 or len(cameras) != n or cameras.dtype.hasobject:
        raise ValueError('cameras must be a non-object vector with one entry per sample')
    if np.issubdtype(cameras.dtype, np.number) and not np.isfinite(cameras).all():
        raise ValueError('Camera labels must be finite')
    return cameras


def validate_support(support, n=None):
    """Return a canonical owned CSR copy, rejecting invalid confidence values."""
    support = _canonical_csr(support)
    if support.shape[0] != support.shape[1] or (n is not None and support.shape != (n, n)):
        raise ValueError('Support dimensions disagree')
    if not np.isfinite(support.data).all() or np.any(support.data < 0) or np.any(support.data > 1):
        raise ValueError('Support must be finite and in [0, 1]')
    delta = support - support.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
        raise ValueError('Support must be symmetric')
    if not np.array_equal(support.diagonal(), np.ones(support.shape[0])):
        raise ValueError('Support diagonal must be exactly one')
    return support


def build_reciprocal_graph(full, k=20, block_rows=64, progress=None):
    """Top-k positive affinities, excluding self; ties use ascending sample ID.

    Only O(block_rows*n + n*k) working storage is used. Affinities are not
    changed. The mutual-neighbor graph is symmetric even if input rows have
    different rank orders. The whole affinity is validated and hashed while
    reading its row blocks.
    """
    full = np.asarray(full)
    if full.ndim != 2 or full.shape[0] != full.shape[1] or len(full) < 1:
        raise ValueError('full must be a nonempty square affinity matrix')
    if not np.issubdtype(full.dtype, np.number) or np.iscomplexobj(full):
        raise ValueError('full must contain real numeric affinities')
    if isinstance(k, (bool, np.bool_)) or int(k) != k or k < 1:
        raise ValueError('k must be a positive integer')
    if block_rows < 1:
        raise ValueError('block_rows must be positive')
    n, k = len(full), int(k)
    effective_k = min(k, n - 1)
    digest, hash_dtype = _dense_hasher(full)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.empty(n * effective_k, dtype=np.int64)
    cursor = 0
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        raw_block = np.asarray(full[start:stop])
        hash_block = np.ascontiguousarray(raw_block, dtype=hash_dtype)
        digest.update(memoryview(hash_block).cast('B'))
        block = np.asarray(raw_block, dtype=np.float64)
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError('full affinity must be finite and nonnegative')
        for offset, row in enumerate(block):
            i = start + offset
            eligible = np.flatnonzero(row > 0)
            eligible = eligible[eligible != i]
            if len(eligible) > effective_k:
                threshold = np.partition(row[eligible], len(eligible) - effective_k)[len(eligible) - effective_k]
                above = eligible[row[eligible] > threshold]
                tied = eligible[row[eligible] == threshold]
                chosen = np.concatenate((above, tied[:effective_k - len(above)]))
            else:
                chosen = eligible
            chosen = np.sort(chosen)
            indices[cursor:cursor + len(chosen)] = chosen
            cursor += len(chosen)
            indptr[i + 1] = cursor
        if progress is not None:
            progress({'stage': 'reciprocal_topk', 'rows_done': stop, 'rows_total': n})
    directed = sparse.csr_matrix((np.ones(cursor), indices[:cursor], indptr), shape=(n, n))
    reciprocal = _canonical_csr(directed.multiply(directed.T))
    reciprocal.setdiag(0)
    reciprocal.eliminate_zeros()
    metadata = dict(n=n, k=k, effective_k=effective_k, full_shape=list(full.shape),
                    full_dtype=full.dtype.str, full_sha256=digest.hexdigest(),
                    reciprocal_sha256=csr_sha256(reciprocal),
                    reciprocal_nnz=int(reciprocal.nnz),
                    tie_rule='descending affinity, ascending sample ID',
                    positive_neighbors_only=True, exclude_self=True)
    return reciprocal, metadata


def support_from_reciprocal(reciprocal, cameras, mode='cb', progress=None):
    """Construct sparse support from two-hop candidates, with endpoints removed.

    For ordinary support, q is Jaccard with the opposite endpoint removed. For camera-balanced
    support, Jaccard is computed separately for each witness camera and then
    averaged over cameras whose endpoint-removed union is nonempty.
    """
    if mode not in _MODES:
        raise ValueError('mode must be ordinary or cb')
    mode = _MODES[mode]
    reciprocal = _canonical_csr(reciprocal)
    n = reciprocal.shape[0]
    cameras = _validate_cameras(cameras, n)
    if reciprocal.shape != (n, n) or np.any(reciprocal.diagonal() != 0):
        raise ValueError('Reciprocal graph must be square with zero diagonal')
    if reciprocal.nnz and not np.all(reciprocal.data == 1):
        raise ValueError('Reciprocal graph must be binary')
    if (reciprocal != reciprocal.T).nnz:
        raise ValueError('Reciprocal graph must be symmetric')
    common = _canonical_csr(reciprocal @ reciprocal)
    common.setdiag(0)
    common.eliminate_zeros()
    common_coo = common.tocoo()
    rows, cols, overlap = common_coo.row, common_coo.col, common_coo.data
    if len(rows) == 0:
        return sparse.eye(n, dtype=np.float64, format='csr')
    adjacent = _pair_values(reciprocal, rows, cols)
    if mode == 'ordinary':
        degrees = np.asarray(reciprocal.sum(axis=1)).ravel()
        union = degrees[rows] + degrees[cols] - 2 * adjacent - overlap
        if np.any(union <= 0):
            raise ValueError('Invalid endpoint-removed union')
        values = overlap / union
    else:
        labels, codes = np.unique(cameras, return_inverse=True)
        values = np.zeros(len(rows), dtype=np.float64)
        active_cameras = np.zeros(len(rows), dtype=np.int32)
        for code in range(len(labels)):
            camera_neighbors = reciprocal[:, codes == code]
            degrees = np.asarray(camera_neighbors.sum(axis=1)).ravel()
            per_camera_common = camera_neighbors @ camera_neighbors.T
            per_camera_overlap = _pair_values(per_camera_common, rows, cols)
            endpoint_count = ((codes[cols] == code).astype(np.int8)
                              + (codes[rows] == code).astype(np.int8))
            union = degrees[rows] + degrees[cols] - adjacent * endpoint_count - per_camera_overlap
            if np.any(union < 0):
                raise ValueError('Invalid camera-specific union')
            active = union > 0
            active_cameras += active
            values[active] += per_camera_overlap[active] / union[active]
            if progress is not None:
                progress({'stage': 'support_cb', 'cameras_done': code + 1,
                          'cameras_total': len(labels), 'candidate_nnz': len(rows)})
        if np.any(active_cameras == 0):
            raise ValueError('Common witnesses without an active camera')
        values /= active_cameras
    result = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    result.setdiag(1.0)
    return validate_support(result, n)


def support_stats(support, graph=None, cameras=None):
    """Statistics include implicit zero entries without densifying support."""
    support = validate_support(support)
    n = support.shape[0]
    rows = np.repeat(np.arange(n), np.diff(support.indptr))
    off = rows != support.indices
    values = support.data[off]
    possible = n * max(0, n - 1)
    counts = np.bincount(rows[off], minlength=n)
    result = dict(n=n, nnz=int(support.nnz), offdiag_positive_nnz=int(len(values)),
                  offdiag_positive_fraction=float(len(values) / possible) if possible else 0.0,
                  offdiag_mean_including_zeros=float(values.sum() / possible) if possible else 0.0,
                  offdiag_positive_quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else [],
                  mean_positive_witness_neighbors=float(counts.mean()) if n else 0.0,
                  max_positive_witness_neighbors=int(counts.max()) if n else 0)
    if graph is not None:
        graph = _canonical_csr(graph)
        if graph.shape != support.shape or np.any(graph.data < 0) or not np.isfinite(graph.data).all():
            raise ValueError('Graph does not match support')
        gr = np.repeat(np.arange(n), np.diff(graph.indptr))
        offdiag = gr != graph.indices
        gr, gc = gr[offdiag], graph.indices[offdiag]
        q = _pair_values(support, gr, gc)
        result.update(graph_offdiag_nnz=int(len(q)), graph_q_positive_fraction=float(np.mean(q > 0)) if len(q) else 0.0,
                      graph_q_mean=float(q.mean()) if len(q) else 0.0)
        if cameras is not None:
            cameras = _validate_cameras(cameras, n)
            same = cameras[gr] == cameras[gc]
            for name, selection in (('same_camera', same), ('cross_camera', ~same)):
                vals = q[selection]
                result[name] = dict(edges=int(len(vals)), q_mean=float(vals.mean()) if len(vals) else None,
                                    q_positive_fraction=float(np.mean(vals > 0)) if len(vals) else None)
    return result


def build_supports(full, cameras, k=20, modes=('ordinary', 'cb'), block_rows=64, progress=None):
    reciprocal, base_metadata = build_reciprocal_graph(full, k, block_rows, progress)
    cameras = _validate_cameras(cameras, base_metadata['n'])
    base_metadata.update(cameras_sha256=array_sha256(cameras),
                         definition_version=DEFINITION_VERSION,
                         diagonal_support=1.0, endpoints_excluded=True)
    supports, metadata = {}, {}
    for requested in modes:
        if requested not in _MODES:
            raise ValueError('mode must be ordinary or cb')
        mode = _MODES[requested]
        if mode in supports:
            continue
        support = support_from_reciprocal(reciprocal, cameras, mode, progress)
        info = dict(base_metadata, mode=mode, support_sha256=csr_sha256(support),
                    stats=support_stats(support))
        supports[mode], metadata[mode] = support, info
    return supports, metadata


def build_support(full, cameras, k=20, mode='cb', block_rows=64, progress=None):
    supports, metadata = build_supports(full, cameras, k, (mode,), block_rows, progress)
    canonical_mode = _MODES[mode]
    return supports[canonical_mode], metadata[canonical_mode]


def _check_metadata(support, metadata):
    required = ('n', 'k', 'mode', 'full_sha256', 'cameras_sha256', 'definition_version', 'support_sha256')
    if any(key not in metadata for key in required):
        raise ValueError('Support metadata lacks required input/hash fields')
    if metadata['definition_version'] != DEFINITION_VERSION or metadata['mode'] not in ('ordinary', 'cb'):
        raise ValueError('Unsupported support definition or mode')
    if metadata['n'] != support.shape[0] or metadata['support_sha256'] != csr_sha256(support):
        raise ValueError('Support shape/content hash mismatch')
    for key in ('full_sha256', 'cameras_sha256', 'support_sha256'):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid SHA-256 metadata: ' + key)


def save_support(path, support, metadata):
    """Atomically save one NPZ with CSR arrays and checksummed JSON metadata."""
    support = validate_support(support)
    metadata = dict(metadata)
    _check_metadata(support, metadata)
    metadata['archive_version'] = ARCHIVE_VERSION
    metadata.pop('metadata_sha256', None)
    metadata['metadata_sha256'] = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    encoded = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, data=support.data, indices=support.indices,
                                indptr=support.indptr, shape=np.asarray(support.shape, dtype=np.int64),
                                metadata_utf8=encoded)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_support(path, expected=None):
    """Verify archive contents and optional exact metadata expectations.

    Example expected: {'full_sha256': ..., 'cameras_sha256': ..., 'k': 20,
                       'mode': 'cb', 'definition_version': DEFINITION_VERSION}.
    expected is a flat mapping and every specified field must match exactly.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_utf8'].tobytes().decode('utf-8'))
        support = sparse.csr_matrix((archive['data'], archive['indices'], archive['indptr']),
                                    shape=tuple(archive['shape'].tolist()))
    digest = metadata.pop('metadata_sha256', None)
    if digest != hashlib.sha256(_json_bytes(metadata)).hexdigest():
        raise ValueError('Support metadata hash mismatch')
    if metadata.get('archive_version') != ARCHIVE_VERSION:
        raise ValueError('Unsupported support archive version')
    support = validate_support(support)
    _check_metadata(support, metadata)
    for key, value in (expected or {}).items():
        if key not in metadata or metadata[key] != value:
            raise ValueError('Support metadata expectation mismatch: ' + key)
    metadata['metadata_sha256'] = digest
    return support, metadata


class ReliabilityIHyRDP(CameraIHyRDP):
    """CSW with a fixed positive gate r=1-lam+lam*q and unchanged solver.

    lam=0 dispatches to the original CSW initialization/bridge implementation,
    providing exact numerical regression. A full all-one Q takes that same
    identity path. For lam>0 support is required and copied to owned CSR.
    The large full affinity is held through a read-only view, not copied;
    callers must not mutate its backing array while this model is in use.
    """

    def __init__(self, graph, full, cameras, eta=0.5, *, support=None, lam=0.0,
                 alpha=5.0, beta=25.0, step=5, support_metadata=None):
        if not np.isfinite(lam) or not 0 <= lam < 1:
            raise ValueError('lam must be finite and in [0, 1)')
        if not np.all(np.isfinite([alpha, beta])) or min(alpha, beta) <= 0:
            raise ValueError('alpha and beta must be finite and positive')
        if isinstance(step, (bool, np.bool_)) or int(step) != step or step < 1:
            raise ValueError('step must be a positive integer')
        n = graph.shape[0]
        cameras = _validate_cameras(cameras, n).copy()
        if support is None and lam != 0:
            raise ValueError('A frozen support matrix is required when lam is positive')
        self.support = None if support is None else validate_support(support, n)
        self.support_metadata = None if support_metadata is None else dict(support_metadata)
        if support_metadata is not None:
            if self.support is None:
                raise ValueError('Support metadata supplied without support')
            _check_metadata(self.support, support_metadata)
            if support_metadata['cameras_sha256'] != array_sha256(cameras):
                raise ValueError('Camera vector does not match support metadata')
            if support_metadata['full_sha256'] != array_sha256(full):
                raise ValueError('Full affinity does not match support metadata')
        self.lam = float(lam)
        self._identity_gate = (lam == 0 or (self.support is not None
                               and self.support.nnz == n * n and np.all(self.support.data == 1)))
        full_view = np.asarray(full).view()
        full_view.flags.writeable = False
        # This is the production CSW path, including its original sum ordering.
        super().__init__(graph, full_view, cameras, eta)
        self.alpha, self.beta, self.step = float(alpha), float(beta), int(step)
        self.base_full_sums = self.full_sums.copy()
        self.input_hashes = dict(graph_sha256=csr_sha256(graph), cameras_sha256=array_sha256(cameras),
                                 support_sha256=None if self.support is None else csr_sha256(self.support))
        if support_metadata is not None:
            self.input_hashes['full_sha256'] = support_metadata['full_sha256']
        if not self._identity_gate:
            rows = np.repeat(np.arange(self.n), np.diff(self.w.indptr))
            q = _pair_values(self.support, rows, self.w.indices)
            self.w.data *= (1 - self.lam + self.lam * q)
            if np.any(self.w.data <= 0):
                raise ValueError('Reliability scaling lost positive graph support through numerical underflow')
            correction = self._support_weighted_full_sums()
            self.full_sums = (1 - self.lam) * self.base_full_sums + self.lam * correction
            if not np.isfinite(self.full_sums).all() or np.any(self.full_sums <= 0):
                raise ValueError('Invalid reliability-weighted full-similarity degrees')
        if self.support is not None:
            for buffer in (self.support.data, self.support.indices, self.support.indptr):
                buffer.flags.writeable = False

    def _support_weighted_full_sums(self, block_rows=128):
        result = np.zeros(self.n, dtype=np.float64)
        for start in range(0, self.n, block_rows):
            stop = min(self.n, start + block_rows)
            begin, end = self.support.indptr[start], self.support.indptr[stop]
            rows = np.repeat(np.arange(start, stop), np.diff(self.support.indptr[start:stop + 1]))
            cols = self.support.indices[begin:end]
            values = np.asarray(self.full[rows, cols], dtype=np.float64)
            values *= np.where(self.cameras[rows] == self.cameras[cols], self.eta, 1.0)
            values *= self.support.data[begin:end]
            result[start:stop] = np.bincount(rows - start, weights=values, minlength=stop - start)
        return result

    def full_kernel_block(self, rows, cols):
        """Return weighted raw affinities for a requested block, not row-normalized."""
        rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
        block = np.asarray(self.full[np.ix_(rows, cols)], dtype=np.float64)
        block *= np.where(self.cameras[rows, None] == self.cameras[cols][None, :], self.eta, 1.0)
        if not self._identity_gate:
            q = self.support[rows][:, cols].toarray()
            block *= (1 - self.lam + self.lam * q)
        return block

    def bridges(self, active_components):
        if self._identity_gate:
            return super().bridges(active_components)
        key = tuple(sorted(active_components))
        if key in self.bridge_cache:
            return self.bridge_cache[key]
        g = np.flatnonzero(np.isin(self.components, key))
        choices = []
        for comp in range(self.nc):
            if comp in active_components:
                continue
            outside = np.flatnonzero(self.components == comp)
            best = None
            for start in range(0, len(outside), 128):
                ids = outside[start:start + 128]
                block = self.full_kernel_block(ids, g)
                block /= self.full_sums[ids, None]
                i, j = np.unravel_index(np.argmax(block), block.shape)
                value = float(block[i, j])
                candidate = (value, int(ids[i]), int(g[j]), comp)
                if best is None or value > best[0] or (value == best[0] and candidate[1:3] < best[1:3]):
                    best = candidate
            if best is not None and best[0] > 0:
                choices.append(best)
        choices.sort(key=lambda item: (-item[0], item[1], item[2]))
        result = []
        for _, i, j, _ in choices[:self.step]:
            value = float(self.full[i, j]) * (self.eta if self.cameras[i] == self.cameras[j] else 1.0)
            value *= 1 - self.lam + self.lam * float(self.support[i, j])
            result.append((i, j, value))
        self.bridge_cache[key] = result
        return result
