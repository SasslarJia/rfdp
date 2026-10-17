"""Unrun mechanism control: match the sparse CSW off-diagonal total mass.

This is NOT an ordinary/camera-balanced reciprocal-witness support. It removes
the original Q's edge allocation while matching its average sparse contraction.
It deliberately does not match per-row degrees or the full-kernel total mass.
No identity labels, relevance annotations or retrieval metrics are accepted.

Import with frozen v2/source on PYTHONPATH and RFDP_PAPER_DIR pointing at the
production snapshot. No runner or existing protocol is changed by this module.
"""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy import sparse

import reliability_model as rm


CONTROL_VERSION = 'uniform-sparse-offdiagonal-mass-matched-v1'
CONTROL_TYPE = 'uniform sparse-offdiagonal-mass-matched control'


def _sha_json(value):
    return hashlib.sha256(rm._json_bytes(value)).hexdigest()


def _checked_scalar(value, name, lower_open=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + ' must be a finite scalar')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must be a finite scalar') from exc
    valid_range = 0 < value <= 1 if lower_open else 0 <= value < 1
    if not math.isfinite(value) or not valid_range:
        raise ValueError(name + (' must be in (0,1]' if lower_open else ' must be in [0,1)'))
    return value


def prepare_uniform_control(graph, full, cameras, *, rho, lam,
                            source_support, source_support_metadata):
    """Derive an identity support and a fully bound independent control contract.

    qbar = sum_{i != j} (CSW(graph))_ij Q_ij / sum_{i != j} (CSW(graph))_ij.
    All stored directed non-diagonal graph entries are counted, including Q=0.
    Sparse duplicate entries are summed first. No per-query statistics are used.
    A zero/non-finite denominator is an explicit error, including at lambda=0.
    """
    rho = _checked_scalar(rho, 'rho', lower_open=True)
    lam = _checked_scalar(lam, 'lambda')
    graph_values = graph.data if sparse.issparse(graph) else graph
    support_values = source_support.data if sparse.issparse(source_support) else source_support
    if np.iscomplexobj(graph_values) or np.iscomplexobj(support_values):
        raise ValueError('graph and source support must be real')
    w = rm._canonical_csr(graph)
    n = w.shape[0]
    if n < 1 or w.shape != (n, n):
        raise ValueError('graph must be nonempty and square')
    if not np.isfinite(w.data).all() or np.any(w.data < 0):
        raise ValueError('graph must be finite and nonnegative')
    delta = w - w.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
        raise ValueError('graph must be symmetric')
    cameras = rm._validate_cameras(cameras, n)
    full = np.asarray(full)
    if full.shape != (n, n) or not np.issubdtype(full.dtype, np.number) or np.iscomplexobj(full):
        raise ValueError('full must be a real square affinity matching graph')
    for start in range(0, n, 128):
        block = full[start:start + 128]
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError('full must be finite and nonnegative')
    q = rm.validate_support(source_support, n)
    if not isinstance(source_support_metadata, dict):
        raise ValueError('source support metadata is required')
    source_meta = copy.deepcopy(source_support_metadata)
    rm._check_metadata(q, source_meta)
    k = source_meta['k']
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('source support k must be a positive integer')
    full_hash = rm.array_sha256(full)
    camera_hash = rm.array_sha256(cameras)
    if source_meta['full_sha256'] != full_hash:
        raise ValueError('Full affinity does not match original support metadata')
    if source_meta['cameras_sha256'] != camera_hash:
        raise ValueError('Camera vector does not match original support metadata')

    rows = np.repeat(np.arange(n), np.diff(w.indptr))
    weighted = w.copy()
    weighted.data *= np.where(cameras[rows] == cameras[w.indices], rho, 1.)
    if not np.isfinite(weighted.data).all() or np.any(weighted.data <= 0):
        raise ValueError('CSW weighting lost positive graph support')
    off = rows != w.indices
    weights = weighted.data[off]
    q_values = rm._pair_values(q, rows[off], w.indices[off])
    try:
        total = math.fsum(weights)
        numerator = math.fsum(weights * q_values)
    except OverflowError as exc:
        raise ValueError('Nonfinite sparse off-diagonal mass') from exc
    if not math.isfinite(total) or total <= 0 or not math.isfinite(numerator):
        raise ValueError('Positive finite sparse off-diagonal mass is required')
    qbar = numerator / total
    if not 0 <= qbar <= 1:
        raise ValueError('Weighted mean Q lies outside [0,1]')
    effective_lambda = lam * (1 - qbar)
    gamma = 1 - effective_lambda  # The exact float factor production will use.
    identity = sparse.eye(n, format='csr', dtype=np.float64)
    parameters = {'rho': rho, 'source_lambda': lam,
                  'lambda_effective': effective_lambda, 'gamma': gamma}
    hashes = {
        'graph_sha256': rm.csr_sha256(w),
        'csw_graph_sha256': rm.csr_sha256(weighted),
        'full_sha256': full_hash,
        'cameras_sha256': camera_hash,
        'source_support_sha256': rm.csr_sha256(q),
        'source_support_metadata_sha256': _sha_json(source_meta),
        'identity_support_sha256': rm.csr_sha256(identity),
        'parameters_sha256': _sha_json(parameters),
        'control_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reliability_source_sha256': hashlib.sha256(Path(rm.__file__).read_bytes()).hexdigest(),
        'production_source_sha256': dict(rm.PRODUCTION_SOURCE_HASHES),
    }
    metadata = {
        'control_type': CONTROL_TYPE,
        'definition_version': CONTROL_VERSION,
        'n': n, 'parameters': parameters, 'inputs': hashes,
        'source_support': source_meta,
        'statistics': {
            'sum_convention': 'all directed i!=j CSR entries after canonicalization; implicit q=0 included',
            'weighting': 'pre-reliability-gate CSW sparse graph',
            'offdiagonal_entry_count': int(np.count_nonzero(off)),
            'offdiagonal_mass': total, 'q_weighted_mass': numerator, 'qbar': qbar,
            'expected_control_mass': gamma * total,
            'expected_source_mass': (1 - lam) * total + lam * numerator,
        },
        'interpretation': {
            'matches': 'global sparse off-diagonal total mass up to floating-point summation error',
            'does_not_match': ['per-row sparse degrees', 'full-kernel total mass', 'per-query subproblems'],
            'identity_labels_used': False,
            'q_is_identity_for_control': True,
            'same_gate_for_sparse_full_sums_and_bridges': True,
        },
    }
    metadata['contract_sha256'] = _sha_json(metadata)
    return identity, metadata


class UniformMassMatchedIHyRDP(rm.ReliabilityIHyRDP):
    """Use production gate/bridge/solver paths with independent control metadata.

    ``lam`` in this API is the paired original reliability lambda. The inherited
    ``self.lam`` is lambda_eff; ``source_lambda`` records the original. New runner
    code MUST persist ``control_metadata`` and explicitly name this control.
    It must not serialize the identity Q through ordinary/CB save_support.
    """

    def __init__(self, graph, full, cameras, *, rho=0.5, lam=0.5,
                 source_support, source_support_metadata,
                 alpha=5.0, beta=25.0, step=5, expected_contract_sha256=None):
        identity, metadata = prepare_uniform_control(
            graph, full, cameras, rho=rho, lam=lam,
            source_support=source_support,
            source_support_metadata=source_support_metadata)
        if expected_contract_sha256 is not None and expected_contract_sha256 != metadata['contract_sha256']:
            raise ValueError('Uniform control contract SHA-256 mismatch')
        super().__init__(graph, full, cameras, eta=metadata['parameters']['rho'],
                         support=identity, lam=metadata['parameters']['lambda_effective'],
                         alpha=alpha, beta=beta, step=step, support_metadata=None)
        self.control_metadata = metadata
        self.source_lambda = metadata['parameters']['source_lambda']
        self.input_hashes.update(full_sha256=metadata['inputs']['full_sha256'],
                                 control_contract_sha256=metadata['contract_sha256'])

