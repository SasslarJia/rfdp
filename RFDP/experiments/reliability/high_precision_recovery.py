"""High-precision recovery of the SAME float64-assembled hybrid equation.

No ridge, graph changes, symmetrization, threshold relaxation, or ranking tail.
The coefficients of the already assembled float64 CSR matrix are lifted exactly
to arbitrary precision. Independent components use MP residual/accumulation with
double-LU corrections, then bounded MP dense LU if refinement cannot converge.

Crucially, final acceptance uses exact rational dot products of the actual
returned values. longdouble is used only if its quantized vector still passes;
otherwise an object array of mpf values is returned for np.lexsort. A positive
test-vector certificate for the exact rounded Z-matrix is also required. Such a
matrix is an M-matrix; it is SPD only when its coefficients are exactly symmetric.

Integrate in a NEW experiment version, after the old precision layer is installed:
    install(PaperFullIterative, events=PaperPrecision.EVENTS)
This mutates only the module's in-memory solver binding, never source files.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import uuid

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


RESIDUAL_TOL = 1e-8
SOLVER_ID = 'same-float64-exact-rational-mp-v1'
EVENTS = []


class RecoveryFailure(np.linalg.LinAlgError):
    pass


def _canonical(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise RecoveryFailure('Noncanonical CSR input is rejected rather than rounding duplicate-coordinate sums')
    matrix.eliminate_zeros()
    return matrix


def system_sha256(matrix, rhs):
    matrix = _canonical(matrix)
    digest = hashlib.sha256()
    for value, dtype in ((np.asarray(matrix.shape), '<i8'), (matrix.indptr, '<i8'),
                         (matrix.indices, '<i8'), (matrix.data, '<f8'), (rhs, '<f8')):
        digest.update(np.asarray(value, dtype=dtype).tobytes())
    return digest.hexdigest()


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, mp.mpf):
        sign, mantissa, exponent, _ = value._mpf_
        if not mp.isfinite(value):
            raise RecoveryFailure('Nonfinite MP score')
        numerator = -int(mantissa) if sign else int(mantissa)
        return Fraction(numerator << exponent, 1) if exponent >= 0 else Fraction(numerator, 1 << (-exponent))
    if not np.isfinite(value):
        raise RecoveryFailure('Nonfinite returned score')
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _mp_from_fraction(value):
    return mp.mpf(value.numerator) / value.denominator


def _mp_exact(value):
    return _mp_from_fraction(_fraction(value))


def _exact_matvec(matrix, vector):
    return [sum((Fraction.from_float(float(matrix.data[k])) * vector[matrix.indices[k]]
                 for k in range(matrix.indptr[i], matrix.indptr[i + 1])), Fraction(0))
            for i in range(matrix.shape[0])]


def certify_solution(matrix, rhs, scores, residual_tol=RESIDUAL_TOL):
    """Exact binary-rational residual/certificate for the vector actually returned.

    The residual-threshold comparison is done by squaring both sides as exact
    Fractions. MP is used only to format the resulting norm for diagnostics.
    """
    matrix = _canonical(matrix)
    rhs = np.asarray(rhs, dtype=np.float64)
    if matrix.shape != (len(rhs), len(rhs)) or len(scores) != len(rhs):
        raise RecoveryFailure('Matrix/vector shape disagreement')
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all() or np.any(rhs <= 0):
        raise RecoveryFailure('Expected finite matrix and strictly positive RHS')
    row_ids = np.repeat(np.arange(len(rhs)), np.diff(matrix.indptr))
    offdiag = row_ids != matrix.indices
    if np.any(matrix.data[offdiag] > 0) or np.any(matrix.diagonal() <= 0):
        raise RecoveryFailure('Rounded hybrid matrix is not a Z-matrix with positive diagonal')
    x = [_fraction(value) for value in scores]
    b = [Fraction.from_float(float(value)) for value in rhs]
    ax = _exact_matvec(matrix, x)
    residual = [bi - axi for bi, axi in zip(b, ax)]
    numerator = sum((value * value for value in residual), Fraction(0))
    denominator = sum((value * value for value in b), Fraction(0))
    squared = numerator / denominator
    limit = Fraction.from_float(float(residual_tol)) ** 2
    positive = all(value > 0 for value in x)
    positive_ax = all(value > 0 for value in ax)
    eta = max(abs(value) / bi for value, bi in zip(residual, b))
    asymmetry = matrix - matrix.T
    exact_symmetric = asymmetry.nnz == 0
    # Strictly positive v and Av for a Z-matrix prove nonsingular M-matrix.
    # With exact symmetry, D A D is symmetric strictly diagonally dominant,
    # hence SPD. This avoids unreliable near-zero eigenvalue estimates.
    certificate = positive and positive_ax
    with mp.workdps(100):
        relative = float(mp.sqrt(_mp_from_fraction(squared)))
        eta_float = float(_mp_from_fraction(eta))
    result = dict(relative_residual=relative, componentwise_relative_residual=eta_float,
                  residual_threshold=float(residual_tol), exact_residual_pass=bool(squared <= limit),
                  positive_solution=positive, positive_matrix_times_solution=positive_ax,
                  exact_positive_scaling_certificate=certificate,
                  exact_symmetric=exact_symmetric,
                  max_asymmetry=float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.,
                  matrix_class=('SPD-certified' if exact_symmetric else 'nonsymmetric-M-matrix-certified')
                               if certificate else 'not-certified',
                  spd_status=('certified-by-exact-positive-scaling' if certificate else 'not-certified')
                             if exact_symmetric else 'not-symmetric-rounded-matrix',
                  system_sha256=system_sha256(matrix, rhs))
    if certificate and eta < 1:
        bound = [eta * abs(value) / (1 - eta) for value in x]
        order = sorted(range(len(x)), key=lambda i: (x[i], i))
        suffix = [Fraction(0)] * len(x)
        if order:
            suffix[-1] = x[order[-1]] - bound[order[-1]]
            for j in range(len(x) - 2, -1, -1):
                suffix[j] = min(suffix[j + 1], x[order[j]] - bound[order[j]])
        k = min(5, len(x) - 1)
        ordered = all(x[order[j]] + bound[order[j]] < suffix[j + 1]
                      or (eta == 0 and x[order[j]] == suffix[j + 1]) for j in range(k))
        result['ordered_first_five_certified'] = bool(ordered)
    else:
        result['ordered_first_five_certified'] = False
    return result


def assemble_hybrid(w_uu, e_u, alpha=5., beta=25.):
    """Exactly the float64 assembly expression used by PaperPrecision.py."""
    w = sparse.csr_matrix(w_uu, dtype=np.float64)
    e = np.asarray(e_u, dtype=np.float64)
    h = np.asarray(w.sum(axis=1)).ravel()
    if np.any(h <= 0) or np.any(e < 0) or min(alpha, beta) <= 0:
        raise ValueError('Invalid hybrid degrees/parameters')
    dh = sparse.diags(1 / np.sqrt(h))
    matrix = (sparse.diags(1 + alpha * e / (h + e)) - dh @ w @ dh).tocsr()
    rhs = np.where(e > 0, alpha, beta)
    return matrix, rhs


def _mp_residual(matrix, data_mp, b_mp, x):
    return [b_mp[i] - mp.fsum(data_mp[k] * x[matrix.indices[k]]
                            for k in range(matrix.indptr[i], matrix.indptr[i + 1]))
            for i in range(len(b_mp))]


def _block_solve(matrix, rhs, dps, max_refinement, dense_limit):
    matrix = _canonical(matrix)
    with mp.workdps(dps):
        data_mp = [mp.mpf(float(value)) for value in matrix.data]
        b_mp = [mp.mpf(float(value)) for value in rhs]
        norm_b = mp.sqrt(mp.fsum(value * value for value in b_mp))
        target = mp.power(10, -min(40, dps // 2))
        history = []
        try:
            lu = splu(matrix.tocsc())
            initial = lu.solve(np.asarray(rhs, dtype=np.float64))
            if not np.isfinite(initial).all():
                raise RuntimeError('Nonfinite double LU candidate')
            x = [mp.mpf(float(value)) for value in initial]
            stale = 0
            previous = mp.inf
            for iteration in range(max_refinement + 1):
                residual = _mp_residual(matrix, data_mp, b_mp, x)
                relative = mp.sqrt(mp.fsum(value * value for value in residual)) / norm_b
                history.append(float(relative))
                if relative <= target and all(value > 0 for value in x):
                    return x, dict(method='mp-residual-double-lu-refinement', dps=dps,
                                   iterations=iteration, residual_history=history)
                stale = stale + 1 if relative >= previous * mp.mpf('.9') else 0
                if stale >= 3 or iteration == max_refinement:
                    break
                previous = relative
                scale = max(abs(value) for value in residual)
                if not scale:
                    break
                delta = lu.solve(np.asarray([float(value / scale) for value in residual]))
                if not np.isfinite(delta).all():
                    break
                x = [value + scale * mp.mpf(float(change)) for value, change in zip(x, delta)]
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        if len(rhs) > dense_limit:
            raise RecoveryFailure(f'MP refinement failed for block size {len(rhs)} > dense_limit={dense_limit}; no unvalidated Krylov substitution')
        dense = mp.matrix(len(rhs))
        for i in range(len(rhs)):
            for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
                dense[i, int(matrix.indices[k])] = data_mp[k]
        try:
            solution = mp.lu_solve(dense, mp.matrix(b_mp))
        except (ZeroDivisionError, ValueError) as error:
            raise RecoveryFailure('MP LU could not factor the unchanged rounded matrix: ' + str(error)) from error
        x = [solution[i] for i in range(len(rhs))]
        residual = _mp_residual(matrix, data_mp, b_mp, x)
        relative = mp.sqrt(mp.fsum(value * value for value in residual)) / norm_b
        if relative > target or not all(mp.isfinite(value) and value > 0 for value in x):
            raise RecoveryFailure('MP LU did not produce an accurate positive solution; rounded model may be invalid')
        return x, dict(method='mp-dense-lu', dps=dps, iterations=0,
                       residual_history=history, mp_relative_residual=float(relative))


def solve_fixed_system(matrix, rhs, *, dps=80, max_refinement=12, dense_limit=256):
    """Solve only the supplied float64 equation and certify actual output values.

    Large-block refinement failure is explicit, not an implicit model change.
    This bounded implementation intentionally does not use CG on a matrix that
    has not been proved exactly symmetric and positive definite.
    """
    started = time.monotonic()
    matrix = _canonical(matrix)
    rhs = np.asarray(rhs, dtype=np.float64)
    before = system_sha256(matrix, rhs)
    if matrix.shape != (len(rhs), len(rhs)) or not len(rhs):
        raise RecoveryFailure('Expected nonempty square system')
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all() or np.any(rhs <= 0):
        raise RecoveryFailure('Invalid matrix/RHS')
    nc, labels = connected_components(matrix, directed=False)
    blocks = []
    with mp.workdps(dps):
        x = [mp.mpf(0) for _ in rhs]
        for component in range(nc):
            ids = np.flatnonzero(labels == component)
            block = matrix[ids][:, ids].tocsr()
            values, info = _block_solve(block, rhs[ids], dps, max_refinement, dense_limit)
            for i, value in zip(ids, values):
                x[i] = value
            blocks.append(dict(component=component, size=len(ids), nnz=int(block.nnz), **info))
        mp_scores = np.asarray(x, dtype=object)
        mp_certificate = certify_solution(matrix, rhs, mp_scores)
        if not mp_certificate['exact_residual_pass'] or not mp_certificate['exact_positive_scaling_certificate']:
            raise RecoveryFailure('MP candidate failed exact residual or positive-scaling certificate: ' + json.dumps(mp_certificate))
        # Quantization is part of the solver contract. Never report a pre-cast
        # residual as though it belonged to a lower-precision returned array.
        long_scores = np.asarray([np.longdouble(mp.nstr(value, dps + 12)) for value in x], dtype=np.longdouble)
        try:
            quantized = certify_solution(matrix, rhs, long_scores)
        except RecoveryFailure as error:
            quantized = dict(exact_residual_pass=False, exact_positive_scaling_certificate=False,
                             ordered_first_five_certified=False, error=str(error))
        if (quantized['exact_residual_pass'] and quantized['exact_positive_scaling_certificate']
                and (quantized['ordered_first_five_certified'] or not mp_certificate['ordered_first_five_certified'])):
            scores, output_type, certificate = long_scores, 'longdouble', quantized
        else:
            scores, output_type, certificate = mp_scores, 'mpf-object', mp_certificate
    if system_sha256(matrix, rhs) != before:
        raise RecoveryFailure('Matrix/RHS changed during recovery')
    report = dict(kind='same-float64-equation-high-precision-recovery', solver_id=SOLVER_ID,
                  size=len(rhs), nnz=int(matrix.nnz), components=nc, blocks=blocks,
                  output_type=output_type, longdouble_nmant=int(np.finfo(np.longdouble).nmant),
                  certificate=certificate, quantized_longdouble_certificate=quantized,
                  internal_mp_certificate=mp_certificate, seconds=time.monotonic() - started)
    return scores, certificate['relative_residual'], report


def save_evidence(matrix, rhs, scores, report, directory):
    """Persist the target and actual output exactly, without pickle or rounding."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    matrix = _canonical(matrix)
    exact = [_fraction(value) for value in scores]
    numerator = np.asarray([str(value.numerator) for value in exact])
    denominator = np.asarray([str(value.denominator) for value in exact])
    output_digest = hashlib.sha256('\n'.join(f'{n}/{d}' for n, d in zip(numerator, denominator)).encode()).hexdigest()
    name = system_sha256(matrix, rhs)[:20] + '-' + uuid.uuid4().hex
    path = directory / (name + '.npz')
    # Each invocation gets an independent immutable evidence pair. In particular,
    # workers may solve the same system without overwriting timing or hashes.
    with path.open('xb') as stream:
        np.savez_compressed(stream, m_data=matrix.data, m_indices=matrix.indices,
                            m_indptr=matrix.indptr, m_shape=matrix.shape,
                            rhs=np.asarray(rhs, dtype=np.float64),
                            score_numerator=numerator, score_denominator=denominator)
    payload = dict(report, evidence_format='float64-csr-and-exact-rational-scores-v1',
                   score_rational_sha256=output_digest,
                   system_sha256=system_sha256(matrix, rhs),
                   npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    metadata_path = path.with_suffix('.json')
    with metadata_path.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return dict(npz=str(path), metadata=str(metadata_path),
                npz_sha256=payload['npz_sha256'],
                metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())


def verify_evidence(path):
    """Recompute the exact certificate from only persisted coefficients/scores."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['npz_sha256']:
        raise RecoveryFailure('Numerical evidence NPZ hash mismatch')
    with np.load(path, allow_pickle=False) as saved:
        matrix = sparse.csr_matrix((saved['m_data'], saved['m_indices'], saved['m_indptr']),
                                   shape=tuple(saved['m_shape']))
        rhs = saved['rhs']
        exact = [Fraction(int(n), int(d)) for n, d in zip(saved['score_numerator'], saved['score_denominator'])]
        output_digest = hashlib.sha256('\n'.join(f'{value.numerator}/{value.denominator}' for value in exact).encode()).hexdigest()
    if output_digest != metadata['score_rational_sha256']:
        raise RecoveryFailure('Numerical evidence score hash mismatch')
    if system_sha256(matrix, rhs) != metadata['system_sha256']:
        raise RecoveryFailure('Numerical evidence target hash mismatch')
    certificate = certify_solution(matrix, rhs, exact)
    if certificate != metadata['certificate']:
        raise RecoveryFailure('Persisted numerical certificate differs from recomputation')
    if not certificate['exact_residual_pass'] or not certificate['exact_positive_scaling_certificate']:
        raise RecoveryFailure('Persisted numerical result does not satisfy solver contract')
    return certificate


def install(module, *, events=None, evidence_dir=None, dps=80, max_refinement=12, dense_limit=256):
    """Wrap the currently installed production solver without touching files."""
    if getattr(module.solve_hybrid_sparse, '_same_equation_mp_recovery', False):
        return module.solve_hybrid_sparse
    previous = module.solve_hybrid_sparse
    event_sink = EVENTS if events is None else events

    def recovered(w_uu, e_u, alpha=5., beta=25., backend='auto', diagnostics=None):
        try:
            return previous(w_uu, e_u, alpha, beta, backend, diagnostics)
        except np.linalg.LinAlgError as failure:
            matrix, rhs = assemble_hybrid(w_uu, e_u, alpha, beta)
            scores, residual, report = solve_fixed_system(matrix, rhs, dps=dps,
                                                         max_refinement=max_refinement,
                                                         dense_limit=dense_limit)
            report['original_error'] = str(failure)
            if evidence_dir is not None:
                report['evidence'] = save_evidence(matrix, rhs, scores, report, evidence_dir)
            event_sink.append(report)
            if diagnostics is not None:
                diagnostics.update(cg_calls=0, cg_iterations=0, direct_calls=1, uncertified_cg=0)
            return scores, residual

    recovered._same_equation_mp_recovery = True
    recovered._previous_solver = previous
    module.solve_hybrid_sparse = recovered
    return recovered
