"""Original HyRDP, with a stable Laplacian assembly and one reusable LU.

No graph weights, objective terms, promotion rules, or certificate tolerances
change. With psi=sqrt(h)*z, solve (L+alpha*h*p)z=sqrt(h)*b. Forming L from
off-diagonal edges avoids subtracting the dominant self loop from one. An
outward extended-float certificate is optional; the original MP fallback stays.
"""
import time

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from . import certified

SOLVER_ID = 'original-stable-laplacian-lu-certified-v1'
MODE = 'stable_lu_certified'


def assemble_laplacian(w, e, alpha):
    h = np.asarray(w.sum(axis=1)).ravel()
    root = np.sqrt(h)
    off = w.copy()
    off.setdiag(0.)
    off.eliminate_zeros()
    degree = np.asarray(off.sum(axis=1)).ravel()
    anchor = alpha * h * e / (h + e)
    matrix = (sparse.diags(degree + anchor) - off).tocsc()
    return matrix, root


class _NormalizedFactor:
    """The same factor expressed in psi coordinates for residual correction."""
    def __init__(self, factor, root):
        self.factor, self.root = factor, root
        self.solves = 0

    def solve(self, rhs):
        self.solves += 1
        return self.root * self.factor.solve(self.root * rhs)


def _accept(w, e, alpha, beta, values, block=None, dps=0):
    values = list(values)

    def certificate():
        if dps:
            return block.certify(values, dps)
        cert = certified.fast_certificate(w, e, alpha, beta, values, 'original', wide=True)
        return cert if cert is not None else certified.fast_certificate(w, e, alpha, beta, values, 'original')

    cert = certificate()
    checked = None if cert is None else certified.check_order(w, e, values, cert, canonicalize=True)
    if checked is None:
        return None
    _, pairs = checked
    if pairs:
        for i, j in pairs:
            values[j] = values[i]
        cert = certificate()
        checked = None if cert is None else certified.check_order(w, e, values, cert)
        if checked is None:
            return None
    return values, cert, checked[0], len(pairs)


def extended_residual(w, e, alpha, beta, values):
    """Candidate correction only; acceptance always uses outward arithmetic."""
    dtype = np.longdouble
    if np.finfo(dtype).nmant <= np.finfo(np.float64).nmant:
        return None
    start, end = w.indptr[:-1], w.indptr[1:]
    width = int(np.max(end - start))
    rows = np.arange(len(e))
    h = np.zeros(len(e), dtype=dtype)
    for offset in range(width):
        active = start + offset < end
        i = rows[active]
        h[i] += w.data[start[active] + offset]
    root = np.sqrt(h)
    x = np.asarray(values, dtype=dtype)
    z = x / root
    lap = np.zeros(len(e), dtype=dtype)
    for offset in range(width):
        active = start + offset < end
        i = rows[active]
        k = start[active] + offset
        j = w.indices[k]
        mask = i != j
        lap[i[mask]] += w.data[k[mask]] * (z[i[mask]] - z[j[mask]])
    anchor = alpha * h * e / (h + e)
    return np.where(e > 0, alpha, beta) - (lap + anchor * z) / root


def solve(w, e, alpha=5., beta=25.):
    start = time.monotonic()
    w, e, _, components = certified.original.validate(w, e, alpha, beta)
    matrix, root = assemble_laplacian(w, e, alpha)
    # One factorization, reused by every correction and precision fallback.
    factor = _NormalizedFactor(splu(matrix), root)
    b = np.where(e > 0, alpha, beta)
    candidate = factor.solve(b)
    if not np.isfinite(candidate).all():
        raise certified.original.ArithmeticFailure('Nonfinite stable LU candidate')
    values = [mp.mpf(float(v)) for v in candidate]
    result = _accept(w, e, alpha, beta, values)
    refinements = []
    dps = 0
    # At most two cheap corrections before the existing precision ladder.
    for _ in range(2):
        if result is not None:
            break
        residual = extended_residual(w, e, alpha, beta, candidate)
        if residual is None or not np.isfinite(residual).all():
            break
        correction = factor.solve(np.asarray(residual, dtype=np.float64))
        updated = np.asarray(candidate.astype(np.longdouble) + correction.astype(np.longdouble), dtype=np.float64)
        refinements.append(dict(method='extended-residual-reused-lu', iterations=1))
        if np.array_equal(updated, candidate) or not np.isfinite(updated).all():
            break
        candidate = updated
        values = [mp.mpf(float(v)) for v in candidate]
        result = _accept(w, e, alpha, beta, values)
    if result is None:
        for dps in certified.PRECISIONS:
            with mp.workdps(dps):
                block = certified.Block(w, e, alpha, beta, 'original')
                block.factorize = lambda: factor
                values, report = block.solve(values, dps)
                refinements.append(report)
                result = _accept(w, e, alpha, beta, values, block, dps)
                if result is not None:
                    break
    if result is None:
        raise certified.original.ArithmeticFailure('Stable original solve did not certify')
    values, cert, order, ties = result
    event = dict(solver_id=SOLVER_ID, source='original', size=len(e), components=components,
                 dps=dps, certificate_backend=cert['backend'], candidate_method='stable-laplacian-sparse-lu',
                 cg_calls=0, cg_iterations=0, direct_calls=1, additional_factorizations=0,
                 linear_systems=1, linear_solve_calls=factor.solves,
                 refinement_solve_calls=factor.solves - 1, refinements=refinements,
                 returned_prefix_local=order[:5], returned_scores_sha256=certified.digest_scores(values),
                 returned_residual_upper=float(cert['relative']), exact_tie_pairs=ties,
                 seconds=time.monotonic() - start)
    return np.asarray(values, dtype=object), float(cert['relative']), event


class StableOriginalSolver:
    def __init__(self, *, events=None):
        self.events = certified.EVENTS if events is None else events

    def reset(self):
        pass

    def __call__(self, w, e, alpha, beta, *, diagnostics, indices, targets, bridges):
        values, residual, event = solve(w, e, alpha, beta)
        self.events.append(event)
        if diagnostics is not None:
            diagnostics.update(cg_calls=0, cg_iterations=0, direct_calls=1, uncertified_cg=0)
        return values, residual
