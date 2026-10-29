"""Arithmetic adapter for the original HyRDP formula, not a new graph mechanism.

The target is the equation assembled in real arithmetic from the supplied binary
W/e, NOT the previously rounded float64 matrix. Every returned solve is checked
with outward interval arithmetic, including legacy solves that did not fail.
No ridge, discarded bridge, relaxed residual, or numerical ranking tail.
"""
import hashlib
import time
import warnings

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

SOLVER_ID = 'original-formula-certified-exact-ties-v3'
RESIDUAL_TOL = 1e-8
DENSE_LIMIT = 256
PRECISIONS = (100, 160, 240)


class ArithmeticFailure(np.linalg.LinAlgError):
    pass


def input_sha(w, e, alpha, beta):
    digest = hashlib.sha256()
    for v, dtype in ((w.shape, '<i8'), (w.indptr, '<i8'), (w.indices, '<i8'),
                     (w.data, '<f8'), (e, '<f8'), ([alpha, beta], '<f8')):
        digest.update(np.asarray(v, dtype=dtype).tobytes())
    return digest.hexdigest()


def validate(w, e, alpha, beta):
    w = sparse.csr_matrix(w, dtype=np.float64).copy()
    w.sort_indices()
    if not w.has_canonical_format:
        raise ArithmeticFailure('Duplicate sparse coordinates are not silently summed')
    w.eliminate_zeros()
    e = np.asarray(e, dtype=np.float64)
    if w.shape != (len(e), len(e)) or not len(e):
        raise ArithmeticFailure('Expected a nonempty square system')
    if (not np.isfinite(w.data).all() or not np.isfinite(e).all() or
            np.any(w.data < 0) or np.any(e < 0) or
            not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0):
        raise ArithmeticFailure('Invalid input weights, boundary or parameters')
    if (w-w.T).nnz:
        raise ArithmeticFailure('Exact symmetry required; no silent symmetrization')
    nc, labels = connected_components(w, directed=False)
    for c in range(nc):
        if not np.any(e[labels == c] > 0):
            raise ArithmeticFailure('Unanchored component; the original problem is singular')
    if np.any(np.diff(w.indptr) == 0):
        raise ArithmeticFailure('Zero internal degree')
    return w, e, labels, nc


def point(x):
    """Enclose the actual binary value without a decimal/float64 round trip."""
    if isinstance(x, np.floating):
        a, b = x.as_integer_ratio()
        return mp.iv.mpf(a)/mp.iv.mpf(b)
    return mp.iv.mpf(x)


def endpoints(x):
    return mp.make_mpf(x._mpi_[0]), mp.make_mpf(x._mpi_[1])


def exact_float(x):
    return mp.mpf(float(x))


def exact_sum(values):
    total = mp.mpf(0)
    for value in values:
        total = mp.fadd(total, value, exact=True)
    return total


def score_value(x):
    if isinstance(x, mp.mpf):
        return x
    a, b = x.as_integer_ratio()
    return mp.mpf(a)/b


class Block:
    def __init__(self, w, e, alpha, beta):
        self.w, self.e, self.n = w, e, len(e)
        self.alpha, self.beta = alpha, beta
        self.rows = [[(int(w.indices[k]), exact_float(w.data[k]))
                      for k in range(w.indptr[i], w.indptr[i+1])]
                     for i in range(self.n)]
        self.h = [exact_sum(v for _, v in row) for row in self.rows]
        if any(v <= 0 for v in self.h):
            raise ArithmeticFailure('Nonpositive exact internal degree')
        self.b = [exact_float(alpha if v > 0 else beta) for v in e]
        self.ee = [exact_float(v) for v in e]

    def coefficients(self):
        hroot = [mp.sqrt(v) for v in self.h]
        d = [exact_float(self.alpha)*h*e/(h+e) for h, e in zip(self.h, self.ee)]
        return hroot, d

    def residual(self, x):
        root, d = self.coefficients()
        z = [v/r for v, r in zip(x, root)]
        # Laplacian edge differences preserve the constant-nullspace identity.
        return [self.b[i]-(mp.fsum(v*(z[i]-z[j]) for j, v in row)+d[i]*z[i])/root[i]
                for i, row in enumerate(self.rows)]

    def certify(self, x, dps):
        from fast_hybrid_certificate import certify as fast_certificate
        fast = fast_certificate(self.w, self.e, self.alpha, self.beta, x) if dps == PRECISIONS[0] else None
        if fast is not None:
            return fast
        old_dps = mp.iv.dps
        mp.iv.dps = dps
        try:
            h = [point(v) for v in self.h]
            root = [mp.iv.sqrt(v) for v in h]
            ee = [point(v) for v in self.ee]
            d = [point(self.alpha)*v*t/(v+t) for v, t in zip(h, ee)]
            xx = [point(v) for v in x]
            z = [v/r for v, r in zip(xx, root)]
            residual = []
            for i, row in enumerate(self.rows):
                lap = sum((point(v)*(z[i]-z[j]) for j, v in row if j != i), mp.iv.mpf(0))
                residual.append(point(self.b[i])-(lap+d[i]*z[i])/root[i])
            upper = [max(abs(t) for t in endpoints(v)) for v in residual]
            eta_iv = max((abs(v)/point(b) for v, b in zip(residual, self.b)),
                         key=lambda v: endpoints(v)[1])
            eta = endpoints(eta_iv)[1]
            relative_iv = mp.iv.sqrt(sum((point(v)**2 for v in upper), mp.iv.mpf(0))/
                                      sum((point(b)**2 for b in self.b), mp.iv.mpf(0)))
            relative = endpoints(relative_iv)[1]
            positive = all(v > 0 and mp.isfinite(v) for v in x)
            accepted = positive and relative <= mp.mpf(RESIDUAL_TOL) and eta < 1
            intervals = []
            if accepted:
                for value in xx:
                    lo = endpoints(value/(1+point(eta)))[0]
                    hi = endpoints(value/(1-point(eta)))[1]
                    intervals.append((lo, hi))
            return dict(accepted=accepted, relative=relative, eta=eta,
                        intervals=intervals, positive=positive)
        finally:
            mp.iv.dps = old_dps

    def rounded_matrix(self):
        """Float64 correction matrix; subclasses must match their own equation."""
        h = np.asarray(self.w.sum(1)).ravel()
        dh = sparse.diags(1/np.sqrt(h))
        return (sparse.diags(1+self.alpha*self.e/(h+self.e))-dh@self.w@dh).tocsc()

    def solve(self, initial, dps):
        """Accurate residual correction, with bounded dense MP fallback."""
        target = mp.power(10, -min(60, dps//2))
        rounded = self.rounded_matrix()
        history = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                lu = splu(rounded)
            x = ([exact_float(v) for v in lu.solve(np.asarray(self.b, dtype=float))]
                 if initial is None else list(initial))
            stale, previous = 0, mp.inf
            for iteration in range(13):
                r = self.residual(x)
                rel = max(abs(v)/b for v, b in zip(r, self.b))
                history.append(mp.nstr(rel, 10))
                if rel < target and all(v > 0 and mp.isfinite(v) for v in x):
                    return x, dict(method='accurate-residual-sparse-lu-refinement',
                                   iterations=iteration, history=history)
                stale = stale+1 if rel >= previous*mp.mpf('.9') else 0
                if stale >= 3 or iteration == 12:
                    break
                previous = rel
                scale = max(abs(v) for v in r)
                if not scale or not mp.isfinite(scale):
                    break
                delta = lu.solve(np.asarray([float(v/scale) for v in r]))
                if not np.isfinite(delta).all():
                    break
                x = [v+scale*exact_float(c) for v, c in zip(x, delta)]
        except (RuntimeError, ValueError, ArithmeticError, Warning):
            pass
        if self.n > DENSE_LIMIT:
            raise ArithmeticFailure(f'Accurate refinement failed for {self.n} nodes; no dense allocation above {DENSE_LIMIT}')
        root, d = self.coefficients()
        lap = mp.matrix(self.n)
        for i, row in enumerate(self.rows):
            # Omit self loops algebraically, not by subtracting large diagonals.
            lap[i, i] = mp.fsum(v for j, v in row if j != i)+d[i]
            for j, v in row:
                if j != i:
                    lap[i, j] = -v
        z = mp.lu_solve(lap, mp.matrix([r*b for r, b in zip(root, self.b)]))
        return [root[i]*z[i] for i in range(self.n)], dict(method='accurately-assembled-mp-laplacian', history=history)


def ordered_certificate(scores, intervals):
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    for j in range(min(5, len(order)-1)):
        if not intervals[order[j]][1] < min(intervals[k][0] for k in order[j+1:]):
            return False
    return True


def solve(w, e, alpha=5., beta=25., legacy=None):
    start = time.monotonic()
    w, e, labels, nc = validate(w, e, alpha, beta)
    before = input_sha(w, e, alpha, beta)
    original, legacy_error = None, None
    if legacy is not None:
        try:
            original, _ = legacy(w, e, alpha, beta)
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            legacy_error = str(exc)
    reports = []
    for dps in PRECISIONS:
        with mp.workdps(dps):
            scores = [None]*len(e)
            intervals = [None]*len(e)
            blocks = []
            max_residual = mp.mpf(0)
            for c in range(nc):
                ids = np.flatnonzero(labels == c)
                block = Block(w[ids][:, ids].tocsr(), e[ids], alpha, beta)
                candidate = None if original is None else [score_value(v) for v in original[ids]]
                cert = None if candidate is None else block.certify(candidate, dps)
                initial_pass = cert is not None and cert['accepted']
                info = dict(method='legacy-candidate-accurate-formula-verified')
                if not initial_pass or dps != PRECISIONS[0]:
                    try:
                        candidate, info = block.solve(candidate, dps)
                    except (ArithmeticError, ValueError, ArithmeticFailure) as exc:
                        reports.append(dict(dps=dps, component=c, solve_error=str(exc)))
                        break
                    cert = block.certify(candidate, dps)
                if not cert['accepted']:
                    break
                max_residual = max(max_residual, cert['relative'])
                for j, value, interval in zip(ids, candidate, cert['intervals']):
                    scores[j], intervals[j] = value, interval
                blocks.append(dict(size=len(ids), anchors=int(np.count_nonzero(e[ids])),
                                   original_candidate_passed=bool(initial_pass),
                                   residual_upper=mp.nstr(cert['relative'], 18),
                                   eta_upper=mp.nstr(cert['eta'], 18), **info))
            ties = None
            strict_order = all(v is not None for v in scores) and ordered_certificate(scores, intervals)
            if all(v is not None for v in scores) and not strict_order:
                from exact_hybrid_ties import certify_ties
                ties = certify_ties(w, e, alpha, beta, scores, intervals, labels, nc, dps, Block)
                if ties is not None:
                    max_residual = max(max_residual, ties['max_residual'])
            if strict_order or ties is not None:
                if input_sha(w, e, alpha, beta) != before:
                    raise ArithmeticFailure('Inputs changed during solve')
                report = dict(solver_id=SOLVER_ID, arithmetic_contract='exact-binary-W-e-original-formula',
                              input_sha256=before, size=len(e), components=nc, dps=dps,
                              relative_residual_upper=mp.nstr(max_residual, 18),
                              ordered_first_five_certified=True, output_type='mpf-object',
                              legacy_error=legacy_error, blocks=blocks, seconds=time.monotonic()-start,
                              exact_tie_groups=[] if ties is None else ties['groups'],
                              tie_certificate=None if ties is None else ties['certificate'])
                return np.asarray(scores, dtype=object), float(max_residual), report
            reports.append(dict(dps=dps, blocks=blocks, completed_blocks=len(blocks)))
    raise ArithmeticFailure('Original-formula residual/order not certified at bounded precisions: '+str(reports))


def install(module, events):
    """Only alter the caller process; frozen source/baseline processes stay intact."""
    legacy = module.solve_hybrid_sparse
    def adapted(w, e, alpha=5., beta=25., backend='auto', diagnostics=None):
        scores, residual, report = solve(w, e, alpha, beta, legacy)
        report['kind'] = 'accurate-formula-interval-certified'
        events.append(report)
        if diagnostics is not None:
            diagnostics.update(cg_calls=0, cg_iterations=0, direct_calls=1, uncertified_cg=0)
        return scores, residual
    module.solve_hybrid_sparse = adapted
