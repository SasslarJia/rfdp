"""D3-v1: remove only the current scoring matrix's fixed diagonal.

The outer graph and boundary weights are unchanged. Positive internal degree
uses the original HyRDP equation; zero internal degree with a positive boundary
has the exact score one. Both parts enter one first-five order certificate.
"""
from fractions import Fraction
from math import isqrt
import time

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, splu

from . import certified as base

original = base.original
SOLVER_ID = 'diagonal-free-hybrid-certified-v1'
SOURCE = 'diagonal_free'
TOL = base.TOL
PRECISIONS = (0,) + base.PRECISIONS


def _prepare(w, e, alpha, beta, source):
    if source != SOURCE:
        raise ValueError('D3 requires source=diagonal_free')
    w = sparse.csr_matrix(w, dtype=np.float64, copy=True)
    w.sort_indices()
    if not w.has_canonical_format:
        raise original.ArithmeticFailure('Duplicate sparse coordinates are not silently summed')
    w.eliminate_zeros()
    e = np.array(e, dtype=np.float64, copy=True)
    if e.ndim != 1 or not len(e) or w.shape != (len(e), len(e)):
        raise original.ArithmeticFailure('Expected a nonempty square system')
    if (not np.isfinite(w.data).all() or not np.isfinite(e).all()
            or np.any(w.data < 0) or np.any(e < 0)
            or not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0):
        raise original.ArithmeticFailure('Invalid input weights, boundary or parameters')
    if (w-w.T).nnz:
        raise original.ArithmeticFailure('Exact symmetry required; no silent symmetrization')
    off = w.copy()
    off.setdiag(0.)
    off.eliminate_zeros()
    # With nonnegative stored weights, empty positive support means exact zero
    # degree. No floating degree threshold decides the analytic branch.
    positive = np.flatnonzero(np.diff(off.indptr) > 0)
    analytic = np.flatnonzero(np.diff(off.indptr) == 0)
    if np.any(e[analytic] == 0):
        raise original.ArithmeticFailure('Zero internal degree without a boundary anchor')
    components, labels = connected_components(off, directed=False)
    if len(np.unique(labels[e > 0])) != components:
        raise original.ArithmeticFailure('Unanchored component; D3 is singular')
    return w, e, off, positive, analytic, labels, components


def _fraction(value):
    """The exact rational value of a stored float or MP point."""
    if isinstance(value, mp.mpf):
        sign, mantissa, exponent, _ = value._mpf_
        numerator = -mantissa if sign else mantissa
        return (Fraction(numerator << exponent) if exponent >= 0
                else Fraction(numerator, 1 << -exponent))
    return Fraction.from_float(float(value))


def _rational_root(value):
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if numerator*numerator == value.numerator and denominator*denominator == value.denominator:
        return Fraction(numerator, denominator)
    return None


def _unit_witnesses(off, e, alpha, beta, values, boxes, labels, analytic):
    """Prove additional exact-one entries by an exact rational residual witness.

    Rational reconstruction only proposes a whole-component vector. Every row
    is then checked with Fraction arithmetic. An unsuccessful reconstruction
    proves nothing; it never creates a tolerance-based tie or changes scores.
    This is a sufficient certificate, not a separate exact linear solver.
    """
    units = set(map(int, analytic))
    needed = {int(labels[i]) for i, (lo, hi) in enumerate(boxes)
              if i not in units and lo <= 1 <= hi}
    aa, bb = _fraction(alpha), _fraction(beta)
    for label in needed:
        ids = list(map(int, np.flatnonzero(labels == label)))
        rows = {i: [(int(off.indices[k]), _fraction(off.data[k]))
                    for k in range(off.indptr[i], off.indptr[i+1])] for i in ids}
        h = {i: sum((v for _, v in rows[i]), Fraction(0)) for i in ids}
        normalized = {}
        rational = True
        for i in ids:
            normalized[i] = []
            for j, weight in rows[i]:
                root = _rational_root(h[i]*h[j])
                if root is None:
                    rational = False
                    break
                normalized[i].append((j, weight/root))
            if not rational:
                break
        if not rational:
            continue
        diagonal = {i: 1 + aa*_fraction(e[i])/(h[i]+_fraction(e[i])) for i in ids}
        points = {i: _fraction(values[i]) for i in ids}
        # These bounds limit proof search, never alter the equation or score
        # tolerance. The accepted witness must satisfy every exact row.
        for bound in (1 << 8, 1 << 16, 1 << 32, 1 << 64):
            witness = {i: value.limit_denominator(bound) for i, value in points.items()}
            if any(v <= 0 for v in witness.values()):
                continue
            if all(diagonal[i]*witness[i]
                   - sum((weight*witness[j] for j, weight in normalized[i]), Fraction(0))
                   == (aa if e[i] > 0 else bb) for i in ids):
                units.update(i for i in ids if witness[i] == 1)
                break
    return units


def _certificate(wp, ep, positive, analytic, values, alpha, beta, dps):
    if any(values[i] != 1 for i in analytic):
        return None
    if len(positive):
        points = [values[i] for i in positive]
        part = (base.fast_certificate(wp, ep, alpha, beta, points, 'original') if not dps
                else base.Block(wp, ep, alpha, beta, 'original').certify(points, dps))
        if part is None or not part['accepted']:
            return None
    else:
        part = dict(accepted=True, relative=mp.mpf(0), eta=mp.mpf(0),
                    intervals=[], backend='analytic-exact-one')
    boxes = [(mp.mpf(1), mp.mpf(1)) for _ in values]
    for i, interval in zip(positive, part['intervals']):
        boxes[i] = interval
    # The positive block's relative residual also bounds the full residual:
    # analytic rows add zero error and positive squared RHS to its denominator.
    return dict(part, intervals=boxes, backend='diagonal-free/'+part['backend'])


def _check_order(off, e, values, cert, units, canonicalize=False):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    changes = {}
    for position, i in enumerate(order[:5]):
        for j in order[position+1:]:
            if cert['intervals'][i][1] < cert['intervals'][j][0]:
                continue
            if i in units and j in units:
                if values[i] != 1 or values[j] != 1:
                    if not canonicalize:
                        return None
                    changes[i] = changes[j] = mp.mpf(1)
            elif base.exact_twins(off, e, i, j):
                if values[i] != values[j]:
                    if not canonicalize:
                        return None
                    changes[j] = changes.get(i, values[i])
            else:
                return None
    return order, list(changes.items())


def _accepted(off, e, wp, ep, positive, analytic, labels, values, alpha, beta, dps,
              canonicalize=False):
    cert = _certificate(wp, ep, positive, analytic, values, alpha, beta, dps)
    if cert is None:
        return None
    units = _unit_witnesses(off, e, alpha, beta, values, cert['intervals'], labels, analytic)
    checked = _check_order(off, e, values, cert, units, canonicalize)
    if checked is None:
        return None
    _, changes = checked
    if changes:
        for i, value in changes:
            values[i] = value
        cert = _certificate(wp, ep, positive, analytic, values, alpha, beta, dps)
        if cert is None:
            return None
        checked = _check_order(off, e, values, cert, units)
        if checked is None:
            return None
    return cert, checked[0], units, len(changes)


def solve(w, e, alpha=5., beta=25., source=SOURCE):
    start = time.monotonic()
    w, e, off, positive, analytic, labels, components = _prepare(w, e, alpha, beta, source)
    if len(e) > 8192:
        raise original.ArithmeticFailure('Local stop limit 8192; no truncation')
    wp, ep = off[positive][:, positive].tocsr(), e[positive]
    candidate = None
    method = 'analytic'
    cg_calls = cg_iterations = direct_calls = 0
    refinements = []
    if len(positive):
        system = base.assemble_system(wp, ep, alpha, beta, 'original')
        if len(positive) > 128:
            cg_calls = 1
            def count_iteration(_):
                nonlocal cg_iterations
                cg_iterations += 1
            candidate, info = cg(system.matrix, system.rhs, rtol=1e-12, atol=0.,
                                 maxiter=10000, callback=count_iteration)
            if info != 0:
                candidate = None
            else:
                method = 'cg'
        if candidate is None:
            direct_calls += 1
            method = 'sparse-lu'
            try:
                candidate = splu(system.matrix.tocsc()).solve(system.rhs)
            except RuntimeError:
                candidate = None
    values = [mp.mpf(1) for _ in e]
    if candidate is not None:
        for i, value in zip(positive, candidate):
            values[i] = mp.mpf(float(value))
    for dps in PRECISIONS:
        with mp.workdps(max(24, dps)):
            if len(positive):
                if not dps and candidate is None:
                    continue
                if dps:
                    block = base.Block(wp, ep, alpha, beta, 'original')
                    points, report = block.solve(None if candidate is None else [values[i] for i in positive], dps)
                    refinements.append(report)
                    candidate = points
                    for i, value in zip(positive, points):
                        values[i] = value
            result = _accepted(off, e, wp, ep, positive, analytic, labels, values,
                               alpha, beta, dps, canonicalize=True)
            if result is None:
                continue
            cert, order, units, changes = result
            event = dict(solver_id=SOLVER_ID, source=source, size=len(e), components=components,
                         input_sha256=original.input_sha(w, e, alpha, beta),
                         scoring_input_sha256=original.input_sha(off, e, alpha, beta),
                         analytic_nodes_local=list(map(int, analytic)),
                         positive_degree_count=len(positive), exact_one_nodes_local=sorted(units),
                         dps=dps, certificate_backend=cert['backend'], candidate_method=method,
                         linear_systems=int(bool(len(positive))), cg_calls=cg_calls,
                         cg_iterations=cg_iterations, direct_calls=direct_calls,
                         refinements=refinements, exact_tie_changes=changes,
                         returned_prefix_local=order[:5], returned_scores_sha256=base.digest_scores(values),
                         returned_residual_upper=float(cert['relative']), seconds=time.monotonic()-start)
            return np.asarray(values, dtype=object), float(cert['relative']), event
    raise original.ArithmeticFailure('D3 first-five order not certified; no numerical tail')


def certify_saved(w, e, values, event, alpha=5., beta=25., source=SOURCE):
    """Validate the actual saved full-U scores, without solving or replacing them."""
    w, e, off, positive, analytic, labels, components = _prepare(w, e, alpha, beta, source)
    if (len(values) != len(e) or event.get('solver_id') != SOLVER_ID
            or event.get('source') != source or event.get('size') != len(e)
            or event.get('input_sha256') != original.input_sha(w, e, alpha, beta)
            or event.get('scoring_input_sha256') != original.input_sha(off, e, alpha, beta)
            or event.get('analytic_nodes_local') != list(map(int, analytic))
            or event.get('positive_degree_count') != len(positive)
            or event.get('components') != components
            or event.get('linear_systems') != int(bool(len(positive)))
            or event.get('returned_scores_sha256') != base.digest_scores(values)):
        raise ValueError('D3 saved-score identity mismatch')
    if not all(mp.isfinite(v) and v > 0 for v in values):
        raise original.ArithmeticFailure('D3 saved points must be finite and positive')
    wp, ep = off[positive][:, positive].tocsr(), e[positive]
    for dps in PRECISIONS:
        with mp.workdps(max(24, dps)):
            result = _accepted(off, e, wp, ep, positive, analytic, labels, values, alpha, beta, dps)
            if result is not None:
                cert, order, units, _ = result
                if event.get('returned_prefix_local') != order[:5]:
                    raise ValueError('D3 saved prefix mismatch')
                if event.get('exact_one_nodes_local') != sorted(units):
                    raise ValueError('D3 saved exact-one declaration mismatch')
                return dict(prefix=order[:5], relative=float(cert['relative']), dps=dps,
                            backend=cert['backend'], linear_systems=int(bool(len(positive))))
    raise original.ArithmeticFailure('D3 saved points not certified; no score correction')


def install(source=SOURCE):
    if source != SOURCE:
        raise ValueError('D3 requires source=diagonal_free')
    from MODEL.Paper_RFDP import PaperFullIterative as model
    previous = model.solve_hybrid_sparse
    def wrapped(w, e, alpha=5., beta=25., backend='auto', diagnostics=None):
        scores, residual, event = solve(w, e, alpha, beta, source)
        base.EVENTS.append(event)
        if diagnostics is not None:
            diagnostics.update(cg_calls=event['cg_calls'], cg_iterations=event['cg_iterations'],
                               direct_calls=event['direct_calls'], uncertified_cg=0)
        return scores, residual
    model.solve_hybrid_sparse = wrapped
    return previous
