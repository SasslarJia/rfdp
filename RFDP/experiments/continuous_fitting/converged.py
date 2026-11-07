"""Converged relative-residual F1 and its converged-reference uniform control.

Only the final nonlinear target, score enclosure and ordered first five can
authorize a return.  A frozen positive-weight linear system supplies an exact
inner solution enclosure; componentwise sqrt(h) shifts then enclose the unique
nonlinear minimizer.  Thus the certificate does not assume that preceding
floating-point MM iterates were exact minimizers of their surrogate problems.
Saved-score verification reconstructs this evidence without running a solver.
"""
import time

import mpmath as mp
import numpy as np
from scipy.sparse.linalg import cg, splu

from . import certified as base
from . import f1

original = base.original
SOURCES = ('f1_robust_converged', 'f1_uniform_converged')
SOLVER_ID = 'f1-converged-nonlinear-shift-certified-v1'
WEIGHT_RULE = 'relative-default-charbonnier-converged-nu1p5-v1'
NU = 1.5
TOL = base.TOL
PRECISIONS = base.PRECISIONS
# A resource guard, never a truncated approximation or a retrieval parameter.
MAX_MM_ITERATIONS = 1000
MAX_LOCAL_N = 8192


def _parameters(source, max_mm_iterations=None):
    if source not in SOURCES:
        raise ValueError('Unknown converged F1 source')
    if max_mm_iterations is not None and (type(max_mm_iterations) is not int
                                          or max_mm_iterations < 0):
        raise ValueError('max_mm_iterations must be a nonnegative integer')


def _upper_float(value):
    """Keep reported float upper bounds outward as well as the MP decision."""
    value = float(value)
    return value if not np.isfinite(value) else float(np.nextafter(value, np.inf))


def _positive(values):
    return all(v > 0 and mp.isfinite(v) for v in values)


def _weight_points(w, e, values, dps):
    if dps:
        return f1.nominal_weights(w, e, values, 'f1_robust', NU, max(40, dps))
    # This is only a candidate MM update.  It is never the final certificate.
    with np.errstate(all='ignore'):
        h = np.asarray(w.sum(axis=1)).ravel()
        p = e/(h+e)
        r = p*np.asarray(values, dtype=float)-1.
        weights = np.where(e > 0, 1./np.sqrt(np.hypot(1., r)), 1.)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        return f1.nominal_weights(w, e, values, 'f1_robust', NU, 40)
    return [mp.mpf(float(v)) for v in weights]


def _screen(w, e, values, alpha, beta):
    """Sparse, noncertifying stationarity screen; cannot authorize output."""
    with np.errstate(all='ignore'):
        x = np.asarray(values, dtype=float)
        h = np.asarray(w.sum(axis=1)).ravel()
        root = np.sqrt(h); z = x/root
        rows = np.repeat(np.arange(len(e)), np.diff(w.indptr))
        lap = np.bincount(rows, weights=w.data*(z[rows]-z[w.indices]), minlength=len(e))/root
        r = e/(h+e)*x-1.
        weight = 1./np.sqrt(np.hypot(1., r))
        rhs = np.where(e > 0, alpha*weight, beta)
        gradient = lap+np.where(e > 0, alpha*weight*r, -beta)
        answer = float(np.max(np.abs(gradient)/rhs))
    return answer if np.isfinite(answer) else np.inf


def _step_change(previous, current, dps):
    with mp.workdps(max(40, dps)):
        if not _positive(current):
            return mp.inf
        return max(abs(x-y)/x for x, y in zip(current, previous))


def _interval_geometry(block):
    point = original.point
    h = [point(v) for v in block.h]
    root = [mp.iv.sqrt(v) for v in h]
    p = [point(v)/(h[i]+point(v)) for i, v in enumerate(block.ee)]
    return root, p


def _laplacian(block, values, root):
    point = original.point
    z = [point(v)/r for v, r in zip(values, root)]
    return [sum((point(weight)*(z[i]-z[j]) for j, weight in row if j != i),
                point(0))/root[i] for i, row in enumerate(block.rows)]


def _linear_enclosure(block, values, root, p, weights):
    """One bound valid for every exact coefficient vector in weights.

    In particular it is valid for the unknown scalar mean of the nonlinear
    reference weights.  The M-matrix comparison uses a positive right-hand
    side and the actual stored points, not a rounded assembled matrix.
    """
    point, ends = original.point, original.endpoints
    if not _positive(values):
        return None
    lap = _laplacian(block, values, root)
    residual, rhs = [], []
    for i, v in enumerate(block.ee):
        b = point(block.alpha)*weights[i] if v > 0 else point(block.beta)
        if ends(b)[0] <= 0:
            return None
        residual.append(b*(1-p[i]*point(values[i]))-lap[i] if v > 0 else b-lap[i])
        rhs.append(b)
    eta = max(ends(abs(r)/b)[1] for r, b in zip(residual, rhs))
    if not mp.isfinite(eta) or eta >= 1:
        return None
    boxes = [(ends(point(v)/(1+point(eta)))[0],
              ends(point(v)/(1-point(eta)))[1]) for v in values]
    return dict(eta=eta, intervals=boxes, laplacian=lap)


def _score_error(values, boxes):
    point, ends = original.point, original.endpoints
    if any(lo <= 0 or not mp.isfinite(hi) for lo, hi in boxes):
        return mp.inf
    return max(ends(abs(point(v)-mp.iv.mpf([lo, hi]))/point(lo))[1]
               for v, (lo, hi) in zip(values, boxes))


def _phi(value):
    return value/mp.iv.sqrt(mp.iv.sqrt(1+value**2))


def _phi_inverse(value):
    return value*mp.iv.sqrt((value**2+mp.iv.sqrt(value**4+4))/2)


def _reference_certificate(w, e, alpha, beta, values, weights, labels, nc, dps):
    """Enclose the nonlinear root through exact inner-solution shifts.

    For the exact frozen-weight solution x, -(Kx)_i/alpha = a_i(p_i*x_i-1)
    on A.  This identity avoids subtracting graph terms in the shift rule.
    Floating interval endpoints enclose exact sub/supersolutions; the endpoints
    themselves are not asserted to have a sign-certified nonlinear residual.
    """
    old = mp.iv.dps; mp.iv.dps = dps
    try:
        with mp.workdps(dps):
            point, ends = original.point, original.endpoints
            block = original.Block(w, e, alpha, beta)
            root, p = _interval_geometry(block)
            a = [point(v) for v in weights]
            inner = _linear_enclosure(block, values, root, p, a)
            if inner is None:
                return None
            shifts = [[] for _ in range(nc)]
            for i, v in enumerate(block.ee):
                if v > 0:
                    lo, hi = inner['intervals'][i]
                    r = p[i]*mp.iv.mpf([lo, hi])-1
                    tau = (_phi_inverse(a[i]*r)-r)/(p[i]*root[i])
                    shifts[int(labels[i])].append(ends(tau))
            ranges = [(min(v[0] for v in component), max(v[1] for v in component))
                      for component in shifts]
            boxes = []
            for i, (lo, hi) in enumerate(inner['intervals']):
                lower_shift, upper_shift = ranges[int(labels[i])]
                x = mp.iv.mpf([lo, hi])
                boxes.append((ends(x+point(lower_shift)*root[i])[0],
                              ends(x+point(upper_shift)*root[i])[1]))
            relative, relative_lower = mp.mpf(0), mp.mpf(0)
            for i, v in enumerate(block.ee):
                r = p[i]*point(values[i])-1
                weight = 1/mp.iv.sqrt(mp.iv.sqrt(1+r**2)) if v > 0 else point(1)
                b = point(alpha)*weight if v > 0 else point(beta)
                gradient = inner['laplacian'][i]+point(alpha)*_phi(r) if v > 0 else inner['laplacian'][i]-b
                lo, hi = ends(abs(gradient)/b)
                relative = max(relative, hi); relative_lower = max(relative_lower, lo)
            error = _score_error(values, boxes)
            accepted = relative <= mp.mpf(TOL) and error <= mp.mpf(TOL)
            return dict(accepted=accepted, relative=relative, relative_lower=relative_lower,
                        relative_score_error=error, eta=inner['eta'], intervals=boxes,
                        backend='outward-mp-nonlinear-sqrt-degree-shifts', dps=dps)
    finally:
        mp.iv.dps = old


def _uniform_certificate(w, e, alpha, beta, values, reference_cert, dps):
    old = mp.iv.dps; mp.iv.dps = dps
    try:
        with mp.workdps(dps):
            block = original.Block(w, e, alpha, beta)
            root, p = _interval_geometry(block)
            weights = f1._mp_weights(block, reference_cert['intervals'], 'f1_uniform', NU)
            inner = _linear_enclosure(block, values, root, p, weights)
            if inner is None:
                return None
            error = _score_error(values, inner['intervals'])
            active = int(np.flatnonzero(e > 0)[0])
            return dict(accepted=inner['eta'] <= mp.mpf(TOL) and error <= mp.mpf(TOL),
                        relative=inner['eta'], eta=inner['eta'], relative_score_error=error,
                        intervals=inner['intervals'],
                        uniform_weight_interval=original.endpoints(weights[active]),
                        backend='outward-mp-converged-reference-uniform', dps=dps)
    finally:
        mp.iv.dps = old


def _canonicalize(w, e, values, cert, recertify):
    if cert is None:
        return None, cert
    checked = base.check_order(w, e, values, cert, canonicalize=True)
    if checked is None:
        return None, cert
    # Exact W/e transpositions preserve the unique nonlinear root.  They also
    # preserve the scalar-weight uniform target, regardless of the last inner
    # system's numerical weights.  Recheck all modified returned points.
    for i, j in checked[1]:
        values[j] = values[i]
    if checked[1]:
        cert = recertify(values)
        if cert is None:
            return None, cert
    return base.check_order(w, e, values, cert), cert


def _candidate(w, e, alpha, beta, weights, stats):
    matrix, rhs = f1.assemble_weighted(w, e, weights, alpha, beta)
    candidate = None
    if len(e) > 128:
        def count(_):
            stats['cg_iterations'] += 1
        stats['cg_calls'] += 1; stats['linear_solve_calls'] += 1
        candidate, flag = cg(matrix, rhs, rtol=1e-12, atol=0., maxiter=10000, callback=count)
        if flag != 0 or not np.isfinite(candidate).all():
            candidate = None
    if candidate is None:
        stats['direct_calls'] += 1; stats['linear_solve_calls'] += 1
        try:
            candidate = splu(matrix.tocsc()).solve(rhs)
        except (RuntimeError, ValueError, ArithmeticError):
            candidate = None
    if candidate is None or not np.isfinite(candidate).all():
        return None
    return [mp.mpf(float(v)) for v in candidate]


def _refine(w, e, alpha, beta, weights, values, dps, stats, stage):
    stats['linear_solve_calls'] += 1
    stats['refinement_solve_calls'] += 1
    stats['direct_calls'] += 1
    with mp.workdps(dps):
        values, report = f1.WeightedBlock(w, e, alpha, beta, weights).solve(values, dps)
    stats['refinements'].append(dict(stage=stage, mm_iteration=stats['mm_iterations'],
                                     dps=dps, **report))
    return values


def _certificate_record(values, weights, cert):
    return dict(dps=cert['dps'], backend=cert['backend'],
                scores_sha256=base.digest_scores(values),
                frozen_weight_points=[base.packed(v) for v in weights],
                frozen_weights_sha256=base.digest_scores(weights),
                nonlinear_residual_upper=_upper_float(cert['relative']),
                relative_score_error_upper=_upper_float(cert['relative_score_error']),
                inner_linear_residual_upper=_upper_float(cert['eta']),
                ordered_first_five_certified=True,
                enclosure='exact-inner-solution-component-sqrt-h-shifts')


def _uniform_attempt(w, e, alpha, beta, reference, reference_cert, dps, stats):
    stats['uniform_systems'] += 1
    weights = f1.nominal_weights(w, e, reference, 'f1_uniform', NU, max(40, dps))
    values = _candidate(w, e, alpha, beta, weights, stats)
    recertify = lambda v: _uniform_certificate(w, e, alpha, beta, v, reference_cert, dps)
    if values is not None:
        cert = recertify(values)
        checked, cert = _canonicalize(w, e, values, cert, recertify)
        if checked is not None:
            return values, weights, cert, checked[0]
    values = _refine(w, e, alpha, beta, weights, values, dps, stats, 'uniform')
    cert = recertify(values)
    checked, cert = _canonicalize(w, e, values, cert, recertify)
    return None if checked is None else (values, weights, cert, checked[0])


def solve(w, e, alpha=5., beta=25., source='f1_robust_converged', *,
          max_mm_iterations=MAX_MM_ITERATIONS):
    """MM until the final objective, relative score error and order certify.

    max_mm_iterations only limits resources: exhausting it raises, and does
    not return the current iterate.  nu is fixed to 1.5 by this method identity.
    """
    _parameters(source, max_mm_iterations); started = time.monotonic()
    w, e, labels, nc = original.validate(w, e, alpha, beta)
    if len(e) > MAX_LOCAL_N:
        raise original.ArithmeticFailure('Local stop limit 8192; no truncation')
    initial, _, initial_event = base.solve(w, e, alpha, beta, 'original')
    initial = list(initial); values = list(initial)
    weights = [mp.mpf(1)]*len(e)
    initial_cg = int(len(e) > 128)
    initial_direct = int(initial_event['candidate_method'] == 'sparse-lu')
    stats = dict(mm_iterations=0, uniform_systems=0, cg_calls=initial_cg,
                 cg_iterations=initial_event['cg_iterations'],
                 direct_calls=initial_direct+len(initial_event['refinements']),
                 linear_solve_calls=initial_cg+initial_direct+len(initial_event['refinements']),
                 refinement_solve_calls=len(initial_event['refinements']), refinements=[])
    dps = 0; step = mp.inf; refined = False
    while True:
        # Screening constants affect when certification work is attempted,
        # never its tolerance or the target.  Stagnation also forces a check.
        near = dps or _screen(w, e, values, alpha, beta) <= 32*TOL or step <= 32*TOL
        if near:
            dps = dps or PRECISIONS[0]
            recertify = lambda v: _reference_certificate(w, e, alpha, beta, v, weights, labels, nc, dps)
            cert = recertify(values)
            checked, cert = _canonicalize(w, e, values, cert, recertify)
            if checked is None and not refined:
                values = _refine(w, e, alpha, beta, weights, values, dps, stats, 'reference')
                refined = True
                cert = recertify(values)
                checked, cert = _canonicalize(w, e, values, cert, recertify)
            if checked is not None:
                reference = list(values)
                reference_record = _certificate_record(reference, weights, cert)
                result = None
                if source == SOURCES[1]:
                    result = _uniform_attempt(w, e, alpha, beta, reference, cert, dps, stats)
                if source == SOURCES[0] or result is not None:
                    reference_weights = f1.nominal_weights(w, e, reference, 'f1_robust', NU, max(40, dps))
                    if source == SOURCES[0]:
                        output, final_weights, final_cert, order = reference, reference_weights, cert, checked[0]
                    else:
                        output, final_weights, final_cert, order = result
                    event = dict(solver_id=SOLVER_ID, source=source, nu=NU, weight_rule=WEIGHT_RULE,
                        target='strictly-convex-converged-relative-residual-f1', size=len(e), components=nc,
                        initial_input_sha256=original.input_sha(w, e, alpha, beta),
                        initial_scores=[base.packed(v) for v in initial],
                        initial_scores_sha256=base.digest_scores(initial), initial_event=initial_event,
                        initial_certificate_dps=initial_event['dps'],
                        initial_residual_upper=initial_event['returned_residual_upper'],
                        dps=dps, certificate_backend=final_cert['backend'],
                        certificate=reference_record if source == SOURCES[0] else dict(
                            dps=dps, backend=final_cert['backend'],
                            relative_score_error_upper=_upper_float(final_cert['relative_score_error']),
                            uniform_weight_interval=[base.packed(v) for v in final_cert['uniform_weight_interval']],
                            ordered_first_five_certified=True),
                        weights=[float(v) for v in final_weights],
                        weight_points=[base.packed(v) for v in final_weights],
                        weights_sha256=base.digest_scores(final_weights), weight_dps=max(40, dps),
                        returned_scores_sha256=base.digest_scores(output), returned_prefix_local=order[:5],
                        returned_residual_upper=_upper_float(final_cert['relative']),
                        relative_score_error_upper=_upper_float(final_cert['relative_score_error']),
                        ordered_first_five_certified=True,
                        linear_systems=1+stats['mm_iterations']+stats['uniform_systems'],
                        max_mm_iterations=max_mm_iterations, candidate_method='sparse-mm-with-certified-refinement',
                        seconds=time.monotonic()-started, **stats)
                    if source == SOURCES[1]:
                        event.update(reference_scores=[base.packed(v) for v in reference],
                            reference_scores_sha256=base.digest_scores(reference),
                            reference_certificate=reference_record,
                            reference_weights=[float(v) for v in reference_weights],
                            reference_weight_points=[base.packed(v) for v in reference_weights])
                    return np.asarray(output, dtype=object), _upper_float(final_cert['relative']), event
            # A small MM step at the current arithmetic floor calls for more
            # precision, not for accepting a plateau as convergence.
            if cert is None or step <= mp.power(10, 2-dps//2):
                index = PRECISIONS.index(dps)
                if index+1 == len(PRECISIONS):
                    raise original.ArithmeticFailure('Converged F1 final target/order not certified at precision limit')
                dps = PRECISIONS[index+1]
        if stats['mm_iterations'] >= max_mm_iterations:
            raise original.ArithmeticFailure('Converged F1 MM safety limit reached without final certificate')
        previous = list(values)
        weights = _weight_points(w, e, values, dps)
        stats['mm_iterations'] += 1
        values = _candidate(w, e, alpha, beta, weights, stats)
        refined = False
        if dps or values is None or not _positive(values):
            dps = dps or PRECISIONS[0]
            values = _refine(w, e, alpha, beta, weights, values, dps, stats, 'reference')
            refined = True
        step = _step_change(previous, values, dps)


def _saved_reference(w, e, alpha, beta, values, record, labels, nc):
    dps = record.get('dps')
    if dps not in PRECISIONS or record.get('scores_sha256') != base.digest_scores(values):
        raise ValueError('Converged F1 reference score/arithmetic binding mismatch')
    weights = base.unpack(record['frozen_weight_points'])
    if (len(weights) != len(e) or record.get('frozen_weights_sha256') != base.digest_scores(weights)
            or any(v > 1 or (e[i] == 0 and v != 1) for i, v in enumerate(weights))):
        raise ValueError('Invalid converged F1 frozen-weight evidence')
    cert = _reference_certificate(w, e, alpha, beta, values, weights, labels, nc, dps)
    checked = None if cert is None else base.check_order(w, e, values, cert)
    if checked is None:
        raise original.ArithmeticFailure('Saved reference fails final nonlinear/score/order certificate')
    expected = _certificate_record(values, weights, cert)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Converged F1 saved reference diagnostics do not match its certificate')
    return cert, checked[0]


def certify_saved(w, e, values, event, alpha=5., beta=25., source='f1_robust_converged'):
    """Verify actual saved points, without solving, refining or replacing them."""
    _parameters(source); w, e, labels, nc = original.validate(w, e, alpha, beta)
    if (len(values) != len(e) or event.get('solver_id') != SOLVER_ID or event.get('source') != source
            or event.get('nu') != NU or event.get('weight_rule') != WEIGHT_RULE
            or event.get('size') != len(e) or event.get('components') != nc
            or event.get('initial_input_sha256') != original.input_sha(w, e, alpha, beta)
            or event.get('returned_scores_sha256') != base.digest_scores(values)):
        raise ValueError('Converged F1 input/definition/returned-score binding mismatch')
    dps = event.get('dps')
    if dps not in PRECISIONS or event.get('weight_dps') != max(40, dps):
        raise ValueError('Invalid converged F1 arithmetic protocol')
    initial = base.unpack(event['initial_scores'])
    if len(initial) != len(e) or event.get('initial_scores_sha256') != base.digest_scores(initial):
        raise ValueError('Converged F1 initial-score binding mismatch')
    base.certify_saved(w, e, initial, alpha, beta, 'original')
    if source == SOURCES[0]:
        reference = list(values); record = event['certificate']
    else:
        reference = base.unpack(event['reference_scores']); record = event['reference_certificate']
        if len(reference) != len(e) or event.get('reference_scores_sha256') != base.digest_scores(reference):
            raise ValueError('Converged F1 uniform reference binding mismatch')
    if record.get('dps') != dps:
        raise ValueError('Converged F1 certificate precision mismatch')
    reference_cert, reference_order = _saved_reference(w, e, alpha, beta, reference, record, labels, nc)
    if source == SOURCES[1]:
        reference_weights = f1.nominal_weights(w, e, reference, 'f1_robust', NU, max(40, dps))
        if (event.get('reference_weights') != [float(v) for v in reference_weights]
                or event.get('reference_weight_points') != [base.packed(v) for v in reference_weights]):
            raise ValueError('Converged F1 reported reference weights do not follow the saved reference')
    nominal_source = 'f1_robust' if source == SOURCES[0] else 'f1_uniform'
    weights = f1.nominal_weights(w, e, reference, nominal_source, NU, max(40, dps))
    if (event.get('weight_points') != [base.packed(v) for v in weights]
            or event.get('weights_sha256') != base.digest_scores(weights)
            or event.get('weights') != [float(v) for v in weights]):
        raise ValueError('Converged F1 reported weights do not follow the final reference')
    cert = reference_cert if source == SOURCES[0] else _uniform_certificate(w, e, alpha, beta, values, reference_cert, dps)
    checked = None if cert is None else base.check_order(w, e, values, cert)
    if checked is None:
        raise original.ArithmeticFailure('Saved scores fail the converged target/score/order certificate')
    if (event.get('returned_residual_upper') != _upper_float(cert['relative'])
            or event.get('relative_score_error_upper') != _upper_float(cert['relative_score_error'])
            or event.get('certificate_backend') != cert['backend']
            or event.get('ordered_first_five_certified') is not True):
        raise ValueError('Converged F1 final diagnostics do not match the reconstructed certificate')
    if source == SOURCES[1]:
        expected = dict(dps=dps, backend=cert['backend'],
                        relative_score_error_upper=_upper_float(cert['relative_score_error']),
                        uniform_weight_interval=[base.packed(v) for v in cert['uniform_weight_interval']],
                        ordered_first_five_certified=True)
        if event.get('certificate') != expected:
            raise ValueError('Converged F1 uniform diagnostics do not match its coupled certificate')
    iterations, uniform_systems = event.get('mm_iterations'), event.get('uniform_systems')
    if (type(iterations) is not int or iterations < 0 or type(uniform_systems) is not int
            or uniform_systems < (1 if source == SOURCES[1] else 0)
            or (source == SOURCES[0] and uniform_systems != 0)
            or event.get('linear_systems') != 1+iterations+uniform_systems
            or event.get('returned_prefix_local') != checked[0][:5]):
        raise ValueError('Converged F1 order/iteration accounting mismatch')
    return dict(prefix=checked[0][:5], relative=_upper_float(cert['relative']), dps=dps,
                relative_score_error_upper=_upper_float(cert['relative_score_error']),
                backend=cert['backend'], linear_systems=event['linear_systems'],
                mm_iterations=iterations, ordered_first_five_certified=True)


def install(source='f1_robust_converged'):
    _parameters(source)
    from MODEL.Paper_RFDP import PaperFullIterative as model
    previous = model.solve_hybrid_sparse
    def wrapped(w, e, alpha=5., beta=25., backend='auto', diagnostics=None):
        values, residual, event = solve(w, e, alpha, beta, source)
        base.EVENTS.append(event)
        if diagnostics is not None:
            diagnostics.update(cg_calls=event['cg_calls'], cg_iterations=event['cg_iterations'],
                               direct_calls=event['direct_calls'], uncertified_cg=0)
        return values, residual
    model.solve_hybrid_sparse = wrapped
    return previous
