"""One robust fitting update and its matched uniform control.

The reference is the original real-arithmetic solution for the supplied binary
W/e. Its certified intervals are propagated through the weight rule, so the
second certificate is not merely a certificate for rounded plug-in weights.
Saved-score verification reconstructs both stages and never runs a solver.
"""
import time

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

from . import certified as base

original = base.original
SOURCES = ('f1_robust', 'f1_uniform')
SOLVER_ID = 'f1-v2-one-mm-certified-v1'
WEIGHT_RULE = 'relative-default-charbonnier-once-v1'
TOL = base.TOL
PRECISIONS = (0,) + base.PRECISIONS
down, up = base.down, base.up


def _parameters(source, nu):
    if source not in SOURCES or nu not in (1.5, 2.):
        raise ValueError('F1 requires robust/uniform source and nu=1.5 (or nu=2 recovery check)')


def _degree_bounds(w):
    start, end = w.indptr[:-1], w.indptr[1:]
    rows = np.arange(w.shape[0]); lo = np.zeros(len(rows)); hi = lo.copy()
    for offset in range(int(np.max(end-start))):
        mask = start+offset < end; i = rows[mask]; k = start[mask]+offset
        lo[i] = down(lo[i]+w.data[k]); hi[i] = up(hi[i]+w.data[k])
    return lo, hi


def _sum_bounds(lo, hi):
    """Pairwise outward summation, without assuming np.sum is correctly rounded."""
    lo, hi = np.asarray(lo).copy(), np.asarray(hi).copy()
    while len(lo) > 1:
        if len(lo) % 2:
            lo = np.append(lo, 0.); hi = np.append(hi, 0.)
        lo = down(lo[::2]+lo[1::2]); hi = up(hi[::2]+hi[1::2])
    return float(lo[0]), float(hi[0])


def _float_weights(w, e, boxes, source, nu):
    """Enclose ideal coefficients using elementary outward operations."""
    hl, hu = _degree_bounds(w)
    if np.any(hl <= 0): return None
    pl = np.maximum(0., down(e/up(hu+e)))
    pu = np.minimum(1., up(e/down(hl+e)))
    direct = e > 0; pl[~direct] = pu[~direct] = 0.
    lower = np.ones(len(e)); upper = lower.copy()
    if nu == 2.: return lower, upper
    xl = np.asarray([float(t[0]) for t in boxes]); xu = np.asarray([float(t[1]) for t in boxes])
    # Conversion from MP intervals, if supplied, must also be outward.
    xl, xu = down(xl), up(xu)
    rl = down(down(pl*xl)-1.); ru = up(up(pu*xu)-1.)
    smallest = np.where((rl <= 0) & (ru >= 0), 0., np.minimum(np.abs(rl), np.abs(ru)))
    largest = np.maximum(np.abs(rl), np.abs(ru))
    with np.errstate(all='ignore'):
        a = up(1.+up(largest*largest))
        b = np.maximum(1., down(1.+np.maximum(0., down(smallest*smallest))))
        wl = down(1./up(np.sqrt(up(np.sqrt(a)))))
        wu = np.minimum(1., up(1./down(np.sqrt(down(np.sqrt(b))))))
    lower[direct], upper[direct] = wl[direct], wu[direct]
    if source == 'f1_uniform':
        nl, nh = _sum_bounds(down(pl[direct]*lower[direct]), up(pu[direct]*upper[direct]))
        dl, dh = _sum_bounds(pl[direct], pu[direct])
        lower[direct] = down(nl/dh); upper[direct] = min(1., up(nh/dl))
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower <= 0) or np.any(lower > upper)):
        return None
    return lower, upper


def _mp_weights(block, boxes, source, nu):
    point = original.point
    h = [point(v) for v in block.h]; e = [point(v) for v in block.ee]
    p = [v/(u+v) for u, v in zip(h, e)]
    weights = [point(1) for _ in e]
    if nu == 2.: return weights
    for i, v in enumerate(block.ee):
        if v > 0:
            a, b = boxes[i]
            r = p[i]*mp.iv.mpf([a, b])-1
            weights[i] = 1/mp.iv.sqrt(mp.iv.sqrt(1+r**2))
    if source == 'f1_uniform':
        active = [i for i, v in enumerate(block.ee) if v > 0]
        value = sum((p[i]*weights[i] for i in active), point(0))/sum((p[i] for i in active), point(0))
        for i in active: weights[i] = value
    return weights


def nominal_weights(w, e, initial, source='f1_robust', nu=1.5, dps=40):
    """Deterministic candidate coefficients; their intervals define the certificate."""
    _parameters(source, nu)
    with mp.workdps(dps):
        block = original.Block(w, e, 1., 1.)
        p = [v/(h+v) for h, v in zip(block.h, block.ee)]
        weights = [mp.mpf(1) if not v or nu == 2. else
                   (1+(p[i]*initial[i]-1)**2)**(-mp.mpf(1)/4)
                   for i, v in enumerate(block.ee)]
        if source == 'f1_uniform':
            active = [i for i, v in enumerate(block.ee) if v > 0]
            value = mp.fsum(p[i]*weights[i] for i in active)/mp.fsum(p[i] for i in active)
            for i in active: weights[i] = value
        return weights


def assemble_weighted(w, e, weights, alpha=5., beta=25.):
    """Float64 candidate matrix, assembled without subtracting large self loops."""
    h = np.asarray(w.sum(axis=1)).ravel()
    omega = np.asarray(weights, dtype=np.float64)
    off = w.copy(); off.setdiag(0.); off.eliminate_zeros()
    degree = np.asarray(off.sum(axis=1)).ravel()
    scale = sparse.diags(1./np.sqrt(h))
    matrix = (sparse.diags(degree/h+alpha*e/(h+e)*omega)-scale@off@scale).tocsr()
    rhs = np.where(e > 0, alpha*omega, beta)
    return matrix, rhs


class WeightedBlock(original.Block):
    """Reuse the original sparse refinement on a fixed candidate weight vector."""
    def __init__(self, w, e, alpha, beta, weights):
        super().__init__(w, e, alpha, beta)
        self.weights = list(weights)
        self.b = [original.exact_float(alpha)*weights[i] if v > 0 else original.exact_float(beta)
                  for i, v in enumerate(e)]

    def coefficients(self):
        root, d = super().coefficients()
        return root, [v*c for v, c in zip(d, self.weights)]

    def rounded_matrix(self):
        return assemble_weighted(self.w, self.e, self.weights, self.alpha, self.beta)[0].tocsc()


def _initial_certificate(w, e, alpha, beta, values, dps):
    if dps == 0: return base.fast_certificate(w, e, alpha, beta, values, 'original')
    with mp.workdps(dps):
        return base.Block(w, e, alpha, beta, 'original').certify(values, dps)


def _fast_certificate(w, e, alpha, beta, values, weight_bounds):
    with np.errstate(all='ignore'):
        x = np.asarray([float(v) for v in values]); n = len(e)
        if not np.isfinite(x).all() or np.any(x <= 0): return None
        xl, xu = down(x), up(x); hl, hu = _degree_bounds(w)
        if np.any(hl <= 0): return None
        rootl, rootu = down(np.sqrt(hl)), up(np.sqrt(hu))
        zl, zu = down(xl/rootu), up(xu/rootl)
        ll = np.zeros(n); lu = ll.copy(); rows = np.arange(n)
        start, end = w.indptr[:-1], w.indptr[1:]
        for offset in range(int(np.max(end-start))):
            active = start+offset < end; i = rows[active]; k = start[active]+offset
            j, weight = w.indices[k], w.data[k]; active = i != j
            i, j, weight = i[active], j[active], weight[active]
            ll[i] = down(ll[i]+down(weight*down(zl[i]-zu[j])))
            lu[i] = up(lu[i]+up(weight*up(zu[i]-zl[j])))
        wl, wu = weight_bounds
        dl = np.maximum(0., down(down(down(down(alpha*hl)*e)/up(hu+e))*wl))
        du = up(up(up(up(alpha*hu)*e)/down(hl+e))*wu)
        dl[e == 0] = du[e == 0] = 0.
        nl, nu = down(ll+down(dl*zl)), up(lu+up(du*zu))
        al = np.minimum(down(nl/rootl), down(nl/rootu))
        au = np.maximum(up(nu/rootl), up(nu/rootu))
        bl = np.where(e > 0, down(alpha*wl), beta)
        bu = np.where(e > 0, up(alpha*wu), beta)
        if np.any(bl <= 0): return None
        error = np.maximum(np.abs(down(bl-au)), np.abs(up(bu-al)))
        eta = float(np.max(up(error/bl)))
        if not np.isfinite(eta) or not 0 <= eta <= TOL: return None
        lower, upper = down(xl/up(1.+eta)), up(xu/down(1.-eta))
        if np.any(lower <= 0) or not np.isfinite(upper).all(): return None
        return dict(accepted=True, relative=eta, eta=eta, intervals=list(zip(lower, upper)),
                    backend='outward-float64-coupled-weights')


def _mp_certificate(w, e, alpha, beta, values, initial_boxes, source, nu, dps):
    old = mp.iv.dps; mp.iv.dps = dps
    try:
        with mp.workdps(dps):
            block = original.Block(w, e, alpha, beta)
            point, ends = original.point, original.endpoints
            h = [point(v) for v in block.h]; ee = [point(v) for v in block.ee]
            root = [mp.iv.sqrt(v) for v in h]
            weights = _mp_weights(block, initial_boxes, source, nu)
            d = [point(alpha)*v*t/(v+t)*c for v, t, c in zip(h, ee, weights)]
            b = [point(alpha)*weights[i] if v > 0 else point(beta) for i, v in enumerate(e)]
            xx = [point(v) for v in values]; z = [v/r for v, r in zip(xx, root)]
            residual = []
            for i, row in enumerate(block.rows):
                lap = sum((point(weight)*(z[i]-z[j]) for j, weight in row if j != i), point(0))
                residual.append(b[i]-(lap+d[i]*z[i])/root[i])
            if any(ends(v)[0] <= 0 for v in b): return None
            eta = max(ends(abs(r)/v)[1] for r, v in zip(residual, b))
            if (not all(v > 0 and mp.isfinite(v) for v in values)
                    or not mp.isfinite(eta) or eta > mp.mpf(TOL)): return None
            boxes = [(ends(v/(1+point(eta)))[0], ends(v/(1-point(eta)))[1]) for v in xx]
            return dict(accepted=True, relative=eta, eta=eta, intervals=boxes,
                        backend='outward-mp-coupled-weights')
    finally:
        mp.iv.dps = old


def _coupled_certificate(w, e, alpha, beta, values, initial_cert, source, nu, dps):
    if dps == 0:
        bounds = _float_weights(w, e, initial_cert['intervals'], source, nu)
        if bounds is None: return None
        return _fast_certificate(w, e, alpha, beta, values, bounds)
    return _mp_certificate(w, e, alpha, beta, values, initial_cert['intervals'], source, nu, dps)


def _canonicalize(w, e, values, cert, recertify):
    # A W/e transposition preserves the unique original solution and therefore
    # the ideal F1 weights. It also preserves the uniform scalar weight rule.
    result = base.check_order(w, e, values, cert, canonicalize=True)
    if result is None: return None, cert
    _, pairs = result
    if pairs:
        for i, j in pairs: values[j] = values[i]
        cert = recertify(values)
        if cert is None: return None, cert
    return base.check_order(w, e, values, cert), cert


def solve(w, e, alpha=5., beta=25., source='f1_robust', nu=1.5):
    _parameters(source, nu); start = time.monotonic()
    w, e, _, nc = original.validate(w, e, alpha, beta)
    if len(e) > 8192: raise original.ArithmeticFailure('Local stop limit 8192; no truncation')
    initial, _, initial_event = base.solve(w, e, alpha, beta, 'original')
    initial = list(initial); initial_refinements = []; second_refinements = []
    cg_iterations = 0; cg_calls = 0; direct_calls = 0
    for dps in PRECISIONS:
        with mp.workdps(max(40, dps)):
            if dps:
                initial, info = base.Block(w, e, alpha, beta, 'original').solve(initial, dps)
                initial_refinements.append(dict(dps=dps, **info))
            ic = _initial_certificate(w, e, alpha, beta, initial, dps)
            if ic is None or not ic['accepted']: continue
            checked, ic = _canonicalize(w, e, initial, ic,
                lambda v: _initial_certificate(w, e, alpha, beta, v, dps))
            if checked is None: continue
            weights = nominal_weights(w, e, initial, source, nu, max(40, dps))
            matrix, rhs = assemble_weighted(w, e, weights, alpha, beta)
            candidate = None; method = 'reuse-original' if nu == 2. else 'sparse-lu'
            if nu == 2.:
                values = initial.copy()
            else:
                if len(e) > 128:
                    def count(_):
                        nonlocal cg_iterations
                        cg_iterations += 1
                    cg_calls += 1
                    candidate, flag = cg(matrix, rhs, rtol=1e-12, atol=0., maxiter=10000, callback=count)
                    if flag != 0: candidate = None
                    else: method = 'cg'
                if candidate is None:
                    direct_calls += 1
                    try: candidate = splu(matrix.tocsc()).solve(rhs)
                    except RuntimeError: candidate = None
                values = None if candidate is None else [mp.mpf(float(v)) for v in candidate]
                if dps:
                    values, info = WeightedBlock(w, e, alpha, beta, weights).solve(values, dps)
                    second_refinements.append(dict(dps=dps, **info))
            if values is None: continue
            recertify = lambda v: _coupled_certificate(w, e, alpha, beta, v, ic, source, nu, dps)
            cert = recertify(values)
            if cert is None: continue
            checked, cert = _canonicalize(w, e, values, cert, recertify)
            if checked is None: continue
            order = checked[0]
            event = dict(solver_id=SOLVER_ID, source=source, nu=nu, updates=1,
                weight_rule=WEIGHT_RULE, size=len(e), components=nc,
                initial_input_sha256=original.input_sha(w, e, alpha, beta),
                initial_scores=[base.packed(v) for v in initial],
                initial_scores_sha256=base.digest_scores(initial), initial_certificate_dps=dps,
                initial_residual_upper=float(ic['relative']), initial_event=initial_event,
                initial_refinements=initial_refinements, weight_dps=max(40,dps),
                weights=[float(v) for v in weights], weight_points=[base.packed(v) for v in weights],
                weights_sha256=base.digest_scores(weights), dps=dps,
                certificate_backend=cert['backend'], candidate_method=method,
                second_refinements=second_refinements,
                returned_scores_sha256=base.digest_scores(values), returned_prefix_local=order[:5],
                returned_residual_upper=float(cert['relative']),
                linear_systems=1 if nu == 2. else 2,
                cg_calls=cg_calls+int(initial_event['cg_iterations']>0),
                cg_iterations=cg_iterations+initial_event['cg_iterations'],
                direct_calls=direct_calls+int(initial_event['candidate_method']=='sparse-lu'),
                seconds=time.monotonic()-start)
            return np.asarray(values, dtype=object), float(cert['relative']), event
    raise original.ArithmeticFailure('F1 coefficient/first-five certification failed; no numerical tail')


def certify_saved(w, e, values, event, alpha=5., beta=25., source='f1_robust', nu=1.5):
    """Rebuild the actual two-stage evidence. Never solve or replace saved points."""
    _parameters(source, nu); w, e, _, _ = original.validate(w, e, alpha, beta)
    if (len(values) != len(e) or event.get('solver_id') != SOLVER_ID or event.get('source') != source
            or event.get('nu') != nu or event.get('updates') != 1
            or event.get('weight_rule') != WEIGHT_RULE or event.get('size') != len(e)
            or event.get('initial_input_sha256') != original.input_sha(w, e, alpha, beta)
            or event.get('returned_scores_sha256') != base.digest_scores(values)):
        raise ValueError('F1 event/input/returned-score binding mismatch')
    dps = event.get('initial_certificate_dps')
    if dps not in PRECISIONS or event.get('dps') != dps or event.get('weight_dps') != max(40,dps):
        raise ValueError('Invalid F1 arithmetic protocol')
    with mp.workdps(max(40,dps)):
        initial = base.unpack(event['initial_scores'])
        if len(initial) != len(e) or event['initial_scores_sha256'] != base.digest_scores(initial):
            raise ValueError('F1 initial-score binding mismatch')
        ic = _initial_certificate(w, e, alpha, beta, initial, dps)
        if ic is None or base.check_order(w, e, initial, ic) is None:
            raise original.ArithmeticFailure('F1 original-state scores not certified')
        weights = nominal_weights(w, e, initial, source, nu, max(40,dps))
        if (event.get('weights') != [float(v) for v in weights]
                or event.get('weight_points') != [base.packed(v) for v in weights]
                or event.get('weights_sha256') != base.digest_scores(weights)):
            raise ValueError('F1 weights do not follow the saved original solution')
        cert = _coupled_certificate(w, e, alpha, beta, values, ic, source, nu, dps)
        checked = None if cert is None else base.check_order(w, e, values, cert)
        if checked is None:
            raise original.ArithmeticFailure('F1 saved scores not certified for the coupled rule')
        if (event.get('returned_prefix_local') != checked[0][:5]
                or event.get('linear_systems') != (1 if nu == 2. else 2)):
            raise ValueError('F1 returned order/solve-count mismatch')
        return dict(prefix=checked[0][:5], relative=float(cert['relative']), dps=dps,
                    backend=cert['backend'], linear_systems=event['linear_systems'])


def install(source, nu=1.5):
    _parameters(source, nu)
    from MODEL.Paper_RFDP import PaperFullIterative as model
    previous = model.solve_hybrid_sparse
    def wrapped(w, e, alpha=5., beta=25., backend='auto', diagnostics=None):
        values, residual, event = solve(w, e, alpha, beta, source, nu)
        base.EVENTS.append(event)
        if diagnostics is not None:
            diagnostics.update(cg_calls=event['cg_calls'], cg_iterations=event['cg_iterations'],
                               direct_calls=event['direct_calls'], uncertified_cg=0)
        return values, residual
    model.solve_hybrid_sparse = wrapped
    return previous
