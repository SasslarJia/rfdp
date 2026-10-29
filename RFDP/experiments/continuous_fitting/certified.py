"""Certified quadratic fitting with an explicitly chosen target and score coordinate.

Float64 candidates and vectorized outward residuals are tried first. Higher
precision and sparse LU refinement are used only when necessary. The frozen
original Block supplies unchanged matrix/refinement utilities; its hard-source
residual certificate and tie classifier are deliberately not used.
"""
import hashlib
import json
import time

import mpmath as mp
import numpy as np
from scipy.sparse.linalg import cg, splu

from experiments.internal_nonlinear.runtime import accurate_hybrid_ties as original
from .model import assemble_system

SOLVER_ID = 'quadratic-fitting-certified-v3'
TOL = 1e-8
PRECISIONS = (24, 40, 80, 120)
EVENTS = []


def packed(value):
    if isinstance(value, mp.mpf):
        return [int(v) for v in value._mpf_]
    return [int(v) for v in mp.mpf(float(value))._mpf_]


def unpack(values):
    out = []
    for t in values:
        if len(t) != 4 or any(type(v) is not int for v in t):
            raise ValueError('Invalid MP score encoding')
        sign, man, exp, bits = t
        if sign != 0 or man <= 0 or man.bit_length() != bits or not man & 1 or abs(exp) > 100000:
            raise ValueError('Nonpositive/noncanonical MP score')
        out.append(mp.make_mpf(tuple(t)))
    return out


def digest_scores(values):
    return hashlib.sha256(json.dumps([packed(v) for v in values], separators=(',', ':')).encode()).hexdigest()


def down(x): return np.nextafter(x, -np.inf)
def up(x): return np.nextafter(x, np.inf)


def fast_certificate(w, e, alpha, beta, scores, source):
    """Outward elementary operations on exact binary W/e and actual scores."""
    with np.errstate(all='ignore'):
        x = np.asarray([float(v) for v in scores])
        if not np.isfinite(x).all() or np.any(x <= 0): return None
        xl, xu = down(x), up(x)
        start, end = w.indptr[:-1], w.indptr[1:]
        width = int(np.max(end-start)); rows = np.arange(len(e))
        hl = np.zeros(len(e)); hu = hl.copy()
        for offset in range(width):
            active = start+offset < end; i = rows[active]; k = start[active]+offset
            hl[i] = down(hl[i]+w.data[k]); hu[i] = up(hu[i]+w.data[k])
        if np.any(hl <= 0) or not np.isfinite(hu).all(): return None
        rl, ru = down(np.sqrt(hl)), up(np.sqrt(hu))
        zl, zu = (xl, xu) if source=='stationary_measure' else (down(xl/ru), up(xu/rl))
        ll = np.zeros(len(e)); lu = ll.copy()
        for offset in range(width):
            active = start+offset < end; i = rows[active]; k = start[active]+offset
            j = w.indices[k]; weight = w.data[k]; mask = i != j
            i, j, weight = i[mask], j[mask], weight[mask]
            ll[i] = down(ll[i]+down(weight*down(zl[i]-zu[j])))
            lu[i] = up(lu[i]+up(weight*up(zu[i]-zl[j])))
        dl = np.maximum(0., down(down(down(alpha*hl)*e)/up(hu+e)))
        du = up(up(up(alpha*hu)*e)/down(hl+e))
        if source == 'boundary_odds':
            dl = np.maximum(0., down(alpha*e)); du = up(alpha*e)
        dl[e == 0] = 0.; du[e == 0] = 0.
        if source == 'two_step_fitting':
            pl = np.maximum(0., down(e/up(hu+e)))
            pu = np.minimum(1., up(e/down(hl+e)))
            pl[e == 0] = 0.; pu[e == 0] = 0.
            sl = np.zeros(len(e)); su = sl.copy()
            for offset in range(width):
                active = (start+offset < end) & (e == 0)
                i = rows[active]; k = start[active]+offset; j = w.indices[k]
                sl[i] = down(sl[i]+down(w.data[k]*pl[j]))
                su[i] = up(su[i]+up(w.data[k]*pu[j]))
            indirect = e == 0
            dl[indirect] = np.maximum(0., down(down(beta*sl[indirect])/2.))
            du[indirect] = up(up(beta*su[indirect])/2.)
        nl = down(ll+down(dl*zl)); nu = up(lu+up(du*zu))
        denominator_l, denominator_u = (hl,hu) if source=='stationary_measure' else (rl,ru)
        al = np.minimum(down(nl/denominator_l), down(nl/denominator_u))
        au = np.maximum(up(nu/denominator_l), up(nu/denominator_u))
        if source in ('original', 'two_step_fitting','stationary_measure'):
            bl = bu = np.where(e > 0, alpha, beta)
        elif source == 'boundary_odds':
            bl = down(alpha*down(1.+down(e/hu)))
            bu = up(alpha*up(1.+up(e/hl)))
            bl[e == 0] = beta; bu[e == 0] = beta
        else:
            # p is enclosed directly; intersect only its proven probability range.
            pl = np.maximum(0., down(e/up(hu+e)))
            pu = np.minimum(1., up(e/down(hl+e)))
            pl[e == 0] = 0.; pu[e == 0] = 0.
            bl = down(down(alpha*pl)+down(beta*down(1.-pu)))
            bu = up(up(alpha*pu)+up(beta*up(1.-pl)))
        if np.any(bl <= 0): return None
        error = np.maximum(np.abs(down(bl-au)), np.abs(up(bu-al)))
        eta = float(np.max(up(error/bl)))
        if not np.isfinite(eta) or not 0 <= eta <= TOL: return None
        lower, upper = down(xl/up(1.+eta)), up(xu/down(1.-eta))
        if np.any(lower <= 0) or not np.isfinite(upper).all(): return None
        return dict(accepted=True, relative=eta, eta=eta,
                    intervals=list(zip(lower, upper)), backend='outward-float64', dps=0)


class Block(original.Block):
    def __init__(self, w, e, alpha, beta, source):
        super().__init__(w, e, alpha, beta)
        self.source = source
        if source == 'stationary_measure':
            self.b = [h*b for h,b in zip(self.h,self.b)]
        elif source == 'continuous':
            self.b = [(mp.mpf(alpha)*t+mp.mpf(beta)*h)/(h+t)
                      for h, t in zip(self.h, self.ee)]
        elif source == 'boundary_odds':
            self.b = [mp.mpf(alpha)*(h+t)/h if t > 0 else mp.mpf(beta)
                      for h, t in zip(self.h, self.ee)]

    def coefficients(self):
        if self.source == 'stationary_measure':
            _, d = super().coefficients()
            return [mp.mpf(1)]*self.n, d
        if self.source == 'two_step_fitting':
            root, d = super().coefficients()
            p = [t/(v+t) for v,t in zip(self.h,self.ee)]
            for i,row in enumerate(self.rows):
                if not self.ee[i]:d[i] = mp.mpf(self.beta)*mp.fsum(v*p[j] for j,v in row)/2
            return root, d
        if self.source != 'boundary_odds': return super().coefficients()
        return [mp.sqrt(h) for h in self.h], [mp.mpf(self.alpha)*t for t in self.ee]

    def rounded_matrix(self):
        if self.source not in ('boundary_odds','two_step_fitting','stationary_measure'): return super().rounded_matrix()
        return assemble_system(self.w, self.e, self.alpha, self.beta, self.source).matrix.tocsc()

    def certify(self, scores, dps):
        previous = mp.iv.dps; mp.iv.dps = dps
        point, bounds = original.point, original.endpoints
        try:
            h = [point(v) for v in self.h]; ee = [point(v) for v in self.ee]
            root = [mp.iv.sqrt(v) for v in h]
            d = [point(self.alpha)*v*t/(v+t) for v, t in zip(h, ee)]
            b = ([point(self.alpha if t > 0 else self.beta) for t in self.ee]
                 if self.source in ('original','two_step_fitting','stationary_measure') else
                 [(point(self.alpha)*t+point(self.beta)*v)/(v+t) for v, t in zip(h, ee)])
            if self.source == 'boundary_odds':
                d = [point(self.alpha)*t for t in ee]
                b = [point(self.alpha)*(v+t)/v if exact > 0 else point(self.beta)
                     for v,t,exact in zip(h,ee,self.ee)]
            if self.source == 'two_step_fitting':
                p = [t/(v+t) for v,t in zip(h,ee)]
                for i,row in enumerate(self.rows):
                    if not self.ee[i]:
                        d[i] = point(self.beta)*sum((point(v)*p[j] for j,v in row),mp.iv.mpf(0))/2
            xx = [point(v) for v in scores]
            z = xx if self.source=='stationary_measure' else [v/r for v,r in zip(xx,root)]
            denominator = h if self.source=='stationary_measure' else root
            residual = []
            for i, row in enumerate(self.rows):
                lap = sum((point(weight)*(z[i]-z[j]) for j, weight in row if j != i), mp.iv.mpf(0))
                residual.append(b[i]-(lap+d[i]*z[i])/denominator[i])
            eta = max(bounds(abs(r)/v)[1] for r, v in zip(residual, b))
            relative = bounds(mp.iv.sqrt(sum((abs(r)**2 for r in residual), mp.iv.mpf(0)) /
                                         sum((v*v for v in b), mp.iv.mpf(0))))[1]
            positive = all(v > 0 and mp.isfinite(v) for v in scores)
            accepted = positive and relative <= mp.mpf(TOL) and eta < 1
            boxes = [] if not accepted else [(bounds(v/(1+point(eta)))[0],
                                              bounds(v/(1-point(eta)))[1]) for v in xx]
            return dict(accepted=accepted, relative=relative, eta=eta,
                        intervals=boxes, backend='outward-mp-row-equation', dps=dps)
        finally:
            mp.iv.dps = previous


def exact_twins(w, e, i, j):
    """Sufficient exact transposition symmetry of W/e, hence of M and source."""
    if e[i] != e[j] or w[i, i] != w[j, j]: return False
    def outside(row):
        a, b = w.indptr[row:row+2]
        return {int(k):float(v) for k, v in zip(w.indices[a:b], w.data[a:b]) if k not in (i, j)}
    return outside(i) == outside(j)


def check_order(w, e, values, cert, canonicalize=False):
    if not cert['accepted']: return None
    order = sorted(range(len(values)), key=lambda i:(values[i], i)); pairs = []
    boxes = cert['intervals']
    for k, i in enumerate(order[:min(5, len(order))]):
        for j in order[k+1:]:
            if boxes[i][1] < boxes[j][0]: continue
            if not exact_twins(w, e, i, j): return None
            if values[i] != values[j]:
                if not canonicalize: return None
                pairs.append((i, j))
    return order, pairs


def certify_saved(w, e, values, alpha=5., beta=25., source='continuous'):
    """Check actual stored points; never solve, refine or replace them."""
    w, e, _, _ = original.validate(w, e, alpha, beta)
    if source not in ('original', 'continuous', 'boundary_odds', 'two_step_fitting','stationary_measure') or len(values) != len(e):
        raise ValueError('Invalid source or score count')
    cert = fast_certificate(w, e, alpha, beta, values, source)
    checked = None if cert is None else check_order(w, e, values, cert)
    if checked is not None:
        return dict(prefix=checked[0][:5], relative=float(cert['relative']), dps=0, backend=cert['backend'])
    for dps in PRECISIONS:
        with mp.workdps(dps):
            cert = Block(w, e, alpha, beta, source).certify(values, dps)
            checked = check_order(w, e, values, cert)
            if checked is not None:
                return dict(prefix=checked[0][:5], relative=float(cert['relative']), dps=dps, backend=cert['backend'])
    raise original.ArithmeticFailure('Saved points not certified; no score correction')


def solve(w, e, alpha=5., beta=25., source='continuous'):
    start = time.monotonic()
    w, e, _, nc = original.validate(w, e, alpha, beta)
    if len(e) > 8192: raise original.ArithmeticFailure('Local stop limit 8192; no truncation')
    # Exact symmetric preconditioning; this changes neither the target nor
    # which actual returned x values the certificate checks.
    score_scale = None
    if source == 'stationary_measure':
        score_scale = np.sqrt(np.asarray(w.sum(axis=1)).ravel())
        parent = assemble_system(w,e,alpha,beta,'original')
        system = parent._replace(rhs=score_scale*parent.rhs)
    else:
        system = assemble_system(w, e, alpha, beta, source)
    candidate = None; method = 'sparse-lu'; refinements = []; factor = None
    cg_iterations = 0
    if len(e) > 128:
        def count_iteration(_):
            nonlocal cg_iterations
            cg_iterations += 1
        candidate, info = cg(system.matrix, system.rhs, rtol=1e-12, atol=0., maxiter=10000,
                             callback=count_iteration)
        if info != 0: candidate = None
        else: method = 'cg'
    if candidate is None:
        try:
            factor = splu(system.matrix.tocsc()); candidate = factor.solve(system.rhs)
        except RuntimeError:
            candidate = None
    if candidate is not None and score_scale is not None:
        candidate = candidate/score_scale
    values = None if candidate is None else [mp.mpf(float(v)) for v in candidate]
    for dps in (0,)+PRECISIONS:
        with mp.workdps(max(24, dps)):
            block = None if dps == 0 else Block(w, e, alpha, beta, source)
            if not dps and values is None: continue
            if dps:
                values, report = block.solve(values, dps)
                refinements.append(report)
            cert = (fast_certificate(w, e, alpha, beta, values, source) if not dps
                    else block.certify(values, dps))
            if cert is None: continue
            checked = check_order(w, e, values, cert, canonicalize=True)
            if checked is None: continue
            _, pairs = checked
            # A proved symmetry needs equal actual returned values. Re-certify
            # the modified points; a proved tie alone is not a residual bound.
            if pairs:
                for i, j in pairs: values[j] = values[i]
                cert = (fast_certificate(w, e, alpha, beta, values, source) if not dps
                        else block.certify(values, dps))
                checked = None if cert is None else check_order(w, e, values, cert)
                if checked is None: continue
            order = checked[0]
            report = dict(solver_id=SOLVER_ID, source=source, size=len(e), components=nc,
                          dps=dps, certificate_backend=cert['backend'], candidate_method=method,
                          cg_iterations=cg_iterations,
                          refinements=refinements, returned_prefix_local=order[:5],
                          returned_scores_sha256=digest_scores(values),
                          returned_residual_upper=float(cert['relative']),
                          exact_tie_pairs=len(pairs), seconds=time.monotonic()-start)
            return np.asarray(values, dtype=object), float(cert['relative']), report
    raise original.ArithmeticFailure('No strict first-five certificate; no numerical tail')


def install(source='continuous'):
    from MODEL.Paper_RFDP import PaperFullIterative as model
    previous = model.solve_hybrid_sparse
    def wrapped(w, e, alpha=5., beta=25., backend='auto', diagnostics=None):
        scores, residual, event = solve(w, e, alpha, beta, source)
        EVENTS.append(event)
        if diagnostics is not None:
            diagnostics.update(cg_calls=int(event['cg_iterations']>0), cg_iterations=event['cg_iterations'],
                               direct_calls=int(event['candidate_method']=='sparse-lu'), uncertified_cg=0)
        return scores, residual
    model.solve_hybrid_sparse = wrapped
    return previous
