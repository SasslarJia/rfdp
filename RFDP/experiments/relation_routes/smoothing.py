"""Certified quadratic smoothing inside the unchanged dynamic HyRDP frontier.

The arithmetic target is the real formula on the supplied binary W/e, including
the camera budgets or target-cosine square roots. A rounded sparse matrix is
only a solver preconditioner, never the reference used by the certificate.
TH uses a sparse operator minus one rank-one term; no dense AA block is formed.
"""
import hashlib
import json
import time

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

from experiments.continuous_fitting.certified import digest_scores, exact_twins
from experiments.internal_nonlinear.runtime import accurate_hybrid_ties as original

SOURCES = ('camera_channel', 'th_target')
SOLVER_ID = 'dynamic-relation-smoothing-certified-v1'
TOL = 1e-8
PRECISIONS = (24, 40, 80, 120)
ArithmeticFailure = original.ArithmeticFailure


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def _mul(a, b, c, d):
    """Outward product of intervals [a,b] and [c,d], any signs."""
    terms = np.asarray([a*c, a*d, b*c, b*d])
    return down(np.min(terms, axis=0)), up(np.max(terms, axis=0))


def _sum_bounds(lo, hi):
    """Pairwise outward sum, with O(n) work rather than a Python scalar loop."""
    lo, hi = np.asarray(lo).ravel().copy(), np.asarray(hi).ravel().copy()
    if not len(lo):
        return 0., 0.
    while len(lo) > 1:
        if len(lo) % 2:
            lo = np.append(lo, 0.); hi = np.append(hi, 0.)
        lo = down(lo[::2]+lo[1::2]); hi = up(hi[::2]+hi[1::2])
    return lo[0], hi[0]


class _SparseRankOneFactor:
    def __init__(self, base, vector):
        self.lu = splu(base.tocsc())
        self.vector = vector
        self.response = self.lu.solve(vector) if np.any(vector) else None
        self.denominator = (1.-float(vector@self.response)
                            if self.response is not None else 1.)
        if not np.isfinite(self.denominator) or self.denominator <= 0:
            raise ArithmeticFailure('Unresolved rank-one denominator; no ridge')

    def solve(self, rhs):
        value = self.lu.solve(rhs)
        if self.response is not None:
            if np.ndim(value) == 1:
                value += self.response*(self.vector@value/self.denominator)
            else:
                value += self.response[:, None]*(self.vector@value/self.denominator)[None, :]
        return value


class System:
    """Float64 candidate operator and immutable context for exact certification."""
    def __init__(self, w, e, alpha, beta, source, cameras, target_count, theta):
        self.w, self.e, _, self.components = original.validate(w, e, alpha, beta)
        self.n = len(self.e)
        if self.n > 8192:
            raise ArithmeticFailure('Local stop limit 8192; no truncation')
        if source not in SOURCES:
            raise ValueError('Unknown smoothing route')
        self.alpha, self.beta, self.source = float(alpha), float(beta), source
        self.cameras = None
        self.target_count = None
        self.theta = 0.
        if source == 'camera_channel':
            cameras = np.asarray(cameras)
            if cameras.shape != (self.n,) or cameras.dtype.kind not in 'iu':
                raise ValueError('cameras must be one integer per current frontier node')
            self.cameras = cameras.astype(np.int64, copy=True)
        else:
            if (isinstance(target_count, (bool, np.bool_)) or
                    not isinstance(target_count, (int, np.integer)) or
                    not 1 <= target_count <= 2**53):
                raise ValueError('target_count must be the positive integer size of L')
            if not np.isfinite(theta) or not 0 <= theta < 1:
                raise ValueError('theta must satisfy 0 <= theta < 1')
            if np.any(self.w.data > 1) or np.any(self.e > target_count):
                raise ValueError('Target-cosine requires W in [0,1] and e <= |L|')
            self.target_count, self.theta = int(target_count), float(theta)
        self.rowids = np.repeat(np.arange(self.n), np.diff(self.w.indptr))
        self.cols = self.w.indices
        self.off = self.rowids != self.cols
        self.h = np.asarray(self.w.sum(axis=1)).ravel()
        if not np.isfinite(self.h).all() or np.any(self.h <= 0):
            raise ArithmeticFailure('Invalid internal degree')
        self.root = np.sqrt(self.h)
        self.p = self.e/(self.h+self.e)
        self.rhs = np.where(self.e > 0, alpha, beta)
        weights = self.w.data.copy()
        if source == 'camera_channel':
            same = self.cameras[self.rowids] == self.cameras[self.cols]
            self.same = same
            ds = np.bincount(self.rowids, weights=np.where(self.off & same, weights, 0.), minlength=self.n)
            dx = np.bincount(self.rowids, weights=np.where(self.off & ~same, weights, 0.), minlength=self.n)
            d = ds+dx; channels = (ds > 0).astype(int)+(dx > 0).astype(int)
            fs = np.divide(d, channels*ds, out=np.zeros(self.n), where=ds > 0)
            fx = np.divide(d, channels*dx, out=np.zeros(self.n), where=dx > 0)
            left = np.where(same, fs[self.rowids], fx[self.rowids])
            right = np.where(same, fs[self.cols], fx[self.cols])
            weights *= (left+right)/2.
            rank_diagonal = np.zeros(self.n)
            self.vector = np.zeros(self.n)
        else:
            m = self.e/self.target_count
            s = np.sqrt(np.maximum(0., (1.-m)*(1.+m)))
            weights *= (1.-theta)+theta*s[self.rowids]*s[self.cols]
            rank_diagonal = theta*m*np.sum(m)/self.h
            self.vector = np.sqrt(theta)*m/self.root
        weights[~self.off] = 0.
        c = sparse.csr_matrix((weights, self.w.indices.copy(), self.w.indptr.copy()), shape=self.w.shape)
        c.eliminate_zeros()
        # Algebraic off-diagonal sums avoid subtracting nearly equal self loops.
        degree = np.asarray(c.sum(axis=1)).ravel()
        invroot = sparse.diags(1./self.root)
        self.base = (sparse.diags(degree/self.h+rank_diagonal+alpha*self.p)-invroot@c@invroot).tocsr()
        self.diagonal = self.base.diagonal()-self.vector*self.vector
        if not np.isfinite(self.base.data).all() or np.any(self.diagonal <= 0):
            raise ArithmeticFailure('Invalid rounded smoothing operator')
        def action(x):
            return self.base@x-self.vector*(self.vector@x)
        def actions(x):
            return self.base@x-self.vector[:, None]*(self.vector@x)[None, :]
        self.matrix = LinearOperator((self.n, self.n), matvec=action, rmatvec=action,
                                     matmat=actions, dtype=np.float64)
        self._factor = None
        state = dict(source=source, theta=self.theta, target_count=self.target_count,
                     cameras=None if self.cameras is None else self.cameras.tolist())
        digest = hashlib.sha256(original.input_sha(self.w, self.e, alpha, beta).encode())
        digest.update(json.dumps(state, sort_keys=True, separators=(',', ':')).encode())
        self.input_sha256 = digest.hexdigest()

    def factorize(self):
        if self._factor is None:
            self._factor = _SparseRankOneFactor(self.base, self.vector)
        return self._factor


def assemble_system(w, e, alpha=5., beta=25., source='camera_channel',
                    cameras=None, target_count=None, theta=.5):
    return System(w, e, alpha, beta, source, cameras, target_count, theta)


def _row_bounds(system, lo, hi, mask=None):
    w = system.w
    lower, upper = np.zeros(system.n), np.zeros(system.n)
    start, end = w.indptr[:-1], w.indptr[1:]
    for offset in range(int(np.max(end-start))):
        active = start+offset < end
        rows = np.flatnonzero(active); k = start[active]+offset
        if mask is not None:
            selected = mask[k]; rows, k = rows[selected], k[selected]
        lower[rows] = down(lower[rows]+lo[k]); upper[rows] = up(upper[rows]+hi[k])
    return lower, upper


def fast_certificate(system, scores):
    """Outward real-formula residual; no reliance on the rounded operator."""
    w, e = system.w, system.e
    with np.errstate(all='ignore'):
        x = np.asarray([float(v) for v in scores])
        if len(x) != system.n or not np.isfinite(x).all() or np.any(x <= 0):
            return None
        xl, xu = down(x), up(x)
        hl, hu = _row_bounds(system, w.data, w.data)
        if np.any(hl <= 0) or not np.isfinite(hu).all():
            return None
        rl, ru = down(np.sqrt(hl)), up(np.sqrt(hu))
        zl, zu = down(xl/ru), up(xu/rl)
        i, j = system.rowids, system.cols
        ml = mu = None
        if system.source == 'camera_channel':
            sl, su = _row_bounds(system, w.data, w.data, system.off & system.same)
            tl, tu = _row_bounds(system, w.data, w.data, system.off & ~system.same)
            dl, du = down(sl+tl), up(su+tu)
            # Channel counts depend on exact support, not rounded sums.
            ns = np.bincount(i, weights=(system.off & system.same).astype(int), minlength=system.n) > 0
            nx = np.bincount(i, weights=(system.off & ~system.same).astype(int), minlength=system.n) > 0
            count = ns.astype(int)+nx.astype(int)
            fl, fu = [], []
            for lower, upper, present in ((sl, su, ns), (tl, tu, nx)):
                if np.any(lower[present] <= 0):
                    return None
                a, b = np.zeros(system.n), np.zeros(system.n)
                a[present] = np.maximum(0., down(dl[present]/up(count[present]*upper[present])))
                b[present] = up(du[present]/down(count[present]*lower[present]))
                fl.append(a); fu.append(b)
            leftl = np.where(system.same, fl[0][i], fl[1][i])
            leftu = np.where(system.same, fu[0][i], fu[1][i])
            rightl = np.where(system.same, fl[0][j], fl[1][j])
            rightu = np.where(system.same, fu[0][j], fu[1][j])
            cl = np.maximum(0., down(w.data*down(down(leftl+rightl)/2.)))
            cu = up(w.data*up(up(leftu+rightu)/2.))
        else:
            ml = np.maximum(0., down(e/system.target_count))
            mu = np.minimum(1., up(e/system.target_count))
            ml[e == 0] = mu[e == 0] = 0.
            ml[e == system.target_count] = mu[e == system.target_count] = 1.
            sl = np.maximum(0., down(np.sqrt(np.maximum(0., down(1.-up(mu*mu))))))
            su = up(np.sqrt(np.maximum(0., up(1.-down(ml*ml)))))
            sl[ml == 1] = su[ml == 1] = 0.
            low = down(down(1.-system.theta)+down(system.theta*down(sl[i]*sl[j])))
            high = up(up(1.-system.theta)+up(system.theta*up(su[i]*su[j])))
            cl = np.maximum(0., down(w.data*low)); cu = up(w.data*high)
        cl[~system.off] = cu[~system.off] = 0.
        pl, pu = _mul(cl, cu, down(zl[i]-zu[j]), up(zu[i]-zl[j]))
        ll, lu = _row_bounds(system, pl, pu, system.off)
        if system.source == 'th_target' and system.theta:
            sml, smu = _sum_bounds(ml, mu)
            mzl, mzu = _mul(ml, mu, zl, zu)
            dotl, dotu = _sum_bounds(mzl, mzu)
            ql, qu = _mul(sml, smu, zl, zu)
            ql, qu = down(ql-dotu), up(qu-dotl)
            ql, qu = _mul(ml, mu, ql, qu)
            ll = down(ll+down(system.theta*ql)); lu = up(lu+up(system.theta*qu))
        dl = np.maximum(0., down(down(down(system.alpha*hl)*e)/up(hu+e)))
        du = up(up(up(system.alpha*hu)*e)/down(hl+e))
        dl[e == 0] = du[e == 0] = 0.
        ql, qu = _mul(dl, du, zl, zu)
        nl, nu = down(ll+ql), up(lu+qu)
        al, au = _mul(nl, nu, down(1./ru), up(1./rl))
        error = np.maximum(np.abs(down(system.rhs-au)), np.abs(up(system.rhs-al)))
        eta = float(np.max(up(error/system.rhs)))
        if not np.isfinite(eta) or not 0 <= eta <= TOL:
            return None
        lower, upper = down(xl/up(1.+eta)), up(xu/down(1.-eta))
        if np.any(lower <= 0) or not np.isfinite(upper).all():
            return None
        return dict(accepted=True, relative=eta, eta=eta,
                    intervals=list(zip(lower, upper)), backend='outward-float64', dps=0)


class Block(original.Block):
    """Accurate residual correction only after the float64 certificate fails."""
    def __init__(self, system):
        self.system = system
        super().__init__(system.w, system.e, system.alpha, system.beta)
        self.ds = self.dx = None
        if system.source == 'camera_channel':
            self.ds = [original.exact_sum(v for j, v in row if i != j and system.cameras[i] == system.cameras[j])
                       for i, row in enumerate(self.rows)]
            self.dx = [original.exact_sum(v for j, v in row if i != j and system.cameras[i] != system.cameras[j])
                       for i, row in enumerate(self.rows)]

    def _parts(self, interval=False):
        ctx = mp.iv if interval else mp
        point = original.point if interval else (lambda x: x)
        root = [ctx.sqrt(point(v)) for v in self.h]
        d = [point(mp.mpf(self.alpha))*point(h)*point(e)/(point(h)+point(e))
             for h, e in zip(self.h, self.ee)]
        rows, m = [], None
        if self.system.source == 'camera_channel':
            fs, fx = [], []
            for ds, dx in zip(self.ds, self.dx):
                count = int(ds > 0)+int(dx > 0)
                total = point(ds)+point(dx)
                fs.append(total/(count*point(ds)) if ds > 0 else point(mp.mpf(0)))
                fx.append(total/(count*point(dx)) if dx > 0 else point(mp.mpf(0)))
            for i, row in enumerate(self.rows):
                converted = []
                for j, weight in row:
                    if j != i:
                        f = fs if self.system.cameras[i] == self.system.cameras[j] else fx
                        converted.append((j, point(weight)*(f[i]+f[j])/2))
                rows.append(converted)
        else:
            m = [point(v)/self.system.target_count for v in self.ee]
            ss = []
            for v, exact in zip(m, self.ee):
                if exact == self.system.target_count:
                    ss.append(point(mp.mpf(0)))
                else:
                    # 1-m^2 is nonnegative by the validated exact input range.
                    inside = (1-v)*(1+v)
                    if interval:
                        a, b = original.endpoints(inside)
                        inside = mp.iv.mpf([max(mp.mpf(0), a), max(mp.mpf(0), b)])
                    ss.append(ctx.sqrt(inside))
            theta = point(mp.mpf(self.system.theta))
            for i, row in enumerate(self.rows):
                rows.append([(j, point(v)*((1-theta)+theta*ss[i]*ss[j])) for j, v in row if i != j])
        return root, d, rows, m

    def residual(self, x):
        root, d, rows, m = self._parts()
        z = [v/r for v, r in zip(x, root)]
        total = mp.fsum(m) if m is not None else 0
        dot = mp.fsum(a*v for a, v in zip(m, z)) if m is not None else 0
        result = []
        for i, row in enumerate(rows):
            lap = mp.fsum(v*(z[i]-z[j]) for j, v in row)
            if m is not None:
                lap += mp.mpf(self.system.theta)*m[i]*(total*z[i]-dot)
            result.append(self.b[i]-(lap+d[i]*z[i])/root[i])
        return result

    def certify(self, scores, dps):
        previous = mp.iv.dps
        mp.iv.dps = dps
        point, endpoints = original.point, original.endpoints
        try:
            root, d, rows, m = self._parts(interval=True)
            xx = [point(v) for v in scores]
            z = [v/r for v, r in zip(xx, root)]
            total = sum(m, mp.iv.mpf(0)) if m is not None else 0
            dot = sum((a*v for a, v in zip(m, z)), mp.iv.mpf(0)) if m is not None else 0
            residual = []
            for i, row in enumerate(rows):
                lap = sum((v*(z[i]-z[j]) for j, v in row), mp.iv.mpf(0))
                if m is not None:
                    lap += point(mp.mpf(self.system.theta))*m[i]*(total*z[i]-dot)
                residual.append(point(self.b[i])-(lap+d[i]*z[i])/root[i])
            eta = max(endpoints(abs(r)/point(b))[1] for r, b in zip(residual, self.b))
            rel = endpoints(mp.iv.sqrt(sum((abs(r)**2 for r in residual), mp.iv.mpf(0)) /
                                       sum((point(b)**2 for b in self.b), mp.iv.mpf(0))))[1]
            accepted = all(v > 0 and mp.isfinite(v) for v in scores) and rel <= mp.mpf(TOL) and eta < 1
            boxes = [] if not accepted else [(endpoints(v/(1+point(eta)))[0],
                                              endpoints(v/(1-point(eta)))[1]) for v in xx]
            return dict(accepted=accepted, relative=rel, eta=eta, intervals=boxes,
                        backend='outward-mp-row-equation', dps=dps)
        finally:
            mp.iv.dps = previous

    def solve(self, initial, dps):
        target = mp.power(10, -min(60, dps//2))
        history = []
        try:
            factor = None
            def correction(rhs):
                nonlocal factor
                if factor is None:
                    factor = self.system.factorize()
                return factor.solve(rhs)
            values = ([mp.mpf(float(v)) for v in correction(np.asarray(self.b, dtype=float))]
                      if initial is None else list(initial))
            stale, previous = 0, mp.inf
            for iteration in range(13):
                r = self.residual(values)
                relative = max(abs(v)/b for v, b in zip(r, self.b))
                history.append(mp.nstr(relative, 10))
                if relative < target and all(v > 0 and mp.isfinite(v) for v in values):
                    return values, dict(method='accurate-residual-sparse-rank-one-refinement',
                                        iterations=iteration, history=history)
                stale = stale+1 if relative >= previous*mp.mpf('.9') else 0
                if stale >= 3 or iteration == 12:
                    break
                previous = relative
                scale = max(abs(v) for v in r)
                if not scale or not mp.isfinite(scale):
                    break
                delta = correction(np.asarray([float(v/scale) for v in r]))
                if not np.isfinite(delta).all():
                    break
                values = [v+scale*mp.mpf(float(c)) for v, c in zip(values, delta)]
        except (RuntimeError, ValueError, ArithmeticError, ArithmeticFailure):
            pass
        if self.n > original.DENSE_LIMIT:
            raise ArithmeticFailure('Accurate smoothing refinement failed; no large dense fallback')
        root, d, rows, m = self._parts()
        lap = mp.matrix(self.n)
        for i, row in enumerate(rows):
            lap[i, i] = mp.fsum(v for _, v in row)+d[i]
            for j, v in row:
                lap[i, j] -= v
        if m is not None:
            theta, total = mp.mpf(self.system.theta), mp.fsum(m)
            for i in range(self.n):
                lap[i, i] += theta*m[i]*total
                for j in range(self.n):
                    lap[i, j] -= theta*m[i]*m[j]
        z = mp.lu_solve(lap, mp.matrix([r*b for r, b in zip(root, self.b)]))
        return [r*v for r, v in zip(root, z)], dict(method='accurate-small-dense-fallback', history=history)


def _exact_twins(system, i, j):
    if system.source == 'camera_channel' and system.cameras[i] != system.cameras[j]:
        return False
    return exact_twins(system.w, system.e, i, j)


def check_order(system, values, cert, canonicalize=False):
    if cert is None or not cert['accepted']:
        return None
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    pairs = []
    for k, i in enumerate(order[:min(5, len(order))]):
        for j in order[k+1:]:
            if cert['intervals'][i][1] < cert['intervals'][j][0]:
                continue
            if not _exact_twins(system, i, j):
                return None
            if values[i] != values[j]:
                if not canonicalize:
                    return None
                pairs.append((i, j))
    return order, pairs


def certify_saved(w, e, values, alpha=5., beta=25., source='camera_channel',
                  cameras=None, target_count=None, theta=.5):
    """Check the actual saved scores, without correction or re-solving."""
    system = assemble_system(w, e, alpha, beta, source, cameras, target_count, theta)
    if len(values) != system.n:
        raise ValueError('Wrong saved score count')
    cert = fast_certificate(system, values)
    checked = check_order(system, values, cert)
    if checked is not None:
        return dict(prefix=checked[0][:5], relative=float(cert['relative']), dps=0,
                    backend=cert['backend'], input_sha256=system.input_sha256)
    for dps in PRECISIONS:
        with mp.workdps(dps):
            cert = Block(system).certify(values, dps)
            checked = check_order(system, values, cert)
            if checked is not None:
                return dict(prefix=checked[0][:5], relative=float(cert['relative']), dps=dps,
                            backend=cert['backend'], input_sha256=system.input_sha256)
    raise ArithmeticFailure('Saved smoothing scores not certified; no score replacement')


def solve(w, e, alpha=5., beta=25., source='camera_channel', cameras=None,
          target_count=None, theta=.5):
    started = time.monotonic()
    system = assemble_system(w, e, alpha, beta, source, cameras, target_count, theta)
    candidate, method, iterations = None, 'sparse-rank-one-lu', 0
    if system.n > 128:
        def count(_):
            nonlocal iterations
            iterations += 1
        candidate, info = cg(system.matrix, system.rhs, rtol=1e-12, atol=0., maxiter=10000,
                             M=sparse.diags(1./system.diagonal), callback=count)
        if info != 0 or not np.isfinite(candidate).all():
            candidate = None
        else:
            method = 'cg'
    if candidate is None:
        try:
            candidate = system.factorize().solve(system.rhs)
        except (RuntimeError, ArithmeticFailure):
            candidate = None
    values = None if candidate is None else [mp.mpf(float(v)) for v in candidate]
    reports = []
    for dps in (0,)+PRECISIONS:
        with mp.workdps(max(24, dps)):
            block = None if dps == 0 else Block(system)
            if dps:
                try:
                    values, report = block.solve(values, dps)
                    reports.append(report)
                except (ArithmeticFailure, ArithmeticError, ValueError) as exc:
                    reports.append(dict(dps=dps, failure=str(exc)))
                    continue
            if values is None:
                continue
            cert = fast_certificate(system, values) if not dps else block.certify(values, dps)
            checked = check_order(system, values, cert, canonicalize=True)
            if checked is None:
                continue
            _, pairs = checked
            if pairs:
                for i, j in pairs:
                    values[j] = values[i]
                cert = fast_certificate(system, values) if not dps else block.certify(values, dps)
                checked = check_order(system, values, cert)
                if checked is None:
                    continue
            event = dict(solver_id=SOLVER_ID, source=source, size=system.n,
                         components=system.components, dps=dps, candidate_method=method,
                         certificate_backend=cert['backend'], cg_iterations=iterations,
                         refinements=reports, exact_tie_pairs=len(pairs),
                         returned_prefix_local=checked[0][:5],
                         returned_scores_sha256=digest_scores(values),
                         returned_residual_upper=float(cert['relative']),
                         input_sha256=system.input_sha256, target_count=system.target_count,
                         theta=system.theta, seconds=time.monotonic()-started)
            return np.asarray(values, dtype=object), float(cert['relative']), event
    raise ArithmeticFailure('No strict smoothing first-five certificate; no numerical tail: '+str(reports))
