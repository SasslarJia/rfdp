"""Per-frontier geometry and per-weight linear systems for converged F1.

Candidates and progress estimates never authorize output. The final nonlinear
certificate remains in converged.py. Nothing is cached across frontiers.
"""
import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

from . import f1


class Workspace:
    def __init__(self, w, e, alpha, beta, labels, nc):
        self.w, self.e, self.alpha, self.beta = w, e, alpha, beta
        self.labels, self.nc = labels, nc
        self.h = np.asarray(w.sum(axis=1)).ravel()
        self.root = np.sqrt(self.h)
        self.p = e/(self.h+e)
        self.active = e > 0
        off = w.copy(); off.setdiag(0.); off.eliminate_zeros()
        self.off = off
        self.rows = np.repeat(np.arange(len(e)), np.diff(off.indptr))
        scale = sparse.diags(1/self.root)
        self.laplacian = (-scale@off@scale).tocsr()
        self.diagonal = np.asarray(off.sum(axis=1)).ravel()/self.h
        self.laplacian.setdiag(self.diagonal)
        # Binary input values and exact degree sums are independent of dps.
        with mp.workdps(40):
            self.exact = f1.original.Block(w, e, alpha, beta)

    def weights(self, values, dps=0):
        if dps:
            with mp.workdps(max(40, dps)):
                p = [v/(h+v) for h, v in zip(self.exact.h, self.exact.ee)]
                return [mp.mpf(1) if not v else
                        (1+(p[i]*values[i]-1)**2)**(-mp.mpf(1)/4)
                        for i, v in enumerate(self.exact.ee)]
        with np.errstate(all='ignore'):
            r = self.p*np.asarray(values, dtype=float)-1.
            weights = np.where(self.active, 1/np.sqrt(np.hypot(1., r)), 1.)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            return self.weights(values, 40)
        return weights

    def progress(self, values, weights):
        """Cheap point estimates of stationarity and component-shift error.

        The shift estimate deliberately omits all rounding/inner-solve bounds.
        It schedules certification only; it is NOT a convergence certificate.
        """
        with np.errstate(all='ignore'):
            x = np.asarray(values, dtype=float)
            z = x/self.root
            lap = np.bincount(self.rows, weights=self.off.data*(
                z[self.rows]-z[self.off.indices]), minlength=len(x))/self.root
            r = self.p*x-1.
            omega = 1/np.sqrt(np.hypot(1., r))
            rhs = np.where(self.active, self.alpha*omega, self.beta)
            gradient = lap+np.where(self.active, self.alpha*omega*r, -self.beta)
            residual = np.max(np.abs(gradient)/rhs)
            a = np.asarray(weights, dtype=float)
            s = a[self.active]*r[self.active]
            inverse = s*np.sqrt((s*s+np.hypot(s*s, 2.))/2.)
            shifts = np.abs((inverse-r[self.active]) / (
                self.p[self.active]*self.root[self.active]))
            worst = np.zeros(self.nc)
            np.maximum.at(worst, self.labels[self.active], shifts)
            shift = np.max(worst[self.labels]*self.root/x)
        return (float(residual) if np.isfinite(residual) else np.inf,
                float(shift) if np.isfinite(shift) else np.inf)

    def system(self, weights, stats):
        return LinearSystem(self, weights, stats)


class LinearSystem:
    def __init__(self, workspace, weights, stats):
        self.workspace, self.weights, self.stats = workspace, list(weights), stats
        omega = np.asarray(weights, dtype=float)
        self.matrix = workspace.laplacian.copy()
        self.matrix.setdiag(workspace.diagonal+workspace.alpha*workspace.p*omega)
        self.rhs = np.where(workspace.active, workspace.alpha*omega, workspace.beta)
        self.factor = None

    def factorize(self):
        if self.factor is None:
            self.factor = splu(self.matrix.tocsc())
            self.stats['additional_factorizations'] += 1
        return self.factor

    def candidate(self, previous, rtol):
        candidate = None
        if len(self.rhs) > 128:
            def count(_):
                self.stats['cg_iterations'] += 1
            self.stats['cg_calls'] += 1
            self.stats['linear_solve_calls'] += 1
            diagonal = self.matrix.diagonal()
            preconditioner = sparse.diags(1/diagonal)
            candidate, flag = cg(self.matrix, self.rhs,
                x0=None if previous is None else np.asarray(previous, dtype=float),
                M=preconditioner, rtol=rtol, atol=0., maxiter=10000, callback=count)
            if flag != 0 or not np.isfinite(candidate).all():
                candidate = None
        if candidate is None:
            self.stats['direct_calls'] += 1
            self.stats['linear_solve_calls'] += 1
            try:
                candidate = self.factorize().solve(self.rhs)
            except (RuntimeError, ValueError, ArithmeticError):
                return None
        return candidate if np.isfinite(candidate).all() else None

    def block(self):
        return _CachedWeightedBlock(self)


class _CachedWeightedBlock(f1.WeightedBlock):
    def __init__(self, system):
        geometry = system.workspace.exact
        for name in ('w', 'e', 'n', 'alpha', 'beta', 'rows', 'h', 'ee'):
            setattr(self, name, getattr(geometry, name))
        self.system, self.weights = system, system.weights
        self.b = [f1.original.exact_float(self.alpha)*self.weights[i]
                  if v > 0 else f1.original.exact_float(self.beta)
                  for i, v in enumerate(self.e)]

    def rounded_matrix(self):
        return self.system.matrix.tocsc()

    def factorize(self):
        return self.system.factorize()
