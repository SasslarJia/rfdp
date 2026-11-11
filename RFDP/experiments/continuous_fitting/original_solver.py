"""Original HyRDP with query-local warm starts and unchanged score certificates.

Only the linear-system algorithm changes. Early exit needs the same outward
residual bound and ordered first-five proof as ``certified.solve``; unresolved
boundaries use its high-precision refinement. No frontier or ranking is pruned.
"""
import time

import mpmath as mp
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, splu

from . import certified
from .model import assemble_system

SOLVER_ID = 'original-warm-certified-v1'
MODE = 'warm_certified'
CHECK_EVERY = 32


class _CertifiedStop(Exception):
    def __init__(self, result):
        self.result = result


def _accept(w, e, alpha, beta, values, dps=0, block=None):
    """Certify actual returned points, including after any proved-tie rewrite."""
    values = list(values)  # Never mutate CG's in-place iterate or the warm cache.
    def certificate():
        return (certified.fast_certificate(w, e, alpha, beta, values, 'original')
                if not dps else block.certify(values, dps))
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


def solve(w, e, alpha=5., beta=25., *, x0=None):
    """Solve one complete original system; x0 is only a numerical initial guess."""
    start = time.monotonic()
    w, e, _, components = certified.original.validate(w, e, alpha, beta)
    system = assemble_system(w, e, alpha, beta, 'original')
    n = len(e)
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64).copy()
        if x0.shape != (n,) or not np.isfinite(x0).all():
            raise ValueError('Invalid warm start')
    iterations = checks = direct_calls = 0
    refinements = []
    candidate = None
    result = None
    info = None
    stopped_early = False
    method = 'sparse-lu'
    if n > 128:
        diagonal = system.matrix.diagonal()
        preconditioner = LinearOperator((n, n), matvec=lambda v: v / diagonal,
                                        dtype=np.float64)
        method = 'warm-jacobi-cg' if x0 is not None else 'jacobi-cg'
        def callback(x):
            nonlocal iterations, checks
            iterations += 1
            if iterations % CHECK_EVERY:
                return
            # Cheap screening only; acceptance always uses exact-input outward
            # arithmetic below. A failed screen merely continues the same CG.
            residual = system.rhs - system.matrix @ x
            if np.max(np.abs(residual) / system.rhs) > certified.TOL:
                return
            checks += 1
            with mp.workdps(24):
                actual = [mp.mpf(float(v)) for v in x.copy()]
                accepted = _accept(w, e, alpha, beta, actual)
            if accepted is not None:
                raise _CertifiedStop(accepted)
        try:
            candidate, info = cg(system.matrix, system.rhs, x0=x0,
                                 M=preconditioner, rtol=1e-12, atol=0.,
                                 maxiter=10000, callback=callback)
        except _CertifiedStop as stopped:
            result = stopped.result
            stopped_early = True
    else:
        try:
            direct_calls += 1
            candidate = splu(system.matrix.tocsc()).solve(system.rhs)
        except RuntimeError:
            candidate = None
    values = (None if candidate is None or not np.isfinite(candidate).all()
              else [mp.mpf(float(v)) for v in candidate])
    dps = 0
    if result is None:
        try:
            for dps in (0,) + certified.PRECISIONS:
                with mp.workdps(max(24, dps)):
                    block = None if not dps else certified.Block(w, e, alpha, beta, 'original')
                    if not dps and values is None:
                        continue
                    if dps:
                        values, report = block.solve(values, dps)
                        refinements.append(report)
                    checks += 1
                    result = _accept(w, e, alpha, beta, values, dps, block)
                    if result is not None:
                        break
        except (certified.original.ArithmeticFailure, RuntimeError, ArithmeticError) as exc:
            refinement_error = type(exc).__name__ + ': ' + str(exc)
        else:
            refinement_error = None
    else:
        refinement_error = None
    if result is None:
        # Recovery only: preserve the old solver's ability to resolve a hard
        # boundary. Never pay for a second cold CG in the successful fast path.
        values, residual, fallback = certified.solve(w, e, alpha, beta, source='original')
        event = dict(fallback)
        event.update(solver_id=SOLVER_ID, candidate_method='legacy-recovery',
                     legacy_recovery=fallback, refinement_error=refinement_error,
                     cg_calls=int(n > 128) + int(fallback['cg_iterations'] > 0),
                     cg_iterations=iterations + fallback['cg_iterations'],
                     direct_calls=direct_calls + int(fallback['candidate_method'] == 'sparse-lu'),
                     refinements=refinements + fallback['refinements'],
                     certificate_checks=checks, stopped_early=False,
                     cg_exit_info=info, seconds=time.monotonic()-start)
        return values, residual, event
    values, cert, order, ties = result
    event = dict(solver_id=SOLVER_ID, source='original', size=n, components=components,
                 dps=dps, certificate_backend=cert['backend'], candidate_method=method,
                 cg_calls=int(n > 128), cg_iterations=iterations, direct_calls=direct_calls,
                 cg_exit_info=info, stopped_early=stopped_early,
                 certificate_checks=checks, refinements=refinements,
                 returned_prefix_local=order[:5],
                 returned_scores_sha256=certified.digest_scores(values),
                 returned_residual_upper=float(cert['relative']), exact_tie_pairs=ties,
                 seconds=time.monotonic()-start)
    return np.asarray(values, dtype=object), float(cert['relative']), event


class WarmOriginalSolver:
    """One query's accepted scores, matched by global node ID across frontiers."""
    def __init__(self, *, events=None):
        self.events = certified.EVENTS if events is None else events
        self.reset()

    def reset(self):
        self._ids = np.empty(0, dtype=np.int64)
        self._scores = np.empty(0, dtype=np.float64)

    def __call__(self, w, e, alpha, beta, *, diagnostics, indices, targets, bridges):
        ids = np.asarray(indices, dtype=np.int64)
        if ids.shape != (len(e),) or np.any(np.diff(ids) <= 0):
            raise ValueError('Frontier IDs must be unique and increasing')
        warm_count = 0
        x0 = None
        if len(self._ids):
            _, old, new = np.intersect1d(self._ids, ids, assume_unique=True, return_indices=True)
            warm_count = len(new)
            if warm_count:
                # Conservative finite initialization for newly exposed nodes;
                # its value affects convergence only, not the accepted solution.
                x0 = np.full(len(ids), max(1., float(self._scores[old].max())))
                x0[new] = self._scores[old]
        values, residual, event = solve(w, e, alpha, beta, x0=x0)
        event.update(warm_reused_nodes=warm_count, warm_new_nodes=len(ids)-warm_count)
        self.events.append(event)
        # Cache ONLY the actual returned and certified values (including ties).
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            cached = np.asarray(values, dtype=np.float64).copy()
        if np.isfinite(cached).all() and np.all(cached > 0):
            self._ids, self._scores = ids.copy(), cached
        else:
            # MP output need not fit float64. An optional initial guess must
            # never turn an otherwise certified query into a later failure.
            self.reset()
        if diagnostics is not None:
            diagnostics.update(cg_calls=event['cg_calls'], cg_iterations=event['cg_iterations'],
                               direct_calls=event['direct_calls'], uncertified_cg=0)
        return values, residual


def binding(cfg):
    """Explicit opt-in: old configs and all non-original research paths stay put."""
    mode = cfg.get('original_solver', 'legacy')
    if mode not in ('legacy', MODE):
        raise ValueError('Unknown original solver mode')
    if mode == 'legacy':
        return {}
    if (cfg['source'] != 'original' or cfg.get('relation_model', 'original') != 'original'
            or cfg.get('admission_policy', 'original') != 'original' or 'relation_schema' in cfg):
        raise ValueError('Warm solver is bound to the original I-HyRDP path only')
    return dict(original_solver=mode, original_solver_id=SOLVER_ID)
