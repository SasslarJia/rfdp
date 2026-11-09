"""Independent rectangular tensor-product RDP control, not hybrid HyRDP.

For the complete current frontier U and target set L, X has shape |U| x |L|.
With symmetric normalized graphs S_U and S_L and effective boundary Y=G_UL,

    J(X) = <X, X-S_U X S_L>/2 + mu*||X-Y||_F**2/2,
    X - diffusion*S_U X S_L = (1-diffusion)*Y.

This is a genuine four-index relationship model: both endpoints propagate.
Its quadratic fitting is the weighted-affinity RDP control, NOT Zheng's
point-to-set L1/L2 fitting. Uniform row means are aggregated only AFTER the
full relationship system has been solved; negative means are distance scores.
The .99 default is a fixed diffusion setting, not an empirically best setting.

No Kronecker matrix is built. Storage is O(|U||L|); each matrix application
costs O(nnz(G_UU)*|L| + |U|*nnz(G_LL)). No frontier/target truncation is allowed.
Input blocks must include the existing self edges and current temporary bridges.

Certification concerns the assembled float64 normalized operators. True
residuals are recomputed independently of CG's stopping flag, with conservative
dot-product roundoff allowances. The positive barrier bounds score errors even
when the RHS has zeros. Saving only row scores does NOT preserve enough data to
recheck the relationship equation: certify_saved requires the full saved X and
the three input blocks. No identity labels or learned parameters are used.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg


DEFINITION_VERSION = 'rectangular-tpg-rdp-v1-affinity-fit-negative-row-mean'


class QuaternaryCertificationError(np.linalg.LinAlgError):
    """A native solution exists but does not satisfy requested certification."""

    def __init__(self, message, diagnostics, scores=None, relations=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.scores = scores
        self.relations = relations


def _gamma(operations):
    z = float(operations) * np.finfo(np.float64).eps
    if z >= .5:
        raise ValueError('Problem dimensions exceed the rounding-bound model')
    return z / (1. - z)


def _up(value):
    return np.nextafter(value, np.inf)


def _norm_bounds(value):
    """Outward Frobenius bounds, including norm accumulation roundoff.

    Scaling puts one entry at magnitude one, so tiny squared terms cannot
    underflow the complete sum. The allowance includes scaling, dot products,
    square root and final rescaling, rather than treating a BLAS norm as exact.
    """
    scale = float(np.max(np.abs(value)))
    if scale == 0.:
        return 0., 0.
    if not np.isfinite(scale):
        return 0., np.inf
    estimate = float(scale * np.linalg.norm(np.asarray(value).ravel() / scale))
    allowance = _gamma(2 * np.size(value) + 16)
    lower = max(0., float(np.nextafter(estimate / (1. + allowance), -np.inf)))
    upper = float(_up(estimate / (1. - allowance)))
    return lower, upper


def _graph(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + ' must be real')
    w = sparse.csr_matrix(value, dtype=np.float64).copy()
    w.sum_duplicates()
    w.eliminate_zeros()
    w.sort_indices()
    if w.shape[0] != w.shape[1] or w.shape[0] == 0:
        raise ValueError(name + ' must be a nonempty square graph')
    if not np.isfinite(w.data).all() or np.any(w.data < 0):
        raise ValueError(name + ' must have finite nonnegative weights')
    delta = w - w.T
    # Exact symmetry is necessary for the advertised symmetric CG equation.
    if delta.nnz:
        raise ValueError(name + ' must be exactly symmetric')
    h = np.asarray(w.sum(axis=1), dtype=np.float64).ravel()
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError(name + ' must retain positive degrees/self edges')
    inv = 1. / np.sqrt(h)
    # Form DWD with a single multiplication per stored edge, symmetrically.
    coo = w.tocoo()
    values = coo.data * (inv[coo.row] * inv[coo.col])
    normalized = sparse.csr_matrix((values, (coo.row, coo.col)), shape=w.shape)
    return normalized, h


@dataclass
class _Problem:
    su: object
    sl: object
    hu: np.ndarray
    hl: np.ndarray
    y: np.ndarray
    diffusion: float

    @property
    def shape(self):
        return self.y.shape

    def propagate(self, x):
        return np.asarray(self.sl @ (self.su @ x).T).T

    def apply(self, vector):
        x = np.asarray(vector).reshape(self.shape)
        return (x - self.diffusion * self.propagate(x)).ravel()


def _prepare(w_uu, w_ul, w_ll, diffusion):
    diffusion = float(diffusion)
    if not np.isfinite(diffusion) or not 0. <= diffusion < 1.:
        raise ValueError('diffusion must lie in [0, 1)')
    su, hu = _graph(w_uu, 'G_UU')
    sl, hl = _graph(w_ll, 'G_LL')
    if np.iscomplexobj(w_ul):
        raise ValueError('G_UL must be real')
    y = (w_ul.toarray() if sparse.issparse(w_ul) else np.asarray(w_ul))
    y = np.array(y, dtype=np.float64, order='C', copy=True)
    if y.shape != (len(hu), len(hl)):
        raise ValueError('Boundary shape must be exactly |U| by |L|')
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('G_UL must have finite nonnegative effective boundary weights')
    return _Problem(su, sl, hu, hl, y, diffusion)


def _ids(value, n):
    if value is None:
        return np.arange(n, dtype=np.int64)
    value = np.asarray(value)
    if (value.shape != (n,) or not np.issubdtype(value.dtype, np.integer)
            or len(np.unique(value)) != n):
        raise ValueError('ids must contain one unique integer ID per frontier row')
    return value


def _certify(problem, x, ids, top_k):
    """Recompute the equation and bound its solution, not CG's recurrence."""
    n, m = problem.shape
    rhs = (1. - problem.diffusion) * problem.y
    product = problem.propagate(x)
    residual = x - problem.diffusion * product - rhs
    rhs_norm = float(np.linalg.norm(rhs))
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / rhs_norm if rhs_norm else residual_norm
    scores = -np.mean(x, axis=1)

    ku = int(np.diff(problem.su.indptr).max(initial=0))
    kl = int(np.diff(problem.sl.indptr).max(initial=0))
    rounding = _gamma(2 * ku + 2 * kl + 16)
    abs_product = (product if np.all(x >= 0) else problem.propagate(np.abs(x)))
    # Includes two sparse products, scalar products and the residual subtraction.
    residual_upper = _up(np.abs(residual) + rounding * (
        np.abs(x) + problem.diffusion * np.abs(abs_product) + np.abs(rhs)))
    rhs_norm_lower, _ = _norm_bounds(rhs)
    _, residual_norm_upper = _norm_bounds(residual_upper)
    exact_zero = bool(not np.any(problem.y) and not np.any(x))
    relative_upper = (0. if exact_zero else
                      float(_up(residual_norm_upper / rhs_norm_lower))
                      if rhs_norm_lower > 0. else np.inf)

    u, v = np.sqrt(problem.hu), np.sqrt(problem.hl)
    barrier = u[:, None] * v[None, :]
    barrier_product = (problem.su @ u)[:, None] * (problem.sl @ v)[None, :]
    mv = barrier - problem.diffusion * barrier_product
    mv_lower = np.nextafter(mv - rounding * (
        barrier + problem.diffusion * np.abs(barrier_product)), -np.inf)
    barrier_valid = bool(np.isfinite(mv_lower).all() and np.all(mv_lower > 0))
    if barrier_valid:
        factor = float(_up(np.max(residual_upper / mv_lower)))
        mean_v_upper = float(_up(np.mean(v) * (1. + _gamma(m + 2))))
        projection_rounding = _gamma(m + 2) * np.mean(np.abs(x), axis=1)
        score_error = _up(factor * u * mean_v_upper + projection_rounding)
        nonnegative = bool(np.all(x >= -factor * barrier))
    else:
        factor = np.inf
        score_error = np.full(n, np.inf)
        nonnegative = False

    order = np.lexsort((ids, scores))
    k = min(int(top_k), max(n - 1, 0))
    lower = np.nextafter(scores - score_error, -np.inf)
    upper = _up(scores + score_error)
    suffix_min = np.minimum.accumulate(lower[order][::-1])[::-1]
    ranked = bool(exact_zero or k == 0 or np.all(
        upper[order[:k]] < suffix_min[1:k + 1]))
    equation_ok = bool(barrier_valid and np.isfinite(relative_upper)
                       and relative_upper <= 1e-8 and nonnegative)
    stats = {
        'definition': DEFINITION_VERSION,
        'model_family': 'independent_rectangular_quaternary_rdp',
        'hybrid_fitting_preserved': False,
        'relation_shape': [n, m],
        'relation_variables': n * m,
        'diffusion': problem.diffusion,
        'relative_residual': relative_upper,
        'relative_residual_upper': relative_upper,
        'relative_residual_native': relative,
        'residual_frobenius_norm': residual_norm,
        'residual_frobenius_upper': residual_norm_upper,
        'rhs_frobenius_lower': rhs_norm_lower,
        'equation_certified': equation_ok,
        'ranking_certified': bool(equation_ok and ranked),
        'certified_top_k': int(top_k),
        'score_absolute_error_max': float(np.max(score_error)),
        'barrier_error_factor': factor,
        'roundoff_gamma': float(rounding),
        'score_order': 'ascending_negative_uniform_relation_mean',
        'certificate_kind': ('zero_rhs_exact_lexicographic' if exact_zero
                             else 'positive_barrier_with_dot_roundoff_bound'),
        'recertification_requires_full_relation_solution': True,
    }
    return scores, stats


def _check_tolerances(rtol, maxiter, top_k):
    if not np.isfinite(rtol) or not 0 < rtol <= 1e-8:
        raise ValueError('rtol must be positive and at most 1e-8')
    for value, name in ((maxiter, 'maxiter'), (top_k, 'top_k')):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(name + ' must be a positive integer')


def solve_quaternary_sparse(w_uu, w_ul, w_ll, *, diffusion=.99, ids=None,
                            rtol=1e-12, maxiter=10000, top_k=5,
                            diagnostics=None, return_relations=False,
                            require_rank_certificate=True):
    """Solve the full U x L relation problem and certify its scalar ranking.

    Returns (scores, relative_residual), or additionally X when
    return_relations=True. Disabling require_rank_certificate is an explicit
    native-precision audit mode; diagnostics then identify uncertified ranking.
    Equation residual and positivity requirements are never disabled.
    """
    _check_tolerances(rtol, maxiter, top_k)
    problem = _prepare(w_uu, w_ul, w_ll, diffusion)
    ids = _ids(ids, problem.shape[0])
    size = problem.y.size
    rhs = ((1. - problem.diffusion) * problem.y).ravel()
    stats = {'cg_calls': 0, 'cg_iterations': 0, 'direct_calls': 0,
             'matrix_free': True, 'truncated_relationships': False}
    if not np.any(rhs) or problem.diffusion == 0:
        x = problem.y.copy()
        info = 0
    else:
        operator = LinearOperator((size, size), matvec=problem.apply, dtype=np.float64)
        diagonal = (1. - problem.diffusion *
                    problem.su.diagonal()[:, None] * problem.sl.diagonal()[None, :]).ravel()
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise np.linalg.LinAlgError('Invalid tensor-product diagonal')
        preconditioner = LinearOperator((size, size), matvec=lambda z: z / diagonal,
                                       dtype=np.float64)

        def callback(_):
            stats['cg_iterations'] += 1

        stats['cg_calls'] = 1
        result, info = cg(operator, rhs, rtol=float(rtol), atol=0.,
                          maxiter=int(maxiter), M=preconditioner, callback=callback)
        x = result.reshape(problem.shape)
    if not np.isfinite(x).all():
        raise np.linalg.LinAlgError('Nonfinite quaternary relation solution')
    scores, certificate = _certify(problem, x, ids, top_k)
    stats.update(certificate)
    stats['cg_info'] = int(info)
    stats['native_precision_audit'] = not require_rank_certificate
    if diagnostics is not None:
        diagnostics.update(stats)
    if info != 0 or not stats['equation_certified']:
        raise QuaternaryCertificationError('Quaternary equation failed numerical acceptance',
                                            stats, scores, x)
    if require_rank_certificate and not stats['ranking_certified']:
        raise QuaternaryCertificationError('Quaternary top-k order is not certified',
                                            stats, scores, x)
    answer = (scores, float(stats['relative_residual']))
    return answer + (x,) if return_relations else answer


def certify_saved(w_uu, w_ul, w_ll, relations, scores, *, diffusion=.99,
                  ids=None, top_k=5, require_rank_certificate=True):
    """Recheck saved X, its row projection and the complete relationship equation.

    Row scores alone are deliberately rejected: many different relationship
    matrices have the same projection and different equation residuals.
    """
    if relations is None:
        raise ValueError('Full saved U x L relations are required to recheck the RDP equation')
    _check_tolerances(1e-12, 1, top_k)
    problem = _prepare(w_uu, w_ul, w_ll, diffusion)
    if np.iscomplexobj(relations) or np.iscomplexobj(scores):
        raise ValueError('Saved relations and scores must be real')
    x = np.asarray(relations, dtype=np.float64)
    saved_scores = np.asarray(scores, dtype=np.float64)
    if x.shape != problem.shape or not np.isfinite(x).all():
        raise ValueError('Saved relationship shape/values disagree')
    if saved_scores.shape != (problem.shape[0],) or not np.isfinite(saved_scores).all():
        raise ValueError('Saved score shape/values disagree')
    projected, stats = _certify(problem, x, _ids(ids, problem.shape[0]), top_k)
    # Serialization of float64 arrays must preserve the actual generated scores.
    if not np.array_equal(saved_scores, projected):
        raise ValueError('Saved scores do not equal the saved relation projection')
    stats['saved_projection_verified'] = True
    if not stats['equation_certified'] or (require_rank_certificate
                                         and not stats['ranking_certified']):
        raise QuaternaryCertificationError('Saved quaternary solution failed certification',
                                            stats, saved_scores, x)
    return stats
