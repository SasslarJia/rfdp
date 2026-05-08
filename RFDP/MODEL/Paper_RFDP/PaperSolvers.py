#
import numpy as np
from MODEL.Paper_RFDP.PaperUtils import safe_divide
from MODEL.Paper_RFDP.PaperUtils import solve_linear


class HyRdpClosedForm(object):
    """Exact HyRDP closed-form solver based on Eq. (13)-(14)."""

    def solve(self, problem, alpha: float, beta: float) -> np.ndarray:
        if problem.empty_unlabeled():
            return np.zeros(0, dtype=float)
        h_u = problem.h_u
        d_u = problem.d_u
        e_u = problem.e_u
        w_uu = problem.w_uu
        a_num = len(problem.a_ids)
        b_num = len(problem.b_ids)
        rhs = np.concatenate([np.full(a_num, alpha, dtype=float),
                              np.full(b_num, beta, dtype=float)])
        lambdas = h_u + alpha * h_u * safe_divide(e_u, d_u)
        inv_sqrt_h = safe_divide(np.ones_like(h_u), np.sqrt(np.maximum(h_u, 1e-12)))
        coef_mat = np.diag(lambdas) - w_uu
        coef_mat = inv_sqrt_h[:, None] * coef_mat * inv_sqrt_h[None, :]
        return solve_linear(coef_mat, rhs)


class HyRdpIterative(object):
    """HyRDP iterative solver based on Eq. (15)-(16)."""

    def __init__(self, max_terms: int = 200, stable_rounds: int = 0, tol: float = 1e-10):
        self._max_terms = max_terms
        self._stable_rounds = stable_rounds
        self._tol = tol

    def solve(self, problem, alpha: float, beta: float) -> np.ndarray:
        if problem.empty_unlabeled():
            return np.zeros(0, dtype=float)
        h_u = problem.h_u
        d_u = problem.d_u
        e_u = problem.e_u
        w_uu = problem.w_uu
        a_num = len(problem.a_ids)
        b_num = len(problem.b_ids)
        rhs = np.concatenate([np.full(a_num, alpha, dtype=float),
                              np.full(b_num, beta, dtype=float)])
        lambdas = h_u + alpha * h_u * safe_divide(e_u, d_u)
        sqrt_h = np.sqrt(np.maximum(h_u, 1e-12))
        inv_lambda = safe_divide(np.ones_like(lambdas), lambdas)
        power = inv_lambda[:, None] * w_uu
        y_vec = inv_lambda * sqrt_h * rhs
        x_vec = np.array(y_vec, copy=True)
        prev_rank = None
        stbl_num = 0
        for _ in range(self._max_terms):
            psi = sqrt_h * x_vec
            nxt_x = power.dot(x_vec) + y_vec
            if np.max(np.abs(nxt_x - x_vec)) <= self._tol:
                x_vec = nxt_x
                break
            cur_rank = tuple(np.argsort(psi).tolist())
            if prev_rank is not None and cur_rank == prev_rank:
                stbl_num += 1
                if self._stable_rounds > 0 and stbl_num >= self._stable_rounds:
                    x_vec = nxt_x
                    break
            else:
                stbl_num = 0
                prev_rank = cur_rank
            x_vec = nxt_x
        return np.sqrt(np.maximum(h_u, 1e-12)) * x_vec


class GmfptClosedForm(object):
    """Closed-form GMFPT solver based on Eq. (20)."""

    def solve(self, problem) -> np.ndarray:
        if problem.empty_unlabeled():
            return np.zeros(0, dtype=float)
        coef_mat = np.eye(problem.p_uu.shape[0], dtype=float) - problem.p_uu
        rhs = np.ones(problem.p_uu.shape[0], dtype=float)
        return solve_linear(coef_mat, rhs)
