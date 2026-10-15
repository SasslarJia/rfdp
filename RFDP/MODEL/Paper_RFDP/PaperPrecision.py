"""Iterative refinement of the same hybrid equation after a failed float64 solve."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from . import PaperFullIterative as mod
old = mod.solve_hybrid_sparse
EVENTS = []

def refined(w_uu, e_u, alpha=5.0, beta=25.0, backend='auto', diagnostics=None):
    try:
        return old(w_uu, e_u, alpha, beta, backend, diagnostics)
    except np.linalg.LinAlgError as exc:
        w = sparse.csr_matrix(w_uu, dtype=np.float64)
        e = np.asarray(e_u, dtype=np.float64)
        h = np.asarray(w.sum(axis=1)).ravel()
        dh = sparse.diags(1 / np.sqrt(h))
        m = (sparse.diags(1 + alpha * e / (h + e)) - dh @ w @ dh).tocsr()
        b = np.where(e > 0, alpha, beta)
        failure = str(exc)
        lu = splu(m.tocsc())
        ml = m.astype(np.longdouble)
        bl = b.astype(np.longdouble)
        x = lu.solve(b).astype(np.longdouble)
        for i in range(12):
            r = bl - ml @ x
            res = float(np.linalg.norm(r) / np.linalg.norm(bl))
            if res <= 1e-08 and np.isfinite(x).all() and np.all(x >= 0):
                if diagnostics is not None:
                    diagnostics.update(cg_calls=0, cg_iterations=0, direct_calls=1, uncertified_cg=0)
                EVENTS.append(dict(size=len(b), original_error=failure, refinement_steps=i, relative_residual=res))
                return (x, res)
            x += lu.solve(np.asarray(r, dtype=np.float64)).astype(np.longdouble)
        raise np.linalg.LinAlgError('Extended precision refinement failed ' + str(res))

def install():
    mod.solve_hybrid_sparse = refined
