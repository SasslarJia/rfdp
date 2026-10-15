"""Zheng 2021 Algorithm 1 with unrestricted dynamic two-hop subgraphs.

CSR storage and incremental boundary weights change computation, not the graph.
No fixed candidate pool, identity labels, or score-dependent experiment selection.
"""
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, cg, spsolve


def solve_hybrid_sparse(w_uu, e_u, alpha=5., beta=25., backend='auto', diagnostics=None):
    """Solve Eq. (13); CG changes the solver, never the target equation.

    Each connected component of U must have an anchored A node. Then the
    symmetric normalized Laplacian plus nonnegative anchoring is SPD and a
    nonsingular M-matrix. For r=b-Mx and eta=max(abs(r)/b), inverse positivity
    gives abs(x-x_exact) <= eta/(1-eta)*abs(x). The first six score intervals
    must be disjoint to certify the ordered first-five selection. Otherwise
    use the direct solver. The independent dense oracle remains authoritative.
    """
    w = sparse.csr_matrix(w_uu, dtype=np.float64)
    e = np.asarray(e_u, dtype=np.float64)
    h = np.asarray(w.sum(axis=1)).ravel()
    if not (alpha > 0 and beta > 0) or np.any(h <= 0) or np.any(e < 0):
        raise ValueError('Invalid hybrid degrees/parameters')
    inv_h = 1. / np.sqrt(h)
    normalized = sparse.diags(inv_h) @ w @ sparse.diags(inv_h)
    matrix = (sparse.diags(1. + alpha * e / (h + e)) - normalized).tocsr()
    rhs = np.where(e > 0, alpha, beta)
    if backend not in ('auto', 'direct', 'cg'):
        raise ValueError('Invalid solver backend')
    stats = dict(cg_calls=0, cg_iterations=0, direct_calls=0, uncertified_cg=0)
    scores = None
    if backend == 'cg' or (backend == 'auto' and len(e) > 128):
        def callback(x):
            stats['cg_iterations'] += 1
        stats['cg_calls'] = 1
        candidate, info = cg(matrix, rhs, rtol=1e-12, atol=0., maxiter=10000,
                             M=sparse.diags(1. / matrix.diagonal()), callback=callback)
        if info == 0 and np.isfinite(candidate).all() and np.all(candidate > 0):
            true_residual = matrix @ candidate - rhs
            eta = float(np.max(np.abs(true_residual) / rhs))
            relative = float(np.linalg.norm(true_residual) / np.linalg.norm(rhs))
            if eta < 1. and relative <= 1e-8:
                bound = eta / (1. - eta) * np.abs(candidate)
                idx = np.argsort(candidate, kind='stable')
                k = min(5, len(idx) - 1)
                # Every selected item is ordered, and fifth is below ALL others.
                left = candidate[idx[:k]] + bound[idx[:k]]
                suffix_min = np.minimum.accumulate((candidate[idx] - bound[idx])[::-1])[::-1]
                if np.all(left < suffix_min[1:k + 1]):
                    scores = candidate
        if scores is None:
            stats['uncertified_cg'] = 1
    if scores is None:
        stats['direct_calls'] = 1
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            try:
                scores = spsolve(matrix.tocsc(), rhs)
            except MatrixRankWarning as exc:
                raise np.linalg.LinAlgError(str(exc)) from exc
    residual = float(np.linalg.norm(matrix @ scores - rhs) / np.linalg.norm(rhs))
    if not np.isfinite(scores).all() or not np.isfinite(residual) or residual > 1e-8:
        raise np.linalg.LinAlgError('Hybrid relative residual ' + str(residual))
    if scores.min() < -1e-10 * max(1., abs(scores).max()):
        raise np.linalg.LinAlgError('Negative hybrid dissimilarity')
    if diagnostics is not None:
        diagnostics.update(stats)
    return np.maximum(scores, 0.), residual


class FullIHyRDP:
    """Dynamic I-HyRDP; nq counts seeds and is n by default.

    Full similarity is used only by the paper's cross-component bridge rule.
    Explicit numerical-tail policy implements the discussion following Eq.31.
    Programming errors, invalid inputs and timeouts never trigger a tail.
    """
    def __init__(self, graph, full_similarity, alpha=5., beta=25., step=5):
        self.w = sparse.csr_matrix(graph, dtype=np.float64)
        self.w.eliminate_zeros()
        self.n = self.w.shape[0]
        if self.w.shape != (self.n, self.n) or full_similarity.shape != self.w.shape:
            raise ValueError('Graph dimensions disagree')
        if not np.isfinite(self.w.data).all() or np.any(self.w.data < 0):
            raise ValueError('Invalid graph weights')
        delta = self.w - self.w.T
        if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
            raise ValueError('HyRDP requires symmetric weights')
        if not np.all(np.isfinite([alpha, beta])) or min(alpha, beta) <= 0 or step < 1:
            raise ValueError('Invalid parameters')
        self.full = full_similarity
        self.alpha, self.beta, self.step = float(alpha), float(beta), int(step)
        self.nc, self.components = connected_components(self.w, directed=False)
        self.component_sizes = np.bincount(self.components)
        self.full_sums = np.asarray(full_similarity.sum(axis=1, dtype=np.float64))
        if not np.isfinite(self.full_sums).all() or np.any(self.full_sums <= 0):
            raise ValueError('Invalid full-similarity degrees')
        self.bridge_cache = {}

    def bridges(self, active_components):
        """Eq.31: strongest row-normalized outside-to-entire-G edge per component."""
        key = tuple(sorted(active_components))
        if key in self.bridge_cache:
            return self.bridge_cache[key]
        g = np.flatnonzero(np.isin(self.components, key))
        choices = []
        for comp in range(self.nc):
            if comp in active_components:
                continue
            outside = np.flatnonzero(self.components == comp)
            best = None
            for start in range(0, len(outside), 128):
                ids = outside[start:start + 128]
                block = np.asarray(self.full[np.ix_(ids, g)], dtype=np.float64)
                block /= self.full_sums[ids, None]
                i, j = np.unravel_index(np.argmax(block), block.shape)
                val = float(block[i, j])
                candidate = (val, int(ids[i]), int(g[j]), comp)
                if best is None or val > best[0] or (val == best[0] and candidate[1:3] < best[1:3]):
                    best = candidate
            if best is not None and best[0] > 0:
                choices.append(best)
        choices.sort(key=lambda item: (-item[0], item[1], item[2]))
        result = [(i, j, float(self.full[i, j])) for _, i, j, _ in choices[:self.step]]
        self.bridge_cache[key] = result
        return result

    def rank(self, seeds, nq=None, input_similarity=None, numerical_tail=False,
             bridge=True, trace=None, progress=None):
        start = time.monotonic()
        seeds = list(map(int, seeds))
        if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or max(seeds) >= self.n:
            raise ValueError('Invalid query seeds')
        nq = self.n if nq is None else int(nq)
        if not len(seeds) <= nq <= self.n:
            raise ValueError('Invalid requested rank length')
        if numerical_tail and input_similarity is None:
            raise ValueError('Numerical tail requires the explicit original input metric')
        labeled = seeds.copy()
        mask = np.zeros(self.n, dtype=bool)
        mask[seeds] = True
        e = np.asarray(self.w[seeds].sum(axis=0)).ravel()
        active = set(self.components[seeds].tolist())
        diagnostics = dict(iterations=0, solver_calls=0, bridge_events=0, bridge_edges=0,
                           max_unlabeled=0, max_relative_residual=0., numerical_tail_count=0,
                           numerical_tail_reason=None, status='running', diffusion_nodes=len(seeds),
                           cg_calls=0, cg_iterations=0, direct_calls=0, uncertified_cg=0)
        while len(labeled) < nq:
            t = diagnostics['iterations']
            edges = []
            ew = e
            g_count = int(self.component_sizes[list(active)].sum())
            if len(labeled) == g_count and g_count < self.n:
                if not bridge:
                    diagnostics['status'] = 'component_exhausted_no_bridge'
                    break
                edges = self.bridges(active)
                if not edges:
                    raise RuntimeError('No positive bridge to remaining components')
                ew = e.copy()
                for i, j, value in edges:
                    assert not mask[i] and mask[j]
                    ew[i] += value
                diagnostics['bridge_events'] += 1
                diagnostics['bridge_edges'] += len(edges)
            a_mask = (ew > 0) & ~mask
            a = np.flatnonzero(a_mask)
            b_mask = np.asarray(self.w @ a_mask.astype(np.float64)).ravel() > 0
            b_mask &= ~mask & ~a_mask
            u = np.flatnonzero(a_mask | b_mask)
            if len(u) == 0:
                raise RuntimeError('Empty frontier before requested ranking is complete')
            diagnostics['max_unlabeled'] = max(diagnostics['max_unlabeled'], len(u))
            residual = 0.
            scores = None
            if len(u) <= self.step:
                chosen = u  # Algorithm 1 line 8; ascending ID within unordered block.
            else:
                try:
                    solver_stats = {}
                    scores, residual = solve_hybrid_sparse(self.w[u][:, u], ew[u], self.alpha,
                                                           self.beta, diagnostics=solver_stats)
                except np.linalg.LinAlgError as exc:
                    if not numerical_tail:
                        raise
                    diagnostics['numerical_tail_reason'] = str(exc)
                    diagnostics['status'] = 'numerical_input_tail'
                    break
                diagnostics['solver_calls'] += 1
                for key, value in solver_stats.items():
                    diagnostics[key] += value
                diagnostics['max_relative_residual'] = max(diagnostics['max_relative_residual'], residual)
                chosen = u[np.lexsort((u, scores))[:self.step]]
            chosen = chosen[:nq - len(labeled)]
            if trace is not None:
                trace(dict(iteration=t, labeled=labeled.copy(), direct=a.tolist(),
                           indirect=np.flatnonzero(b_mask).tolist(), selected=chosen.tolist(),
                           bridges=edges, unlabeled=u.tolist(),
                           scores=None if scores is None else scores.tolist(), residual=residual))
            mask[chosen] = True
            labeled.extend(chosen.tolist())
            # Only newly labeled rows are accumulated; original graph is immutable.
            for node in chosen:
                row_start, row_end = self.w.indptr[node:node + 2]
                e[self.w.indices[row_start:row_end]] += self.w.data[row_start:row_end]
            active.update(self.components[chosen].tolist())
            diagnostics['iterations'] += 1
            diagnostics['diffusion_nodes'] = len(labeled)
            if progress is not None and diagnostics['iterations'] % 100 == 0:
                progress(dict(diagnostics, elapsed_seconds=time.monotonic() - start))
        if len(labeled) < nq:
            if input_similarity is None:
                raise RuntimeError('Unfinished ranking without explicit original-distance tail')
            sim = np.asarray(input_similarity).reshape(-1)
            if sim.shape != (self.n,) or not np.isfinite(sim).all():
                raise ValueError('Invalid original-distance tail')
            remaining = np.flatnonzero(~mask)
            remaining = remaining[np.lexsort((remaining, -sim[remaining]))][:nq - len(labeled)]
            diagnostics['numerical_tail_count'] = len(remaining) if diagnostics['status'] == 'numerical_input_tail' else 0
            diagnostics['tail_count'] = len(remaining)
            labeled.extend(remaining.tolist())
        else:
            diagnostics['status'] = 'complete_diffusion'
            diagnostics['tail_count'] = 0
        diagnostics['elapsed_seconds'] = time.monotonic() - start
        if len(labeled) != nq or len(set(labeled)) != nq:
            raise RuntimeError('Incomplete or duplicated output')
        return np.asarray(labeled, dtype=np.int32), diagnostics
