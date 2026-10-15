"""Independent dense oracle and behavioral tests for Algorithm 1."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP, solve_hybrid_sparse


def dense_oracle(w, full, query, step=2, alpha=5., beta=25.):
    """Literal, slow dense transcription, independent of production helpers."""
    n = len(w)
    labeled = [query]
    count, labels = connected_components(csr_matrix(w), directed=False)
    traces = []
    p = full / full.sum(axis=1)[:, None]
    while len(labeled) < n:
        work = w.copy()
        g = np.flatnonzero(np.isin(labels, labels[labeled]))
        edges = []
        if len(labeled) == len(g):
            options = []
            for c in range(count):
                if c in labels[labeled]:
                    continue
                ids = np.flatnonzero(labels == c)
                block = p[np.ix_(ids, g)]
                i, j = np.unravel_index(np.argmax(block), block.shape)
                options.append((float(block[i, j]), int(ids[i]), int(g[j])))
            for _, i, j in sorted(options, key=lambda v: (-v[0], v[1], v[2]))[:step]:
                work[i, j] = full[i, j]
                work[j, i] = full[j, i]
                edges.append((i, j))
        a = sorted(set(np.flatnonzero(work[:, labeled].sum(axis=1) > 0)) - set(labeled))
        b = sorted(set(np.flatnonzero(work[:, a].sum(axis=1) > 0)) - set(labeled) - set(a))
        u = sorted(a + b)
        if len(u) <= step:
            chosen = u
        else:
            wu = work[np.ix_(u, u)]
            h = wu.sum(axis=1)
            e = work[np.ix_(u, labeled)].sum(axis=1)
            m = np.eye(len(u)) - wu / np.sqrt(h[:, None] * h[None, :])
            m += np.diag(alpha * e / (h + e))
            scores = np.linalg.solve(m, np.where(e > 0, alpha, beta))
            chosen = np.asarray(u)[np.lexsort((u, scores))[:step]].tolist()
        traces.append(dict(labeled=labeled.copy(), direct=a, indirect=b, selected=chosen, bridges=edges))
        labeled.extend(chosen)
    return labeled, traces


class TestFullIHyRDP(unittest.TestCase):
    def test_cg_same_equation_and_selection(self):
        rng = np.random.default_rng(241)
        a = rng.random((151, 151)); w = (a + a.T) / 2
        e = rng.random(151); e[1::2] = 0
        direct, _ = solve_hybrid_sparse(w, e, backend='direct')
        stats = {}
        iterative, _ = solve_hybrid_sparse(w, e, backend='cg', diagnostics=stats)
        np.testing.assert_allclose(iterative, direct, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(np.argsort(iterative)[:5], np.argsort(direct)[:5])
        self.assertEqual(stats['cg_calls'], 1)
        self.assertEqual(stats['direct_calls'], 0)

    def test_cg_ties_use_direct_solver(self):
        w = np.ones((150, 150)); e = np.ones(150)
        stats = {}
        solve_hybrid_sparse(w, e, backend='cg', diagnostics=stats)
        self.assertEqual(stats['uncertified_cg'], 1)
        self.assertEqual(stats['direct_calls'], 1)

    def test_cg_false_convergence_retries_direct(self):
        rng = np.random.default_rng(27)
        a = rng.random((150,150)); w = (a+a.T)/2; e = rng.random(150)
        expected,_ = solve_hybrid_sparse(w,e,backend='direct')
        stats = {}
        with patch('MODEL.Paper_RFDP.PaperFullIterative.cg', return_value=(expected*.999,0)):
            actual,_ = solve_hybrid_sparse(w,e,diagnostics=stats)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(stats['direct_calls'],1)

    def test_sparse_equation_matches_dense(self):
        rng = np.random.default_rng(912)
        for n in [4, 13, 35]:
            a = rng.random((n, n)); w = (a + a.T) / 2
            e = rng.random(n); e[1::2] = 0
            scores, residual = solve_hybrid_sparse(w, e)
            h = w.sum(axis=1)
            m = np.diag(1 + 5 * e / (h + e)) - w / np.sqrt(h[:, None] * h[None, :])
            expected = np.linalg.solve(m, np.where(e > 0, 5., 25.))
            np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)
            self.assertLess(residual, 1e-10)

    def test_stepwise_dense_oracle_across_components(self):
        rng = np.random.default_rng(183)
        for step in [1, 2, 5]:
            a = rng.uniform(.1, 1., (17, 17)); full = (a + a.T) / 2
            w = np.eye(17)
            for lo, hi in [(0, 5), (5, 12), (12, 17)]:
                for i in range(lo, hi - 1):
                    w[i, i + 1] = w[i + 1, i] = full[i, i + 1]
            expected, ref_trace = dense_oracle(w, full, 0, step)
            trace = []
            order, diag = FullIHyRDP(w, full, step=step).rank([0], trace=trace.append)
            self.assertEqual(order.tolist(), expected)
            for a, b in zip(trace, ref_trace):
                for key in ['labeled', 'direct', 'indirect', 'selected']:
                    self.assertEqual(a[key], b[key])
                self.assertEqual([(i, j) for i, j, _ in a['bridges']], b['bridges'])
            self.assertEqual(diag['tail_count'], 0)

    def test_bridge_only_after_component_exhausted(self):
        w = np.eye(8)
        for i in range(5):
            w[i, i + 1] = w[i + 1, i] = .8
        full = np.full((8, 8), .3); np.fill_diagonal(full, 1)
        full[7, 0] = full[0, 7] = .9
        trace = []
        FullIHyRDP(w, full, step=1).rank([0], trace=trace.append)
        first = next(t for t in trace if t['bridges'])
        self.assertEqual(set(first['labeled']), set(range(6)))
        self.assertIn((7, 0), [(i, j) for i, j, _ in first['bridges']])

    def test_full_universe_beyond_500(self):
        n = 510
        w = np.eye(n) + .4 * np.eye(n, k=1) + .4 * np.eye(n, k=-1)
        order, diag = FullIHyRDP(w, np.ones((n, n)), step=5).rank([0])
        np.testing.assert_array_equal(order, np.arange(n))
        self.assertEqual(diag['diffusion_nodes'], n)
        self.assertEqual(diag['status'], 'complete_diffusion')

    def test_programming_errors_are_never_tail(self):
        w = np.ones((5, 5))
        with patch('MODEL.Paper_RFDP.PaperFullIterative.solve_hybrid_sparse', side_effect=ValueError('bad input')):
            with self.assertRaises(ValueError):
                FullIHyRDP(w, w, step=1).rank([0], numerical_tail=True, input_similarity=np.arange(5.))

    def test_explicit_numerical_failure_policy(self):
        w = np.ones((5, 5))
        with patch('MODEL.Paper_RFDP.PaperFullIterative.solve_hybrid_sparse', side_effect=np.linalg.LinAlgError('test failure')):
            with self.assertRaises(np.linalg.LinAlgError):
                FullIHyRDP(w, w, step=1).rank([0])
            order, diag = FullIHyRDP(w, w, step=1).rank([0], numerical_tail=True, input_similarity=np.arange(5.))
            self.assertEqual(order.tolist(), [0, 4, 3, 2, 1])
            self.assertEqual(diag['numerical_tail_count'], 4)
            self.assertEqual(diag['status'], 'numerical_input_tail')

    def test_no_bridge_is_explicit_ablation(self):
        w = np.eye(4)
        order, diag = FullIHyRDP(w, np.ones((4, 4))).rank([0], bridge=False, input_similarity=np.arange(4.))
        self.assertEqual(order.tolist(), [0, 3, 2, 1])
        self.assertEqual(diag['status'], 'component_exhausted_no_bridge')

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            FullIHyRDP(np.eye(2), np.ones((2, 2)), step=0)


if __name__ == '__main__':
    unittest.main()
