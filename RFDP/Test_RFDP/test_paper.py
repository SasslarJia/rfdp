#
import os
import sys
import unittest as untest
import numpy as np
#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Iput_RFDP.IputWhl import IputWhl
from MODEL.Paper_RFDP.PaperUtils import add_temporary_connections
from MODEL.Paper_RFDP.PaperUtils import build_paper_problem
from MODEL.Paper_RFDP.PaperSolvers import GmfptClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm
from MODEL.Paper_RFDP.PaperSolvers import HyRdpIterative
from MODEL.Paper_RFDP.PaperRanker import HyRdpPaper
from MODEL.Paper_RFDP.PaperIterative import IHyRdpPaper


class TestPaperRFDP(untest.TestCase):
    """Unit tests for the paper-accurate RFDP implementation."""

    def setUp(self):
        self.err = 1e-5
        self.asst = OperAsstNumpy()

    def test_problem_split(self):
        smth_w = np.array([
            [1.0, 0.8, 0.0, 0.0],
            [0.8, 1.0, 0.6, 0.0],
            [0.0, 0.6, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0]
        ])
        problem = build_paper_problem(smth_w, [0, 1, 2, 3], [0])
        self.assertEqual(problem.a_ids, [1])
        self.assertEqual(problem.b_ids, [2, 3])
        self.assertEqual(problem.unlabeled_ids, [1, 2, 3])

    def test_hyrdp_closed_vs_iterative(self):
        smth_w = np.array([
            [1.0, 0.8, 0.0, 0.0],
            [0.8, 1.0, 0.6, 0.2],
            [0.0, 0.6, 1.0, 0.7],
            [0.0, 0.2, 0.7, 1.0]
        ])
        problem = build_paper_problem(smth_w, [0, 1, 2, 3], [0])
        closed = HyRdpClosedForm().solve(problem, 5.0, 25.0)
        iterative = HyRdpIterative(max_terms=500, stable_rounds=0).solve(problem, 5.0, 25.0)
        self.assertTrue(np.allclose(closed, iterative, atol=5e-4, rtol=5e-4))
        self.assertEqual(np.argsort(closed).tolist(), np.argsort(iterative).tolist())

    def test_gmfpt_equation(self):
        smth_w = np.array([
            [1.0, 0.8, 0.0, 0.0],
            [0.8, 1.0, 0.6, 0.2],
            [0.0, 0.6, 1.0, 0.7],
            [0.0, 0.2, 0.7, 1.0]
        ])
        problem = build_paper_problem(smth_w, [0, 1, 2, 3], [0])
        mu = GmfptClosedForm().solve(problem)
        lhs = (np.eye(problem.p_uu.shape[0]) - problem.p_uu).dot(mu)
        self.assertTrue(np.allclose(lhs, np.ones_like(mu), atol=self.err, rtol=self.err))

    def test_temporary_connections(self):
        base_w = np.array([
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.8],
            [0.0, 0.0, 0.8, 1.0]
        ])
        full_w = np.array([
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.8, 0.3],
            [0.2, 0.8, 1.0, 0.8],
            [0.1, 0.3, 0.8, 1.0]
        ])
        work_w, chosen = add_temporary_connections(base_w, full_w, [0, 1], 1)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0], (2, 1))
        self.assertAlmostEqual(work_w[2, 1], full_w[2, 1], places=8)
        self.assertAlmostEqual(work_w[1, 2], full_w[1, 2], places=8)
        self.assertEqual(float(work_w[3, 1]), 0.0)

    def test_iterative_hyrdp_recovers_hidden_relevant(self):
        base_w = np.array([
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.85],
            [0.0, 0.0, 0.85, 1.0]
        ])
        full_w = np.array([
            [1.0, 0.9, 0.10, 0.75],
            [0.9, 1.0, 0.80, 0.20],
            [0.10, 0.80, 1.0, 0.85],
            [0.75, 0.20, 0.85, 1.0]
        ])
        mdl_iput = IputWhl(smth_w=base_w, fit_w=base_w, full_w=full_w, asst_rela=self.asst)
        base_ranker = HyRdpPaper(mdl_iput, solver=HyRdpClosedForm(), alpha=5.0, beta=25.0,
                                 incl_qry=False)
        base_result = base_ranker.solve_query([0], 3)
        self.assertEqual(base_result[:2], [0, 1])
        self.assertNotIn(2, base_result[:3])
        iterative_result = IHyRdpPaper(base_ranker, nq=3, lambda_step=1)([[0]])[0]
        self.assertEqual(iterative_result, [0, 1, 2])


# if __name__ == '__main__':
#     untest.main()
