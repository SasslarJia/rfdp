import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Iput_RFDP.IputWhl import IputWhl
from MODEL.Paper_RFDP.PaperRanker import HyRdpPaper
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm


class RankingSafetyTests(unittest.TestCase):
    def make_ranker(self, solver=None):
        graph = np.array([[1., .8, 0., 0.], [.8, 1., 0., 0.],
                          [0., 0., 1., .7], [0., 0., .7, 1.]])
        affinity = np.array([[1., .8, .9, .2], [.8, 1., .3, .1],
                             [.9, .3, 1., .7], [.2, .1, .7, 1.]])
        model = IputWhl(graph, graph, affinity, OperAsstNumpy())
        return HyRdpPaper(model, solver=solver or HyRdpClosedForm(),
                         fallback_similarity=lambda ids: np.array([1., .8, .5, .6]))

    def test_original_distance_tail_and_unreachable_count(self):
        ranker = self.make_ranker()
        result = ranker.solve_query([0], 4, sub_ids=[0, 1, 2, 3])
        self.assertEqual(result, [0, 1, 3, 2])
        self.assertEqual(ranker.diagnostics['dropped_unreachable_nodes'], 2)
        self.assertEqual(ranker.diagnostics['fallback_count'], 0)
        self.assertLess(ranker.diagnostics['max_relative_residual'], 1e-8)

    def test_failed_solver_is_recorded_and_uses_input_metric(self):
        class Failure:
            def solve(self, *args):
                raise np.linalg.LinAlgError('deliberate test failure')
        ranker = self.make_ranker(Failure())
        ranker._fallback_similarity = lambda ids: np.array([1., .1, .9, .8])
        self.assertEqual(ranker.solve_query([0], 4), [0, 2, 3, 1])
        self.assertEqual(ranker.diagnostics['fallback_count'], 1)
        self.assertEqual(ranker.diagnostics['fallback_queries'][0]['query_ids'], [0])


if __name__ == '__main__':
    unittest.main()
