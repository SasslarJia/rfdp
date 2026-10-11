#
import os
import sys
import unittest as untest

#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EVAL.ReIDEval import evaluate_reid_rankings


class TestReIDEval(untest.TestCase):

    def test_market1501_style_metrics(self):
        rankings = [
            [0, 3, 4, 5, 6, 7],
            [1, 6, 7, 3, 4, 5],
        ]
        query_ids = [0, 1]
        query_pids = [1, 2]
        query_camids = [1, 1]
        gallery_pids = [1, 1, 2, 2, -1]
        gallery_camids = [1, 2, 1, 2, 3]
        metrics = evaluate_reid_rankings(
            rankings=rankings,
            query_ids=query_ids,
            query_pids=query_pids,
            query_camids=query_camids,
            gallery_pids=gallery_pids,
            gallery_camids=gallery_camids,
            gallery_offset=2,
            max_rank=5,
        )
        self.assertAlmostEqual(metrics['rank-1'], 0.5)
        self.assertAlmostEqual(metrics['rank-5'], 1.0)
        self.assertAlmostEqual(metrics['mAP'], 0.75)
        self.assertEqual(metrics['valid_queries'], 2)


    def test_truncated_ranking_miss_counts_as_zero(self):
        metrics = evaluate_reid_rankings(
            rankings=[[2], [2]], query_ids=[0, 1],
            query_pids=[1, 2], query_camids=[1, 1],
            gallery_pids=[1, 2], gallery_camids=[2, 2],
            gallery_offset=2, max_rank=2,
        )
        self.assertEqual(metrics['valid_queries'], 2)
        self.assertAlmostEqual(metrics['mAP'], 0.5)
        self.assertAlmostEqual(metrics['rank-1'], 0.5)
        self.assertEqual(metrics['cmc'], [0.5, 0.5])

    def test_empty_ranking_with_gallery_positive_counts_as_zero(self):
        metrics = evaluate_reid_rankings(
            rankings=[[]], query_ids=[0],
            query_pids=[1], query_camids=[1],
            gallery_pids=[1, 2], gallery_camids=[2, 2],
            gallery_offset=1, max_rank=2,
        )
        self.assertEqual(metrics['valid_queries'], 1)
        self.assertEqual(metrics['mAP'], 0.0)
        self.assertEqual(metrics['cmc'], [0.0, 0.0])

    def test_only_filtered_entries_counts_as_zero(self):
        metrics = evaluate_reid_rankings(
            rankings=[[1, 2]], query_ids=[0],
            query_pids=[1], query_camids=[1],
            gallery_pids=[1, -1, 1], gallery_camids=[1, 2, 2],
            gallery_offset=1, max_rank=3,
        )
        self.assertEqual(metrics['valid_queries'], 1)
        self.assertEqual(metrics['mAP'], 0.0)
        self.assertEqual(metrics['cmc'], [0.0, 0.0, 0.0])

    def test_query_without_cross_camera_positive_is_still_skipped(self):
        metrics = evaluate_reid_rankings(
            rankings=[[2, 3], [2, 3]], query_ids=[0, 1],
            query_pids=[1, 2], query_camids=[1, 1],
            gallery_pids=[1, 2], gallery_camids=[2, 1],
            gallery_offset=2, max_rank=2,
        )
        self.assertEqual(metrics['valid_queries'], 1)
        self.assertEqual(metrics['mAP'], 1.0)
        self.assertEqual(metrics['cmc'], [1.0, 1.0])


if __name__ == '__main__':
    untest.main()
