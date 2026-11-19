"""Numerical and storage oracles for the official MSMT17 execution path."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse

from experiments.msmt17.graph import PackedSymmetric, transform, write_distances
from experiments.msmt17.rankings import checked_row
from experiments.msmt17.resources import DISK_MARGIN_BYTES, GIB, require_disk_space
from experiments.reid_adaptation import graph
from Test_RFDP.test_adaptation_graph import frozen_features, historical_oracle


class FullInputTests(unittest.TestCase):
    def test_disk_admission_uses_planned_bytes_on_the_output_volume(self):
        rank_bytes = 11659 * 82161 * 4 + 4096
        needed = rank_bytes + 2 * GIB
        disk = SimpleNamespace(total=110 * GIB, free=29 * GIB)
        with patch('experiments.msmt17.resources.shutil.disk_usage', return_value=disk):
            require_disk_space('/data', needed)
            disk.free = needed + DISK_MARGIN_BYTES
            require_disk_space('/data', needed)
            disk.free -= 1
            with self.assertRaises(OSError):
                require_disk_space('/data', needed)

    def test_resume_checks_a_five_digit_query_before_removing_its_duplicate(self):
        from experiments.msmt17.diffusion import commit_rows, recover_row, saved_row_hash
        from experiments.msmt17.extract import save_json
        from experiments.msmt17.models import digest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"queries").mkdir()
            (root/"accepted").mkdir()
            q = 10003
            table = np.lib.format.open_memmap(root/"rankings.npy", mode="w+",
                                              dtype=np.int32, shape=(q+1, 3))
            row = np.array([11002, 11000, 11001], dtype=np.int32)
            duplicate = root/"queries"/f"{q}.rankings.npy"
            np.save(duplicate, row)
            checked = root/"accepted"/f"{q}.json"
            save_json(checked, dict(query=q, signature="example"))
            save_json(root/"queries"/f"{q}.json", dict(signature="example",
                      readback_sha256=digest(checked), rankings_sha256=saved_row_hash(row)))
            self.assertTrue(recover_row(root, table, q, "example"))
            commit_rows(table, root, [q])
            self.assertFalse(duplicate.exists())
            self.assertTrue(recover_row(root, table, q, "example"))
            table[q, 0] = 0
            with self.assertRaises(ValueError):
                recover_row(root, table, q, "example")

    def test_packed_graph_exact_against_historical_operators(self):
        for n, width in ((37, 6), (131, 64), (17, 1)):
            with self.subTest(n=n, width=width), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                features = frozen_features(n, dimensions=32)
                features[1] = features[2]
                kd = ks = min(20, n-2)
                expected_a, expected_g = historical_oracle(features, kd, ks, .4, width)
                triangle = np.lib.format.open_memmap(root/"triangle.npy", mode="w+", dtype=np.float32,
                                                    shape=(n*(n+1)//2,))
                packed = PackedSymmetric(triangle, n)
                write_distances(features, packed, width, lambda *_: None)
                scales = graph._distance_sums(packed, kd, width)
                sums, body_sha = transform(packed, scales, kd, .4, width, lambda *_: None)
                np.testing.assert_array_equal(packed.symmetric_rows(0, n), expected_a)
                np.testing.assert_array_equal(sums, expected_a.sum(axis=1, dtype=np.float64))
                self.assertEqual(body_sha, hashlib.sha256(expected_a.tobytes(order="C")).hexdigest())
                graph._write_mutual_graph(packed, root/"graph.npz", ks, width)
                np.testing.assert_array_equal(sparse.load_npz(root/"graph.npz").toarray(), expected_g)
                subset = np.ix_([3, 7], [2, 3, 1])
                np.testing.assert_array_equal(packed[subset], expected_a[subset])
                self.assertEqual(packed[2, 1], expected_a[2, 1])

    def test_full_row_includes_same_camera_negatives_excludes_junk(self):
        pids = np.array([2, -1, 2, 2, 8, 8])
        cams = np.array([0, 1, 1, 2, 0, 1])
        row = np.array([500, 501, 504, 502, 503, 505], dtype=np.int32)
        r = checked_row(row, 2, 0, pids, cams, 500, 6)
        self.assertEqual((r["first_positive"], r["last_positive"]), (2, 3))
        self.assertAlmostEqual(r["AP"], 7/12)
        self.assertAlmostEqual(r["INP"], 2/3)
        with self.assertRaises(ValueError):
            checked_row(row[[0, 1, 2, 3, 4, 4]], 2, 0, pids, cams, 500, 6)
