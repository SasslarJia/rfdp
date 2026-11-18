import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.msmt17 import extract


class TestMsmtExtraction(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_official_order_pid_zero_camera_ends(self):
        raw = b"0007/0007_03_15_001.jpg 0\n0001/0001_11_01_002.jpg 19\n"
        rows = extract.parse_list(raw, "query", 2)
        self.assertEqual([r["pid"] for r in rows], [0, 19])
        self.assertEqual([r["camid"] for r in rows], [14, 0])
        self.assertEqual([r["camera_original"] for r in rows], [15, 1])
        self.assertEqual(rows[0]["path"], "0007/0007_03_15_001.jpg")

    def test_official_count_and_path_validation(self):
        for raw in (b"../0001_00_01_1.jpg 2", b"/0001_00_01_1.jpg 2",
                    b"0001_00_00_1.jpg 2", b"0001_00_16_1.jpg 2", b"0001_00_01_1.jpg -1"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                extract.parse_list(raw, "gallery", 1)
        with self.assertRaises(ValueError):
            extract.parse_list(b"0001_00_01_1.jpg 2", "query", 11659)

    def test_release_root(self):
        actual = self.root / "MSMT17_V2"
        (actual / "mask_test_v2").mkdir(parents=True)
        (actual / "list_query.txt").touch()
        self.assertEqual(extract.data_root(self.root, "V2"), actual)
        self.assertEqual(extract.data_root(actual, "V2"), actual)
        with self.assertRaises(ValueError):
            extract.data_root(actual, "V1")

    def test_population_order_and_duplicates(self):
        rows = [dict(path=str(i), pid=i % 3060, camid=i % 15,
                     split="query" if i < 11659 else "gallery") for i in range(93820)]
        extract.validate_rows(rows)
        rows[11659]["split"] = "query"
        with self.assertRaises(ValueError):
            extract.validate_rows(rows)
        rows[11659].update(split="gallery", path="0")
        with self.assertRaises(ValueError):
            extract.validate_rows(rows)

    def test_features_preserve_rows_and_fsync_resume(self):
        signature = {"model": "fixture"}
        features = np.eye(5, dtype=np.float32)
        with patch.object(extract, "CHUNK", 2):
            arr, state, start = extract.open_progress(self.root, features.shape, signature)
            self.assertEqual(start, 0)
            arr[:3] = features[:3]
            extract.flush_chunk(self.root, arr, state, 0, 2, 0.1)
            arr, state, start = extract.open_progress(self.root, features.shape, signature)
            self.assertEqual(start, 2)
            for end in (4, 5):
                arr[start:end] = features[start:end]
                extract.flush_chunk(self.root, arr, state, start, end, 0.1)
                start = end
            saved = self.root / "features.npy"
            sha, error = extract.audit_array(saved, features.shape)
            self.assertEqual(sha, hashlib.sha256(saved.read_bytes()).hexdigest())
            self.assertEqual(error, 0)
            np.testing.assert_array_equal(np.load(saved), features)
            with self.assertRaises(ValueError):
                extract.open_progress(self.root, features.shape, {"model": "different"})
            arr[1] = arr[2]
            arr.flush()
            with self.assertRaises(ValueError):
                extract.open_progress(self.root, features.shape, signature)

    def test_vector_acceptance(self):
        for block in (np.array([[np.nan]], np.float32), np.array([[0]], np.float32),
                      np.array([[1]], np.float64), np.array([[0.9]], np.float32)):
            with self.subTest(block=block), self.assertRaises(ValueError):
                extract.check_vectors(block)

    def test_image_bytes_and_order_on_resume(self):
        import io
        try:
            from PIL import Image
            import torchvision
        except ImportError:
            self.skipTest("GPU image frontend dependencies are unavailable")
        buf = io.BytesIO()
        Image.new("RGB", (32, 16), (80, 127, 150)).save(buf, format="JPEG")
        path = self.root/"fixture.jpg"
        path.write_bytes(buf.getvalue())
        row = dict(path=path.name, sha256=extract.digest(path), camid=14)
        images = extract.Images(self.root, [{}, row], "transreid", start=1)
        row_index, pixels, camera = images[0]
        self.assertEqual((row_index, len(images), tuple(pixels.shape), camera), (1, 1, (3, 256, 128), 14))
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed at official row 1"):
            images[0]

    def test_actual_batch_writer_partial_tail_and_receiving_verifier(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        rows = list(range(5))
        with ExitStack() as context:
            patcher = context.enter_context
            patcher(patch.object(extract, "TOTAL", 5))
            patcher(patch.object(extract, "CHUNK", 4))
            patcher(patch.object(extract, "SPECS", {"osnet": (2, 2, [], [])}))
            patcher(patch.object(extract, "runtime", return_value={"device": "unit fixture"}))
            patcher(patch.object(extract, "Images", side_effect=lambda root, records, name, start:
                          [(i, torch.tensor([1., float(i)]), 14) for i in records[start:]]))
            patcher(patch.object(extract, "load_model", return_value=torch.nn.Identity()))
            patcher(patch.object(torch.Tensor, "cuda", lambda tensor, **kw: tensor))
            patcher(patch.object(torch.cuda, "reset_peak_memory_stats"))
            meta = {"manifest_sha256": "fixture"}
            result = extract.extract_one("osnet", self.root, self.root, rows, meta, self.root, 0)
            folder = self.root / "osnet"
            self.assertEqual(result, extract.verify_one(folder, "osnet", meta)["feature_sha256"])
            expected = torch.nn.functional.normalize(torch.tensor([[1., float(i)] for i in rows]), dim=1)
            np.testing.assert_array_equal(np.load(folder/"features.npy"), expected.numpy())
            self.assertEqual([c[0] for c in extract.read_json(folder/"progress.json")["chunks"]], [4, 5])
            with patch.object(extract, "load_model", side_effect=AssertionError("repeat forward")):
                self.assertEqual(result, extract.extract_one("osnet", self.root, self.root, rows, meta, self.root, 0))

    def test_handoff_binds_receipts_and_shared_inputs(self):
        meta = {"counts": {"query": 2, "gallery": 1}, "version": "V2"}
        shapes = {"osnet": (2, 128, [], []), "transreid": (3, 16, [], [])}
        with patch.object(extract, "TOTAL", 3), patch.object(extract, "SPECS", shapes), \
             patch.object(extract, "pinned_assets", return_value={}), \
             patch.object(extract, "software_hashes", return_value={}):
            for name, (dim, batch, _, _) in shapes.items():
                folder = self.root / name
                folder.mkdir()
                vectors = np.zeros((3, dim), dtype=np.float32)
                vectors[:, 0] = 1
                np.save(folder / "features.npy", vectors)
                signature = dict(backbone=name, dimensions=dim, batch=batch, dataset=meta,
                                 count=3, assets={}, source={})
                receipt = dict(signature=signature, feature_sha256=extract.digest(folder/"features.npy"))
                extract.save_json(folder/"COMPLETE.json", receipt)
                self.assertEqual(extract.verify_one(folder, name, meta), receipt)
                with self.assertRaises(ValueError):
                    extract.verify_one(folder, name, {**meta, "version": "V1"})
                vectors[:, :2] = [0, 1]
                np.save(folder/"features.npy", vectors)
                with self.assertRaises(ValueError):
                    extract.verify_one(folder, name, meta)


if __name__ == "__main__":
    unittest.main()
