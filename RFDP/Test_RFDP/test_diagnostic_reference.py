"""Preserved baseline formats used by the bounded mechanism review."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.continuous_fitting.diagnose import load_query_reference


class ReferenceFormatTests(unittest.TestCase):
    def test_original_and_metric_export_agree(self):
        rows = [dict(query_id=0, AP=.8, rank1=1.),
                dict(query_id=1, AP=.4, rank1=0.)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'baseline.json'
            for key in ('queries', 'per_query'):
                with self.subTest(key=key):
                    text = json.dumps({key: rows})
                    path.write_text(text)
                    self.assertEqual(load_query_reference(path), rows)
                    self.assertEqual(path.read_text(), text)

    def test_misordered_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'baseline.json'
            path.write_text(json.dumps({'per_query': [dict(query_id=1, AP=.8, rank1=1.)]}))
            with self.assertRaisesRegex(AssertionError, 'Reference query order'):
                load_query_reference(path)


if __name__ == '__main__':
    unittest.main()
