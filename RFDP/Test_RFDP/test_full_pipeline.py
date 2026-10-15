"""Portable CLI smoke using synthetic input; never contributes report metrics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'RFDP/experiments/full_reid'
READY = importlib.util.find_spec('sklearn') is not None and (ENTRY / 're_ranking_ranklist.py').exists() and (ENTRY / 'research/classic/ecn/ecn.py').exists()


@unittest.skipUnless(READY, 'Install requirements-cpu.txt and run fetch_controls.py first')
class TestFullPipeline(unittest.TestCase):
    def test_five_methods_resume_and_integrity(self):
        with tempfile.TemporaryDirectory(prefix='rfdp-test-') as directory:
            base = Path(directory)
            rng = np.random.default_rng(7)
            features = rng.normal(size=(46, 12)).astype(np.float32)
            mat = base / 'toy.mat'
            savemat(mat, dict(query_f=features[:4], gallery_f=features[4:],
                             query_label=np.arange(4), query_cam=np.ones(4, dtype=int),
                             gallery_label=np.arange(42) % 4, gallery_cam=2 + np.arange(42) % 3))
            out = base / 'results'
            command = [sys.executable, str(ENTRY / 'run_full.py'), '--mat-path', str(mat),
                       '--cache-dir', str(base / 'cache'), '--output-dir', str(out),
                       '--depth', '25', '--workers', '2']
            def run(args, success=True):
                result = subprocess.run(args, capture_output=True, text=True, timeout=120)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            run(command)
            self.assertEqual(json.loads((out / 'COMPLETE.json').read_text())['status'], 'verified')
            paths = list((out / 'ihyrdp').glob('*')) + list((out / 'csw').glob('*'))
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            run(command)
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            self.assertIn('different configuration', run(command + ['--rho', '.6'], False).stderr)
            rankfile = out / 'csw.rankings.npy'
            saved = rankfile.read_bytes()
            rank = np.load(rankfile)
            rank[0, 0] = rank[0, 1]
            np.save(rankfile, rank)
            verify = [sys.executable, str(ENTRY / 'verify_full.py'), '--mat-path', str(mat),
                      '--output-dir', str(out)]
            run(verify, False)
            self.assertFalse((out / 'COMPLETE.json').exists())
            rankfile.write_bytes(saved)
            run(verify)


if __name__ == '__main__':
    unittest.main()
