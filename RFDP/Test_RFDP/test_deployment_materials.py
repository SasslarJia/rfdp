"""Deployment inputs remain external without replacing executable sources."""
from pathlib import Path
import tempfile
import unittest

from experiments.internal_nonlinear.prepare_deployment import export_model


class DeploymentMaterialsTests(unittest.TestCase):
    def test_code_and_external_materials_are_combined_without_run_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, materials, target = [root / name for name in ('code', 'materials', 'export')]
            source.mkdir()
            (materials / 'population').mkdir(parents=True)
            (source / 'solver.py').write_text('value = 1\n')
            (source / 'result.json').write_text('{"generated": true}\n')
            manifest = b'{"frozen": "unchanged"}\n'
            (materials / 'population/source-lock.json').write_bytes(manifest)
            (materials / 'population/protocol.md').write_text('Frozen protocol\n')
            export_model(source, materials, target)
            self.assertEqual((target / 'solver.py').read_bytes(), (source / 'solver.py').read_bytes())
            self.assertEqual((target / 'population/source-lock.json').read_bytes(), manifest)
            self.assertTrue((target / 'population/protocol.md').is_file())
            self.assertFalse((target / 'result.json').exists())

    def test_invalid_materials_fail_before_creating_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, materials, target = [root / name for name in ('code', 'materials', 'export')]
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                export_model(source, materials, target)
            self.assertFalse(target.exists())
            materials.mkdir()
            (materials / 'solver.py').write_text('value = 2\n')
            with self.assertRaises(ValueError):
                export_model(source, materials, target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
