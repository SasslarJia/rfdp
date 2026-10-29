"""Export the versioned solver with explicit machine paths; never start a job."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--mat-file', required=True, type=Path)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.runtime_root.resolve()
    target = root / 'internal-nonlinear-smoothing-v1-20261028'
    if target.exists():
        raise FileExistsError('Export into a new location; do not overwrite a running deployment')
    original_inputs = json.loads((here / 'model/population-adaptive-v2/analysis-inputs.json').read_text())
    mapping = {
        '/root/autodl-tmp/rfdp-graph-routes-20261026': str(root),
        '/root/rfdp-stage2-20261020/repo': str(args.repo_root.resolve()),
        '/root/rfdp-stage2-20261020/early-osnet-official-v1/inputs/cache': str(args.cache_dir.resolve()),
        original_inputs['mat']: str(args.mat_file.resolve()),
        '/root/miniconda3/bin/python': args.python,
    }
    # Preflight the helper files before creating the export.
    for source in (here / 'runtime').iterdir():
        existing = root / source.name
        if existing.exists() and existing.read_bytes() != source.read_bytes():
            raise FileExistsError(str(existing))
    shutil.copytree(here / 'model', target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in target.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.json', '.sh', '.md', '.txt'}:
            text = path.read_text()
            for before, after in sorted(mapping.items(), key=lambda item: -len(item[0])):
                text = text.replace(before, after)
            path.write_text(text)
    for source in (here / 'runtime').iterdir():
        shutil.copy2(source, root / source.name)
    # Re-lock only the two production packages. Earlier mathematical-check locks
    # refer to their original server evidence and are retained as provenance.
    for name in ['population-v1', 'population-adaptive-v2']:
        folder = target / name
        lock_path = folder / 'source-lock.json'
        lock = json.loads(lock_path.read_text())
        for group in ['files', 'dependencies']:
            for name in lock[group]:
                lock[group][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    print(target)
    print('Source exported only. Supply baseline/qualification reference inputs before launching.')


if __name__ == '__main__':
    main()
