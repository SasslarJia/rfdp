"""Run the frozen uniform integration fixture in an isolated source workspace."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / 'RFDP/experiments/full_reid/build_feature_cache.py').is_file():
        raise ValueError('Repository cache builder missing')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, source / path.name)
    # Copy source only, excluding images, feature files, caches, logs and Git.
    copied_repo = root / 'repo'
    for path in (repo / 'RFDP').rglob('*.py'):
        if path.name.startswith('._') or path.is_symlink():
            continue
        target = copied_repo / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', str(source / 'synthetic_uniform_integration.py'),
                    '--repo', str(copied_repo), '--output-dir', str(root / 'results')],
                   env=environment, check=True)


if __name__ == '__main__':
    main()
