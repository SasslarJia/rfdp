"""Export the versioned solver with explicit machine paths; never start a job."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def export_model(source, materials, target):
    """Combine code with external frozen manifests, excluding generated outputs."""
    if not materials.is_dir():
        raise FileNotFoundError(f'Missing external deployment materials: {materials}')
    artifacts = [p for p in materials.rglob('*') if p.is_file()]
    if any(p.suffix not in {'.json', '.md'} for p in artifacts):
        raise ValueError('Deployment materials must contain only JSON manifests and Markdown protocols')
    target.mkdir(parents=True, exist_ok=False)
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.sh', '.txt'}:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for path in artifacts:
        destination = target / path.relative_to(materials)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def verify_frozen_repository(repo_root, references):
    """A cleaned working tree is not a replacement for a frozen run's inputs."""
    prefix = references['repo'].rstrip('/') + '/'
    mismatches = []
    for original, expected in references['locked_files'].items():
        if not original.startswith(prefix):
            continue
        relative = original[len(prefix):]
        path = repo_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(
            'Frozen repository references differ: ' + ', '.join(mismatches)
            + '. Use a separate checkout of 5e5ec194ef362c9bf179d73f8bd6ea0a10d71601 '
              'for --repo-root; do not rewrite the historical reference hashes.'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--mat-file', required=True, type=Path)
    parser.add_argument('--python', required=True)
    parser.add_argument('--materials-root', required=True, type=Path,
                        help='External research-materials root, preserving repository-relative paths')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.runtime_root.resolve()
    target = root / 'internal-nonlinear-smoothing-v1-20261028'
    if target.exists():
        raise FileExistsError('Export into a new location; do not overwrite a running deployment')
    materials = args.materials_root.resolve() / 'RFDP/experiments/internal_nonlinear/model'
    references = json.loads((materials / 'population-adaptive-v2/references.json').read_text())
    verify_frozen_repository(args.repo_root.resolve(), references)
    original_inputs = json.loads((materials / 'population-adaptive-v2/analysis-inputs.json').read_text())
    mapping = {
        '/root/autodl-tmp/rfdp-graph-routes-20261026': str(root),
        '/root/rfdp-stage2-20261020/repo': str(args.repo_root.resolve()),
        '/root/rfdp-stage2-20261020/early-osnet-official-v1/inputs/cache': str(args.cache_dir.resolve()),
        original_inputs['mat']: str(args.mat_file.resolve()),
        '/root/miniconda3/bin/python': args.python,
    }
    # Preflight the helper files before creating the export.
    for source in (here / 'runtime').glob('*.py'):
        existing = root / source.name
        if existing.exists() and existing.read_bytes() != source.read_bytes():
            raise FileExistsError(str(existing))
    export_model(here / 'model', materials, target)
    for path in target.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.json', '.sh', '.md', '.txt'}:
            text = path.read_text()
            for before, after in sorted(mapping.items(), key=lambda item: -len(item[0])):
                text = text.replace(before, after)
            path.write_text(text)
    for source in (here / 'runtime').glob('*.py'):
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
